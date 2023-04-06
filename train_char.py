from functools import partial
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import jax
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from flax.jax_utils import replicate, unreplicate
from orbax.checkpoint import (
    Checkpointer,
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointHandler,
)
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter

from cleanrlhf.model import GPT, GPTConfig, GPTConfigPreset, param_decay_mask

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991


@dataclass(frozen=True)
class CosineDecayScheduleConfig:
    init_value: float = 0
    peak_value: float = 0.001
    warmup_steps: int = 100
    decay_steps: int = 5000
    end_value: float = 0.0001


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanrlhf"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # logistics args
    data_dir: str = "./data/shakespeare_char"
    """the data_dir to use"""
    ckpt_dir: Optional[str] = None
    """the checkpoint directory to use"""

    # training args
    local_batch_size: int = 64
    """the local batch size to use"""
    batch_size: Optional[int] = None
    """TO BE UPDATED IN RUNTIME: the effective batch size (calculated as local_batch_size * jax.device_count() * gradient_accumulation_steps)"""
    learning_rate: CosineDecayScheduleConfig = field(default_factory=CosineDecayScheduleConfig)
    """the learning rate schedule"""
    betas: Tuple[int] = (0.9, 0.99)
    """the betas for the Adam optimizer"""
    weight_decay: float = 0.1  # only applied on matmul weights
    """the weight decay to use"""
    grad_norm_clip: float = 1.0
    """the gradient norm clipping to use"""
    max_iters: int = 5000
    """the maximum number of iterations to run"""
    block_size: int = 256
    """the block size to use"""
    vocab_size: Optional[int] = None
    """TO BE UPDATED IN RUNTIME: the vocab size of the dataset"""
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes; *8 simulates 8 GPUs
    """the number of gradient accumulation steps to use"""
    gpt: GPTConfigPreset = GPTConfig(n_layer=6, n_head=6, embd_dim=384, use_bias=False)
    """the model's hyperparameters"""

    # distributed args
    distributed: bool = False
    """if toggled, this experiment will be distributed"""
    world_size: int = 1
    """TO BE UPDATED IN RUNTIME: the number of processes to use"""
    local_rank: int = 0
    """TO BE UPDATED IN RUNTIME: the local rank of the process"""
    global_devices: Optional[List[int]] = None
    """TO BE UPDATED IN RUNTIME: the global devices to use"""
    local_devices: Optional[List[int]] = None
    """TO BE UPDATED IN RUNTIME: the local devices to use"""


def init_model(key: jax.random.PRNGKey, args: Args) -> TrainState:
    gpt = GPT(
        config=args.gpt,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
    )
    x = jax.random.randint(
        key, (args.local_batch_size, args.block_size), minval=0, maxval=args.vocab_size
    )  # B, T; or local_batch_size, sequence_length
    y = jax.random.randint(
        key, (args.local_batch_size, args.block_size), minval=0, maxval=args.vocab_size
    )  # B; or local_batch_size, sequence_length
    gpt_params = gpt.init(key, x, y, deterministic=True)
    print(gpt.tabulate(key, x, y, deterministic=True))
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_norm_clip),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=optax.warmup_cosine_decay_schedule(**asdict(args.learning_rate)),
            b1=args.betas[0],
            b2=args.betas[1],
            weight_decay=args.weight_decay,
            mask=param_decay_mask(gpt_params),
        ),
    )
    # optax.MultiSteps handles schedule properly (https://optax.readthedocs.io/en/latest/gradient_accumulation.html#interaction-of-optax-multistep-with-schedules)
    # it works properly with `clip_by_global_norm` by only clip the grad norm of the accumulated gradients.
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=args.gradient_accumulation_steps)
    return TrainState.create(
        apply_fn=gpt.apply,
        params=gpt_params,
        tx=optimizer,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # load data's vocab_size
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {vocab_size} (inside {meta_path})")
    if vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        vocab_size = 50304
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # jax distributed
    if args.distributed:
        jax.distributed.initialize()
    local_devices = jax.local_devices()
    global_devices = jax.devices()

    # fill in the args
    args.vocab_size = vocab_size
    args.batch_size = args.local_batch_size * jax.local_device_count() * args.gradient_accumulation_steps * args.world_size
    if args.ckpt_dir is None:
        args.ckpt_dir = f"runs/{run_name}/models"
    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.global_devices = [str(item) for item in global_devices]
    args.local_devices = [str(item) for item in local_devices]

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=asdict(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    pprint(args)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, params_key, dropout_key = jax.random.split(key, 3)
    dropout_key = jax.random.fold_in(dropout_key, jax.process_index())
    dropout_keys = jax.random.split(dropout_key, jax.local_device_count())

    # poor man's data loader
    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = np.random.randint(len(data) - args.block_size, size=(args.local_batch_size,))
        xs, ys = [], []
        for _ in range(jax.local_device_count()):
            x = np.stack([(data[i : i + args.block_size]).astype(np.int64) for i in ix])
            y = np.stack([(data[i + 1 : i + 1 + args.block_size]).astype(np.int64) for i in ix])
            xs.append(x)
            ys.append(y)
        xs = jax.device_put_sharded(xs, local_devices)
        ys = jax.device_put_sharded(ys, local_devices)
        return xs, ys

    # initialize model
    train_state = init_model(params_key, args)
    train_state = replicate(train_state)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    mngr = CheckpointManager(
        args.ckpt_dir,
        checkpointers={"train_state": Checkpointer(PyTreeCheckpointHandler())},
        options=CheckpointManagerOptions(max_to_keep=1, save_interval_steps=20),
        metadata={"args": tyro.to_yaml(args)},
    )

    # setup the training loop
    iter_num = 0
    iter_time = 0.0
    iter_dt = 0.0
    X, Y = get_batch("train")  # fetch the very first batch

    @partial(jax.pmap, axis_name='batch', devices=global_devices)
    def update(train_state: TrainState, x, y, dropout_keys):
        dropout_keys = jax.random.fold_in(dropout_keys, train_state.step)
        (loss, logits), grads = jax.value_and_grad(train_state.apply_fn, has_aux=True)(
            train_state.params, x, y, deterministic=False, rngs={"dropout": dropout_keys}
        )
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, (loss, logits)

    while True:
        for micro_step in range(args.gradient_accumulation_steps):
            train_state, (loss, logits) = update(train_state, X, Y, dropout_keys)
            # print(f"iter {iter_num}, micro_step {micro_step}: loss {loss.item()}")
            X, Y = get_batch("train")
            # immediately async prefetch next batch while model is doing the forward pass on the GPU

        writer.add_scalar("train/loss", loss[-1].item(), iter_num)
        writer.add_scalar("charts/learning_rate", train_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(), iter_num)
        if iter_num % 10 == 0:
            print(f"iter_dt {iter_dt * 1000:.2f}ms; iter {iter_num}: train loss {loss[-1].item():.5f}")

        mngr.save(iter_num, {"train_state": unreplicate(train_state)})
        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if iter_num >= args.max_iters:
            break
