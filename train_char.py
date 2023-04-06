import os
import pickle
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrlhf.model import GPT, MODELS_PRESET, GPTConfig, param_decay_mask

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991


@dataclass
class CosineDecayScheduleConfig:
    init_value: float = 0
    peak_value: float = 0.001
    warmup_steps: int = 100
    decay_steps: int = 5000
    end_value: float = 0.0001


@dataclass
class TrainerConfig:
    batch_size: Optional[int] = None
    local_batch_size = 64
    learning_rate: CosineDecayScheduleConfig = field(default_factory=CosineDecayScheduleConfig)
    betas: Tuple[int] = (0.9, 0.99)
    weight_decay: float = 0.1  # only applied on matmul weights
    grad_norm_clip: float = 1.0
    num_workers: int = 0
    max_iters: int = 10000
    block_size: int = 256
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes; *8 simulates 8 GPUs


@dataclass
class Args:
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

    data_dir: str = "./data/shakespeare_char"
    """the data_dir to use"""
    model_type: str = "gpt-small"
    """the type of model"""

    gpt: GPTConfig = field(default_factory=GPTConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.model_type:
        assert args.model_type in MODELS_PRESET, f"model_type {args.model_type} not found in {MODELS_PRESET.keys()}"
        args.gpt = MODELS_PRESET[args.model_type]
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, params_key, actor_key, critic_key = jax.random.split(key, 4)
    block_size = args.trainer.block_size

    # poor man's data loader
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {vocab_size} (inside {meta_path})")
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = np.random.randint(len(data) - block_size, size=(args.trainer.local_batch_size,))
        x = np.stack([(data[i : i + block_size]).astype(np.int64) for i in ix])
        y = np.stack([(data[i + 1 : i + 1 + block_size]).astype(np.int64) for i in ix])
        x, y = jax.device_put((x, y))
        return x, y

    # initialize model
    gpt = GPT(
        config=args.gpt,
        vocab_size=vocab_size,
        block_size=block_size,
    )
    x = jax.random.randint(
        key, (args.trainer.local_batch_size, block_size), minval=0, maxval=vocab_size
    )  # B, T; or local_batch_size, sequence_length
    y = jax.random.randint(
        key, (args.trainer.local_batch_size, block_size), minval=0, maxval=vocab_size
    )  # B; or local_batch_size, sequence_length
    gpt_params = gpt.init(params_key, x, y, deterministic=True)
    print(gpt.tabulate(key, x, y, deterministic=True))
    gpt_loss, (gpt_y) = gpt.apply(gpt_params, x, y, deterministic=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.trainer.grad_norm_clip),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=optax.warmup_cosine_decay_schedule(**asdict(args.trainer.learning_rate)),
            b1=args.trainer.betas[0],
            b2=args.trainer.betas[1],
            weight_decay=args.trainer.weight_decay,
            mask=param_decay_mask(gpt_params),
        ),
    )
    # optax.MultiSteps handles schedule properly (https://optax.readthedocs.io/en/latest/gradient_accumulation.html#interaction-of-optax-multistep-with-schedules)
    # it works properly with `clip_by_global_norm` by only clip the grad norm of the accumulated gradients.
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=args.trainer.gradient_accumulation_steps)
    train_state = TrainState.create(
        apply_fn=gpt.apply,
        params=gpt_params,
        tx=optimizer,
    )

    # setup the training loop
    iter_num = 0
    iter_time = 0.0
    iter_dt = 0.0
    X, Y = get_batch("train")  # fetch the very first batch

    @jax.jit
    def update(train_state: TrainState, x, y, key):
        dropout_key = jax.random.fold_in(key, train_state.step)
        (loss, logits), grads = jax.value_and_grad(train_state.apply_fn, has_aux=True)(
            train_state.params, x, y, deterministic=False, rngs={"dropout": dropout_key}
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, (loss, logits)

    def generate(train_state, key, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        B, T = input_tokens.shape
        padding = jnp.zeros((B, max(block_size - T, max_new_tokens)), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=-1)
        indexes = jnp.arange(T, T + max_new_tokens)
        start_indexes = (indexes - block_size).clip(min=0)
        # print("B, T, max_new_tokens, tokens", B, T, max_new_tokens, tokens.shape)
        # tokens index -> tokens None
        def scan_f(tokens, item):
            (i, start_i) = item
            # l: x y
            # t: a b - -
            # i: 0 1 2 3
            step_key = jax.random.fold_in(key, i)
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.args.block_size else idx[:, -self.args.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = train_state.apply_fn(
                train_state.params,
                jax.lax.dynamic_slice(tokens, (0, start_i), (B, block_size)),
                targets=None,
                deterministic=False,
                rngs={"dropout": step_key},
            )  # TODO: (0, 0) is going to be problematic
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, i - 1, :] / temperature
            # optionally crop the logits to only the top k options
            # sample from the distribution
            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)
                # logits = jnp.where(logits < v[:, -1:], float('-inf'), logits)
            # append sampled index to the running sequence and continue
            tokens = tokens.at[:, i].set(next_token)

            return tokens, None

        tokens, _ = jax.lax.scan(scan_f, tokens, (indexes, start_indexes))

        return tokens

    while True:
        for micro_step in range(args.trainer.gradient_accumulation_steps):
            train_state, (loss, logits) = update(train_state, X, Y, key)
            print(f"iter {iter_num}, micro_step {micro_step}: loss {loss.item()}")
            X, Y = get_batch("train")
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        writer.add_scalar("train/loss", loss.item(), iter_num)
        writer.add_scalar("charts/learning_rate", train_state.opt_state[2][1].hyperparams["learning_rate"].item(), iter_num)
        if iter_num % 10 == 0:
            print(f"iter_dt {iter_dt * 1000:.2f}ms; iter {iter_num}: train loss {loss.item():.5f}")

        if iter_num % 500 == 0:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        {
                            "args": asdict(args),
                            "params": train_state.params,
                        }
                    )
                )
            print(f"model saved to {model_path}")

        #     eval_len = 500
        #     # sample from the model...
        #     context = "O God, O God!"
        #     key, subkey = jax.random.split(key, 2)
        #     x = np.array([train_dataset.stoi[s] for s in context], dtype=np.int32)[None, ...]
        #     y = generate(train_state, subkey, x, eval_len, temperature=0.9, top_k=5)[0][: len(x[0]) + eval_len]
        #     completion = "".join([train_dataset.itos[int(i)] for i in y])
        #     print(len(completion), completion)

        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if iter_num >= args.trainer.max_iters:
            break
