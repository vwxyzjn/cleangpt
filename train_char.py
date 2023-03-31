import argparse
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from distutils.util import strtobool

import hyperstate
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training.train_state import TrainState
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cleanrlhf.model import GPT, MODELS_PRESET, GPTConfig

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanrlhf",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--model-type", type=str, default="gpt-mini",
        help="the type of model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--hps", nargs="+", help="Override hyperparameter value")
    args = parser.parse_args()
    # fmt: on
    return args


@dataclass
class TrainerConfig:
    batch_size = 64
    learning_rate = 0.0005
    betas = (0.9, 0.95)
    weight_decay = 0.1  # only applied on matmul weights
    grad_norm_clip = 1.0
    num_workers = 0
    max_iters = 10000
    # gradient_accumulation_steps: int = 5    # used to simulate larger batch sizes


@dataclass
class Config:
    gpt: GPTConfig
    trainer: TrainerConfig


# -----------------------------------------------------------------------------


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, block_size, data):
        self.block_size = block_size

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    args = parse_args()
    config = hyperstate.load(Config, file=args.config, overrides=args.hps)
    gpt_config = config.gpt
    if args.model_type:
        assert args.model_type in MODELS_PRESET, f"model_type {args.model_type} not found in {MODELS_PRESET.keys()}"
        gpt_config = MODELS_PRESET[args.model_type]
    config.gpt = gpt_config
    print(config)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
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

    # construct the training dataset
    if not os.path.exists("tinyshakespeare.txt"):
        urllib.request.urlretrieve(
            "https://github.com/karpathy/char-rnn/raw/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
            "tinyshakespeare.txt",
        )
    text = open("tinyshakespeare.txt").read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(block_size=128, data=text)

    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_block_size()

    # initialize model
    gpt = GPT(
        config=config.gpt,
        vocab_size=vocab_size,
        block_size=block_size,
    )
    x = jax.random.randint(key, (1, block_size), minval=0, maxval=vocab_size)  # B, T; or batch_size, sequence_length
    y = jax.random.randint(key, (1, block_size), minval=0, maxval=vocab_size)  # B; or batch_size, sequence_length
    gpt_params = gpt.init(params_key, x, y, deterministic=True)
    print(gpt.tabulate(key, x, y, deterministic=True))
    gpt_loss, (gpt_y) = gpt.apply(gpt_params, x, y, deterministic=True)
    train_state = TrainState.create(
        apply_fn=gpt.apply,
        params=gpt_params,
        tx=optax.chain(
            optax.clip_by_global_norm(config.trainer.grad_norm_clip),
            optax.inject_hyperparams(optax.adamw)(
                config.trainer.learning_rate,
                b1=config.trainer.betas[0],
                b2=config.trainer.betas[1],
            ),
        ),
    )

    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        # pin_memory=True,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
    )

    # setup the training loop
    iter_num = 0
    iter_time = 0.0
    iter_dt = 0.0
    data_iter = iter(train_loader)

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
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
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
        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = [np.array(t) for t in batch]
        x, y = batch
        # raise

        train_state, (loss, logits) = update(train_state, x, y, key)

        writer.add_scalar("train/loss", loss.item(), iter_num)
        writer.add_scalar("charts/learning_rate", train_state.opt_state[1].hyperparams["learning_rate"].item(), iter_num)
        if iter_num % 100 == 0:
            print(f"iter_dt {iter_dt * 1000:.2f}ms; iter {iter_num}: train loss {loss.item():.5f}")

        if iter_num % 500 == 0:
            eval_len = 500

            # sample from the model...
            context = "O God, O God!"
            key, subkey = jax.random.split(key, 2)
            x = np.array([train_dataset.stoi[s] for s in context], dtype=np.int32)[None, ...]
            y = generate(train_state, subkey, x, eval_len, temperature=0.9, top_k=5)[0][: len(x[0]) + eval_len]
            completion = "".join([train_dataset.itos[int(i)] for i in y])
            print(len(completion), completion)
            # # save the latest model
            # print("saving model")
            # ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            # torch.save(model.state_dict(), ckpt_path)

        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if iter_num >= config.trainer.max_iters:
            break
