import argparse
import os
import pickle
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool

import hyperstate
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training.train_state import TrainState
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cleanrlhf.model import GPT, GPTConfig, MODELS_PRESET

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
    learning_rate = 5e-4
    betas = (0.9, 0.95)
    weight_decay = 0.1  # only applied on matmul weights
    grad_norm_clip = 1.0
    num_workers = 0
    max_iters = 2000


@dataclass
class Config:
    gpt: GPTConfig
    trainer: TrainerConfig


class SortDataset:
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {"train", "test"}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000  # ...

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):

        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unique digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = "test" if h % 4 == 0 else "train"  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok

        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[: self.length - 1] = -1
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

    # set up dataset
    train_dataset = SortDataset("train")
    test_dataset = SortDataset("test")
    x, y = train_dataset[0]
    for a, b in zip(x, y):
        print(int(a), int(b))
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
        (loss, logits), grads = jax.value_and_grad(train_state.apply_fn, has_aux=True)(train_state.params, x, y, deterministic=False, rngs={'dropout': dropout_key})
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
                train_state.params, jax.lax.dynamic_slice(tokens, (0, start_i), (B, block_size)), targets=None, deterministic=False, rngs={'dropout': step_key}
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

        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if iter_num >= config.trainer.max_iters:
            break

    n = train_dataset.length # naugy direct access shrug
    inp = jnp.array([[0, 0, 2, 1, 0, 1]], dtype=jnp.int32)
    cat = generate(train_state, key, inp, n)
    sol = jnp.sort(inp)
    sol_candidate = cat[:, n:]
    print('input sequence  :', inp)
    print('predicted sorted:', sol_candidate)
    print('gt sort         :', sol)
    print('matches         :', bool((sol == sol_candidate).all()))

    def eval_split(split, max_batches, key):
        dataset = {"train": train_dataset, "test": test_dataset}[split]
        n = train_dataset.length  # naugy direct access shrug
        results = []
        mistakes_printed_already = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            # isolate the input pattern alone
            x, y = np.array(x), np.array(y)
            inp = x[:, :n]
            sol = y[:, -n:]
            key, subkey = jax.random.split(key, 2)
            # let the model sample the rest of the sequence
            cat = generate(train_state, subkey, inp, n, top_k=1)  # using greedy argmax, not sampling
            sol_candidate = cat[:, n:]  # isolate the filled in sequence
            # compare the predicted sequence to the true sequence
            correct = (sol == sol_candidate).all(1)  # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(len(x)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 3:  # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print(f"GPT claims that {inp[i]} sorted is {sol_candidate[i]} but gt is {sol[i]}")
            if max_batches is not None and b + 1 >= max_batches:
                break
        rt = jnp.array(results, dtype=jnp.float32)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100 * rt.mean()))
        return rt.sum()

    # run a lot of examples from both train and test through the model and verify the output correctness
    train_score = eval_split("train", max_batches=50, key=key)
    test_score = eval_split("test", max_batches=50, key=key)
