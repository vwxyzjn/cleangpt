import argparse
import os
import pickle
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool

import hyperstate
import jax
import numpy as np
import optax
import torch
from flax.training.train_state import TrainState
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cleanrlhf.model import GPT, GPTConfig

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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--hps", nargs="+", help="Override hyperparameter value")
    args = parser.parse_args()
    # fmt: on
    return args


@dataclass
class TrainerConfig:
    max_iters = None
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
            monitor_gym=True,
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
        c=config.gpt,
        vocab_size=vocab_size,
        block_size=block_size,
    )
    x = jax.random.randint(key, (1, block_size), minval=0, maxval=vocab_size)  # B, T; or batch_size, sequence_length
    y = jax.random.randint(key, (1, block_size), minval=0, maxval=vocab_size)  # B; or batch_size, sequence_length
    gpt_params = gpt.init(params_key, x, y, key)
    gpt_y = gpt.apply(gpt_params, x, y, key)
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

    # def loss(params, x, y, key):
    #     loss, logits = train_state.apply_fn(params, x, x, key)
    #     return loss

    @jax.jit
    def update(train_state: TrainState, x, y, key):
        (loss, (logits, key)), grads = jax.value_and_grad(train_state.apply_fn, has_aux=True)(train_state.params, x, y, key)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, (loss, logits, key)

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

        train_state, (loss, logits, key) = update(train_state, x, y, key)

        if iter_num % 100 == 0:
            print(f"iter_dt {iter_dt * 1000:.2f}ms; iter {iter_num}: train loss {loss.item():.5f}")

        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if iter_num >= config.trainer.max_iters:
            break

    # # create a Trainer object
    # from mingpt.trainer import Trainer

    # train_config = Trainer.get_default_config()
    # train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    # train_config.max_iters = 2000
    # train_config.num_workers = 0
    # trainer = Trainer(train_config, model, train_dataset)

    # def batch_end_callback(trainer):
    #     if trainer.iter_num % 100 == 0:
    #         print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    # trainer.set_callback('on_batch_end', batch_end_callback)

    # trainer.run()

    # # now let's perform some evaluation
    # model.eval()

    # def eval_split(trainer, split, max_batches):
    #     dataset = {'train':train_dataset, 'test':test_dataset}[split]
    #     n = train_dataset.length # naugy direct access shrug
    #     results = []
    #     mistakes_printed_already = 0
    #     loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    #     for b, (x, y) in enumerate(loader):
    #         x = x.to(trainer.device)
    #         y = y.to(trainer.device)
    #         # isolate the input pattern alone
    #         inp = x[:, :n]
    #         sol = y[:, -n:]
    #         # let the model sample the rest of the sequence
    #         cat = model.generate(inp, n, do_sample=False) # using greedy argmax, not sampling
    #         sol_candidate = cat[:, n:] # isolate the filled in sequence
    #         # compare the predicted sequence to the true sequence
    #         correct = (sol == sol_candidate).all(1).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
    #         for i in range(x.size(0)):
    #             results.append(int(correct[i]))
    #             if not correct[i] and mistakes_printed_already < 3: # only print up to 5 mistakes to get a sense
    #                 mistakes_printed_already += 1
    #                 print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
    #         if max_batches is not None and b+1 >= max_batches:
    #             break
    #     rt = torch.tensor(results, dtype=torch.float)
    #     print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    #     return rt.sum()

    # # run a lot of examples from both train and test through the model and verify the output correctness
    # with torch.no_grad():
    #     train_score = eval_split(trainer, 'train', max_batches=50)
    #     test_score  = eval_split(trainer, 'test',  max_batches=50)

    # # let's run a random given sequence through the model as well
    # n = train_dataset.length # naugy direct access shrug
    # inp = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to(trainer.device)
    # assert inp[0].nelement() == n
    # with torch.no_grad():
    #     cat = model.generate(inp, n, do_sample=False)
    # sol = torch.sort(inp[0])[0]
    # sol_candidate = cat[:, n:]
    # print('input sequence  :', inp.tolist())
    # print('predicted sorted:', sol_candidate.tolist())
    # print('gt sort         :', sol.tolist())
    # print('matches         :', bool((sol == sol_candidate).all()))
