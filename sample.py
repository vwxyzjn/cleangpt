import os
import pickle

import jax
import numpy as np
import tiktoken
import tyro
from orbax.checkpoint import (
    Checkpointer,
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointHandler,
)

from cleanrlhf.model import generate
from train_char import Args, init_model

mngr = CheckpointManager(
    f"runs/train_char__1__1680814128/models",
    checkpointers={"train_state": Checkpointer(PyTreeCheckpointHandler())},
    options=CheckpointManagerOptions(max_to_keep=1, save_interval_steps=500),
)
args = tyro.from_yaml(Args, mngr.metadata()["args"])
print(args)
key = jax.random.PRNGKey(args.seed)
train_state = init_model(key, args)
restored = mngr.restore(mngr.latest_step(), items={"train_state": train_state})
train_state = restored["train_state"]

# look for the meta pickle in case it is available in the dataset folder
meta_path = os.path.join(args.data_dir, "meta.pkl")
load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

start = "O god O god,"
eval_len = 500
start_ids = encode(start)
x = np.array(start_ids, dtype=np.int32)[None, ...]
y = generate(train_state, args.block_size, key, x, eval_len, temperature=0.9, top_k=5)[0][: len(x[0]) + eval_len]
# completion = "".join([train_dataset.itos[int(i)] for i in y])
# print(len(completion), completion)
print(decode(y.tolist()))
print("---------------")
