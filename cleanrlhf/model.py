from dataclasses import dataclass
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def dropout(x: jnp.ndarray, rate: float, key: jax.random.KeyArray) -> jnp.ndarray:
    """
    # nn.Dropout is a bit confusing to me... @Costa
    Functional dropout implementation. In contrast to the flax.linen module, this can
    be used inside of standard JAX function transforms.
    Note that we could also use the lifted transforms provided by Flax, but this
    is more general.
    taken from https://github.com/brentyi/minGPT-flax/blob/7927b564e04b929e4df219a9334d86de9486dfb0/mingpt/attention.py#L11
    """
    keep_prob = 1.0 - rate
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    out = jnp.where(mask, x / keep_prob, 0.0)
    assert out.shape == x.shape
    return out


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    @nn.compact
    def __call__(self, x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    embd_dim: int  # alias: C
    n_head: int  # alias: nh
    attn_pdrop: int
    resid_pdrop: int
    block_size: int  # alias: T, sequence_length

    @nn.compact
    def __call__(self, x: jnp.array, attn_pdrop_key: jax.random.KeyArray, resid_pdrop_key: jax.random.KeyArray):
        assert self.embd_dim % self.n_head == 0, "embd_dim must be divisible by num_heads"
        B, T, C = jnp.shape(x)  # batch size, sequence length, embedding dimensionality (embd_dim
        head_dim = C // self.n_head  # alias: hm
        bias = jnp.tril(jnp.ones((self.block_size, self.block_size))).reshape(1, 1, self.block_size, self.block_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        c_attn = nn.Dense(3 * C)(x)  # (B, T, 3 * C), `c_attn` means `concatenated attention`
        q, k, v = jnp.split(c_attn, 3, axis=-1)  # each has shape (B, T, C)
        q = q.reshape(B, T, self.n_head, head_dim).swapaxes(1, 2)  # (B, nh, T, hm), nh: n_head, hm: head dimensionality
        k = k.reshape(B, T, self.n_head, head_dim).swapaxes(1, 2)  # (B, nh, T, hm)
        v = v.reshape(B, T, self.n_head, head_dim).swapaxes(1, 2)  # (B, nh, T, hm)
        attn = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim)  # (B, nh, T, T), attention scores
        attn = jnp.where(bias == 0, float("-inf"), attn)  # (B, nh, T, T), mask out the future tokens
        attn = nn.softmax(attn, axis=-1)  # (B, nh, T, T), attention weights (probabilities)
        attn = dropout(attn, rate=self.attn_pdrop, key=attn_pdrop_key)
        y = attn @ v  # (B, nh, T, hm)
        y = y.swapaxes(1, 2)  # (B, T, nh, hm)
        y = y.reshape(B, T, C)  # (B, T, C)
        y = nn.Dense(C)(y)  # (B, T, C)
        y = dropout(y, rate=self.resid_pdrop, key=resid_pdrop_key)  # (B, T, C)
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    embd_dim: int  # alias: C
    n_head: int  # alias: nh
    attn_pdrop: int
    resid_pdrop: int
    block_size: int  # alias: T, sequence_length

    @nn.compact
    def __call__(self, x: jnp.array, key: jax.random.KeyArray):
        key, attn_pdrop_key, resid_pdrop_key, resid_pdrop_key2 = jax.random.split(key, 4)
        oldx = x
        x = nn.LayerNorm(self.embd_dim)(x)
        x = CausalSelfAttention(self.embd_dim, self.n_head, self.attn_pdrop, self.resid_pdrop, self.block_size)(
            x, attn_pdrop_key, resid_pdrop_key
        )
        x = oldx + x

        oldx = x
        x = nn.Dense(4 * self.embd_dim)(x)
        x = NewGELU()(x)
        x = nn.Dense(self.embd_dim)(x)
        x = dropout(x, rate=self.resid_pdrop, key=resid_pdrop_key2)
        x = oldx + x
        return x

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    n_layer: int
    n_head: int
    embd_dim: int
    # these options must be filled in externally
    vocab_size: int
    block_size: int
    # dropout hyperparameters
    embd_pdrop: int = 0.1
    resid_pdrop: int = 0.1
    attn_pdrop: int = 0.1

    # self.transformer = nn.ModuleDict(dict(
    #     wte = nn.Embed(self.vocab_size, self.embd_dim), # word to embedding
    #     wpe = nn.Embed(self.block_size, self.embd_dim), # word to position embedding
    #     drop = nn.Dropout(self.embd_pdrop),
    #     h = nn.ModuleList([Block(self) for _ in range(self.n_layer)]),
    #     ln_f = nn.LayerNorm(self.embd_dim),
    # ))
    # self.lm_head = nn.Linear(self.embd_dim, self.vocab_size, bias=False)

    @nn.compact
    def __call__(self, idx: jnp.array, target: jnp.array, key: jax.random.KeyArray):
        key, embd_pdrop_key = jax.random.split(key, 2)
        _, T = jnp.shape(idx)  # B, T
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = jnp.array([jnp.arange(0, T, dtype=jnp.int32)])  # shape (1, T)

        # forward the GPT model itself
        tok_embd = nn.Embed(self.vocab_size, self.embd_dim)(idx)  # token embeddings of shape (B, T, embd_dim)
        pos_embd = nn.Embed(self.block_size, self.embd_dim)(pos)  # position embeddings of shape (1, T, embd_dim)
        x = dropout(tok_embd + pos_embd, rate=self.embd_pdrop, key=embd_pdrop_key)
        for _ in range(self.n_layer):
            x = Block(
                self.embd_dim,
                self.n_head,
                self.attn_pdrop,
                self.resid_pdrop,
                self.block_size,
            )(x, key)
        x = nn.LayerNorm(self.embd_dim)(x)
        logits = nn.Dense(self.vocab_size, use_bias=False)(x)

        # Costa: mask out the gradient for indices with values = -1
        # should be equivalent to `ignore_index=-1` in F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        logits = jnp.where(target == -1, -1e8, logits)
        loss = optax.softmax_cross_entropy(logits.reshape(-1, jnp.shape(logits)[-1]), target.reshape(-1))
        return loss, logits


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    embd_dim: int


MODELS_PRESET: Dict[str, GPTConfig] = {
    "openai-gpt": GPTConfig(n_layer=12, n_head=12, embd_dim=768),  # 117M params
    # GPT-2 configs
    "gpt2": GPTConfig(n_layer=12, n_head=12, embd_dim=768),  # 124M params
    "gpt2-medium": GPTConfig(n_layer=24, n_head=16, embd_dim=1024),  # 350M params
    "gpt2-large": GPTConfig(n_layer=36, n_head=20, embd_dim=1280),  # 774M params
    "gpt2-xl": GPTConfig(n_layer=48, n_head=25, embd_dim=1600),  # 1558M params
    # Gophers
    "gopher-44m": GPTConfig(n_layer=8, n_head=16, embd_dim=512),
    # (there are a number more...)
    # I made these tiny models up
    "gpt-mini": GPTConfig(n_layer=6, n_head=6, embd_dim=192),
    "gpt-micro": GPTConfig(n_layer=4, n_head=4, embd_dim=128),
    "gpt-nano": GPTConfig(n_layer=3, n_head=3, embd_dim=48),
}


if __name__ == "__main__":
    block_size = 3
    embd_dim = 12
    n_head = 3
    key = jax.random.PRNGKey(0)
    key, params_key, attn_pdrop_key, resid_pdrop_key = jax.random.split(key=key, num=4)
    x = jax.random.normal(key, (1, block_size, embd_dim))  # B, T, C; or batch_size, sequence_length, embedding_dimensionality

    # CausalSelfAttention Demo
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    attn = CausalSelfAttention(
        embd_dim=embd_dim, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, block_size=block_size
    )
    attn_params = attn.init(params_key, x, attn_pdrop_key, resid_pdrop_key)
    attn_y = attn.apply(attn_params, x, attn_pdrop_key, resid_pdrop_key)

    # Block Demo
    block = Block(embd_dim=embd_dim, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, block_size=block_size)
    block_params = block.init(params_key, x, key)
    block_y = block.apply(block_params, x, key)

    # GPT Demo
    n_layer = 3
    vocab_size = 10
    gpt = GPT(
        n_layer=n_layer,
        n_head=n_head,
        embd_dim=embd_dim,
        vocab_size=vocab_size,
        block_size=block_size,
    )
    x = jax.random.randint(key, (1, block_size), minval=0, maxval=vocab_size)  # B, T; or batch_size, sequence_length
    gpt_params = gpt.init(params_key, x, x, key)
    gpt_y = gpt.apply(gpt_params, x, x, key)
