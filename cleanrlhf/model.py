import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import chex
import functools

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
    embed_dim: int   # alias: C
    n_head: int      # alias: nh
    attn_pdrop: int
    resid_pdrop: int
    block_size: int  # alias: T, sequence_length
    @nn.compact
    def __call__(self, x: jnp.array, attn_pdrop_key: jax.random.KeyArray, resid_pdrop_key: jax.random.KeyArray):
        assert self.embed_dim % self.n_head == 0, "embed_dim must be divisible by num_heads"
        B, T, C = jnp.shape(x) # batch size, sequence length, embedding dimensionality (embed_dim
        head_dim = C // self.n_head # alias: hm
        bias = jnp.tril(jnp.ones((self.block_size, self.block_size))).reshape(1, 1, self.block_size, self.block_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        c_attn = nn.Dense(3 * C)(x) # (B, T, 3 * C), `c_attn` means `concatenated attention`
        q, k, v = jnp.split(c_attn, 3, axis=-1) # each has shape (B, T, C)
        q = q.reshape(B, T, self.n_head, head_dim).swapaxes(1,2) # (B, nh, T, hm), nh: n_head, hm: head dimensionality
        k = k.reshape(B, T, self.n_head, head_dim).swapaxes(1,2) # (B, nh, T, hm)
        v = v.reshape(B, T, self.n_head, head_dim).swapaxes(1,2) # (B, nh, T, hm)
        attn = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim) # (B, nh, T, T), attention scores
        attn = jnp.where(bias == 0, float('-inf'), attn) # (B, nh, T, T), mask out the future tokens
        attn = nn.softmax(attn, axis=-1) # (B, nh, T, T), attention weights (probabilities)
        attn = dropout(attn, rate=self.attn_pdrop, key=attn_pdrop_key)
        y = attn @ v # (B, nh, T, hm)
        y = y.swapaxes(1, 2) # (B, T, nh, hm)
        y = y.reshape(B, T, C) # (B, T, C)
        y = nn.Dense(C)(y) # (B, T, C)
        y = dropout(y, rate=self.resid_pdrop, key=resid_pdrop_key)  # (B, T, C)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """
    embed_dim: int   # alias: C
    n_head: int      # alias: nh
    attn_pdrop: int
    resid_pdrop: int
    block_size: int  # alias: T, sequence_length
    # def __init__(self, config):
    #     super().__init__()
    #     self.ln_1 = nn.LayerNorm(config.embed_dim)
    #     self.attn = CausalSelfAttention(config)
    #     self.ln_2 = nn.LayerNorm(config.embed_dim)
    #     self.mlp = nn.ModuleDict(dict(
    #         c_fc    = nn.Linear(config.embed_dim, 4 * config.embed_dim),
    #         c_proj  = nn.Linear(4 * config.embed_dim, config.embed_dim),
    #         act     = NewGELU(),
    #         dropout = nn.Dropout(config.resid_pdrop),
    #     ))
    #     m = self.mlp
    #     self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward


    # def forward(self, x):
    #     x = x + self.attn(self.ln_1(x))
    #     x = x + self.mlpf(self.ln_2(x))
    #     return x
    @nn.compact
    def __call__(self, x: jnp.array, key: jax.random.KeyArray):
        key, attn_pdrop_key, resid_pdrop_key, resid_pdrop_key2 = jax.random.split(key, 4)
        oldx = x
        x = nn.LayerNorm(self.embed_dim)(x)
        x = CausalSelfAttention(
            self.embed_dim,
            self.n_head,
            self.attn_pdrop,
            self.resid_pdrop,
            self.block_size
        )(x, attn_pdrop_key, resid_pdrop_key)
        x = oldx + x

        oldx = x
        x = nn.Dense(4 * self.embed_dim)(x)
        x = NewGELU()(x)
        x = nn.Dense(self.embed_dim)(x)
        x = dropout(x, rate=self.resid_pdrop, key=resid_pdrop_key2)
        x = oldx + x
        return x


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


if __name__ == "__main__":
    block_size = 3
    embed_dim = 12
    key = jax.random.PRNGKey(0)
    key, params_key, attn_pdrop_key, resid_pdrop_key = jax.random.split(key=key, num=4)
    # genereate a random input
    x = jax.random.normal(key, (1, block_size, embed_dim)) # B, T, C; or batch_size, sequence_length, embedding_dimensionality


    # CausalSelfAttention Demo
    attn = CausalSelfAttention(embed_dim=embed_dim, n_head=3, attn_pdrop=0.1, resid_pdrop=0.1, block_size=block_size)
    attn_params = attn.init(params_key, x, attn_pdrop_key, resid_pdrop_key)
    attn_y = attn.apply(attn_params, x, attn_pdrop_key, resid_pdrop_key)

    # Block Demo
    block = Block(embed_dim=embed_dim, n_head=3, attn_pdrop=0.1, resid_pdrop=0.1, block_size=block_size)
    block_params = block.init(params_key, x, key)
    block_y = block.apply(block_params, x, key)