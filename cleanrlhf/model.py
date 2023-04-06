from dataclasses import dataclass
from typing import Dict, Optional
import flax

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
    use_bias: bool
    deterministic: Optional[bool] = None
    dtype: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.array, deterministic=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        assert self.embd_dim % self.n_head == 0, "embd_dim must be divisible by num_heads"
        B, T, C = jnp.shape(x)  # batch size, sequence length, embedding dimensionality (embd_dim
        head_dim = C // self.n_head  # alias: hd
        bias = jnp.tril(jnp.ones((self.block_size, self.block_size))).reshape(1, 1, self.block_size, self.block_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        c_attn = nn.Dense(3 * C, use_bias=self.use_bias)(x)  # (B, T, 3 * C), `c_attn` means `concatenated attention`
        q, k, v = jnp.split(c_attn, 3, axis=-1)  # each has shape (B, T, C)
        q = q.reshape(B, T, self.n_head, head_dim).swapaxes(1, 2)  # (B, nh, T, hd), nh: n_head, hd: head dimensionality
        k = k.reshape(B, T, self.n_head, head_dim).swapaxes(1, 2)  # (B, nh, T, hd)
        v = v.reshape(B, T, self.n_head, head_dim).swapaxes(1, 2)  # (B, nh, T, hd)
        attn = q @ k.swapaxes(-1, -2) / jnp.sqrt(head_dim)  # (B, nh, T, T), attention scores
        attn = jnp.where(bias[:, :, :T, :T] == 0, float("-inf"), attn)  # (B, nh, T, T), mask out the future tokens
        attn = nn.softmax(attn, axis=-1)  # (B, nh, T, T), attention weights (probabilities)
        attn = nn.Dropout(self.attn_pdrop)(attn, deterministic=deterministic)
        y = attn @ v  # (B, nh, T, hd)
        y = y.swapaxes(1, 2)  # (B, T, nh, hd)
        y = y.reshape(B, T, C)  # (B, T, C)
        c_proj = nn.Dense(C, use_bias=self.use_bias)(y)  # (B, T, C)
        x = nn.Dropout(rate=self.resid_pdrop)(c_proj, deterministic=deterministic)
        return x


class MLP(nn.Module):
    n_head: int  # alias: nh
    attn_pdrop: int
    resid_pdrop: int
    block_size: int  # alias: T, sequence_length
    use_bias: bool

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = nn.Dense(4 * C, use_bias=self.use_bias, name="c_fc")(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, use_bias=self.use_bias, name="c_proj")(x)
        x = nn.Dropout(self.resid_pdrop)(x, deterministic)
        return x


class Block(nn.Module):
    embd_dim: int  # alias: C
    n_head: int  # alias: nh
    attn_pdrop: int
    resid_pdrop: int
    block_size: int  # alias: T, sequence_length
    use_bias: bool
    dtype: Optional[str] = None

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, use_bias=self.use_bias)
        self.attn = CausalSelfAttention(
            self.embd_dim, self.n_head, self.attn_pdrop, self.resid_pdrop, self.block_size, self.use_bias, dtype=self.dtype
        )
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, use_bias=self.use_bias)
        self.mlp = MLP(self.n_head, self.attn_pdrop, self.resid_pdrop, self.block_size, self.use_bias)

    def __call__(self, x, deterministic=None):
        x = x + self.attn(self.ln_1(x), deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x


@dataclass
class GPTConfig:
    n_layer: int = 3
    n_head: int = 3
    embd_dim: int = 48
    # dropout hyperparameters
    embd_pdrop: int = 0.1
    resid_pdrop: int = 0.1
    attn_pdrop: int = 0.1
    use_bias: bool = True
    dtype: Optional[str] = None


class GPT(nn.Module):
    config: GPTConfig

    # these options must be filled in externally
    vocab_size: int = None
    block_size: int = None

    @nn.compact
    def __call__(self, idx, targets: jnp.array, deterministic=None):
        _, T = jnp.shape(idx)  # B, T
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = jnp.arange(0, T)[None]  # shape (1, T)

        wte = nn.Embed(self.vocab_size, self.config.embd_dim, dtype=self.config.dtype, name="wte")
        wpe = nn.Embed(self.block_size, self.config.embd_dim, dtype=self.config.dtype, name="wpe")

        token_embed = wte(idx)  # token embeddings of shape (B, T, embd_dim)
        pos_embed = wpe(pos)  # position embeddings of shape (1, T, embd_dim)
        x = nn.Dropout(self.config.embd_pdrop)(token_embed + pos_embed, deterministic)

        for i in range(self.config.n_layer):
            x = Block(
                self.config.embd_dim,
                self.config.n_head,
                self.config.attn_pdrop,
                self.config.resid_pdrop,
                self.block_size,
                self.config.use_bias,
                self.config.dtype,
                name=str(i),
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name="ln_f")(x)
        logits = wte.attend(x)
        if targets is None:
            return logits
        # Costa: the following should be equivalent to `ignore_index=-1`
        # in F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        valid_targets = jnp.where(targets == -1, 0, targets)  # remove the mask from the integer labels for cross entropy
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, jnp.shape(logits)[-1]), valid_targets.reshape(-1)
        )
        loss = loss.mean(where=targets.reshape(-1) != -1)  # only calculate the mean for indices that are ignored
        return loss, logits


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
    "gpt-small": GPTConfig(n_layer=6, n_head=6, embd_dim=384, use_bias=False),
    "gpt-mini": GPTConfig(n_layer=6, n_head=6, embd_dim=192),
    "gpt-micro": GPTConfig(n_layer=4, n_head=4, embd_dim=128),
    "gpt-nano": GPTConfig(n_layer=3, n_head=3, embd_dim=48),
}


def param_decay_mask(params: flax.core.FrozenDict) -> flax.core.FrozenDict:
    """ pytree mask for non-bias parameters """
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_param_mask = {k: k[-1] not in ('bias', 'embedding', 'scale') for k in flat_params.keys()}
    param_mask = flax.traverse_util.unflatten_dict(flat_param_mask)
    return flax.core.frozen_dict.freeze(param_mask)


if __name__ == "__main__":
    block_size = 3
    embd_dim = 12
    n_head = 3
    key = jax.random.PRNGKey(0)
    key, params_key, dropout_key = jax.random.split(key=key, num=3)
    x = jax.random.normal(key, (1, block_size, embd_dim))  # B, T, C; or batch_size, sequence_length, embedding_dimensionality

    # CausalSelfAttention Demo
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    attn = CausalSelfAttention(
        embd_dim=embd_dim,
        n_head=n_head,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        block_size=block_size,
        use_bias=False,
    )
    attn_params = attn.init(params_key, x, deterministic=True)
    attn_y = attn.apply(attn_params, x, deterministic=True)
    attn_y = attn.apply(attn_params, x, deterministic=False, rngs={"dropout": dropout_key})

    # Block Demo
    block = Block(
        embd_dim=embd_dim, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, block_size=block_size, use_bias=False
    )
    block_params = block.init(params_key, x, deterministic=True)
    block_y = block.apply(block_params, x, deterministic=True)

    # GPT Demo
    n_layer = 3
    vocab_size = 10
    # x = jax.random.randint(key, (1, block_size), minval=0, maxval=vocab_size)  # B, T; or batch_size, sequence_length
    # y = jax.random.randint(key, (1,), minval=0, maxval=vocab_size)  # B; or batch_size, sequence_length
    x = jnp.array([[0, 1, 1, 2, 2, 1, 0, 1, 1, 1, 2], [0, 1, 1, 2, 0, 2, 0, 0, 1, 1, 2], [0, 1, 2, 2, 1, 0, 0, 0, 1, 1, 2]])
    y = jnp.array(
        [
            [-1, -1, -1, -1, -1, 0, 1, 1, 1, 2, 2],
            [-1, -1, -1, -1, -1, 0, 0, 1, 1, 2, 2],
            [-1, -1, -1, -1, -1, 0, 0, 1, 1, 2, 2],
        ]
    )
    gpt = GPT(
        config=GPTConfig(
            n_layer=n_layer,
            n_head=n_head,
            embd_dim=embd_dim,
        ),
        vocab_size=3,
        block_size=11,
    )
    gpt_params = gpt.init(params_key, x, y, deterministic=True)
    gpt_loss, (gpt_y) = gpt.apply(gpt_params, x, y, deterministic=True)
    x = jnp.array([[0, 1, 1, 2, 2, 1], [0, 1, 1, 2, 0, 2], [0, 1, 2, 2, 1, 0]])
    logits = gpt.apply(gpt_params, x, targets=None, deterministic=True)
