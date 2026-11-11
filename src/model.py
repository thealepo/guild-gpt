import math

import jax
import jax.numpy as jnp
from flax import nnx

from .config import GuildGPTConfig


class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, config: GuildGPTConfig, rngs: nnx.Rngs):
        self.n_heads = config.n_heads
        self.head_size = config.hidden_size // config.n_heads
        
        init = jax.nn.initializers.normal(stddev=0.02)
        resid_init = jax.nn.initializers.normal(stddev=0.02 / math.sqrt(2 * config.n_layers))

        self.Wq = nnx.Linear(config.hidden_size, config.hidden_size, kernel_init=init, rngs=rngs)
        self.Wk = nnx.Linear(config.hidden_size, config.hidden_size, kernel_init=init, rngs=rngs)
        self.Wv = nnx.Linear(config.hidden_size, config.hidden_size, kernel_init=init, rngs=rngs)
        self.Wo = nnx.Linear(config.hidden_size, config.hidden_size, kernel_init=resid_init, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, seq_len, _ = x.shape

        ...


class MultiLayerPerceptron(nnx.Module):
    def __init__(self, config: GuildGPTConfig, rngs: nnx.Rngs):
        init = jax.nn.initializers.normal(stddev=0.02)
        resid_init = jax.nn.initializers.normal(stddev=0.02 / math.sqrt(2 * config.n_layers))

        self.fc1 = nnx.Linear(config.hidden_size, config.mlp_hidden_size, kernel_init=init, rngs=rngs)
        self.fc2 = nnx.Linear(config.mlp_hidden_size, config.hidden_size, kernel_init=resid_init, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nnx.Module):
    def __init__(self, config: GuildGPTConfig, rngs: nnx.Rngs):
        self.mhsa = MultiHeadSelfAttention(config, rngs=rngs)
        self.mlp = MultiLayerPerceptron(config, rngs=rngs)

        self.ln1 = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)
        self.ln2 = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

