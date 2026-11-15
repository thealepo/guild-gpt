import jax
import jax.numpy as jnp
from flax import nnx

from .attention import MultiHeadSelfAttention


class MultiLayerPerceptron(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32
        args = {"dtype": dtype, "param_dtype": jnp.float32, "rngs": rngs}
        self.fc1 = nnx.Linear(config.hidden_size, config.mlp_hidden_size, **args)
        self.fc2 = nnx.Linear(config.mlp_hidden_size, config.hidden_size, **args)

    def __call__(self, x):
        return self.fc2(jax.nn.gelu(self.fc1(x)))


class TransformerLayer(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.mhsa = MultiHeadSelfAttention(config, rngs)
        self.mlp = MultiLayerPerceptron(config, rngs)
        self.ln1 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.ln2 = nnx.LayerNorm(config.hidden_size, rngs=rngs)

    def __call__(self, x):
        x = x + self.mhsa(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GuildGPT(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.wpe = nnx.Embed(config.max_seq_len, config.hidden_size, rngs=rngs)
        self.layers = nnx.List([TransformerLayer(config, rngs) for _ in range(config.n_layers)])
        self.ln_f = nnx.LayerNorm(config.hidden_size, rngs=rngs)

    def __call__(self, input_ids):
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input exceeds model context length")
        x = self.wte(input_ids) + self.wpe(jnp.arange(input_ids.shape[1]))
        for layer in self.layers:
            x = layer(x)
        return self.wte.attend(self.ln_f(x))
