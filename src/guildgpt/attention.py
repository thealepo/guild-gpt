import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.n_heads = config.n_heads
        self.head_size = config.hidden_size // config.n_heads
        self.backend = config.attention_backend
        self.dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32
        args = {"param_dtype": jnp.float32, "dtype": self.dtype, "rngs": rngs}
        self.Wq = nnx.Linear(config.hidden_size, config.hidden_size, **args)
        self.Wk = nnx.Linear(config.hidden_size, config.hidden_size, **args)
        self.Wv = nnx.Linear(config.hidden_size, config.hidden_size, **args)
        self.Wo = nnx.Linear(config.hidden_size, config.hidden_size, **args)

    def __call__(self, x):
        b, n, _ = x.shape

        def heads(projection):
            return projection(x).reshape(b, n, self.n_heads, self.head_size).transpose(0, 2, 1, 3)

        q, k, v = heads(self.Wq), heads(self.Wk), heads(self.Wv)
        if self.backend == "sdpa":
            # JAX SDPA expects [batch, sequence, heads, head_dim].
            out = jax.nn.dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                is_causal=True,
                implementation="xla",
            )
        else:
            scores = jnp.einsum("bhid,bhjd->bhij", q, k).astype(jnp.float32)
            scores *= self.head_size**-0.5
            scores = jnp.where(jnp.tril(jnp.ones((n, n), dtype=bool)), scores, -jnp.inf)
            weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
            out = jnp.einsum("bhij,bhjd->bhid", weights, v).transpose(0, 2, 1, 3)
        out = out.reshape(b, n, self.n_heads * self.head_size)
        return self.Wo(out)
