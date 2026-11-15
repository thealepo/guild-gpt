import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from guildgpt import GuildGPT, GuildGPTConfig


def test_full_parameter_count_without_allocating_weights():
    config = GuildGPTConfig()
    shape_tree = jax.eval_shape(
        lambda: nnx.to_pure_dict(nnx.state(GuildGPT(config, nnx.Rngs(0)), nnx.Param))
    )
    assert sum(x.size for x in jax.tree.leaves(shape_tree)) == 124_439_808
    assert config.parameter_count() == 124_439_808


def test_shape_tying_and_causality(tiny_config):
    model = GuildGPT(tiny_config, nnx.Rngs(0))
    a = jnp.array([[1, 2, 3, 4]])
    b = jnp.array([[1, 2, 9, 8]])
    logits_a, logits_b = model(a), model(b)
    assert logits_a.shape == (1, 4, 24)
    np.testing.assert_allclose(logits_a[:, :2], logits_b[:, :2], atol=1e-6)
    assert "lm_head" not in nnx.to_pure_dict(nnx.state(model, nnx.Param))


def test_attention_backend_parity(tiny_config):
    from dataclasses import replace

    reference = GuildGPT(tiny_config, nnx.Rngs(0))
    optimized = GuildGPT(replace(tiny_config, attention_backend="sdpa"), nnx.Rngs(1))
    nnx.update(optimized, nnx.state(reference, nnx.Param))
    ids = jnp.array([[1, 2, 3, 4]])
    np.testing.assert_allclose(reference(ids), optimized(ids), atol=2e-5, rtol=2e-5)
