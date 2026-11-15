import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import load_file, save_file

from guildgpt import GuildGPT
from guildgpt.hub import load_pretrained, save_pretrained


def test_safetensors_round_trip_and_incompatible_keys(tmp_path, tiny_config):
    model = GuildGPT(tiny_config, nnx.Rngs(0))
    save_pretrained(model, tmp_path)
    restored = load_pretrained(tmp_path)
    ids = jnp.array([[1, 2, 3]])
    np.testing.assert_allclose(model(ids), restored(ids), atol=0)
    tensors = load_file(str(tmp_path / "model.safetensors"))
    tensors.pop(next(iter(tensors)))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Tensor key mismatch"):
        load_pretrained(tmp_path)
