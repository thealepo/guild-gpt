import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from guildgpt import GuildGPT
from guildgpt.checkpoints import restore_checkpoint, save_checkpoint
from guildgpt.data import WindowLoader
from guildgpt.training import create_optimizer, train_update


def test_resume_matches_uninterrupted(tmp_path, tiny_run):
    loader = WindowLoader(np.arange(20, dtype=np.uint16), 4, 2)
    model = GuildGPT(tiny_run.model, nnx.Rngs(0))
    optimizer, _ = create_optimizer(model, tiny_run, 2)
    train_update(model, optimizer, loader, 0, 0, tiny_run)
    path = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        {"global_step": 1, "epoch": 0, "batch_index": 2},
        tiny_run.as_dict(),
        {"sha256": "example"},
    )
    resumed = GuildGPT(tiny_run.model, nnx.Rngs(999))
    resumed_optimizer, _ = create_optimizer(resumed, tiny_run, 2)
    meta = restore_checkpoint(
        path,
        resumed,
        resumed_optimizer,
        config=tiny_run.as_dict(),
        dataset_manifest={"sha256": "example"},
    )
    assert meta["progress"]["batch_index"] == 2
    ids = jnp.array([[1, 2]])
    np.testing.assert_allclose(model(ids), resumed(ids), atol=0)
    for left, right in zip(
        jax.tree.leaves(nnx.state(optimizer)), jax.tree.leaves(nnx.state(resumed_optimizer))
    ):
        np.testing.assert_allclose(left, right, atol=0)
    train_update(model, optimizer, loader, 0, 2, tiny_run)
    train_update(resumed, resumed_optimizer, loader, 0, 2, tiny_run)
    np.testing.assert_allclose(model(ids), resumed(ids), atol=1e-6)
    assert jax.tree.structure(nnx.state(optimizer)) == jax.tree.structure(
        nnx.state(resumed_optimizer)
    )
