import numpy as np
import pytest
from flax import nnx

from guildgpt import GuildGPT
from guildgpt.data import WindowLoader
from guildgpt.training import (
    create_optimizer,
    evaluate,
    make_schedule,
    train_update,
    updates_per_epoch,
)


def test_update_schedule_and_accounting(tiny_run):
    loader = WindowLoader(np.arange(20, dtype=np.uint16), 4, 2)
    assert updates_per_epoch(loader, 2) == 2
    model = GuildGPT(tiny_run.model, nnx.Rngs(0))
    optimizer, schedule = create_optimizer(model, tiny_run, 2)
    before = model.wte.embedding.value.copy()
    first = train_update(model, optimizer, loader, 0, 0, tiny_run)
    second = train_update(model, optimizer, loader, 0, 2, tiny_run)
    assert first["tokens"] + second["tokens"] == 19
    assert first["micro_steps"] == 2 and second["micro_steps"] == 1
    assert np.isfinite(first["loss"])
    assert float(schedule(0)) > float(schedule(1))
    assert not np.array_equal(before, model.wte.embedding.value)
    assert evaluate(model, loader)["tokens"] == 19


def test_schedule_rejects_stale_short_plan(tiny_run):
    from dataclasses import replace

    with pytest.raises(ValueError):
        make_schedule(replace(tiny_run, warmup_steps=100), 10)


def test_accumulation_matches_full_batch(tiny_run):
    from dataclasses import replace

    tokens = np.arange(17, dtype=np.uint16)
    accumulated_loader = WindowLoader(tokens, 4, 2, shuffle=False)
    full_loader = WindowLoader(tokens, 4, 4, shuffle=False)
    accumulated = GuildGPT(tiny_run.model, nnx.Rngs(0))
    full = GuildGPT(tiny_run.model, nnx.Rngs(0))
    full_run = replace(tiny_run, micro_batch_size=4, gradient_accumulation_steps=1)
    accumulated_opt, _ = create_optimizer(accumulated, tiny_run, 2)
    full_opt, _ = create_optimizer(full, full_run, 2)
    train_update(accumulated, accumulated_opt, accumulated_loader, 0, 0, tiny_run)
    train_update(full, full_opt, full_loader, 0, 0, full_run)
    np.testing.assert_allclose(
        accumulated.wte.embedding.value, full.wte.embedding.value, atol=2e-6, rtol=2e-6
    )


def test_bfloat16_microstep_is_finite(tiny_run):
    from dataclasses import replace

    config = replace(tiny_run.model, compute_dtype="bfloat16")
    run = replace(tiny_run, model=config)
    model = GuildGPT(config, nnx.Rngs(0))
    optimizer, _ = create_optimizer(model, run, 2)
    loader = WindowLoader(np.arange(17, dtype=np.uint16), 4, 2)
    assert np.isfinite(train_update(model, optimizer, loader, 0, 0, run)["loss"])
