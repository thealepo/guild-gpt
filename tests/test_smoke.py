import numpy as np
from flax import nnx

from guildgpt import GuildGPT
from guildgpt.checkpoints import restore_checkpoint, save_checkpoint
from guildgpt.data import WindowLoader
from guildgpt.generation import generate_tokens
from guildgpt.hub import load_pretrained, save_pretrained
from guildgpt.training import create_optimizer, evaluate, train_update


def test_complete_synthetic_workflow(tmp_path, tiny_run):
    train = WindowLoader(np.arange(20, dtype=np.uint16), 4, 2)
    validation = WindowLoader(np.arange(10, dtype=np.uint16), 4, 2, shuffle=False)
    model = GuildGPT(tiny_run.model, nnx.Rngs(0))
    optimizer, _ = create_optimizer(model, tiny_run, 2)
    train_update(model, optimizer, train, 0, 0, tiny_run)
    assert evaluate(model, validation)["tokens"] == 9
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints",
        model,
        optimizer,
        {"global_step": 1, "epoch": 0, "batch_index": 2},
        tiny_run.as_dict(),
        {"dataset": "synthetic"},
    )
    restored = GuildGPT(tiny_run.model, nnx.Rngs(1))
    restored_optimizer, _ = create_optimizer(restored, tiny_run, 2)
    restore_checkpoint(checkpoint, restored, restored_optimizer)
    train_update(restored, restored_optimizer, train, 0, 2, tiny_run)
    exported = save_pretrained(restored, tmp_path / "export")
    inference = load_pretrained(exported)
    tokens = generate_tokens(inference, [1, 2], max_new_tokens=3, top_k=5, seed=4)
    assert len(tokens) == 5
    assert tokens == generate_tokens(inference, [1, 2], max_new_tokens=3, top_k=5, seed=4)
