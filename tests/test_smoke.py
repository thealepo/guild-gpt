import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import yaml
from flax import nnx
from transformers import GPT2TokenizerFast

from guildgpt import GuildGPT
from guildgpt.checkpoints import restore_checkpoint, save_checkpoint
from guildgpt.data import WindowLoader, file_sha256
from guildgpt.generation import generate_tokens
from guildgpt.hub import load_pretrained, save_pretrained
from guildgpt.training import create_optimizer, evaluate, train_update
from scripts import train as train_script


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


def test_no_wandb_training_does_not_fetch_sample_tokenizer(tmp_path, tiny_run, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_path = data_dir / "train.bin"
    validation_path = data_dir / "validation.bin"
    np.arange(20, dtype=np.uint16).tofile(train_path)
    np.arange(10, dtype=np.uint16).tofile(validation_path)
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "andrewelawrence/writingPrompts-merged",
                "tokenizer_vocab_size": tiny_run.model.vocab_size,
                "train_sha256": file_sha256(train_path),
                "validation_sha256": file_sha256(validation_path),
            }
        )
    )
    config = replace(
        tiny_run,
        data_dir=str(data_dir),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        eval_every=1,
        checkpoint_every=10,
        checkpoint_minutes=0,
        log_every=1,
        sample_every=1,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.as_dict()))

    def fail_if_fetched(*args, **kwargs):
        raise AssertionError("No-W&B training must not fetch a tokenizer")

    monkeypatch.setattr(GPT2TokenizerFast, "from_pretrained", fail_if_fetched)
    monkeypatch.setattr(
        train_script, "jax", SimpleNamespace(default_backend=lambda: "gpu", devices=jax.devices)
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    train_script.main()

    assert (tmp_path / "checkpoints" / "step_00000002" / "metadata.json").exists()
