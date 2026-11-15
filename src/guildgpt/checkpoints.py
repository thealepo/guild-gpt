from __future__ import annotations

import json
import shutil
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

RESUME_MUTABLE_FIELDS = {
    "data_dir",
    "checkpoint_dir",
    "keep_checkpoints",
    "checkpoint_every",
    "checkpoint_minutes",
    "eval_every",
    "log_every",
    "sample_every",
}


def _training_config(config):
    return {key: value for key, value in config.items() if key not in RESUME_MUTABLE_FIELDS}


def save_checkpoint(
    root,
    model,
    optimizer,
    progress: dict,
    config: dict,
    dataset_manifest: dict,
    wandb_run_id: str | None = None,
    keep: int | None = None,
) -> Path:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    step = int(progress["global_step"])
    directory = root / f"step_{step:08d}"
    if directory.exists():
        raise FileExistsError(directory)
    payload = {
        "model": nnx.to_pure_dict(nnx.state(model, nnx.Param)),
        "optimizer": nnx.to_pure_dict(nnx.state(optimizer)),
    }
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(directory / "state", payload)
    metadata = {
        "progress": progress,
        "config": config,
        "dataset_manifest": dataset_manifest,
        "wandb_run_id": wandb_run_id,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if keep is not None:
        if keep < 1:
            raise ValueError("keep must be positive")
        complete = sorted(p for p in root.glob("step_*") if (p / "metadata.json").exists())
        for old in complete[:-keep]:
            shutil.rmtree(old)
    return directory


def restore_checkpoint(directory, model, optimizer, *, config=None, dataset_manifest=None):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    if config is not None and _training_config(metadata["config"]) != _training_config(config):
        raise ValueError("Checkpoint configuration differs from current run")
    if dataset_manifest is not None and metadata["dataset_manifest"] != dataset_manifest:
        raise ValueError("Checkpoint dataset manifest differs from current data")
    with ocp.PyTreeCheckpointer() as checkpointer:
        payload = checkpointer.restore(directory / "state")
    model_state = nnx.state(model, nnx.Param)
    optimizer_state = nnx.state(optimizer)
    nnx.replace_by_pure_dict(model_state, payload["model"])
    nnx.replace_by_pure_dict(optimizer_state, payload["optimizer"])
    nnx.update(model, model_state)
    nnx.update(optimizer, optimizer_state)
    return metadata


def latest_checkpoint(root):
    root = Path(root)
    if not root.exists():
        return None
    complete = sorted(p for p in root.glob("step_*") if (p / "metadata.json").exists())
    return complete[-1] if complete else None


def load_model_checkpoint(directory, model):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    with ocp.PyTreeCheckpointer() as checkpointer:
        payload = checkpointer.restore(directory / "state")
    state = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(state, payload["model"])
    nnx.update(model, state)
    return metadata
