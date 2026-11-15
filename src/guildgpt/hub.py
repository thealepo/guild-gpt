from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from safetensors.numpy import load_file, save_file

from .config import GuildGPTConfig
from .model import GuildGPT


def _flatten(tree, prefix=()):
    result = {}
    for key, value in tree.items():
        path = prefix + (str(key),)
        if isinstance(value, dict):
            result.update(_flatten(value, path))
        else:
            result[".".join(path)] = np.asarray(value)
    return result


def _unflatten(tensors, template):
    def build(tree, prefix=()):
        result = {}
        for key, value in tree.items():
            path = prefix + (str(key),)
            name = ".".join(path)
            if isinstance(value, dict):
                result[key] = build(value, path)
            else:
                array = tensors[name]
                if array.shape != value.shape or array.dtype != np.asarray(value).dtype:
                    raise ValueError(f"Incompatible tensor {name}: {array.shape}, {array.dtype}")
                result[key] = jax.device_put(array)
        return result

    expected = set(_flatten(template))
    if set(tensors) != expected:
        raise ValueError(
            f"Tensor key mismatch: missing={expected - set(tensors)}, extra={set(tensors) - expected}"
        )
    return build(template)


def save_pretrained(model, output_dir, tokenizer=None, model_card=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    save_file(_flatten(params), str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(asdict(model.config), indent=2) + "\n")
    if tokenizer is not None:
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match model")
        tokenizer.save_pretrained(output_dir)
    if model_card is not None:
        (output_dir / "README.md").write_text(model_card)
    return output_dir


def load_pretrained(path, *, seed=0):
    path = Path(path)
    config = GuildGPTConfig(**json.loads((path / "config.json").read_text()))
    model = GuildGPT(config, nnx.Rngs(seed))
    state = nnx.state(model, nnx.Param)
    pure = nnx.to_pure_dict(state)
    tensors = load_file(str(path / "model.safetensors"))
    nnx.replace_by_pure_dict(state, _unflatten(tensors, pure))
    nnx.update(model, state)
    return model


def load_from_hub(repo_id: str, *, revision=None, token=None):
    from huggingface_hub import snapshot_download
    from transformers import GPT2TokenizerFast

    path = snapshot_download(
        repo_id,
        revision=revision,
        token=token,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer*",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    return load_pretrained(path), GPT2TokenizerFast.from_pretrained(path)


def upload_to_hub(directory, repo_id, *, token=None, private=False):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(directory), repo_type="model")
