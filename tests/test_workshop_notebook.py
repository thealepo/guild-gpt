"""Offline checks for the attendee-facing Colab notebook."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from safetensors.numpy import load_file

from guildgpt.config import GuildGPTConfig as ProductionConfig
from guildgpt.hub import save_pretrained
from guildgpt.model import GuildGPT as ProductionModel

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "guildgpt_workshop.ipynb"


def test_workshop_cells_compile_and_diagrams_parse():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert not any(cell.get("outputs") for cell in notebook["cells"])
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    diagram_count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"cell-{index}", "exec")
        for svg in re.findall(r"<svg\b.*?</svg>", source, flags=re.DOTALL):
            ElementTree.fromstring(svg)
            diagram_count += 1
    assert diagram_count == 2


def test_workshop_model_load_and_generation_offline(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    # Colab supplies IPython and network packages; the offline test only needs
    # the symbols used in the model, tiny training, load, and generation cells.
    namespace = {
        "dataclass": dataclass,
        "Path": Path,
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "optax": optax,
        "nnx": nnx,
        "load_file": load_file,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tags = set(cell["metadata"].get("tags", []))
        if "network" in tags or "diagram" in tags:
            continue
        if tags & {"model", "training-explanation", "hub", "generation", "demo"}:
            if "demo" in tags and "synthetic" not in tags:
                continue
            exec("".join(cell["source"]), namespace)  # noqa: S102 - trusted repo notebook

    settings = {
        "max_seq_len": 64,
        "vocab_size": 64,
        "n_layers": 2,
        "n_heads": 4,
        "hidden_size": 64,
        "mlp_hidden_size": 128,
    }
    production = ProductionModel(ProductionConfig(**settings), nnx.Rngs(7))
    save_pretrained(production, tmp_path)
    workshop = namespace["GuildGPT"](namespace["GuildGPTConfig"](**settings), nnx.Rngs(9))
    assert namespace["load_weights_into"](workshop, tmp_path) > 0
    ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    np.testing.assert_allclose(production(ids), workshop(ids), atol=0)

    class ToyTokenizer:
        eos_token_id = 63

        def encode(self, prompt, add_special_tokens=False):
            assert prompt
            return [1, 2, 3]

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(str(token) for token in tokens)

    generated = namespace["generate_text"](
        workshop, ToyTokenizer(), "test", max_new_tokens=2, top_k=5, seed=1
    )
    assert generated.startswith("1 2 3")
