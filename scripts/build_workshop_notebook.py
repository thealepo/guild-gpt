"""Build the self-contained, inference-first AI Guild Colab notebook.

The original experiment notebook is deliberately left untouched. Run this file
after editing the cells below to regenerate notebooks/guildgpt_workshop.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

cells = []


def add_markdown(source: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"guild-{len(cells):03d}",
            "metadata": {},
            "source": (dedent(source).strip("\n") + "\n").splitlines(keepends=True),
        }
    )


def add_code(source: str, *, tags: list[str] | None = None) -> None:
    metadata = {"tags": tags} if tags else {}
    cells.append(
        {
            "cell_type": "code",
            "id": f"guild-{len(cells):03d}",
            "metadata": metadata,
            "source": (dedent(source).strip("\n") + "\n").splitlines(keepends=True),
            "execution_count": None,
            "outputs": [],
        }
    )


add_markdown(
    """
    # Build Your Own GPT Model — AI Guild workshop

    **Alex E. Sanchez** · JAX + Flax NNX · WritingPrompts

    Today we will look inside a small GPT-style language model, build its blocks
    in code, and then use **weights already trained from scratch** to generate a
    story. We use the GPT-2 *tokenizer* (the text-to-number dictionary), **not**
    pretrained GPT-2 model weights.

    **Run the cells from top to bottom.** The full 217-million-token training run
    happened beforehand; this notebook never downloads the whole dataset or
    launches a substantial training job. The only training here is one tiny,
    synthetic demonstration step.

    The model repository ID is set to `thealepo/GuildGPT` near the end of the
    notebook. Until those weights are published, the download cell cannot run.
    """
)

add_markdown(
    """
    ## 0. Colab setup

    In Colab, choose **Runtime → Change runtime type → GPU** if one is available.
    A GPU is recommended for live generation, but Colab does not guarantee one.
    Run the next cell to install the versions used by the trained model. The
    first setup may take a few minutes. If Colab says that already-imported
    packages need a restart, restart the runtime and run from the top again.
    """
)

add_code(
    """
    import shutil
    import subprocess
    import sys

    has_nvidia_gpu = shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
    ).returncode == 0
    jax_package = "jax[cuda12]==0.7.2" if has_nvidia_gpu else "jax==0.7.2"
    packages = [
        jax_package,
        "flax==0.12.0",
        "optax==0.2.6",
        "numpy==2.5.4",
        "transformers==4.57.6",
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *packages]
    )
    print("Installed workshop packages. NVIDIA GPU detected:", has_nvidia_gpu)
    """,
    tags=["setup", "network"],
)

add_markdown(
    """
    These are all the imports used below. If JAX reports `cpu` even though you
    selected a GPU runtime, stop and check the Colab runtime: a CPU session will
    work but generate more slowly. No API token is needed for a **public** model.
    """
)

add_code(
    """
    import json
    import os
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from urllib.parse import urlencode
    from urllib.request import urlopen

    # Avoid a common Colab/Hugging Face setting that requires an optional package.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from huggingface_hub import snapshot_download
    from IPython.display import SVG, display
    from safetensors.numpy import load_file
    from transformers import GPT2TokenizerFast

    print("JAX:", jax.__version__, "| backend:", jax.default_backend())
    print("Devices:", jax.devices())
    """,
    tags=["setup"],
)

add_markdown(
    """
    ## 1. What are we building?

    A language model repeatedly answers one question: **given the tokens so
    far, which token might come next?** It predicts a probability for each of
    the 50,257 token IDs in the vocabulary, picks one, adds it to the text, and
    repeats.
    """
)

add_code(
    r"""
    display(SVG(r'''<svg xmlns="http://www.w3.org/2000/svg" width="870" height="160" viewBox="0 0 870 160" role="img" aria-label="Prompt to tokenizer to GuildGPT to next token">
      <style>text{font:17px sans-serif;fill:#18324a}.small{font:13px sans-serif;fill:#52697c}.box{stroke:#436887;stroke-width:2;rx:16}.arrow{stroke:#49677f;stroke-width:3;fill:none;marker-end:url(#tip)}</style>
      <defs><marker id="tip" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="#49677f"/></marker></defs>
      <rect x="5" y="30" width="175" height="88" class="box" fill="#e9f4ff"/><text x="92" y="67" text-anchor="middle">Story prompt</text><text x="92" y="91" class="small" text-anchor="middle">words we type</text>
      <path d="M185 74 H223" class="arrow"/>
      <rect x="230" y="30" width="175" height="88" class="box" fill="#e7f8f5"/><text x="317" y="67" text-anchor="middle">Tokenizer</text><text x="317" y="91" class="small" text-anchor="middle">words → token IDs</text>
      <path d="M410 74 H448" class="arrow"/>
      <rect x="455" y="30" width="175" height="88" class="box" fill="#f4edff"/><text x="542" y="67" text-anchor="middle">GuildGPT</text><text x="542" y="91" class="small" text-anchor="middle">12 transformer layers</text>
      <path d="M635 74 H673" class="arrow"/>
      <rect x="680" y="30" width="180" height="88" class="box" fill="#fff4df"/><text x="770" y="67" text-anchor="middle">Next token</text><text x="770" y="91" class="small" text-anchor="middle">append and repeat</text>
    </svg>'''))
    """,
    tags=["diagram"],
)

add_markdown(
    """
    ## 2. The dataset, without downloading it all

    We trained on [WritingPrompts merged](https://huggingface.co/datasets/andrewelawrence/writingPrompts-merged),
    a collection of **303,358 prompt/story pairs**. The *story* is what the
    model learned to continue. The prompt was used to keep stories sharing a
    prompt on the same side of our train/validation split; the model was **not**
    trained as a prompt→story instruction model.

    The actual run used **288,720 training stories** and **14,638 validation
    stories**. We appended an end-of-story token to each story, then processed
    about **216.9 million training tokens**. Below we request only **two rows**
    through Hugging Face's [dataset viewer API](https://huggingface.co/docs/dataset-viewer/rows),
    not the ~604 MB dataset. The preview is optional: if the viewer is down,
    continue with the rest of the notebook. Because these are user-written
    stories, the facilitator should review the preview before showing it live.
    """
)

add_code(
    """
    DATASET_ID = "andrewelawrence/writingPrompts-merged"
    viewer_base = "https://datasets-server.huggingface.co/"

    def viewer_json(endpoint, **params):
        url = viewer_base + endpoint + "?" + urlencode(params)
        with urlopen(url, timeout=20) as response:
            return json.load(response)

    try:
        splits = viewer_json("splits", dataset=DATASET_ID)["splits"]
        train_split = next(item for item in splits if item["split"] == "train")
        preview = viewer_json(
            "rows", dataset=DATASET_ID, config=train_split["config"],
            split="train", offset=0, length=2,
        )
        for item in preview["rows"]:
            row = item["row"]
            print("PROMPT:", str(row.get("prompt", ""))[:220])
            print("STORY OPENING:", str(row.get("story", ""))[:350], "…")
            print("-" * 70)
    except Exception as exc:
        print("Dataset preview is temporarily unavailable; the rest of the workshop still works.")
        print(type(exc).__name__, str(exc)[:180])
    """,
    tags=["network", "optional"],
)

add_markdown(
    """
    ### Turning text into token IDs

    The GPT-2 tokenizer is a vocabulary/encoding rule, not a pretrained model.
    It converts a sentence into integer IDs; the neural network consumes those
    integers. A token can be a word, a word fragment, or punctuation.
    """
)

add_code(
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    example_text = "The lighthouse blinked twice."
    example_ids = tokenizer.encode(example_text, add_special_tokens=False)
    print("Text:", example_text)
    print("IDs:", example_ids)
    print("Token pieces:", tokenizer.convert_ids_to_tokens(example_ids))
    print("Vocabulary size:", tokenizer.vocab_size)
    """,
    tags=["network"],
)

add_markdown(
    """
    ## 3. Build the model, block by block

    The configuration says *how large* each part is. The little model we use to
    demonstrate shapes and one training step has only two layers. Later, the
    Hugging Face `config.json` will construct the **real 12-layer model** before
    we put the trained weights into it. The architecture and parameter names
    below match the actual trained GuildGPT code.
    """
)

add_code(
    """
    @dataclass(frozen=True)
    class GuildGPTConfig:
        max_seq_len: int = 1024
        vocab_size: int = 50257
        n_layers: int = 12
        n_heads: int = 12
        hidden_size: int = 768
        mlp_hidden_size: int | None = None
        attention_backend: str = "reference"
        compute_dtype: str = "float32"

        def __post_init__(self):
            if self.mlp_hidden_size is None:
                object.__setattr__(self, "mlp_hidden_size", 4 * self.hidden_size)
            if self.hidden_size % self.n_heads:
                raise ValueError("hidden_size must be divisible by n_heads")
            if self.attention_backend not in {"reference", "sdpa"}:
                raise ValueError("Unknown attention backend")
            if self.compute_dtype not in {"float32", "bfloat16"}:
                raise ValueError("Unknown compute dtype")
    """,
    tags=["model"],
)

add_markdown(
    """
    ### Block A: causal multi-head self-attention

    Each token makes a **query**, **key**, and **value** vector. Attention
    compares queries with keys to decide how much of each earlier value to
    use. The triangular *causal mask* prevents a position from looking at
    future tokens. Multiple heads learn different relationships in parallel.
    """
)

add_code(
    """
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
    """,
    tags=["model"],
)

add_markdown(
    """
    ### Block B: the MLP

    Attention mixes information **between positions**. The multilayer
    perceptron (MLP) then transforms the features **within each position**:
    expand → GELU nonlinearity → shrink back to the hidden size.
    """
)

add_code(
    """
    class MultiLayerPerceptron(nnx.Module):
        def __init__(self, config, rngs: nnx.Rngs):
            dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32
            args = {"dtype": dtype, "param_dtype": jnp.float32, "rngs": rngs}
            self.fc1 = nnx.Linear(config.hidden_size, config.mlp_hidden_size, **args)
            self.fc2 = nnx.Linear(config.mlp_hidden_size, config.hidden_size, **args)

        def __call__(self, x):
            return self.fc2(jax.nn.gelu(self.fc1(x)))
    """,
    tags=["model"],
)

add_markdown(
    """
    ### Block C: one transformer layer

    Layer normalization stabilizes the inputs. The two `x + ...` expressions
    are **residual connections**: they preserve an easy path for existing
    information while attention and the MLP add refinements.
    """
)

add_code(
    r"""
    display(SVG(r'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="150" viewBox="0 0 760 150" role="img" aria-label="Transformer layer: input, attention residual, MLP residual, output">
      <style>text{font:16px sans-serif;fill:#18324a}.small{font:12px sans-serif;fill:#52697c}.box{stroke:#436887;stroke-width:2;rx:14}.arrow{stroke:#49677f;stroke-width:3;fill:none;marker-end:url(#tip)}</style>
      <defs><marker id="tip" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="#49677f"/></marker></defs>
      <rect x="5" y="38" width="116" height="70" class="box" fill="#e9f4ff"/><text x="63" y="78" text-anchor="middle">Input x</text>
      <path d="M124 73 H149" class="arrow"/>
      <rect x="156" y="38" width="174" height="70" class="box" fill="#e7f8f5"/><text x="243" y="69" text-anchor="middle">LayerNorm +</text><text x="243" y="89" text-anchor="middle">Attention</text>
      <path d="M333 73 H358" class="arrow"/>
      <rect x="365" y="38" width="105" height="70" class="box" fill="#fff4df"/><text x="417" y="78" text-anchor="middle">Add x</text>
      <path d="M473 73 H498" class="arrow"/>
      <rect x="505" y="38" width="125" height="70" class="box" fill="#f4edff"/><text x="567" y="69" text-anchor="middle">LayerNorm</text><text x="567" y="89" text-anchor="middle">+ MLP</text>
      <path d="M633 73 H658" class="arrow"/>
      <rect x="665" y="38" width="88" height="70" class="box" fill="#fff4df"/><text x="709" y="69" text-anchor="middle">Add</text><text x="709" y="89" text-anchor="middle">input</text>
      <text x="382" y="137" class="small" text-anchor="middle">Stack 12 such layers, each with its own weights</text>
    </svg>'''))
    """,
    tags=["diagram"],
)

add_code(
    """
    class TransformerLayer(nnx.Module):
        def __init__(self, config, rngs: nnx.Rngs):
            self.mhsa = MultiHeadSelfAttention(config, rngs)
            self.mlp = MultiLayerPerceptron(config, rngs)
            self.ln1 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
            self.ln2 = nnx.LayerNorm(config.hidden_size, rngs=rngs)

        def __call__(self, x):
            x = x + self.mhsa(self.ln1(x))
            return x + self.mlp(self.ln2(x))
    """,
    tags=["model"],
)

add_markdown(
    """
    ### Block D: the complete GPT

    `wte` is the token embedding table; `wpe` is the learned position table.
    The transformer layers process their sum. `ln_f` normalizes the final
    features, and `wte.attend` turns them back into vocabulary scores. Reusing
    the embedding table there is called **weight tying**.
    """
)

add_code(
    """
    class GuildGPT(nnx.Module):
        def __init__(self, config, rngs: nnx.Rngs):
            self.config = config
            self.wte = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
            self.wpe = nnx.Embed(config.max_seq_len, config.hidden_size, rngs=rngs)
            self.layers = nnx.List([TransformerLayer(config, rngs) for _ in range(config.n_layers)])
            self.ln_f = nnx.LayerNorm(config.hidden_size, rngs=rngs)

        def __call__(self, input_ids):
            if input_ids.shape[1] > self.config.max_seq_len:
                raise ValueError("Input exceeds model context length")
            x = self.wte(input_ids) + self.wpe(jnp.arange(input_ids.shape[1]))
            for layer in self.layers:
                x = layer(x)
            return self.wte.attend(self.ln_f(x))
    """,
    tags=["model"],
)

add_markdown(
    """
    ### A quick shape check

    This is a **randomly initialized toy model**, not the trained one. For one
    input sequence, its output has one vocabulary-score vector per token.
    """
)

add_code(
    """
    toy_config = GuildGPTConfig(
        max_seq_len=16, vocab_size=64, n_layers=2, n_heads=4,
        hidden_size=64, mlp_hidden_size=128,
    )
    toy_model = GuildGPT(toy_config, nnx.Rngs(0))
    toy_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    toy_logits = toy_model(toy_ids)
    assert toy_logits.shape == (1, 5, 64)
    print("Input IDs:", toy_ids.shape, "→ scores:", toy_logits.shape)
    """,
    tags=["demo", "synthetic"],
)

add_markdown(
    """
    ## 4. What training does (tiny demonstration only)

    For each position, we compare the model's prediction to the **next** token.
    Cross-entropy is low when the correct next token gets high probability.
    Automatic differentiation computes gradients; AdamW changes the weights.

    The next two cells run **one** update on five made-up token IDs. This is
    only to make training visible; it is **not** the real WritingPrompts
    training procedure. The actual run used masks for partial windows,
    gradient accumulation, validation, a learning-rate schedule, and
    checkpoints. We deliberately do not replay it in Colab.
    """
)

add_code(
    """
    def compute_loss(model, input_ids):
        logits = model(input_ids[:, :-1]).astype(jnp.float32)
        targets = input_ids[:, 1:]
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return losses.mean()

    @nnx.jit
    def toy_train_step(model, optimizer, input_ids):
        loss, grads = nnx.value_and_grad(compute_loss)(model, input_ids)
        optimizer.update(model, grads)
        return loss
    """,
    tags=["training-explanation", "synthetic"],
)

add_code(
    """
    toy_optimizer = nnx.Optimizer(toy_model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)
    loss_before = float(compute_loss(toy_model, toy_ids))
    loss_during = float(toy_train_step(toy_model, toy_optimizer, toy_ids))
    loss_after = float(compute_loss(toy_model, toy_ids))
    print(f"Toy loss before: {loss_before:.3f} | after one update: {loss_after:.3f}")
    print("The real model below is separate; these toy weights are not used again.")
    """,
    tags=["training-explanation", "synthetic"],
)

add_markdown(
    """
    ### What happened in the actual training run?

    A **124,439,808-parameter** GuildGPT learned from scratch for one epoch on
    about **216.9 million training tokens**. The effective batch was 8 windows
    of up to 1,024 tokens, assembled through eight micro-steps. At the final
    checkpoint (step **26,482**), held-out validation loss was **3.357** and
    perplexity was **28.71**. Those numbers show learning, not a guarantee of
    polished stories: this small model can be repetitive or incoherent.

    The complete training code lives in the project repository, not in the
    attendee notebook. We now switch from **learning weights** to **using the
    already learned weights**.
    """
)

add_markdown(
    """
    ## 5. Load the trained weights from Hugging Face

    Hugging Face holds a `config.json` (model dimensions), a
    `model.safetensors` file (learned numbers), and tokenizer files. The code
    we just wrote defines *what those numbers mean*. The next helper checks
    every weight name, shape, and dtype before placing values into **our
    notebook's own `GuildGPT` object**.

    This is a custom Flax NNX model, so `AutoModelForCausalLM.from_pretrained`
    is **not** the right loader. The public model repository must be uploaded
    by the facilitator before this section can run.
    """
)

add_code(
    """
    def flatten_weights(tree, prefix=()):
        result = {}
        for key, value in tree.items():
            path = prefix + (str(key),)
            if isinstance(value, dict):
                result.update(flatten_weights(value, path))
            else:
                result[".".join(path)] = value
        return result

    def load_weights_into(model, model_dir):
        tensors = load_file(str(Path(model_dir) / "model.safetensors"))
        state = nnx.state(model, nnx.Param)
        template = nnx.to_pure_dict(state)
        expected = flatten_weights(template)
        if set(tensors) != set(expected):
            missing = sorted(set(expected) - set(tensors))[:5]
            extra = sorted(set(tensors) - set(expected))[:5]
            raise ValueError(f"Weight names differ: missing={missing}, extra={extra}")

        def build(tree, prefix=()):
            result = {}
            for key, value in tree.items():
                path = prefix + (str(key),)
                if isinstance(value, dict):
                    result[key] = build(value, path)
                else:
                    name = ".".join(path)
                    array = tensors[name]
                    if array.shape != value.shape or array.dtype != np.asarray(value).dtype:
                        raise ValueError(f"Wrong shape or dtype for {name}: {array.shape}, {array.dtype}")
                    result[key] = jax.device_put(array)
            return result

        nnx.replace_by_pure_dict(state, build(template))
        nnx.update(model, state)
        return len(tensors)
    """,
    tags=["hub"],
)

add_markdown(
    """
    **Facilitator setup:** publish the export as the public model repository
    `thealepo/GuildGPT`. Then attendees can run this cell without credentials.
    The download is around **0.5 GB** and
    is cached within that Colab session; allow time for it on workshop Wi-Fi.
    The first model construction and generation also take extra time.
    """
)

add_code(
    """
    MODEL_REPO_ID = "thealepo/GuildGPT"

    try:
        model_dir = Path(snapshot_download(
            repo_id=MODEL_REPO_ID,
            allow_patterns=[
                "config.json", "model.safetensors", "tokenizer*",
                "special_tokens_map.json", "vocab.json", "merges.txt",
            ],
        ))
    except Exception as exc:
        raise RuntimeError(
            "Could not download the model. Check that MODEL_REPO_ID is correct "
            "and the Hugging Face repository is public."
        ) from exc

    trained_config = GuildGPTConfig(**json.loads((model_dir / "config.json").read_text()))
    if trained_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match trained model")
    trained_model = GuildGPT(trained_config, nnx.Rngs(42))
    n_tensors = load_weights_into(trained_model, model_dir)
    print(f"Loaded {n_tensors} trained weight tensors from {MODEL_REPO_ID}")
    print(f"Layers: {trained_config.n_layers} | Hidden size: {trained_config.hidden_size}")
    """,
    tags=["hub", "network"],
)

add_markdown(
    """
    ## 6. Generate a story together

    We repeatedly score the next token, keep the most likely `top_k` options,
    and sample one. `temperature` controls randomness: lower is safer and more
    repetitive; higher is more surprising and sometimes less coherent.

    This version keeps a **fixed-length input** during each generation so JAX
    compiles the model only once for that length. Later positions are merely
    placeholders; causal attention prevents them from changing predictions at
    earlier positions. The **first** generation may pause for compilation.
    Keep the prompt and `max_new_tokens` modest for a live workshop.
    """
)

add_code(
    """
    @nnx.jit
    def next_token_logits(model, padded_ids, position):
        return model(padded_ids)[0, position].astype(jnp.float32)

    def generate_text(model, tokenizer, prompt, *, max_new_tokens=40,
                      temperature=0.8, top_k=40, seed=42):
        if max_new_tokens < 0 or temperature <= 0 or top_k < 1:
            raise ValueError("Use nonnegative max_new_tokens, positive temperature/top_k")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("Please enter a nonempty prompt")
        total_length = len(ids) + max_new_tokens
        if total_length > model.config.max_seq_len:
            raise ValueError("Prompt + generated tokens exceed this model's 1024-token context")
        context_size = next(
            (size for size in (64, 128, 256, 512, 1024) if size >= total_length),
            None,
        )
        buffer = np.full((1, context_size), tokenizer.eos_token_id, dtype=np.int32)
        buffer[0, :len(ids)] = ids
        key = jax.random.PRNGKey(seed)

        for _ in range(max_new_tokens):
            logits = next_token_logits(
                model, jnp.asarray(buffer), jnp.asarray(len(ids) - 1, dtype=jnp.int32)
            ) / temperature
            values, options = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
            key, subkey = jax.random.split(key)
            choice = int(options[jax.random.categorical(subkey, values)])
            if choice == tokenizer.eos_token_id:
                break
            buffer[0, len(ids)] = choice
            ids.append(choice)
        return tokenizer.decode(ids, skip_special_tokens=True)
    """,
    tags=["generation"],
)

add_code(
    r"""
    PROMPT = "The door had been locked for a hundred years. When it opened,"
    MAX_NEW_TOKENS = 40
    TEMPERATURE = 0.8
    TOP_K = 40
    SEED = 42

    started = time.perf_counter()
    story = generate_text(
        trained_model, tokenizer, PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE, top_k=TOP_K, seed=SEED,
    )
    print(story)
    print(f"\nGenerated in {time.perf_counter() - started:.1f}s (includes first-use compilation)")
    """,
    tags=["generation", "demo"],
)

add_markdown(
    """
    ### Your turn

    Change `PROMPT` and rerun **only the last cell**. Try a mystery opening, a
    science-fiction premise, or a sentence you invent. Then change
    `TEMPERATURE` between `0.6` and `1.0` or `TOP_K` between `20` and `50`.
    What changes? Where does the model sound convincing, and where does it
    lose the plot?

    **Takeaway:** you built the architecture in front of you, saw a tiny
    training step, loaded the real trained weights, and used them for next-token
    generation. The model is a workshop artifact, not a fact-checker or a
    production writing assistant; user-generated training text can contain
    biases or offensive material.
    """
)


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
output = Path(__file__).resolve().parents[1] / "notebooks" / "guildgpt_workshop.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(f"Wrote {output} ({len(cells)} cells)")
