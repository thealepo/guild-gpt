# GuildGPT

GuildGPT is a decoder-only language model implemented in JAX and Flax NNX for
the AI Guild workshop. The network is our own `GuildGPT(nnx.Module)`, with
separate Q/K/V projections, causal attention, pre-layer-normalized residual
blocks, GELU MLPs, learned token and position embeddings, and tied output
weights. The full configuration has 12 layers, 12 heads, a 768-dimensional
hidden state, a 3,072-dimensional MLP, and **124,439,808 parameters**. No
pretrained model weights are used. The GPT-2 tokenizer supplies only the
original 50,257-token vocabulary.

The original, experimental notebook is preserved unchanged at
`notebooks/prototype_archive.ipynb`. It is not a sequential training script.
The production path uses only WritingPrompts stories; its Gutenberg exploration
is not part of preprocessing.

## Local setup and checks

Use Python 3.11 or 3.12 on Linux. In a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check src/guildgpt scripts tests
python scripts/train.py --config configs/smoke.yaml
```

The smoke configuration creates 75 synthetic train tokens and 29 validation
tokens locally, trains a tiny model, evaluates, and saves Orbax checkpoints.
It never contacts Hugging Face or W&B. The complete integration test also
restores a checkpoint, continues training, exports SafeTensors, reloads the
model, and generates tokens. The smoke training command writes to
`outputs/smoke-checkpoints`; use a new `checkpoint_dir` for another fresh run,
or `--resume outputs/smoke-checkpoints/step_00000003` to load the final saved
smoke run. Resume from the newest checkpoint in a directory; use a new
`checkpoint_dir` if you intentionally start from an older checkpoint.

## Data and full training

WritingPrompts is loaded from `andrewelawrence/writingPrompts-merged`. Only
`story` text is trained on. Normalized prompts are grouped by a SHA-256 hash;
the first eight digest bytes modulo 100 assign about 5% of prompt groups to
validation. The exact split algorithm is in `src/guildgpt/data.py` and recorded
in `manifest.json`. Stories are tokenized without new tokens, terminated with
GPT-2 EOS, concatenated, and stored as `uint16` memmaps. EOS makes crossing
story boundaries explicit. The manifest records counts, dataset fingerprint,
revision if provided, tokenizer identity, and binary checksums. Existing
files are verified before reuse; use `--force` only when you intend to replace
prepared data.

```bash
python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("andrewelawrence/writingPrompts-merged").sha)'
python scripts/prepare_data.py --output data/processed --revision DATASET_COMMIT_SHA
python scripts/benchmark.py --config configs/gpt2-small.yaml --hourly-price 1.50
python scripts/train.py --config configs/gpt2-small.yaml
```

Replace `DATASET_COMMIT_SHA` with a revision you have checked on the dataset
page, and replace the hourly price with your actual rental price. Dataset
preparation downloads and tokenizes the full corpus; do it only when ready.
The benchmark uses synthetic tokens and warmup before timing; it reads the
prepared manifest only to estimate epoch time and cost. Compare both attention
backends and FP32/BF16 using measured throughput before selecting a full-run
configuration. The shipped full config defaults to reference attention and
FP32 until GPU measurements justify a change.

An epoch covers every next-token target position from `1` through `N-1` once.
Windows start every `seq_len` tokens; the final partial window and batch use
loss masks. Windows shuffle deterministically by epoch and seed. One optimizer
update consumes up to `gradient_accumulation_steps` microbatches without
materializing their concatenation. The schedule is built from the actual
window, batch, accumulation, and epoch counts. `global_step` counts optimizer
updates, `micro_step` counts real microbatches, and `tokens_seen` counts valid
targets. Validation walks every window in stable order and reports token-
weighted cross-entropy and perplexity.

The full config enables W&B project `GuildGPT`. Log in with `wandb login` or
set `WANDB_API_KEY`. Its run ID is stored in each checkpoint and reused on
resume. To disable tracking, set `wandb_project: null`. Checkpoints include
the model, complete Optax optimizer state, data position, counts, configuration,
and dataset manifest. The shuffle order is rebuilt from seed and epoch, and
the accumulation group is completed before each checkpoint. Checkpoints are
saved every configured number of updates or minutes, plus the final state;
the oldest complete checkpoints beyond `keep_checkpoints` are removed. Put
`checkpoint_dir` on a persistent volume before cloud training.

```bash
python scripts/train.py --config configs/gpt2-small.yaml --resume /persistent/checkpoints/step_LATEST
python scripts/evaluate.py --checkpoint /persistent/checkpoints/step_LATEST
python scripts/generate.py --checkpoint /persistent/checkpoints/step_LATEST --prompt 'The last human on Earth'
```

Replace `step_LATEST` with the actual newest numbered checkpoint. The resume
command requires the same configuration and data manifest. For
generation, only the checkpoint is needed; the GPT-2 tokenizer may be fetched
if not cached.

## A100 setup and benchmarking

On a fresh Linux host with an NVIDIA driver supporting CUDA 12, create a
Python 3.11/3.12 environment and install the package with its CUDA extra.
The extra includes the matching JAX CUDA 12 plugin and pip-managed CUDA/cuDNN
libraries for the pinned JAX version:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,cuda]'
python -c 'import jax; print(jax.__version__, jax.default_backend(), jax.devices())'
nvidia-smi
```

If JAX reports `cpu`, resolve the driver/CUDA plugin mismatch before paying
for training. The benchmark reports device, parameter count, GPU memory when
available, compilation time, forward/backward time, update time, token rate,
and projected epoch time and price. These are measurements from the target
machine, never estimates based on the notebook's smaller 77M-parameter pilot.
Run the benchmark with several microbatch and accumulation settings by editing
the YAML (or separate copies) and compare reference and `sdpa` attention.

On Debian/Ubuntu, if `python -m venv .venv` reports that `ensurepip` is
unavailable, install the matching `python3.12-venv` (or `python3.11-venv`)
system package, then recreate the environment. Do not mix JAX CUDA 12 and
CUDA 13 plugin packages in one environment. If Orbax warns about missing
sharding metadata on restore, compare model outputs after restoring on the
target machine before continuing a paid run.

## Export and distribution

First complete the model card template in `docs/MODEL_CARD_TEMPLATE.md` with
actual training measurements, generations, dataset provenance, and licensing
review. Local export does not need Hub credentials:

```bash
python scripts/export_hub.py --checkpoint /persistent/checkpoints/step_FINAL --output outputs/hub-export
```

To upload after the card is complete, use your chosen Hub repository ID and
an authenticated `huggingface-cli login` or `HF_TOKEN`:

```bash
python scripts/export_hub.py --checkpoint /persistent/checkpoints/step_FINAL --output outputs/hub-export --model-card docs/completed-model-card.md --repo-id MY_USERNAME/GuildGPT
```

`step_FINAL` stands for the actual numbered final checkpoint directory.
The exported repository contains `config.json`, `model.safetensors`, GPT-2
tokenizer files, and `README.md`. It contains no optimizer state or dataset.
Load it into our class with:

```python
from guildgpt.hub import load_from_hub
from guildgpt import generate

model, tokenizer = load_from_hub("MY_USERNAME/GuildGPT")
print(generate(model, tokenizer, "The spaceship landed on an unknown planet."))
```

The model repository does not implement Transformers `AutoModelForCausalLM`.
The source package must be installed to reconstruct GuildGPT. Checkpoint
restoration may warn about missing explicit sharding metadata on a different
device topology; validate a cross-host restore before relying on it in a
long GPU run. GPU throughput, BF16 speed, Hub upload, and W&B online resume
still require the target hardware or credentials.

See `docs/DATA_FLOW.md` for the training data and state path.
