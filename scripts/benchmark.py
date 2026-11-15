import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from guildgpt import GuildGPT, load_config
from guildgpt.data import WindowLoader
from guildgpt.training import compute_loss_sum, create_optimizer, train_update


def gpu_info():
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return raw.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hourly-price", type=float, default=None)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("Use at least two measured steps")
    config = load_config(args.config)
    if not config.synthetic and jax.default_backend() != "gpu":
        raise RuntimeError("Full-size benchmark requires a working JAX GPU backend")
    rng = np.random.default_rng(config.seed)
    n = (
        config.model.max_seq_len * config.micro_batch_size * config.gradient_accumulation_steps * 4
        + 1
    )
    tokens = rng.integers(0, config.model.vocab_size, size=n, dtype=np.uint16)
    loader = WindowLoader(tokens, config.model.max_seq_len, config.micro_batch_size)
    model = GuildGPT(config.model, nnx.Rngs(config.seed))
    optimizer, _ = create_optimizer(model, config, max(config.warmup_steps + 2, 100))
    batch = loader.get_batch(0, 0)
    ids, mask = jnp.asarray(batch.tokens), jnp.asarray(batch.mask)
    grad_fn = nnx.jit(nnx.value_and_grad(compute_loss_sum))
    start = time.perf_counter()
    grad_fn(model, ids, mask)[0].block_until_ready()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    grad_fn(model, ids, mask)[0].block_until_ready()
    forward_backward_seconds = time.perf_counter() - start
    train_update(model, optimizer, loader, 0, 0, config)["loss"].block_until_ready()
    start = time.perf_counter()
    processed = 0
    for _ in range(args.steps):
        result = train_update(model, optimizer, loader, 0, 0, config)
        processed += result["tokens"]
    result["loss"].block_until_ready()
    step_seconds = (time.perf_counter() - start) / args.steps
    rate = processed / (step_seconds * args.steps)
    manifest_path = Path(config.data_dir) / "manifest.json"
    corpus_tokens = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        corpus_tokens = manifest.get("train_tokens")
    epoch_seconds = (corpus_tokens - 1) / rate if corpus_tokens and rate else None
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "devices": list(map(str, jax.devices())),
                "gpu": gpu_info(),
                "parameter_count": config.model.parameter_count(),
                "micro_batch_size": config.micro_batch_size,
                "accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": config.micro_batch_size
                * config.gradient_accumulation_steps,
                "compile_seconds": compile_seconds,
                "forward_backward_seconds_per_microbatch": forward_backward_seconds,
                "optimizer_update_seconds": step_seconds,
                "tokens_per_second": rate,
                "epoch_hours": epoch_seconds / 3600 if epoch_seconds else None,
                "epoch_cost_usd": epoch_seconds / 3600 * args.hourly_price
                if epoch_seconds and args.hourly_price is not None
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
