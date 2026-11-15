import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import jax
from flax import nnx

from guildgpt import GuildGPT, load_config
from guildgpt.checkpoints import latest_checkpoint, restore_checkpoint, save_checkpoint
from guildgpt.data import WindowLoader, file_sha256, load_tokens, prepare_synthetic
from guildgpt.training import create_optimizer, evaluate, train_update, updates_per_epoch

SAMPLE_PROMPTS = [
    "The robot had been programmed never to lie. Today, it broke that rule.",
    "The last human colony received a transmission from Earth.",
    "When the stars began disappearing, nobody believed the astronomers.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Complete checkpoint directory")
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.synthetic and jax.default_backend() != "gpu":
        raise RuntimeError("Full-corpus training requires a working JAX GPU backend")
    existing = latest_checkpoint(config.checkpoint_dir)
    if existing and (not args.resume or Path(args.resume).resolve() != existing.resolve()):
        raise FileExistsError(
            f"Checkpoint directory already contains {existing}; resume that checkpoint "
            "or choose a new checkpoint_dir"
        )
    data_dir = Path(config.data_dir)
    if config.synthetic:
        manifest = prepare_synthetic(data_dir, config.model.vocab_size, config.seed)
    else:
        manifest = json.loads((data_dir / "manifest.json").read_text())
        if manifest["dataset"] != "andrewelawrence/writingPrompts-merged":
            raise ValueError("Expected prepared WritingPrompts data")
        if manifest["tokenizer_vocab_size"] != config.model.vocab_size:
            raise ValueError("Tokenizer vocabulary differs from model")
        for split in ("train", "validation"):
            if file_sha256(data_dir / f"{split}.bin") != manifest[f"{split}_sha256"]:
                raise ValueError(f"{split} token file differs from manifest")
    train_loader = WindowLoader(
        load_tokens(data_dir, "train"),
        config.model.max_seq_len,
        config.micro_batch_size,
        config.seed,
    )
    val_loader = WindowLoader(
        load_tokens(data_dir, "validation"),
        config.model.max_seq_len,
        config.micro_batch_size,
        config.seed,
        shuffle=False,
    )
    total_updates = (
        updates_per_epoch(train_loader, config.gradient_accumulation_steps) * config.epochs
    )
    model = GuildGPT(config.model, nnx.Rngs(config.seed))
    optimizer, schedule = create_optimizer(model, config, total_updates)
    progress = {"global_step": 0, "micro_step": 0, "tokens_seen": 0, "epoch": 0, "batch_index": 0}
    run_id = None
    if args.resume:
        metadata = restore_checkpoint(
            args.resume, model, optimizer, config=config.as_dict(), dataset_manifest=manifest
        )
        progress = metadata["progress"]
        run_id = metadata["wandb_run_id"]
    wandb_run = None
    if config.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config={
                **config.as_dict(),
                "total_updates": total_updates,
                "dataset_manifest": manifest,
            },
            id=run_id,
            resume="must" if run_id else "allow",
        )
        run_id = wandb_run.id
        wandb_run.log(
            {
                "system/backend": jax.default_backend(),
                "system/devices": ", ".join(map(str, jax.devices())),
                "system/python": platform.python_version(),
                "system/jax": importlib.metadata.version("jax"),
                "system/flax": importlib.metadata.version("flax"),
                "system/optax": importlib.metadata.version("optax"),
            },
            step=progress["global_step"],
        )
    tokenizer = None
    if not config.synthetic and config.sample_every:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    started = last_checkpoint = time.monotonic()
    last_log_tokens = progress["tokens_seen"]
    last_log_time = started
    for epoch in range(progress["epoch"], config.epochs):
        first = progress["batch_index"] if epoch == progress["epoch"] else 0
        for index in range(first, train_loader.num_batches, config.gradient_accumulation_steps):
            metrics = train_update(model, optimizer, train_loader, epoch, index, config)
            progress["global_step"] += 1
            progress["micro_step"] += metrics["micro_steps"]
            progress["tokens_seen"] += metrics["tokens"]
            next_index = min(index + config.gradient_accumulation_steps, train_loader.num_batches)
            progress["epoch"], progress["batch_index"] = (
                (epoch + 1, 0) if next_index == train_loader.num_batches else (epoch, next_index)
            )
            step = progress["global_step"]
            if step % config.log_every == 0:
                loss_value = float(metrics["loss"])
                grad_norm_value = float(metrics["grad_norm"])
                now = time.monotonic()
                elapsed = max(now - last_log_time, 1e-9)
                rate = (progress["tokens_seen"] - last_log_tokens) / elapsed
                report = {
                    "train/loss": loss_value,
                    "train/grad_norm": grad_norm_value,
                    "train/learning_rate": float(schedule(step - 1)),
                    "train/tokens_seen": progress["tokens_seen"],
                    "train/global_step": step,
                    "train/epoch": epoch,
                    "train/global_batch_size": config.micro_batch_size
                    * config.gradient_accumulation_steps,
                    "train/tokens_per_second": rate,
                    "train/elapsed_seconds": now - started,
                }
                memory = jax.devices()[0].memory_stats()
                if memory:
                    if "bytes_in_use" in memory:
                        report["system/gpu_memory_bytes"] = memory["bytes_in_use"]
                    if "peak_bytes_in_use" in memory:
                        report["system/gpu_peak_memory_bytes"] = memory["peak_bytes_in_use"]
                print(json.dumps(report), flush=True)
                if wandb_run:
                    wandb_run.log(report, step=step)
                last_log_tokens, last_log_time = progress["tokens_seen"], now
            if step % config.eval_every == 0:
                val = evaluate(model, val_loader)
                print(json.dumps({"step": step, "validation": val}), flush=True)
                if wandb_run:
                    wandb_run.log(
                        {f"validation/{key}": value for key, value in val.items()}, step=step
                    )
            if tokenizer and wandb_run and step % config.sample_every == 0:
                import wandb

                from guildgpt.generation import generate

                table = wandb.Table(columns=["prompt", "generation"])
                for prompt in SAMPLE_PROMPTS:
                    table.add_data(
                        prompt,
                        generate(model, tokenizer, prompt, max_new_tokens=100, seed=config.seed),
                    )
                wandb_run.log({"samples": table}, step=step)
            due_time = (
                config.checkpoint_minutes
                and time.monotonic() - last_checkpoint >= 60 * config.checkpoint_minutes
            )
            if step % config.checkpoint_every == 0 or due_time:
                save_checkpoint(
                    config.checkpoint_dir,
                    model,
                    optimizer,
                    progress.copy(),
                    config.as_dict(),
                    manifest,
                    run_id,
                    config.keep_checkpoints,
                )
                last_checkpoint = time.monotonic()
    final = Path(config.checkpoint_dir) / f"step_{progress['global_step']:08d}"
    if not final.exists():
        save_checkpoint(
            config.checkpoint_dir,
            model,
            optimizer,
            progress,
            config.as_dict(),
            manifest,
            run_id,
            config.keep_checkpoints,
        )
    print(json.dumps({"final_checkpoint": str(final), "validation": evaluate(model, val_loader)}))
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
