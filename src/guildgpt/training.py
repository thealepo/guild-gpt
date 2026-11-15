from __future__ import annotations

import math

import jax.numpy as jnp
import optax
from flax import nnx


def updates_per_epoch(loader, accumulation_steps: int) -> int:
    return math.ceil(loader.num_batches / accumulation_steps)


def make_schedule(config, total_steps: int):
    if total_steps <= config.warmup_steps:
        raise ValueError("Warmup must be shorter than the actual optimizer-update plan")
    return optax.warmup_cosine_decay_schedule(
        init_value=config.init_lr,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=total_steps,
        end_value=config.final_lr,
    )


def create_optimizer(model, config, total_steps: int):
    schedule = make_schedule(config, total_steps)
    tx = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(schedule, weight_decay=config.weight_decay),
        ),
        every_k_schedule=config.gradient_accumulation_steps,
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param), schedule


def compute_loss_sum(model, ids, mask):
    logits = model(ids[:, :-1]).astype(jnp.float32)
    targets = ids[:, 1:]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.sum(losses * mask)


@nnx.jit
def micro_step(model, optimizer, ids, mask, normalization):
    # MultiSteps averages k gradients; multiply each microstep by k so the
    # final update is a token-weighted mean even for a partial final batch.
    def loss_fn(model):
        return compute_loss_sum(model, ids, mask) * normalization

    scaled_loss, grads = nnx.value_and_grad(loss_fn)(model)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return scaled_loss / normalization, grad_norm


@nnx.jit
def eval_step(model, ids, mask):
    return compute_loss_sum(model, ids, mask)


def train_update(model, optimizer, loader, epoch, first_batch, config):
    k = config.gradient_accumulation_steps
    batches = [
        loader.get_batch(epoch, i)
        for i in range(first_batch, min(first_batch + k, loader.num_batches))
    ]
    token_count = sum(int(b.mask.sum()) for b in batches)
    if token_count == 0:
        raise ValueError("No valid targets in update")
    loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
    grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
    for batch in batches:
        loss, norm = micro_step(
            model, optimizer, jnp.asarray(batch.tokens), jnp.asarray(batch.mask), k / token_count
        )
        loss_sum = loss_sum + loss
        grad_norm = grad_norm + norm
    # Complete the accumulation group with zero gradients. A single microbatch
    # stays resident on device; the effective batch is never concatenated.
    if len(batches) < k:
        empty = jnp.zeros_like(jnp.asarray(batches[0].tokens))
        zero_mask = jnp.zeros_like(jnp.asarray(batches[0].mask))
        for _ in range(k - len(batches)):
            micro_step(model, optimizer, empty, zero_mask, k / token_count)
    return {
        "loss": loss_sum / token_count,
        "grad_norm": grad_norm / len(batches),
        "tokens": token_count,
        "micro_steps": len(batches),
    }


def evaluate(model, loader):
    loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
    tokens = 0.0
    for i in range(loader.num_batches):
        batch = loader.get_batch(0, i)
        count = float(batch.mask.sum())
        if count:
            loss_sum = loss_sum + eval_step(
                model, jnp.asarray(batch.tokens), jnp.asarray(batch.mask)
            )
            tokens += count
    mean = float(loss_sum) / tokens
    return {"loss": mean, "perplexity": math.exp(min(mean, 80)), "tokens": int(tokens)}
