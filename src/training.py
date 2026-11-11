from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def _eval_loss(model, ids, mask):
    logits = model(ids[:, :-1]).astype(jnp.float32)
    target = ids[:, 1:]

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, target)

    return jnp.sum(per_token_loss * mask) / jnp.maximum(1., jnp.sum(mask))


@nnx.jit
def train_step(model, optimizer, ids, mask, scale):
    def scaled_loss(model):
        logits = model(ids[:, :-1]).astype(jnp.float32)
        target = ids[:, 1:]

        losses = optax.softmax_cross_entropy_with_integer_labels(logits, target)

        return jnp.sum(losses * mask) * scale

    loss, grads = jax.value_and_grad(scaled_loss)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, input_ids, mask):
    return _eval_loss(model, input_ids, mask)


def create_optimizer(model, config, total_steps):
    if total_steps <= config.warmup_steps:
        raise ValueError("Total steps must be greater than warmup steps")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=total_steps,
        end_value=config.min_lr
    )

    tx = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
        ),
        every_k_schedule=config.grad_accumulation_steps,
    )

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    return optimizer, schedule


def evaluate(model, loader, max_batches: int = 16) -> float:
    weighted_sum, token_count = 0.0, 0.0

    for index in range(min(max_batches, loader.num_batches)):
        batch = loader.get_batch(epoch=0, batch_index=index)
        count = float(batch.mask.sum())

        if not count:
            continue

        loss = eval_step(model, jnp.asarry(batch.tokens), jnp.asarray(batch.mask))
        weighted_sum += float(loss) * count
        token_count += count

    return weighted_sum / max(1.0, token_count)