import jax
import jax.numpy as jnp
import numpy as np


def generate_tokens(
    model,
    input_ids,
    *,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=None,
    seed=42,
    eos_token_id=None,
):
    if temperature <= 0 or max_new_tokens < 0 or top_k < 1:
        raise ValueError("Invalid sampling configuration")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    ids = list(map(int, input_ids))
    if not ids:
        raise ValueError("Prompt needs at least one token")
    key = jax.random.PRNGKey(seed)
    for _ in range(max_new_tokens):
        context = jnp.asarray(ids[-model.config.max_seq_len :], dtype=jnp.int32)[None, :]
        logits = model(context)[0, -1].astype(jnp.float32) / temperature
        k = min(top_k, logits.shape[-1])
        values, indices = jax.lax.top_k(logits, k)
        if top_p is not None:
            probs = jax.nn.softmax(values)
            cumulative = jnp.cumsum(probs)
            keep = (cumulative - probs) < top_p
            values = jnp.where(keep, values, -jnp.inf)
        key, subkey = jax.random.split(key)
        token = int(indices[jax.random.categorical(subkey, values)])
        if token == eos_token_id:
            break
        ids.append(token)
    return ids


def generate(model, tokenizer, prompt: str, **kwargs) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    result = generate_tokens(model, ids, eos_token_id=tokenizer.eos_token_id, **kwargs)
    return tokenizer.decode(np.asarray(result), skip_special_tokens=True)
