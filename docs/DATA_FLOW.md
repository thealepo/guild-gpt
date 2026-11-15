# Data and state flow

WritingPrompts `story` records are assigned to train or validation by the
normalized `prompt` hash. This prevents stories from one writing prompt from
appearing in both splits. The GPT-2 tokenizer emits token IDs, and an EOS token
is appended to each nonempty story. Each split becomes a single `uint16`
binary file and a manifest with counts and SHA-256 checksums.

The loader maps a binary file without reading it all into memory. For a
sequence length `L`, window `i` starts at token `i * L` and contains up to
`L+1` tokens. Its input is the first `L` tokens and its targets are the next
`L`. Thus adjacent windows cover every prediction target exactly once. The
last window may be short. Padding provides fixed tensor shapes; the loss mask
excludes padded targets. The order of windows is reshuffled for each epoch,
while token order inside a window remains unchanged.

One microbatch is passed through the custom GuildGPT model. Cross-entropy is
computed in FP32 and weighted by valid target count. Optax accumulates several
microbatch gradients before one clipped AdamW update. The learning rate uses
the count of actual updates derived from the manifest and run settings.

Orbax checkpoints store all parameters, Optax state, optimizer update count,
microbatch count, target count, next data position, config, dataset manifest,
and W&B run ID. Because checkpoints are written after complete accumulation
groups, there is no pending partial gradient to recover. Validation walks a
fixed nonshuffled loader and logs token-weighted metrics. SafeTensors export
extracts model parameters only; the Hub helper rebuilds GuildGPT from the
JSON config and loads those tensors into its NNX state.
