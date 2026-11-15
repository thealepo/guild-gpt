---
library_name: custom
tags:
  - jax
  - flax
  - text-generation
datasets:
  - andrewelawrence/writingPrompts-merged
---

# GuildGPT

GuildGPT is a decoder-only GPT trained from scratch on WritingPrompts stories.
It has 12 layers, 12 heads, a 768-dimensional hidden state, a 3,072-dimensional
GELU MLP, learned absolute positions, and tied GPT-2 token embeddings. It uses
the original GPT-2 tokenizer vocabulary but no pretrained GPT-2 weights.

Before publishing, replace this template with measured information:

- Dataset revision and provenance: [complete after preparation]
- Number of non-padding training target positions: [complete after training]
- Hardware and runtime: [complete after benchmarking]
- Optimizer, schedule, precision, batch size, and epochs: [complete after training]
- Final validation loss and perplexity: [complete after evaluation]
- Representative fixed-prompt generations: [complete after evaluation]
- GitHub source repository URL: [complete before publication]
- Licensing and redistribution review for Reddit-contributed stories: [complete before publication]

This model is intended for education and research into autoregressive language
modeling. It is not designed for safety-critical decisions. It may reproduce
biases, offensive language, or memorized phrases from user-generated stories;
evaluate outputs before reuse. The dataset repository's license should not be
assumed to cover every underlying story's redistribution rights.

Install the GuildGPT Python package from the source repository, then:

```python
from guildgpt.hub import load_from_hub
from guildgpt import generate

model, tokenizer = load_from_hub("MY_USERNAME/GuildGPT")
print(generate(model, tokenizer, "The spaceship landed on an unknown planet."))
```
