import argparse
import json

from flax import nnx

from guildgpt import GuildGPT, GuildGPTConfig, generate
from guildgpt.checkpoints import load_model_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from pathlib import Path

    metadata = json.loads((Path(args.checkpoint) / "metadata.json").read_text())
    config = GuildGPTConfig(**metadata["config"]["model"])
    model = GuildGPT(config, nnx.Rngs(args.seed))
    load_model_checkpoint(args.checkpoint, model)
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
