import argparse
import json
from pathlib import Path

from flax import nnx

from guildgpt import GuildGPT, GuildGPTConfig
from guildgpt.checkpoints import load_model_checkpoint
from guildgpt.hub import save_pretrained, upload_to_hub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/hub-export")
    parser.add_argument("--repo-id", help="Hub repository to upload to; omit for local export")
    parser.add_argument("--model-card", help="Completed model card Markdown for publishing")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((Path(args.checkpoint) / "metadata.json").read_text())
    if metadata["config"]["synthetic"]:
        raise ValueError("Synthetic smoke checkpoints must not be published")
    model = GuildGPT(
        GuildGPTConfig(**metadata["config"]["model"]), nnx.Rngs(metadata["config"]["seed"])
    )
    load_model_checkpoint(args.checkpoint, model)
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if args.repo_id and not args.model_card:
        raise ValueError("Publishing requires --model-card with measured metrics and provenance")
    card = Path(args.model_card or "docs/MODEL_CARD_TEMPLATE.md").read_text()
    if args.repo_id and ("[complete" in card or "MY_USERNAME/GuildGPT" in card):
        raise ValueError("Model card still contains template placeholders")
    save_pretrained(model, args.output, tokenizer=tokenizer, model_card=card)
    print(f"Exported to {args.output}")
    if args.repo_id:
        upload_to_hub(args.output, args.repo_id, private=args.private)
        print(f"Uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
