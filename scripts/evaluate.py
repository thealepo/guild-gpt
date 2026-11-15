import argparse
import json
from pathlib import Path

from flax import nnx

from guildgpt import GuildGPT, GuildGPTConfig
from guildgpt.checkpoints import load_model_checkpoint
from guildgpt.config import RunConfig
from guildgpt.data import WindowLoader, load_tokens
from guildgpt.training import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    metadata = json.loads((Path(args.checkpoint) / "metadata.json").read_text())
    config = RunConfig(
        model=GuildGPTConfig(**metadata["config"]["model"]),
        **{k: v for k, v in metadata["config"].items() if k != "model"},
    )
    val = WindowLoader(
        load_tokens(config.data_dir, "validation"),
        config.model.max_seq_len,
        config.micro_batch_size,
        shuffle=False,
    )
    model = GuildGPT(config.model, nnx.Rngs(config.seed))
    load_model_checkpoint(args.checkpoint, model)
    print(json.dumps(evaluate(model, val), indent=2))


if __name__ == "__main__":
    main()
