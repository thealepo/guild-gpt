import argparse
import json

from guildgpt.data import prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--revision", required=True, help="Pinned dataset commit SHA")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.output, revision=args.revision, force=args.force), indent=2))


if __name__ == "__main__":
    main()
