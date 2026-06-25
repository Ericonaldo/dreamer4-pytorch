from __future__ import annotations

import argparse
from pathlib import Path

from dreamer4.config import load_config
from dreamer4.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dreamer4")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. train.batch_size=32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
