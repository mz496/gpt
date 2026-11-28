"""
The same comparison as compare_tokenizers (char, bpe and regex_bpe at each vocab size, same names and colors, one wandb view),
but with every run trained on the `scaleup` training config instead of the default one.

Usage (from the repo root, with wandb logged in):
    just compare-tokenizers-scaleup
    just compare-tokenizers-scaleup --tokenizers char regex_bpe --max-parallel 5 --entity my-team
    just compare-tokenizers-scaleup --modal --tokenizers char regex_bpe --max-parallel 5 --train-args --source tiny-stories
    uv run --locked -m scripts.compare_tokenizers_scaleup --help

The arguments are compare_tokenizers's (see its docstring); only the default view name differs ("tokenizers-scaleup").
"""

import argparse

from scripts.compare_tokenizers import build_specs, parse_args as parse_compare_args, run_comparison
from src.tiny_shakespeare.wandb_views import resolve_entity

CONFIG = "scaleup"
DEFAULT_NAME = "tokenizers-scaleup"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_compare_args(argv, default_name=DEFAULT_NAME)


def main():
    args = parse_args()

    run_comparison(build_specs(args.sizes, args.train_args, args.tokenizers), args.name, resolve_entity(args.entity), args.project, args.max_parallel, args.log_dir, config=CONFIG, modal=args.modal)


if __name__ == "__main__":
    main()
