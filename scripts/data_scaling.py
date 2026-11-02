"""
Learning-curve experiment: how does validation loss scale with the amount of training data?

Trains char, bpe (vocab 2048) and regex_bpe (vocab 2048) on the `default` config, each on 12.5%, 25%, 50% and
100% of the train split, sampled as random 1000-char chunks that are nested across fractions (the validation split is always the full one), and collects all 12 runs in one wandb view.
Colors are per tokenizer: char = amber-600, bpe = lime-600, regex_bpe = indigo-600.

To read it: plot each run's best (minimum) validation/loss_bpc against the train_chars config value, log-log, one line per tokenizer.

Usage (from the repo root, with wandb logged in):
    just data-scaling
    just data-scaling --max-parallel 2 --fractions 0.25 1.0
    uv run --locked -m scripts.data_scaling --help

Arguments:
    --name NAME           view name suffix; the view is named "<timestamp>_<NAME>" (default: data-scaling)
    --fractions F [F ...] fractions of the train split to train on (default: 0.125 0.25 0.5 1.0)
    --max-parallel N      CAP on trainings running at once (default: 4)
    --entity ENTITY       wandb entity (default: your default entity)
    --project PROJECT     wandb project (default: gpt-experiments)
    --log-dir DIR         where each run's training output goes (default: a fresh temp dir, printed at start)
"""

import argparse
from pathlib import Path

from scripts.compare_tokenizers import AMBER_600, INDIGO_600, LIME_600, RunSpec, run_comparison
from scripts.wandb_views import DEFAULT_PROJECT, resolve_entity

BPE_VOCAB_SIZE = 2048
DEFAULT_FRACTIONS = [0.125, 0.25, 0.5, 1.0]


def build_specs(fractions: list[float]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for tokenizer, color in [("char", AMBER_600), ("bpe", LIME_600), ("regex_bpe", INDIGO_600)]:
        vocab_size = None if tokenizer == "char" else BPE_VOCAB_SIZE
        for fraction in fractions:
            specs.append(RunSpec(
                label=f"{tokenizer}_{fraction:g}",
                tokenizer=tokenizer,
                bpe_vocab_size=vocab_size,
                color=color,
                display_name=f"tokenizer={tokenizer} | train_fraction={fraction:g}",
                extra_args=["--train-fraction", str(fraction)],
            ))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Train each tokenizer on increasing fractions of the train split.")
    parser.add_argument("--name", default="data-scaling")
    parser.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    parser.add_argument("--max-parallel", type=int, default=4, help="cap on trainings running at once")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    run_comparison(build_specs(args.fractions), args.name, resolve_entity(args.entity), args.project, args.max_parallel, args.log_dir)


if __name__ == "__main__":
    main()
