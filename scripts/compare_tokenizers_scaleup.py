"""
Compares char vs regex_bpe (vocab size 2048) on the `scaleup` training config, both trained in parallel.

Same flow as compare_tokenizers: train, then add the runs to a new wandb view, renamed and colored
(char = amber-600, regex_bpe = indigo-600).

Usage (from the repo root, with wandb logged in):
    just compare-tokenizers-scaleup
    just compare-tokenizers-scaleup --entity my-team --max-parallel 1
    uv run --locked -m scripts.compare_tokenizers_scaleup --help

Arguments:
    --name NAME           view name suffix; the view is named "<timestamp>_<NAME>" (default: tokenizers-scaleup)
    --max-parallel N      CAP on trainings running at once (default: 2, i.e. both together)
    --entity ENTITY       wandb entity (default: your default entity)
    --project PROJECT     wandb project (default: gpt-experiments)
    --log-dir DIR         where each run's training output goes (default: a fresh temp dir, printed at start)
"""

import argparse
from pathlib import Path

from scripts.compare_tokenizers import AMBER_600, INDIGO_300_TO_600, RunSpec, run_comparison
from scripts.wandb_views import DEFAULT_PROJECT, resolve_entity

CONFIG = "scaleup"
BPE_VOCAB_SIZE = 2048


def build_specs() -> list[RunSpec]:
    return [
        RunSpec("char", "char", None, AMBER_600, "tokenizer=char"),
        RunSpec(
            label=f"regex_bpe_{BPE_VOCAB_SIZE}",
            tokenizer="regex_bpe",
            bpe_vocab_size=BPE_VOCAB_SIZE,
            color=INDIGO_300_TO_600[-1],
            display_name=f"tokenizer=regex_bpe | bpe_vocab_size={BPE_VOCAB_SIZE}",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=f"Compare char vs regex_bpe-{BPE_VOCAB_SIZE} on the {CONFIG} config.")
    parser.add_argument("--name", default="tokenizers-scaleup")
    parser.add_argument("--max-parallel", type=int, default=1, help="cap on trainings running at once")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    run_comparison(build_specs(), args.name, resolve_entity(args.entity), args.project, args.max_parallel, args.log_dir, config=CONFIG)


if __name__ == "__main__":
    main()
