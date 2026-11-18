"""
Compares char vs regex_bpe (vocab size 2048) on the `scaleup` training config, both trained in parallel.

Same flow as compare_tokenizers: train, then add the runs to a new wandb view, renamed and colored
(char = amber-600, regex_bpe = indigo-600).

Usage (from the repo root, with wandb logged in):
    just compare-tokenizers-scaleup
    just compare-tokenizers-scaleup --entity my-team --max-parallel 1
    just compare-tokenizers-scaleup --modal
    just compare-tokenizers-scaleup --modal --train-args --source tiny-stories --tokenizer-text-only
    uv run --locked -m scripts.compare_tokenizers_scaleup --help

Arguments:
    --name NAME           view name suffix; the view is named "<timestamp>_<NAME>" (default: tokenizers-scaleup)
    --max-parallel N      CAP on trainings running at once (default: 2, i.e. both together)
    --modal               train on Modal (`just modal-train`) instead of locally; the wandb view is still built locally
    --entity ENTITY       wandb entity (default: your default entity)
    --project PROJECT     wandb project (default: gpt-experiments)
    --log-dir DIR         where each run's training output goes (default: a fresh temp dir, printed at start)
    --train-args ...      everything after this is passed to both trainings, e.g. --source tiny-stories --tokenizer-text-only. Put it last.
"""

import argparse
from pathlib import Path

from scripts.compare_tokenizers import AMBER_600, INDIGO_300_TO_600, RunSpec, run_comparison
from scripts.wandb_views import DEFAULT_PROJECT, resolve_entity

CONFIG = "scaleup"
BPE_VOCAB_SIZE = 2048


def build_specs(train_args: list[str] | None = None) -> list[RunSpec]:
    """The two runs. `train_args` are extra arguments for the training of each, e.g. ["--source", "tiny-stories"]."""
    return [
        RunSpec("char", "char", None, AMBER_600, "tokenizer=char", extra_args=list(train_args or [])),
        RunSpec(
            label=f"regex_bpe_{BPE_VOCAB_SIZE}",
            tokenizer="regex_bpe",
            bpe_vocab_size=BPE_VOCAB_SIZE,
            color=INDIGO_300_TO_600[-1],
            display_name=f"tokenizer=regex_bpe | bpe_vocab_size={BPE_VOCAB_SIZE}",
            extra_args=list(train_args or []),
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Compare char vs regex_bpe-{BPE_VOCAB_SIZE} on the {CONFIG} config.")
    parser.add_argument("--name", default="tokenizers-scaleup")
    parser.add_argument("--max-parallel", type=int, default=1, help="cap on trainings running at once")
    parser.add_argument("--modal", action="store_true", help="train on Modal instead of locally")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--log-dir", type=Path, default=None)
    # REMAINDER rather than one quoted string, so that it also works through `just`, which pastes its arguments into the command unquoted.
    parser.add_argument("--train-args", nargs=argparse.REMAINDER, default=[], help="everything after this is passed to both trainings, e.g. --train-args --source tiny-stories --tokenizer-text-only (put it last)")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    run_comparison(build_specs(args.train_args), args.name, resolve_entity(args.entity), args.project, args.max_parallel, args.log_dir, config=CONFIG, modal=args.modal)


if __name__ == "__main__":
    main()
