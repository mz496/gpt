"""
A vibed script to compare char + BPE on diff vocab sizes + regex-then-BPE on diff vocab sizes, then collect everything into one W&B view.

For every run it: trains with `just gpt-train`, renames the wandb run to "tokenizer=T | bpe_vocab_size=N",
adds it to a new saved workspace view, and colors it (char = lime-600, bpe = teal 300..600, regex_bpe = blue 300..600,
with the shade rising with vocab size).

Usage (from the repo root, with wandb logged in):
    just compare-tokenizers
    just compare-tokenizers --sizes 300 512 --max-parallel 2 --entity my-team
    uv run --locked -m scripts.compare_tokenizers --help

Arguments:
    --name NAME           view name suffix; the view is named "<timestamp>_<NAME>" (default: tokenizers-test)
    --sizes N [N ...]     bpe_vocab_size values for bpe and regex_bpe (default: 300 512 1024 2048)
    --max-parallel N      CAP on trainings running at once (default: 3). Others wait for a free slot.
    --entity ENTITY       wandb entity (default: your default entity)
    --project PROJECT     wandb project (default: gpt-experiments)
    --log-dir DIR         where each run's training output goes (default: a fresh temp dir, printed at start)

The wandb runs are also tagged with the view name as their wandb group.
Runs are trained in parallel (up to --max-parallel), then added to the view in a single edit afterwards.
If some trainings fail, the successful ones are still added and the script exits 1, listing the failures.
Note that renaming a run changes its name everywhere, not only in the view.
"""

import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import wandb

from scripts.wandb_views import DEFAULT_PROJECT, add_run, create_view, load_view, resolve_entity, set_run_color

# Tailwind v3 hex values (without the #).
LIME_600 = "65a30d"
TEAL_300_TO_600 = ["5eead4", "2dd4bf", "14b8a6", "0d9488"]
BLUE_300_TO_600 = ["93c5fd", "60a5fa", "3b82f6", "2563eb"]


@dataclass
class RunSpec:
    label: str
    tokenizer: str
    bpe_vocab_size: int | None
    color: str
    display_name: str


def shade_for(index: int, count: int, shades: list[str]) -> str:
    """Spreads `count` ordered items across the shades, lightest to darkest."""
    position = round(index * (len(shades) - 1) / max(count - 1, 1))
    return shades[position]


def build_specs(sizes: list[int]) -> list[RunSpec]:
    specs = [RunSpec("char", "char", None, LIME_600, "tokenizer=char")]
    for i, size in enumerate(sizes):
        for tokenizer, shades in [("bpe", TEAL_300_TO_600), ("regex_bpe", BLUE_300_TO_600)]:
            specs.append(RunSpec(
                label=f"{tokenizer}_{size}",
                tokenizer=tokenizer,
                bpe_vocab_size=size,
                color=shade_for(i, len(sizes), shades),
                display_name=f"tokenizer={tokenizer} | bpe_vocab_size={size}",
            ))
    return specs


def parse_wandb_run_id(log: str) -> str:
    """gpt-train ends with a line 'run_id=<timestamp>_<wandb id>'; returns the wandb id."""
    lines = [line for line in log.splitlines() if line.startswith("run_id=")]
    if not lines:
        raise ValueError("no run_id= line in the training output")
    return lines[-1].removeprefix("run_id=").rsplit("_", 1)[-1]


def train(spec: RunSpec, group: str, log_dir: Path) -> str:
    """Trains one run, writing its output to a log file. Returns the wandb run id. Raises if training fails."""
    command = ["just", "gpt-train", "--tokenizer", spec.tokenizer, "--group", group]
    if spec.bpe_vocab_size is not None:
        command += ["--bpe-vocab-size", str(spec.bpe_vocab_size)]
    log_path = log_dir / f"{spec.label}.log"
    with open(log_path, "w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return parse_wandb_run_id(log_path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Train and compare tokenizers, collected in one wandb view.")
    parser.add_argument("--name", default="tokenizers-test")
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 512, 1024, 2048])
    parser.add_argument("--max-parallel", type=int, default=3, help="cap on trainings running at once")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    entity = resolve_entity(args.entity)
    view_name = f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}_{args.name}"
    view_id = create_view(entity, args.project, view_name)
    log_dir = args.log_dir or Path(tempfile.mkdtemp())
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"view {view_id} ({view_name}); training logs in {log_dir}")

    specs = build_specs(args.sizes)

    # Phase 1: train. The pool size is what caps the parallelism.
    run_ids: dict[str, str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {spec.label: pool.submit(train, spec, view_name, log_dir) for spec in specs}
        for spec in specs:
            try:
                run_ids[spec.label] = futures[spec.label].result()
            except (subprocess.CalledProcessError, ValueError) as e:
                failures.append(f"{spec.label}: {e}")

    # Phase 2: rename the runs, then add and color them all in one load/save of the view.
    api = wandb.Api()
    workspace = load_view(entity, args.project, view_id)
    for spec in specs:
        run_id = run_ids.get(spec.label)
        if run_id is None:
            continue
        run = api.run(f"{entity}/{args.project}/{run_id}")
        run.name = spec.display_name
        run.update()
        add_run(workspace, run_id)
        set_run_color(workspace, run_id, spec.color)
    workspace.save()

    print(f"{len(run_ids)}/{len(specs)} runs added to view {view_id}")
    if failures:
        print("failed (see logs):\n  " + "\n  ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
