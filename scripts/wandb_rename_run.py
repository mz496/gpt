"""
Set a wandb run's display name. This renames the run everywhere (the workspaces API can't rename a run only within one view).
"""

import argparse

import wandb

from scripts.wandb_views import add_common_args, resolve_entity


def main():
    parser = argparse.ArgumentParser(description="Rename a wandb run.")
    parser.add_argument("run_id", help="wandb run id, e.g. 8u06ouw5")
    parser.add_argument("name", help="new display name")
    add_common_args(parser)
    args = parser.parse_args()

    run = wandb.Api().run(f"{resolve_entity(args.entity)}/{args.project}/{args.run_id}")
    run.name = args.name
    run.update()
    print(f"run {args.run_id} is now named {args.name!r}")


if __name__ == "__main__":
    main()
