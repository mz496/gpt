"""
Add a run to a saved workspace view. The view then shows only the runs added to it.
"""

import argparse

import wandb

from scripts.wandb_views import add_common_args, add_run, load_view, resolve_entity


def main():
    parser = argparse.ArgumentParser(description="Add a wandb run to a saved workspace view.")
    parser.add_argument("run_id", help="wandb run id, e.g. 8u06ouw5")
    parser.add_argument("view_id", help="the nw= value of the view's URL")
    add_common_args(parser)
    args = parser.parse_args()

    entity = resolve_entity(args.entity)
    # Fail early if the run doesn't exist.
    wandb.Api().run(f"{entity}/{args.project}/{args.run_id}")

    workspace = load_view(entity, args.project, args.view_id)
    add_run(workspace, args.run_id)
    workspace.save()
    print(f"view {args.view_id} now has {len(workspace.runset_settings.run_settings)} run(s)")


if __name__ == "__main__":
    main()
