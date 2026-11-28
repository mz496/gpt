"""
Add a run to a saved workspace view. The view then shows only the runs added to it.
"""

import argparse

from src.tiny_shakespeare.wandb_views import add_common_args, add_run_to_view, resolve_entity


def main():
    parser = argparse.ArgumentParser(description="Add a wandb run to a saved workspace view.")
    parser.add_argument("run_id", help="wandb run id, e.g. 8u06ouw5")
    parser.add_argument("view_id", help="the nw= value of the view's URL")
    add_common_args(parser)
    args = parser.parse_args()

    n_runs = add_run_to_view(resolve_entity(args.entity), args.project, args.view_id, args.run_id)
    print(f"view {args.view_id} now has {n_runs} run(s)")


if __name__ == "__main__":
    main()
