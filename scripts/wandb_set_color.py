"""
Set the color of a run in a saved workspace view. Colors are stored per view, so the run has to be in the view first.
"""

import argparse

from scripts.wandb_views import add_common_args, load_view, normalize_hex, resolve_entity, set_run_color


def main():
    parser = argparse.ArgumentParser(description="Color a run in a saved wandb workspace view.")
    parser.add_argument("run_id", help="wandb run id, e.g. 8u06ouw5")
    parser.add_argument("view_id", help="the nw= value of the view's URL")
    parser.add_argument("color", help="6-digit hex color without the #, e.g. ff0000")
    add_common_args(parser)
    args = parser.parse_args()

    color = normalize_hex(args.color)
    workspace = load_view(resolve_entity(args.entity), args.project, args.view_id)
    set_run_color(workspace, args.run_id, color)
    workspace.save()
    print(f"run {args.run_id} is {color} in view {args.view_id}")


if __name__ == "__main__":
    main()
