"""
Create an empty saved workspace view and print only its view id (the nw= value in its URL) to stdout.
"""

import argparse

from scripts.wandb_views import add_common_args, create_view, resolve_entity


def main():
    parser = argparse.ArgumentParser(description="Create a saved wandb workspace view and print its id.")
    parser.add_argument("--name", default="view", help="display name of the view")
    add_common_args(parser)
    args = parser.parse_args()

    print(create_view(resolve_entity(args.entity), args.project, args.name))


if __name__ == "__main__":
    main()
