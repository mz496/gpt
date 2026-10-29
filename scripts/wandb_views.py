"""
Shared helpers for the wandb-*-view scripts.

A "view" is a saved workspace. Its id is the `nw=` value in the workspace URL (e.g. mgg03pctxen).
We treat the keys of the view's run_settings as the runs in the view, and keep the view's run filter in sync with them.
"""

import argparse
import re

import wandb
import wandb_workspaces.workspaces as ws

DEFAULT_PROJECT = "gpt-experiments"
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
# wandb stores the id of view `abc` as `nw-abc-v`.
_INTERNAL_NAME = re.compile(r"nw-(.+)-v")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entity", default=None, help="wandb entity (defaults to your default entity)")
    parser.add_argument("--project", default=DEFAULT_PROJECT)


def resolve_entity(entity: str | None) -> str:
    return entity or wandb.Api().default_entity


def normalize_hex(color: str) -> str:
    """'ff0000' or '#FF0000' -> '#ff0000'. Raises ValueError if it isn't a 6-digit hex color."""
    color = color if color.startswith("#") else f"#{color}"
    if not HEX_COLOR.fullmatch(color):
        raise ValueError(f"expected a 6-digit hex color like ff0000, got {color!r}")
    return color.lower()


def view_id_from_internal_name(internal_name: str) -> str:
    """'nw-mgg03pctxen-v' -> 'mgg03pctxen'"""
    match = _INTERNAL_NAME.fullmatch(internal_name)
    if match is None:
        raise ValueError(f"unexpected view name {internal_name!r}")
    return match.group(1)


def create_view(entity: str, project: str, name: str) -> str:
    """Saves a new empty view and returns its id."""
    workspace = ws.Workspace(entity=entity, project=project, name=name, auto_generate_panels=True)
    workspace.save()
    return view_id_from_internal_name(workspace._internal_name)  # pyright: ignore[reportPrivateUsage]


def load_view(entity: str, project: str, view_id: str) -> ws.Workspace:
    # from_url only reads the path and the nw= query, so the host doesn't matter.
    return ws.Workspace.from_url(f"https://example.invalid/{entity}/{project}?nw={view_id}")


def sync_run_filter(workspace: ws.Workspace) -> None:
    """Make the view show exactly the runs in its run_settings. With no runs, the filter is left alone (the view shows everything)."""
    run_ids = sorted(workspace.runset_settings.run_settings)
    if run_ids:
        workspace.runset_settings.filters = f"ID in {run_ids!r}"


def add_run(workspace: ws.Workspace, run_id: str) -> None:
    """Add the run to the view, keeping its color if it already has one."""
    workspace.runset_settings.run_settings.setdefault(run_id, ws.RunSettings())
    sync_run_filter(workspace)


def set_run_color(workspace: ws.Workspace, run_id: str, color: str) -> None:
    """Color a run that's already in the view."""
    run_settings = workspace.runset_settings.run_settings
    if run_id not in run_settings:
        raise KeyError(f"run {run_id} is not in this view; add it first with wandb-add-run-to-view")
    run_settings[run_id] = ws.RunSettings(color=normalize_hex(color), disabled=run_settings[run_id].disabled)
