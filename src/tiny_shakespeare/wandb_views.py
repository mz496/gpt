"""
Helpers for saved wandb workspace views, shared by the wandb-*-view scripts (in scripts/) and by anything that makes a view as part of a run (in src/, which is what
a Modal job has, not scripts/). Nothing here knows what is logged or what a view is for.

A "view" is a saved workspace. Its id is the `nw=` value in the workspace URL (e.g. mgg03pctxen).
We treat the keys of the view's run_settings as the runs in the view, and keep the view's run filter in sync with them.
"""

import argparse
import re

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

DEFAULT_PROJECT = "gpt-experiments"
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
# wandb stores the id of view `abc` as `nw-abc-v`.
_INTERNAL_NAME = re.compile(r"nw-(.+)-v")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entity", default=None, help="wandb entity (defaults to your default entity)")
    parser.add_argument("--project", default=DEFAULT_PROJECT)


def resolve_entity(entity: str | None) -> str:
    resolved = entity or wandb.Api().default_entity
    if resolved is None:
        raise ValueError("no wandb entity: pass --entity or log in to set a default")
    return resolved


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


def create_view(entity: str, project: str, name: str, sections: list[ws.Section] | None = None) -> str:
    """
    Saves a new view and returns its id. With no sections, the view is empty and wandb makes a panel for every key that is logged.
    With sections, it has just those panels.
    """
    workspace = ws.Workspace(entity=entity, project=project, name=name, sections=sections or [], auto_generate_panels=sections is None)
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


def set_line_colors(workspace: ws.Workspace, run_id: str, colors: dict[str, str]) -> None:
    """
    Colors the lines of the run in the line plots of the view. `colors` maps the name of a plotted metric to a hex color, and every plot that has that metric
    gets the color for this run's line of it. The colors of other lines, and of other runs, are kept.
    """
    for section in workspace.sections:
        for panel in section.panels:
            if isinstance(panel, wr.LinePlot):
                chosen = {f"{run_id}:{key}": colors[key] for key in panel.y if isinstance(key, str) and key in colors}
                if chosen:
                    panel.line_colors = {**(panel.line_colors or {}), **chosen}


def add_run_to_view(entity: str, project: str, view_id: str, run_id: str, line_colors: dict[str, str] | None = None) -> int:
    """
    Adds the run to the saved view, which then shows only the runs that were added to it, with its lines colored as `line_colors` says (see set_line_colors).
    Returns how many runs the view has. Raises if the run doesn't exist, before the view is touched.
    """
    wandb.Api().run(f"{entity}/{project}/{run_id}")
    workspace = load_view(entity, project, view_id)
    add_run(workspace, run_id)
    if line_colors:
        set_line_colors(workspace, run_id, line_colors)
    workspace.save()
    return len(workspace.runset_settings.run_settings)


def view_url(entity: str, project: str, view_id: str) -> str:
    """Where the view is, on wandb.ai."""
    return f"https://wandb.ai/{entity}/{project}?nw={view_id}"
