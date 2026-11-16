import re

import wandb
import wandb_workspaces.workspaces as ws
from wandb_workspaces._graphql import get_app_url  # pyright: ignore

ENTITY = wandb.Api().default_entity
PROJECT = "gpt-experiments"

app_url = get_app_url(wandb.Api())
print("app_url:", app_url)

workspace = ws.Workspace(entity=ENTITY, project=PROJECT, name="repro-view", auto_generate_panels=True)
workspace.save()

nw = ""#3v6zhxyxhzn"

loaded = ws.Workspace.from_url(f"https://invalid.url/{ENTITY}/{PROJECT}?nw={nw}")
print("loaded", loaded.name)

"""
# wandb stores the id of view `abc` as `nw-abc-v`.
view_id = re.fullmatch(r"nw-(.+)-v", workspace._internal_name).group(1)  # pyright: ignore
print("saved view id:", view_id)

# BUILD side: workspace.url keeps only the scheme and host of app_url, so on Forge the /wandb is missing.
correct_url = f"{app_url.rstrip('/')}/{ENTITY}/{PROJECT}?nw={view_id}"
print("workspace.url:", workspace.url)
print("expected:     ", correct_url)
print("build side matches:", workspace.url == correct_url)

# PARSE side: reading the correct link back splits the path at fixed positions, so the prefix breaks it.
try:
    loaded = ws.Workspace.from_url(correct_url)
    print("loaded:", loaded.name)
except Exception as e:
    print(f"Workspace.from_url(expected) failed: {type(e).__name__}: {e}")

# # from_url only reads the path and the nw= query, so the host doesn't matter.
# loaded = ws.Workspace.from_url(f"https://example.invalid/{ENTITY}/{PROJECT}?nw={view_id}")
# print("loaded:", loaded)

"""