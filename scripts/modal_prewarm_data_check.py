"""
Lists the Hugging Face datasets already cached in the "gpt-data" Modal volume (what scripts/modal_prewarm_data.py fills):

    uv run modal run scripts/modal_prewarm_data_check.py
"""

import modal

from modal_constants import DATA_DIR, DATA_VOLUME_NAME, HF_HOME

app = modal.App("gpt-prewarm-data-check")
# No create_if_missing: if the volume isn't there, nothing has been prewarmed, and we want to hear that rather than make an empty one.
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub").add_local_python_source("modal_constants")

@app.function(image=image, volumes={DATA_DIR: data_volume.read_only()})
def check() -> None:
    from pathlib import Path

    from huggingface_hub import scan_cache_dir  # pyright: ignore[reportUnknownVariableType] Partially unknown overload types.

    hub = Path(HF_HOME) / "hub"
    if not hub.exists():
        print(f"No HF cache at {hub}: nothing has been prewarmed.")
        return
    datasets = sorted((repo for repo in scan_cache_dir(hub).repos if repo.repo_type == "dataset"), key=lambda repo: repo.repo_id)
    if not datasets:
        print(f"No datasets in {hub}.")
        return
    for repo in datasets:
        print(f"{repo.repo_id}: {repo.size_on_disk_str}, {repo.nb_files} files")
        for revision in repo.revisions:
            for file in sorted(revision.files, key=lambda file: file.file_name):
                print(f"  {revision.commit_hash[:8]} {file.file_name} ({file.size_on_disk_str})")

@app.local_entrypoint()
def main() -> None:
    check.remote()
