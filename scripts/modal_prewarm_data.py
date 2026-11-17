"""
Downloads the TinyStories dataset into the "gpt-data" Modal volume, so that training runs don't each download it:

    uv run modal run scripts/modal_prewarm_data.py

The files land in the Hugging Face cache layout under HF_HOME, so with the volume mounted at DATA_DIR and the same HF_HOME set,
hf_hub_download in TinyStoriesSource finds them already cached. scripts/modal_train.py must mount the volume and set HF_HOME the same way.

Check what's there with:

    uv run modal volume ls gpt-data hf/hub
"""

import sys

import modal

from modal_constants import DATA_DIR, DATA_VOLUME_NAME, HF_HOME

app = modal.App("gpt-prewarm-data")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # --no-dev: the dev group (pytest, pyright, modal) isn't needed to download.
    .uv_sync(extra_options="--no-dev")
    .env({"HF_HOME": HF_HOME})
    # Imported only for the dataset's repo and file names, so they live in one place.
    .add_local_dir("src", remote_path="/root/src")
    .add_local_python_source("modal_constants")
)

@app.function(image=image, timeout=60 * 60, volumes={DATA_DIR: data_volume})
def prewarm() -> None:
    sys.path.insert(0, "/root")

    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType] Partially unknown overload types.

    from src.tiny_shakespeare.data.tiny_stories_source import FILENAMES, REPO_ID

    for filename in FILENAMES.values():
        # Returns right away if the file is already in the volume's cache.
        path = hf_hub_download(REPO_ID, filename, repo_type="dataset")
        print(f"cached {path}")
    # Modal also commits on container exit; this makes the files durable before the function returns.
    data_volume.commit()

@app.local_entrypoint()
def main() -> None:
    prewarm.remote()
