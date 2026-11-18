"""
Runs src.tiny_shakespeare.models.gpt.train on a Modal GPU worker. Arguments are forwarded to train.py unchanged:

    just modal-train --config default --tokenizer bpe --suffix foo

(which runs modal run scripts/modal_train.py --args "--config default ...")

Checkpoints are written to the "gpt-checkpoints" Modal volume as checkpoints/<run_id>/model.pt and tokenizer.json.
Generate from one on Modal with `just modal-generate <run_id>` (scripts/modal_generate.py). To use one locally, download the checkpoints/<run_id> directory:

    uv run modal volume get gpt-checkpoints checkpoints/<run_id> <local dir>   # check where the files land: generate needs them in checkpoints/<run_id>/

Needs a Modal secret named "wandb" containing WANDB_API_KEY:

    uv run modal secret create wandb WANDB_API_KEY=...
"""

import os
import shlex
import sys

import modal

from modal_constants import CHECKPOINT_DIR, CHECKPOINT_VOLUME_NAME, DATA_DIR, DATA_VOLUME_NAME, HF_HOME

# Set MODAL_GPU="" to run on CPU (GPU functions need a payment method on the Modal account).
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("MODAL_GPU", "L4") or None

app = modal.App("gpt-train")
volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
# Writable, so that a dataset that wasn't prewarmed is downloaded into the volume once rather than on every run.
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # --no-dev: the dev group (pytest, pyright, modal) isn't needed to train.
    .uv_sync(extra_options="--no-dev")
    # hf_hub_download caches under HF_HOME, so datasets come from the data volume.
    .env({"HF_HOME": HF_HOME})
    # add_local_dir (not add_local_python_source) so that the data, configs, and tokenizer files come along with the .py files.
    .add_local_dir("src", remote_path="/root/src")
    .add_local_python_source("modal_constants")
)

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("wandb")],
    volumes={CHECKPOINT_DIR: volume, DATA_DIR: data_volume},
)
def train(argv: list[str]) -> None:
    # A container started before a prewarm committed would otherwise see the volume as it was then.
    data_volume.reload()
    # save_checkpoint writes to the relative path checkpoints/<run_id>/model.pt, so run from the volume.
    os.chdir(CHECKPOINT_DIR)
    sys.path.insert(0, "/root")
    sys.argv = ["train", *argv]

    from src.tiny_shakespeare.models.gpt.train import main

    main()
    # Modal also commits on container exit; this makes the checkpoint durable before the function returns.
    volume.commit()
    data_volume.commit()

@app.local_entrypoint()
def main(args: str = "") -> None:
    # modal run turns entrypoint parameters into options, so train.py's flags are passed as one quoted string.
    train.remote(shlex.split(args))
