"""
Runs src.tiny_shakespeare.gpt.train on a Modal GPU worker. Arguments are forwarded to train.py unchanged:

    just modal-train --config default --tokenizer bpe --suffix foo

(which runs modal run scripts/modal_train.py --args "--config default ...")

Checkpoints are written to the "gpt-checkpoints" Modal volume as <run_id>/model.pt. To use one locally:

    uv run modal volume get gpt-checkpoints <run_id>/model.pt checkpoints/<run_id>/model.pt

Needs a Modal secret named "wandb" containing WANDB_API_KEY:

    uv run modal secret create wandb WANDB_API_KEY=...
"""

import os
import shlex
import sys

import modal

# Set MODAL_GPU="" to run on CPU (GPU functions need a payment method on the Modal account).
GPU = os.environ.get("MODAL_GPU", "T4") or None
CHECKPOINT_DIR = "/vol"

app = modal.App("gpt-train")
volume = modal.Volume.from_name("gpt-checkpoints", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # --no-dev: the dev group (pytest, pyright, modal) isn't needed to train.
    .uv_sync(extra_options="--no-dev")
    # add_local_dir (not add_local_python_source) so that the data, configs, and tokenizer files come along with the .py files.
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("wandb")],
    volumes={CHECKPOINT_DIR: volume},
)
def train(argv: list[str]) -> None:
    # save_checkpoint writes to the relative path checkpoints/<run_id>/model.pt, so run from the volume.
    os.chdir(CHECKPOINT_DIR)
    sys.path.insert(0, "/root")
    sys.argv = ["train", *argv]

    from src.tiny_shakespeare.gpt.train import main

    main()
    # Modal also commits on container exit; this makes the checkpoint durable before the function returns.
    volume.commit()

@app.local_entrypoint()
def main(args: str = "") -> None:
    # modal run turns entrypoint parameters into options, so train.py's flags are passed as one quoted string.
    train.remote(shlex.split(args))
