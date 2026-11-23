"""
Finetunes a training run's checkpoint on a Modal GPU worker. Arguments are forwarded to sft.py unchanged:

    just modal-sft <run_id> --config scaleup --source reverse

(which runs modal run scripts/modal_sft.py --run-id <run_id> --args "--config scaleup ...")

The checkpoint is read from the "gpt-checkpoints" Modal volume, where scripts/modal_train.py wrote checkpoints/<run_id>/model.pt and tokenizer.json.
The finetuned checkpoint is written to the same volume, and can be generated from with `just modal-generate <new run id>`.
The SFT datasets are files in the repo (src/tiny_shakespeare/data/sft), so they come along with the code and no data volume is needed.

Needs a Modal secret named "wandb" containing WANDB_API_KEY, as for modal_train.py.
"""

import os
import shlex
import sys

import modal

from modal_constants import CHECKPOINT_DIR, CHECKPOINT_VOLUME_NAME

# Set MODAL_GPU="" to run on CPU, as for modal_train.py.
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("MODAL_GPU", "L4") or None

app = modal.App("gpt-sft")
# No create_if_missing: if there is no checkpoint volume, there is nothing to finetune, and we want to hear that rather than make an empty one.
volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # --no-dev: the dev group (pytest, pyright, modal) isn't needed to train.
    .uv_sync(extra_options="--no-dev")
    # add_local_dir (not add_local_python_source) so that the data, configs, and tokenizer files come along with the .py files.
    .add_local_dir("src", remote_path="/root/src")
    .add_local_python_source("modal_constants")
)

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("wandb")],
    # Writable: the finetuned checkpoint is saved alongside the one it started from.
    volumes={CHECKPOINT_DIR: volume},
)
def sft(run_id: str, argv: list[str]) -> None:
    # load_checkpoint and save_checkpoint use the relative path checkpoints/<run_id>/, so run from the volume.
    os.chdir(CHECKPOINT_DIR)
    sys.path.insert(0, "/root")
    sys.argv = ["sft", run_id, *argv]

    from src.tiny_shakespeare.models.gpt.sft import main

    main()
    # Modal also commits on container exit; this makes the checkpoint durable before the function returns.
    volume.commit()

@app.local_entrypoint()
def main(run_id: str, args: str = "") -> None:
    # modal run turns entrypoint parameters into options, so sft.py's flags are passed as one quoted string.
    sft.remote(run_id, shlex.split(args))
