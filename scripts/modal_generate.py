"""
Generates text from a training run's checkpoint on a Modal worker. Arguments are forwarded to generate.py unchanged:

    just modal-generate <run_id> --max-tokens 500 --generate-seed 3

(which runs modal run scripts/modal_generate.py --run-id <run_id> --args "--max-tokens 500 ...")

The checkpoint is read from the "gpt-checkpoints" Modal volume, where scripts/modal_train.py wrote checkpoints/<run_id>/model.pt and tokenizer.json.
The tokenizer comes with the checkpoint, so no dataset is needed. The text is printed from the worker's output, here.
"""

import os
import shlex
import sys

import modal

from modal_constants import CHECKPOINT_DIR, CHECKPOINT_VOLUME_NAME

# Set MODAL_GPU="" to run on CPU, as for modal_train.py. Generating is brief, so this is cheap either way.
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("MODAL_GPU", "L4") or None

app = modal.App("gpt-generate")
# No create_if_missing: if there is no checkpoint volume, we want to hear that rather than make an empty one.
volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # --no-dev: the dev group (pytest, pyright, modal) isn't needed to generate.
    .uv_sync(extra_options="--no-dev")
    # add_local_dir (not add_local_python_source) so that the data, configs, and tokenizer files come along with the .py files.
    .add_local_dir("src", remote_path="/root/src")
    .add_local_python_source("modal_constants")
)

@app.function(image=image, gpu=GPU, timeout=10 * 60, volumes={CHECKPOINT_DIR: volume.read_only()})
def generate(run_id: str, argv: list[str]) -> None:
    # load_checkpoint reads the relative path checkpoints/<run_id>/model.pt, so run from the volume.
    os.chdir(CHECKPOINT_DIR)
    sys.path.insert(0, "/root")
    sys.argv = ["generate", run_id, *argv]

    from src.tiny_shakespeare.models.gpt.generate import main

    main()

@app.local_entrypoint()
def main(run_id: str, args: str = "") -> None:
    # modal run turns entrypoint parameters into options, so generate.py's flags are passed as one quoted string.
    generate.remote(run_id, shlex.split(args))
