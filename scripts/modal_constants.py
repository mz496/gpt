"""
Names and paths shared by the modal_*.py scripts. Each script's image needs .add_local_python_source("modal_constants") to import this in the container.
"""

# The volume holding the checkpoints, written by modal_train.py and read by modal_generate.py. A run is in it at checkpoints/<run_id>/.
CHECKPOINT_VOLUME_NAME = "gpt-checkpoints"
CHECKPOINT_DIR = "/vol"
# The volume holding the datasets, filled by modal_prewarm_data.py and read by modal_train.py.
DATA_VOLUME_NAME = "gpt-data"
DATA_DIR = "/data"
# hf_hub_download caches under here, so setting this in an image makes it find (and fill) the data volume.
HF_HOME = f"{DATA_DIR}/hf"
