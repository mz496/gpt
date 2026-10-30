from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_NAMES = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


@dataclass(frozen=True)
class TrainConfig:
    # Training
    train_steps: int
    # Seeds weight init and dropout (global RNG), and the generator that picks training batches.
    train_seed: int
    # Seeds the generator that picks the (fixed) batches used for loss estimation.
    eval_seed: int
    # how many independent sequences will we look at in parallel?
    batch_size: int
    learning_rate: float
    # for loss estimation
    eval_iters: int

    # Model (vocab_size comes from the data)
    n_embd: int
    n_head: int
    n_layer: int
    # what is the max context length for predictions?
    block_size: int
    dropout: float


def load_config(name: str) -> TrainConfig:
    """Load configs/<name>.yaml. Unknown or missing keys raise a TypeError."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return TrainConfig(**yaml.safe_load(f))
