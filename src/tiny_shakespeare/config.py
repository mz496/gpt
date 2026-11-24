from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_NAMES = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


@dataclass(frozen=True)
class TrainConfig:
    """How to train: everything that doesn't change what the model is."""

    train_steps: int
    # Seeds weight init and dropout (global RNG), and the generator that picks training batches.
    train_seed: int
    # Seeds the generator that picks the (fixed) batches used for loss estimation.
    eval_seed: int
    # how many independent sequences will we look at in parallel?
    batch_size: int
    learning_rate: float
    # for loss estimation: how many batches to estimate over
    eval_iters: int
    # for loss estimation: how many of those batches to run in parallel (as one concatenated batch) per forward pass.
    # Fewer, bigger forwards are faster when launch-bound, but activation memory grows with it. eval_chunk_size == eval_iters means "all at once".
    eval_chunk_size: int

    def __post_init__(self):
        if self.eval_chunk_size < 1:
            raise ValueError(f"eval_chunk_size must be >= 1, got {self.eval_chunk_size}")
        if self.eval_chunk_size > self.eval_iters:
            raise ValueError(f"eval_chunk_size {self.eval_chunk_size} greater than eval_iters {self.eval_iters} is meaningless")


@dataclass(frozen=True)
class ModelConfig:
    """The shape of the model (vocab_size comes from the data). A trained checkpoint carries its own, so this only matters when making a new model."""

    n_embd: int
    n_head: int
    n_layer: int
    # what is the max context length for predictions?
    block_size: int
    dropout: float


@dataclass(frozen=True)
class FlagConfig:
    """Switches between implementations."""

    # Batched attention heads vs. one module per head (changes the architecture; saved in the checkpoint's GPTConfig).
    flag_use_parallel_heads: bool
    # Stock torch.nn.functional.scaled_dot_product_attention vs. our handrolled one (same function; speed only).
    # Note this only takes effect when use_parallel_heads is true.
    flag_use_pytorch_attention_impl: bool


@dataclass(frozen=True)
class Config:
    train: TrainConfig
    model: ModelConfig
    flags: FlagConfig


def load_config(name: str) -> Config:
    """Load configs/<name>.yaml, which has a train, a model and a flags section. Unknown or missing sections or keys raise a TypeError."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        raw = yaml.safe_load(f)
    if raw.keys() != {"train", "model", "flags"}:
        raise TypeError(f"configs/{name}.yaml must have exactly the sections ['flags', 'model', 'train'], got {sorted(raw)}")
    return Config(train=TrainConfig(**raw["train"]), model=ModelConfig(**raw["model"]), flags=FlagConfig(**raw["flags"]))
