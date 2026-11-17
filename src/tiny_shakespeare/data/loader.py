import random
from pathlib import Path
from typing import Any, Literal, assert_never

import numpy as np
import torch

Dataset = Literal["tiny-shakespeare"]

def get_data(slug: Dataset) -> tuple[str, str]:
    """
    Fetch the dataset with this slug as (train, validation), split 90/10.
    """
    match slug:
        case "tiny-shakespeare":
            data = (Path(__file__).parent / "tiny_shakespeare.txt").read_text()
        case _:
            assert_never(slug)
    n = len(data)
    train_split_len = int(n * 0.9)
    train = data[:train_split_len]
    validation = data[train_split_len+1:]
    return train, validation

def get_batch(tokenized_data: np.ndarray[Any, Any], batch_size: int, block_size: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull `batch_size` windows of length `block_size` starting at random points in `tokenized_data`.
    - `generator` pins the choice of window starts. Without one we draw from torch's global RNG, which is shared with weight init and dropout,
      so the batches would depend on how much randomness the model itself happened to consume.
    - `tokenized_data` is a 1-D array of token ids (e.g. an np.memmap, so the data needn't fit in RAM).

    Returns two int64 tensors (contexts, targets) of size [B, T] on the CPU; the caller moves them to a device.
    - B = batch_size
    - T = block_size
    where targets[i][j] is the token that follows the prefix context[i][:j+1].
    """
    # [B] random numbers drawn from [0, n-block_size)
    # Do we need to prevent wraparound here...?
    starts: np.ndarray[Any, Any] = torch.randint(0, len(tokenized_data) - block_size, (batch_size,), generator=generator).numpy()  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    # [B, T]: Broadcasts each of the B random numbers to length T using [0..T]; each pos is the index to take from the data.
    indices = starts[:, None] + np.arange(block_size)
    # Token ids may be stored in a narrow unsigned dtype (e.g. uint16), which torch doesn't fully support, so widen to int64.
    x = torch.from_numpy(tokenized_data[indices].astype(np.int64))  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    y = torch.from_numpy(tokenized_data[indices + 1].astype(np.int64))  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return x, y

def take_fraction(text: str, fraction: float, chunk_size: int = 1000, seed: int = 0) -> str:
    """
    A representative subsample of `text`: `fraction` of its chunk_size-character chunks, picked at random and kept in their original order.

    The pick order depends only on the seed, so the subsamples are nested (the 25% sample contains the 12.5% sample), and fraction=1.0 returns the text unchanged.
    Sampling chunks (not taking a prefix) keeps the sample from being closer to or further from the validation split's part of the file as it grows.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)
    keep = sorted(order[:max(1, round(len(chunks) * fraction))])
    return "".join(chunks[i] for i in keep)
