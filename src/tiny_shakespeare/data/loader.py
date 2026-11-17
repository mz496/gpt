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

def get_batch(tokenized_data: torch.Tensor | np.ndarray[Any, Any], batch_size: int, block_size: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull `batch_size` windows of length `block_size` starting at random points in `tokenized_data`.
    - `generator` pins the choice of window starts.
    - `tokenized_data` is a tensor, or a numpy array (e.g. an np.memmap, so the data needn't fit in RAM).
    Windows from a numpy array are copied out as int64 tensors on the CPU; the caller moves them to a device.

    Returns two tensors (contexts, targets) of size [B, T]
    - B = batch_size
    - T = block_size
    where targets[i][j] is the token that follows the prefix context[i][:j+1].
    """
    # [B] random numbers drawn from [0, n-block_size)
    # Do we need to prevent wraparound here...?
    starts = torch.randint(0, len(tokenized_data) - block_size, (batch_size,), generator=generator)
    if isinstance(tokenized_data, np.ndarray):
        # Token ids are stored in a narrow unsigned dtype (e.g. uint16), which torch doesn't fully support, so widen each window.
        def window(start: int) -> torch.Tensor:
            return torch.from_numpy(tokenized_data[start:start+block_size].astype(np.int64))  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
        python_starts: list[int] = starts.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] Bad torch type stubs.
        x = torch.stack([window(i) for i in python_starts])
        y = torch.stack([window(i + 1) for i in python_starts])
    else:
        x = torch.stack([tokenized_data[i:i+block_size] for i in starts])
        y = torch.stack([tokenized_data[i+1:i+block_size+1] for i in starts])
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
