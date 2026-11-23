from typing import Any, NamedTuple

import numpy as np
import torch

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

# The target for positions that get no loss. F.cross_entropy ignores targets equal to this by default, so the model needs no changes.
IGNORE_INDEX = -100

class SftData(NamedTuple):
    """
    One split of an SFT dataset: one example per row, each already cut into inputs and targets and right-padded to a common width.
    Both arrays are [N, width]. targets[i][j] is the token that follows inputs[i][:j+1], or IGNORE_INDEX where the model isn't trained to predict it:
    in the prompt, and in the padding.
    """
    inputs: np.ndarray[Any, Any]
    targets: np.ndarray[Any, Any]

def get_sft_batch(data: SftData, batch_size: int, block_size: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull `batch_size` random examples (rows, with replacement) from `data`. The counterpart of get_batch for SFT: the same arguments and the same return value.
    - `generator` pins the choice of rows, for the same reason as in get_batch.
    - Rows are already `data.inputs.shape[1]` wide, so `block_size` isn't used to cut anything; it only guards that a row fits in the model's context.

    Returns two int64 tensors (contexts, targets) of size [B, width] on the CPU; the caller moves them to a device.
    """
    width = data.inputs.shape[1]
    if width > block_size:
        raise ValueError(f"examples are {width} tokens wide, which doesn't fit in block_size {block_size}")
    rows: np.ndarray[Any, Any] = torch.randint(0, len(data.inputs), (batch_size,), generator=generator).numpy()  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    # Token ids may be stored in a narrow unsigned dtype (e.g. uint16), which torch doesn't fully support, so widen to int64. Targets are already int64.
    x = torch.from_numpy(data.inputs[rows].astype(np.int64))  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    y = torch.from_numpy(data.targets[rows])  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return x, y
