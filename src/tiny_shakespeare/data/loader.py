from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, assert_never

import numpy as np
import torch

Dataset = Literal["tiny-shakespeare"]
Split = Literal["train", "validation"]
# Cap on the text the tokenizer trains on; BPE training is pure Python, so it needs a sample of a big dataset, not all of it.
TOKENIZER_TEXT_MAX_CHARS = 100_000_000

def _read_splits(slug: Dataset) -> tuple[str, str]:
    """
    The whole dataset with this slug as (train, validation), split 90/10. Only for datasets small enough to hold as one string; callers use iter_text.
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

def iter_text(slug: Dataset, split: Split) -> Iterator[str]:
    """
    The text of one split of the dataset as consecutive chunks, so that big datasets never need to be held in memory as one string.

    Concatenating the chunks gives the whole split. A chunk never ends in the middle of a document (for tiny shakespeare the whole split is one chunk).
    """
    train, validation = _read_splits(slug)
    match split:
        case "train":
            yield train
        case "validation":
            yield validation
        case _:
            assert_never(split)

def take_chars(chunks: Iterable[str], max_chars: int) -> str:
    """
    Concatenates chunks until at least max_chars chars have been collected; the chunk that crosses the cap is kept whole, so documents aren't cut in half.
    """
    if max_chars <= 0:
        return ""
    taken: list[str] = []
    total = 0
    for chunk in chunks:
        taken.append(chunk)
        total += len(chunk)
        # Check here rather than at the top of the loop, so we never pull a chunk we won't use.
        if total >= max_chars:
            break
    return "".join(taken)

def get_tokenizer_text(slug: Dataset, max_chars: int = TOKENIZER_TEXT_MAX_CHARS) -> str:
    """
    The text to train the tokenizer on: the start of the train split, capped at (about) max_chars.
    Only the train split, so the tokenizer learns nothing from the validation text.
    """
    return take_chars(iter_text(slug, "train"), max_chars)

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
