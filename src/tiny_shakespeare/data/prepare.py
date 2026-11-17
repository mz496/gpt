from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tokenizers.tokenizer import Tokenizer
from .data_source import DataSource, Split

@dataclass
class PreparedData:
    train: np.ndarray[Any, Any]
    validation: np.ndarray[Any, Any]
    # Chars of text behind each split, so losses per token can be converted to losses per char.
    train_chars: int
    validation_chars: int

def encode_split(source: DataSource, split: Split, tokenizer: Tokenizer, dtype: np.dtype[Any]) -> tuple[np.ndarray[Any, Any], int]:
    """
    Tokenizes one split chunk by chunk. Returns the token ids and the number of chars they came from.

    Tokens never span chunks, so for a BPE tokenizer a chunk boundary can change the tokens near it versus encoding the whole split at once.
    This is why chunks end between documents (see DataSource.iter_text).
    """
    arrays: list[np.ndarray[Any, Any]] = []
    num_chars = 0
    for chunk in source.iter_text(split):
        num_chars += len(chunk)
        arrays.append(np.asarray(tokenizer.encode_string(chunk), dtype=dtype))
    return (np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)), num_chars

def prepare(source: DataSource, tokenizer: Tokenizer) -> PreparedData:
    """
    The train and validation splits of the source as arrays of token ids, using an already-trained tokenizer.
    """
    # Narrowest unsigned dtype that fits every id; get_batch widens each window to int64.
    dtype = np.dtype(np.uint16 if len(tokenizer.get_vocab()) <= 2**16 else np.uint32)
    train, train_chars = encode_split(source, "train", tokenizer, dtype)
    validation, validation_chars = encode_split(source, "validation", tokenizer, dtype)
    return PreparedData(train=train, validation=validation, train_chars=train_chars, validation_chars=validation_chars)
