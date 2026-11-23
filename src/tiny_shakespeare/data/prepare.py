import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tokenizers.tokenizer import Tokenizer
from .data_source import DataSource, Split
from .sft.sft_data_source import SFTDataSource

@dataclass
class PreparedData:
    train: np.ndarray[Any, Any]
    validation: np.ndarray[Any, Any]
    # Chars of text behind each split, so losses per token can be converted to losses per char.
    train_chars: int
    validation_chars: int

def _encode_split(source: DataSource, split: Split, tokenizer: Tokenizer, dtype: np.dtype[Any], max_chars: int | None = None) -> tuple[np.ndarray[Any, Any], int]:
    """
    Tokenizes one split chunk by chunk. Returns the token ids and the number of chars they came from.

    If max_chars is given, it stops after the chunk that reaches it, the same rule as DataSource.get_tokenizer_text,
    so that a big split can be tried out on its first part.

    Tokens never span chunks, so for a BPE tokenizer a chunk boundary in the middle of text can change the tokens near it versus encoding the whole split at once.
    """
    arrays: list[np.ndarray[Any, Any]] = []
    num_chars = 0
    if max_chars is not None and max_chars <= 0:
        return np.empty(0, dtype=dtype), 0
    for chunk in source.iter_text(split):
        num_chars += len(chunk)
        arrays.append(np.asarray(tokenizer.encode_string(chunk), dtype=dtype))
        # Checked here rather than before reading a chunk, so we never pull a chunk we won't use.
        if max_chars is not None and num_chars >= max_chars:
            break
    return (np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)), num_chars

def train_tokenizer(source: DataSource, tokenizer: Tokenizer) -> None:
    """
    Registers the source's special tokens with the tokenizer, then trains it on the source's text.

    Most tokenizers learn from a sample of the train split (DataSource.get_tokenizer_text), because training is costly (BPE).
    A tokenizer that can learn from a stream (Tokenizer.trains_on_stream, like char, whose vocab is the set of characters) gets both whole splits instead,
    so that it has seen every character it will be asked to encode. Its vocab then includes characters that only the validation split has.
    """
    tokenizer.register_special_tokens(source.special_tokens)
    if tokenizer.trains_on_stream:
        tokenizer.train_on_chunks(itertools.chain(source.iter_text("train"), source.iter_text("validation")))
    else:
        tokenizer.train(source.get_tokenizer_text())

def prepare(source: DataSource, tokenizer: Tokenizer, max_train_chars: int | None = None) -> PreparedData:
    """
    The train and validation splits of the source as arrays of token ids, using an already-trained tokenizer.

    max_train_chars limits the train split to its first chunks totalling at least that many chars (see encode_split); the validation split is always whole.
    train_chars says how much text was actually used.
    """
    # Narrowest unsigned dtype that fits every id; get_batch widens each window to int64.
    dtype = np.dtype(np.uint16 if len(tokenizer.get_vocab()) <= 2**16 else np.uint32)
    train, train_chars = _encode_split(source, "train", tokenizer, dtype, max_train_chars)
    validation, validation_chars = _encode_split(source, "validation", tokenizer, dtype)
    return PreparedData(train=train, validation=validation, train_chars=train_chars, validation_chars=validation_chars)

def _encode_split_sft(source: SFTDataSource, split: Split, tokenizer: Tokenizer, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    # TODO: Raise if any example would exceed block_size from cfg
    arrays: list[np.ndarray[Any, Any]] = []
    for example in source.iter_examples(split):
        example_tokens = tokenizer.encode_string(example.to_prompt_string()) + tokenizer.encode_string(example.to_response_string())
        arrays.append(np.asarray(example_tokens, dtype=dtype))
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)

def prepare_sft(source: SFTDataSource, tokenizer: Tokenizer, block_size: int) -> PreparedData:
    # Narrowest unsigned dtype that fits every id; get_batch widens each window to int64.
    dtype = np.dtype(np.uint16 if len(tokenizer.get_vocab()) <= 2**16 else np.uint32)
    train = _encode_split_sft(source, "train", tokenizer, dtype)
    validation = _encode_split_sft(source, "validation", tokenizer, dtype)
    return PreparedData(train=train, validation=validation, train_chars=0, validation_chars=0)
    