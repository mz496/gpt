import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tokenizers.tokenizer import Tokenizer
from .data_source import DataSource, Split
from .loader import IGNORE_INDEX, SftData
from .sft.sft_data_source import SFTDataSource, SftExample

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

# Pads the inputs. Any valid id works: padding is never predicted (its targets are IGNORE_INDEX), and under causal attention no earlier token sees it.
PAD_ID = 0

def _encode_examples(examples: Iterable[SftExample], tokenizer: Tokenizer, dtype: np.dtype[Any], block_size: int, label: str) -> SftData:
    """
    Tokenizes the examples one by one, as the request followed by the response, and returns them as one row each, in the order given.
    A full example of [t1 t2 t3 t4 t5] is:
    - Cut at the beginning and end (so target[j] follows input[:j+1])
        - Input:  [t1 t2 t3 t4]
        - Target: [t2 t3 t4 t5] (shifted by 1)
    - Masked so the target receives IGNORE_INDEX everywhere where the input's request falls (so we only learn the response)
        - Input:  [t1 t2 t3 t4]
        - Target: [II II t4 t5]
    - Padded to the longest example given (so that among them, we have a consistent block_size)
        - Input:  [t1 t2 t3 t4 PI PI PI]
        - Target: [II II t4 t5 II II II]

    Everything that is trained on, and everything a trained model is scored on, goes through here, so that "the response" means the same thing for both.
    `label` says what the examples are (e.g. "train"), for the error messages. Raises if there are none, or one doesn't fit in block_size.
    """
    # Holds one input row per example.
    inputs_per_example: list[np.ndarray[Any, Any]] = []
    # Holds one output row per example.
    targets_per_example: list[np.ndarray[Any, Any]] = []
    for example in examples:
        request_tokens = tokenizer.encode_string(example.to_request_string())
        response_tokens = tokenizer.encode_string(example.to_response_string())
        if not response_tokens:
            raise ValueError(f"{label} example {example.input!r} has an empty response, so there would be nothing to train on")
        tokens = np.asarray(request_tokens + response_tokens, dtype=dtype)
        if len(tokens) - 1 > block_size:
            raise ValueError(f"{label} example {example.input[:30]!r} is {len(tokens) - 1} tokens long once cut, which doesn't fit in block_size {block_size}")
        # The ids are unsigned, which can't hold IGNORE_INDEX, so the targets are a (copied) int64 array.
        targets = tokens[1:].astype(np.int64)
        # targets[j] is tokens[j+1], and the first response token is tokens[len(request_tokens)], so the first target to keep is at len(request_tokens) - 1.
        targets[:len(request_tokens) - 1] = IGNORE_INDEX
        inputs_per_example.append(tokens[:-1])
        targets_per_example.append(targets)
    if not inputs_per_example:
        raise ValueError(f"there are no {label} examples to encode")

    width = max(len(inputs) for inputs in inputs_per_example)
    inputs = np.full((len(inputs_per_example), width), PAD_ID, dtype=dtype)
    targets = np.full((len(targets_per_example), width), IGNORE_INDEX, dtype=np.int64)
    for i, (example_inputs, example_targets) in enumerate(zip(inputs_per_example, targets_per_example)):
        inputs[i, :len(example_inputs)] = example_inputs
        targets[i, :len(example_targets)] = example_targets
    return SftData(inputs=inputs, targets=targets)

def _id_dtype(tokenizer: Tokenizer) -> np.dtype[Any]:
    """Narrowest unsigned dtype that fits every id; get_sft_batch widens each batch to int64."""
    return np.dtype(np.uint16 if len(tokenizer.get_vocab()) <= 2**16 else np.uint32)

def _encode_split_sft(source: SFTDataSource, split: Split, tokenizer: Tokenizer, dtype: np.dtype[Any], block_size: int) -> SftData:
    return _encode_examples(source.iter_examples(split), tokenizer, dtype, block_size, label=split)

def prepare_sft(source: SFTDataSource, tokenizer: Tokenizer, block_size: int) -> dict[Split, SftData]:
    """
    The train and validation splits of the source as the SftData that get_sft_batch takes, using an already-trained tokenizer.
    Raises if any example doesn't fit in block_size.
    """
    dtype = _id_dtype(tokenizer)
    splits: tuple[Split, ...] = ("train", "validation")
    return {split: _encode_split_sft(source, split, tokenizer, dtype, block_size) for split in splits}

def prepare_sft_validation_slices(slices: dict[str, list[SftExample]], tokenizer: Tokenizer, block_size: int) -> dict[str, SftData]:
    """
    The validation slices (see SFTDataSource.get_validation_slices) as SftData, one per slice, encoded the same way as the train and validation splits.
    The rows of each are in the order of its examples. Each slice is as wide as its own longest example. Raises if any example doesn't fit in block_size.
    """
    dtype = _id_dtype(tokenizer)
    return {name: _encode_examples(examples, tokenizer, dtype, block_size, label=f"slice {name}") for name, examples in slices.items()}
