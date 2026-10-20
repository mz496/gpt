import numpy as np

from src.tiny_shakespeare.main import decode, encode, get_batch


def test_decode_encode_roundtrip() -> None:
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    vocab = sorted(set(text))

    assert decode(encode("ac", vocab), vocab) == "ac"


def test_get_batch_returns_contiguous_windows_with_next_token_as_target() -> None:
    # Values are offset from indices so a start index can't be mistaken for a token.
    tokenized = list(range(100, 120))
    batch_size, input_size = 50, 4

    batch = get_batch(tokenized, np.random.default_rng(0), batch_size, input_size)

    assert len(batch) == batch_size
    for example in batch:
        start = example.context[0] - 100
        assert example.context == tokenized[start : start + input_size]
        assert example.target == tokenized[start + input_size]
