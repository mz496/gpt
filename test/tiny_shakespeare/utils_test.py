import torch

from src.tiny_shakespeare.gpt.train import get_batch
from src.tiny_shakespeare.utils import decode, encode


def test_decode_encode_roundtrip() -> None:
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    vocab = sorted(set(text))

    assert decode(encode("ac", vocab), vocab) == "ac"


def test_get_batch_returns_contiguous_windows_with_next_token_as_target() -> None:
    # Values are offset from indices so a start index can't be mistaken for a token.
    tokenized = torch.arange(100, 120)
    batch_size, block_size = 50, 4

    contexts, targets = get_batch(tokenized, batch_size, block_size)

    assert contexts.shape == (batch_size, block_size)
    assert targets.shape == (batch_size, block_size)
    for context, target in zip(contexts, targets):
        start = int(context[0]) - 100
        assert context.tolist() == tokenized[start : start + block_size].tolist()
        # targets are the context shifted by one token
        assert target.tolist() == tokenized[start + 1 : start + block_size + 1].tolist()
