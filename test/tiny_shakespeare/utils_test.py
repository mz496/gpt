import torch

from src.tiny_shakespeare.gpt.train import get_batch


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


def test_get_batch_with_same_generator_seed_is_reproducible_and_ignores_global_rng() -> None:
    tokenized = torch.arange(1000)

    first = get_batch(tokenized, 8, 4, torch.Generator().manual_seed(0))
    # Burn some global randomness, as dropout would between batches.
    torch.rand(17)
    second = get_batch(tokenized, 8, 4, torch.Generator().manual_seed(0))

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
