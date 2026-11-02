import torch

from src.tiny_shakespeare.gpt.train import get_batch
from src.tiny_shakespeare.utils import take_fraction


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


def test_take_fraction_one_returns_the_text_unchanged() -> None:
    text = "".join(chr(97 + i % 26) for i in range(5_321))
    assert take_fraction(text, 1.0) == text


def chunks_of(s: str) -> set[str]:
    return {s[i:i + 1000] for i in range(0, len(s), 1000)}


def test_take_fraction_is_nested_and_deterministic() -> None:
    text = "".join(f"{i:04d}" for i in range(2_500))  # 10,000 chars, each 1000-char chunk distinguishable
    eighth, quarter, half = (take_fraction(text, f) for f in (0.125, 0.25, 0.5))
    assert len(eighth) < len(quarter) < len(half) < len(text)
    assert chunks_of(eighth) <= chunks_of(quarter) <= chunks_of(half) <= chunks_of(text)
    assert take_fraction(text, 0.25) == quarter


def test_take_fraction_keeps_chunks_in_original_order() -> None:
    text = "".join(f"{i:04d}" for i in range(2_500))
    sample = take_fraction(text, 0.5)
    starts = [int(sample[i:i + 4]) for i in range(0, len(sample), 1000)]
    assert starts == sorted(starts)


def test_take_fraction_is_not_a_prefix() -> None:
    text = "".join(f"{i:04d}" for i in range(2_500))
    assert not text.startswith(take_fraction(text, 0.5))
