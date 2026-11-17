import pytest
import torch

from src.tiny_shakespeare.models.gpt.model import GPT, GPTConfig

VOCAB_SIZE = 6


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    model = GPT(GPTConfig(vocab_size=VOCAB_SIZE, n_embd=8, n_head=2, n_layer=1, block_size=4, dropout=0.0))
    model.eval()
    return model


def generate(model: GPT, batch_size: int = 1, max_tokens: int = 20, stop_token_ids: set[int] | None = None, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    with torch.no_grad():
        return model.generate(torch.zeros((batch_size, 1), dtype=torch.long), max_tokens=max_tokens, stop_token_ids=stop_token_ids)


@pytest.mark.parametrize("stop_token_ids", [None, set()])
def test_without_stop_tokens_it_generates_max_tokens(model: GPT, stop_token_ids: set[int] | None) -> None:
    assert generate(model, stop_token_ids=stop_token_ids).shape == (1, 21)


def test_stopping_on_every_token_stops_after_the_first_one(model: GPT) -> None:
    assert generate(model, stop_token_ids=set(range(VOCAB_SIZE))).shape == (1, 2)


def test_a_stop_token_the_model_cannot_produce_never_stops_it(model: GPT) -> None:
    assert generate(model, stop_token_ids={VOCAB_SIZE + 10}).shape == (1, 21)


def test_it_stops_right_after_the_first_stop_token_and_matches_the_unstopped_run(model: GPT) -> None:
    full = generate(model)
    generated = full[0, 1:]
    stop_token = int(generated[5])
    first = int((generated == stop_token).nonzero()[0])  # index of the first time that token comes up

    stopped = generate(model, stop_token_ids={stop_token})

    # Stopping changes nothing about the sampling, so it's a prefix of the unstopped output, ending with the stop token.
    assert stopped.shape == (1, 1 + first + 1)
    assert torch.equal(stopped, full[:, :1 + first + 1])
    assert int(stopped[0, -1]) == stop_token


def test_a_batch_keeps_going_until_every_element_has_stopped(model: GPT) -> None:
    full = generate(model, batch_size=3)
    # The step at which each element first produces the token (stop tokens only count once seen).
    stop_token = int(full[0, 4])
    first_steps = [int((full[b, 1:] == stop_token).nonzero()[0]) if bool((full[b, 1:] == stop_token).any()) else None for b in range(3)]

    stopped = generate(model, batch_size=3, stop_token_ids={stop_token})

    if all(step is not None for step in first_steps):
        assert stopped.shape[1] == 1 + max(first_steps) + 1  # pyright: ignore[reportArgumentType]
    else:
        assert stopped.shape == full.shape  # one element never stops, so the batch runs to max_tokens


def test_max_tokens_must_be_positive(model: GPT) -> None:
    with pytest.raises(Exception, match="max_tokens"):
        generate(model, max_tokens=0)
