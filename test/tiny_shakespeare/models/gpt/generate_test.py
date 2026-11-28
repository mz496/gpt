import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from src.tiny_shakespeare.data.sft.sft_data_source import SftExample
from src.tiny_shakespeare.data.special_tokens import END_OF_TEXT, IM_END, IM_START
from src.tiny_shakespeare.models.gpt import generate as generate_module
from src.tiny_shakespeare.models.gpt.generate import generate_response
from src.tiny_shakespeare.models.gpt.model import GPT, GPTConfig, MultiHeadAttentionParallelHeads, load_checkpoint, save_checkpoint
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer

VOCAB_SIZE = 6


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    model = GPT(GPTConfig(vocab_size=VOCAB_SIZE, n_embd=8, n_head=2, n_layer=1, block_size=4, dropout=0.0))
    model.eval()
    return model


def generate(model: GPT, batch_size: int = 1, max_tokens: int = 20, stop_token_ids: set[int] | None = None, seed: int = 0, greedy: bool = False) -> torch.Tensor:
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    with torch.no_grad():
        return model.generate(torch.zeros((batch_size, 1), dtype=torch.long), max_tokens=max_tokens, stop_token_ids=stop_token_ids, greedy=greedy)


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


def test_greedy_does_not_depend_on_the_seed_and_takes_the_most_likely_token(model: GPT) -> None:
    first, second = generate(model, seed=1, greedy=True), generate(model, seed=2, greedy=True)
    assert torch.equal(first, second)
    with torch.no_grad():
        logits, _ = model(torch.zeros((1, 1), dtype=torch.long), None)
    assert int(first[0, 1]) == int(logits[0, -1].argmax())


def test_sampling_is_still_the_default_and_still_depends_on_the_seed(model: GPT) -> None:
    outputs = [generate(model, seed=seed, max_tokens=30) for seed in range(5)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


# --- generate_response

SPECIALS = [END_OF_TEXT, IM_START, IM_END]

@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.from_state({"vocab": list("\nabcdefghijklmnopqrstuvwxyz"), "special_tokens": SPECIALS})


def make_model(tokenizer: CharTokenizer, block_size: int) -> GPT:
    return GPT(GPTConfig(vocab_size=len(tokenizer.get_vocab()), n_embd=8, n_head=2, n_layer=1, block_size=block_size, dropout=0.0))


class Recorder:
    """Stands in for model.generate: remembers how it was called, and 'generates' fixed tokens after the prompt."""

    def __init__(self, answer: list[int]):
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, idx: torch.Tensor, max_tokens: int, stop_token_ids: set[int] | None = None, greedy: bool = False) -> torch.Tensor:
        self.calls.append({"idx": idx, "max_tokens": max_tokens, "stop_token_ids": stop_token_ids, "greedy": greedy})
        return torch.cat([idx, torch.tensor([self.answer])], dim=1)


@pytest.mark.parametrize("greedy", [True, False])
def test_generate_response_asks_with_the_request_template_and_returns_only_the_answer(tokenizer: CharTokenizer, monkeypatch: pytest.MonkeyPatch, greedy: bool) -> None:
    model = make_model(tokenizer, block_size=64)
    recorder = Recorder(tokenizer.encode_string("olleh\n"))
    monkeypatch.setattr(model, "generate", recorder)

    response = generate_response(model, tokenizer, "hello", max_tokens=50, greedy=greedy, device="cpu")

    assert response == "olleh\n"
    (call,) = recorder.calls
    # Exactly the text training used for the request, with nothing of the response in it.
    assert call["idx"].tolist() == [tokenizer.encode_string(SftExample(input="hello", output="olleh").to_request_string())]
    assert call["greedy"] is greedy
    assert call["max_tokens"] == 50
    # It stops at the end of a response, and at any special token.
    assert tokenizer.encode_string("\n")[0] in call["stop_token_ids"]
    assert set(tokenizer.special_token_ids.values()) <= call["stop_token_ids"]


def test_generate_response_rejects_a_request_that_leaves_no_room_to_answer(tokenizer: CharTokenizer) -> None:
    model = make_model(tokenizer, block_size=8)
    with pytest.raises(ValueError, match="block_size"):
        generate_response(model, tokenizer, "hello", max_tokens=5, greedy=True, device="cpu")


def test_the_generate_command_runs_the_model_with_the_builtin_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # checkpoints are read from the relative path checkpoints/<run_id>/
    tokenizer = CharTokenizer.from_state({"vocab": list("\nabc"), "special_tokens": []})
    save_checkpoint(GPT(GPTConfig(vocab_size=4, n_embd=8, n_head=2, n_layer=1, block_size=8, dropout=0.0)), "run", tokenizer)
    loaded: list[GPT] = []

    def spy(run_id: str, device: str, flag_use_pytorch_attention_impl: bool = False) -> GPT:
        model = load_checkpoint(run_id, device, flag_use_pytorch_attention_impl=flag_use_pytorch_attention_impl)
        loaded.append(model)
        return model

    monkeypatch.setattr(generate_module, "load_checkpoint", spy)
    monkeypatch.setattr(generate_module, "get_device", lambda: "cpu")
    monkeypatch.setattr(sys, "argv", ["generate", "run", "--max-tokens", "3", "--greedy"])
    generate_module.main()

    (model,) = loaded
    assert [m.flag_use_pytorch_attention_impl for m in model.modules() if isinstance(m, MultiHeadAttentionParallelHeads)] == [True]
