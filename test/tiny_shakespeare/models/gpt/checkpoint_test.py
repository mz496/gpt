from pathlib import Path

import pytest
import torch

from src.tiny_shakespeare.models.gpt.model import (
    GPT,
    GPTConfig,
    MultiHeadAttentionParallelHeads,
    checkpoint_tokenizer_path,
    load_checkpoint,
    load_checkpoint_tokenizer,
    save_checkpoint,
)
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare.tokenizers.utils import TOKENIZER_NAMES, make_tokenizer

TEXT = "the quick brown fox jumps over the lazy dog, then the dog chases the fox. " * 20


@pytest.fixture(autouse=True)
def run_in_a_temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # checkpoints are written to the relative path checkpoints/<run_id>/
    monkeypatch.chdir(tmp_path)


def trained(name: str) -> Tokenizer:
    tokenizer = make_tokenizer(name, 280)
    tokenizer.train(TEXT)
    return tokenizer


def small_model(tokenizer: Tokenizer) -> GPT:
    config = GPTConfig(vocab_size=len(tokenizer.get_vocab()), n_embd=8, n_head=2, n_layer=1, block_size=4, dropout=0.0)
    return GPT(config)


@pytest.mark.parametrize("name", TOKENIZER_NAMES)
def test_the_checkpoint_ships_a_tokenizer_that_matches_the_one_trained(name: str) -> None:
    tokenizer = trained(name)

    save_checkpoint(small_model(tokenizer), "run", tokenizer)

    loaded = load_checkpoint_tokenizer("run")
    assert loaded.get_vocab() == tokenizer.get_vocab()
    assert loaded.encode_string(TEXT) == tokenizer.encode_string(TEXT)


def test_the_tokenizer_is_saved_next_to_the_model() -> None:
    tokenizer = trained("char")
    path = save_checkpoint(small_model(tokenizer), "run", tokenizer)
    assert checkpoint_tokenizer_path("run") == path.with_name("tokenizer.json")
    assert path.is_file() and checkpoint_tokenizer_path("run").is_file()


def test_a_checkpoint_with_no_tokenizer_raises_when_asked_for_it() -> None:
    tokenizer = trained("char")
    save_checkpoint(small_model(tokenizer), "run", tokenizer)
    checkpoint_tokenizer_path("run").unlink()  # only model.pt left

    with pytest.raises(FileNotFoundError, match="no tokenizer"):
        load_checkpoint_tokenizer("run")


def test_a_run_that_does_not_exist_raises() -> None:
    with pytest.raises(FileNotFoundError, match="no tokenizer"):
        load_checkpoint_tokenizer("nope")


def test_the_saved_model_generates_the_same_as_before_saving() -> None:
    tokenizer = trained("char")
    model = small_model(tokenizer)
    model.eval()
    save_checkpoint(model, "run", tokenizer)
    reloaded = load_checkpoint("run", "cpu")

    idx = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    expected = model.generate(idx, max_tokens=10)
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    actual = reloaded.generate(idx, max_tokens=10)

    assert torch.equal(expected, actual)


def attention_flags(model: GPT) -> list[bool]:
    return [module.flag_use_pytorch_attention_impl for module in model.modules() if isinstance(module, MultiHeadAttentionParallelHeads)]


def test_a_loaded_model_uses_the_attention_it_is_asked_to_and_the_handrolled_one_by_default() -> None:
    tokenizer = trained("char")
    save_checkpoint(small_model(tokenizer), "run", tokenizer)

    assert attention_flags(load_checkpoint("run", "cpu")) == [False]
    assert attention_flags(load_checkpoint("run", "cpu", flag_use_pytorch_attention_impl=True)) == [True]


def test_the_two_attentions_compute_the_same_thing_on_the_same_weights() -> None:
    tokenizer = trained("char")
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    save_checkpoint(small_model(tokenizer), "run", tokenizer)
    handrolled, builtin = load_checkpoint("run", "cpu"), load_checkpoint("run", "cpu", flag_use_pytorch_attention_impl=True)

    idx = torch.randint(0, len(tokenizer.get_vocab()), (3, 4))
    with torch.no_grad():
        expected, actual = handrolled(idx, None)[0], builtin(idx, None)[0]
    # The same function, so it differs by no more than rounding.
    assert torch.allclose(expected, actual, atol=1e-5)


def test_the_flag_does_nothing_for_a_checkpoint_without_parallel_heads() -> None:
    tokenizer = trained("char")
    config = GPTConfig(vocab_size=len(tokenizer.get_vocab()), n_embd=8, n_head=2, n_layer=1, block_size=4, dropout=0.0, flag_use_parallel_heads=False)
    save_checkpoint(GPT(config), "run", tokenizer)

    idx = torch.randint(0, len(tokenizer.get_vocab()), (2, 4))
    with torch.no_grad():
        assert torch.equal(load_checkpoint("run", "cpu")(idx, None)[0], load_checkpoint("run", "cpu", flag_use_pytorch_attention_impl=True)(idx, None)[0])
