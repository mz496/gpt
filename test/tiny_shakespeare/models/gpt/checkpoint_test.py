from pathlib import Path

import pytest
import torch

from src.tiny_shakespeare.models.gpt.model import (
    GPT,
    GPTConfig,
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
