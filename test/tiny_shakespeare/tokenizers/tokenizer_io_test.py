import json
from pathlib import Path

import pytest

from src.tiny_shakespeare.tokenizers.bpe_tokenizer import BPETokenizer
from src.tiny_shakespeare.tokenizers.regex_bpe_tokenizer import RegexBPETokenizer
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare.tokenizers.utils import TOKENIZER_NAMES, load_tokenizer, make_tokenizer

TEXT = "the quick brown fox jumps over the lazy dog, then the dog chases the fox. " * 20
UNSEEN = "the lazy fox, the quick dog."


def trained(name: str, vocab_size: int = 290) -> Tokenizer:
    tokenizer = make_tokenizer(name, vocab_size)
    tokenizer.train(TEXT)
    return tokenizer


@pytest.mark.parametrize("name", TOKENIZER_NAMES)
def test_loaded_tokenizer_matches_the_original(name: str, tmp_path: Path) -> None:
    original = trained(name)
    original.save(tmp_path / "tokenizer.json")

    loaded = load_tokenizer(tmp_path / "tokenizer.json")

    assert type(loaded) is type(original)
    assert loaded.get_vocab() == original.get_vocab()
    assert loaded.encode_string(TEXT) == original.encode_string(TEXT)
    assert loaded.encode_string(UNSEEN) == original.encode_string(UNSEEN)
    assert loaded.decode_as_string(loaded.encode_string(UNSEEN)) == UNSEEN


@pytest.mark.parametrize("name", TOKENIZER_NAMES)
def test_saving_a_loaded_tokenizer_gives_the_same_file(name: str, tmp_path: Path) -> None:
    trained(name).save(tmp_path / "a.json")
    load_tokenizer(tmp_path / "a.json").save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()


@pytest.mark.parametrize("cls", [BPETokenizer, RegexBPETokenizer])
def test_a_bpe_that_halted_early_keeps_its_target_and_merges(cls: type[BPETokenizer | RegexBPETokenizer], tmp_path: Path) -> None:
    original = cls(1000)
    original.train("abababababab")
    assert len(original.merges) < 1000 - 256  # ran out of pairs to merge before reaching the target
    original.save(tmp_path / "tokenizer.json")

    loaded = load_tokenizer(tmp_path / "tokenizer.json")

    assert isinstance(loaded, cls)
    assert loaded.merges == original.merges
    assert loaded.state()["target_vocab_size"] == 1000
    assert loaded.encode_string("ababab") == original.encode_string("ababab")


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "run" / "tokenizer.json"
    trained("char").save(path)
    assert path.is_file()


def write_with(path: Path, saved: dict[str, object]) -> Path:
    path.write_text(json.dumps(saved))
    return path


def test_an_unsupported_format_version_raises(tmp_path: Path) -> None:
    trained("char").save(tmp_path / "t.json")
    saved = json.loads((tmp_path / "t.json").read_text())
    with pytest.raises(ValueError, match="format"):
        load_tokenizer(write_with(tmp_path / "bad.json", {**saved, "format": 99}))


def test_an_unknown_tokenizer_name_raises(tmp_path: Path) -> None:
    trained("char").save(tmp_path / "t.json")
    saved = json.loads((tmp_path / "t.json").read_text())
    with pytest.raises(ValueError, match="unknown tokenizer"):
        load_tokenizer(write_with(tmp_path / "bad.json", {**saved, "name": "wordpiece"}))


def test_a_changed_split_pattern_raises(tmp_path: Path) -> None:
    trained("regex_bpe").save(tmp_path / "t.json")
    saved = json.loads((tmp_path / "t.json").read_text())
    with pytest.raises(ValueError, match="split pattern"):
        load_tokenizer(write_with(tmp_path / "bad.json", {**saved, "pattern": r"\w+"}))
