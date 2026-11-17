from typing import Iterator

import numpy as np
import pytest

from src.tiny_shakespeare.data import prepare as prepare_module
from src.tiny_shakespeare.data.loader import Dataset, Split, get_tokenizer_text, iter_text
from src.tiny_shakespeare.data.prepare import prepare
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare.tokenizers.utils import make_tokenizer

TRAIN = "".join(iter_text("tiny-shakespeare", "train"))
VALIDATION = "".join(iter_text("tiny-shakespeare", "validation"))


def trained(name: str) -> Tokenizer:
    tokenizer = make_tokenizer(name, 270)
    tokenizer.train(get_tokenizer_text("tiny-shakespeare"))
    return tokenizer


@pytest.mark.parametrize("name", ["char", "bpe", "regex_bpe"])
def test_prepared_tokens_equal_encode_string(name: str) -> None:
    tokenizer = trained(name)

    prepared = prepare("tiny-shakespeare", tokenizer)

    assert prepared.train.tolist() == tokenizer.encode_string(TRAIN)
    assert prepared.validation.tolist() == tokenizer.encode_string(VALIDATION)


def test_prepared_chars_are_the_text_lengths() -> None:
    prepared = prepare("tiny-shakespeare", trained("char"))
    assert prepared.train_chars == len(TRAIN)
    assert prepared.validation_chars == len(VALIDATION)


def test_small_vocab_uses_uint16() -> None:
    prepared = prepare("tiny-shakespeare", trained("char"))
    assert prepared.train.dtype == prepared.validation.dtype == np.uint16


def test_vocab_over_65536_uses_uint32() -> None:
    class BigVocab(CharTokenizer):
        def get_vocab(self) -> dict[int, bytes]:
            return {i: b"" for i in range(2**16 + 1)}
    tokenizer = BigVocab()
    tokenizer.train(get_tokenizer_text("tiny-shakespeare"))

    prepared = prepare("tiny-shakespeare", tokenizer)

    assert prepared.train.dtype == np.uint32


def test_several_chunks_are_tokenized_into_one_array(monkeypatch: pytest.MonkeyPatch) -> None:
    def three_chunks(_slug: Dataset, split: Split) -> Iterator[str]:
        text = TRAIN if split == "train" else VALIDATION
        third = len(text) // 3
        yield text[:third]
        yield text[third:2 * third]
        yield text[2 * third:]
    monkeypatch.setattr(prepare_module, "iter_text", three_chunks)
    tokenizer = trained("char")

    prepared = prepare("tiny-shakespeare", tokenizer)

    assert prepared.train.tolist() == tokenizer.encode_string(TRAIN)
    assert prepared.validation.tolist() == tokenizer.encode_string(VALIDATION)
    assert prepared.train_chars == len(TRAIN)


def test_an_empty_split_gives_an_empty_array(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_chunks(_slug: Dataset, _split: Split) -> Iterator[str]:
        return iter(())
    monkeypatch.setattr(prepare_module, "iter_text", no_chunks)

    prepared = prepare("tiny-shakespeare", trained("char"))

    assert len(prepared.train) == len(prepared.validation) == 0
    assert prepared.train.dtype == np.uint16
    assert prepared.train_chars == 0
