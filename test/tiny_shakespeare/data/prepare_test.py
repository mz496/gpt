from typing import Iterator, override

import numpy as np
import pytest

from src.tiny_shakespeare.data.data_source import DataSource, Split
from src.tiny_shakespeare.data.prepare import prepare
from src.tiny_shakespeare.data.tiny_shakespeare_source import TinyShakespeareSource
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare.tokenizers.utils import make_tokenizer

SOURCE = TinyShakespeareSource()
TRAIN = "".join(SOURCE.iter_text("train"))
VALIDATION = "".join(SOURCE.iter_text("validation"))


class ChunkedSource(DataSource):
    """Yields the given chunks for each split."""

    name = "chunked"

    def __init__(self, train: list[str], validation: list[str]):
        self.chunks: dict[Split, list[str]] = {"train": train, "validation": validation}

    @override
    def iter_text(self, split: Split) -> Iterator[str]:
        yield from self.chunks[split]


def trained(name: str) -> Tokenizer:
    tokenizer = make_tokenizer(name, 270)
    tokenizer.train(SOURCE.get_tokenizer_text())
    return tokenizer


@pytest.mark.parametrize("name", ["char", "bpe", "regex_bpe"])
def test_prepared_tokens_equal_encode_string(name: str) -> None:
    tokenizer = trained(name)

    prepared = prepare(SOURCE, tokenizer)

    assert prepared.train.tolist() == tokenizer.encode_string(TRAIN)
    assert prepared.validation.tolist() == tokenizer.encode_string(VALIDATION)


def test_prepared_chars_are_the_text_lengths() -> None:
    prepared = prepare(SOURCE, trained("char"))
    assert prepared.train_chars == len(TRAIN)
    assert prepared.validation_chars == len(VALIDATION)


def test_small_vocab_uses_uint16() -> None:
    prepared = prepare(SOURCE, trained("char"))
    assert prepared.train.dtype == prepared.validation.dtype == np.uint16


def test_vocab_over_65536_uses_uint32() -> None:
    class BigVocab(CharTokenizer):
        @override
        def get_vocab(self) -> dict[int, bytes]:
            return {i: b"" for i in range(2**16 + 1)}
    tokenizer = BigVocab()
    tokenizer.train(SOURCE.get_tokenizer_text())

    prepared = prepare(SOURCE, tokenizer)

    assert prepared.train.dtype == np.uint32


def test_several_chunks_are_tokenized_into_one_array() -> None:
    third = len(TRAIN) // 3
    source = ChunkedSource([TRAIN[:third], TRAIN[third:2 * third], TRAIN[2 * third:]], [VALIDATION[:500], VALIDATION[500:]])
    tokenizer = trained("char")

    prepared = prepare(source, tokenizer)

    assert prepared.train.tolist() == tokenizer.encode_string(TRAIN)
    assert prepared.validation.tolist() == tokenizer.encode_string(VALIDATION)
    assert prepared.train_chars == len(TRAIN)
    assert prepared.validation_chars == len(VALIDATION)


def test_an_empty_split_gives_an_empty_array() -> None:
    prepared = prepare(ChunkedSource([], []), trained("char"))

    assert len(prepared.train) == len(prepared.validation) == 0
    assert prepared.train.dtype == np.uint16
    assert prepared.train_chars == 0
