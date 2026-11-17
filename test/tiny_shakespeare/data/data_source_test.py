from typing import Iterator, override

from src.tiny_shakespeare.data.data_source import DataSource, Split


class FakeSource(DataSource):
    """Yields the given chunks for each split, and records how many train chunks were pulled."""

    name = "fake"

    def __init__(self, train: list[str], validation: list[str] | None = None):
        self.train = train
        self.validation = validation or []
        self.train_chunks_pulled = 0

    @override
    def iter_text(self, split: Split) -> Iterator[str]:
        if split == "validation":
            yield from self.validation
            return
        for chunk in self.train:
            self.train_chunks_pulled += 1
            yield chunk


def test_get_tokenizer_text_joins_train_chunks_up_to_the_chunk_that_reaches_the_cap() -> None:
    source = FakeSource(["aaaa", "bbbb", "cccc", "dddd"])
    assert source.get_tokenizer_text(1) == "aaaa"
    assert source.get_tokenizer_text(4) == "aaaa"
    assert source.get_tokenizer_text(5) == "aaaabbbb"
    assert source.get_tokenizer_text(100) == "aaaabbbbccccdddd"


def test_get_tokenizer_text_does_not_pull_chunks_past_the_cap() -> None:
    source = FakeSource(["aaaa", "bbbb", "cccc", "dddd"])
    source.get_tokenizer_text(8)
    assert source.train_chunks_pulled == 2


def test_get_tokenizer_text_with_a_zero_cap_reads_nothing() -> None:
    source = FakeSource(["aaaa"])
    assert source.get_tokenizer_text(0) == ""
    assert source.train_chunks_pulled == 0


def test_get_tokenizer_text_of_an_empty_source_is_empty() -> None:
    assert FakeSource([]).get_tokenizer_text(10) == ""


def test_get_tokenizer_text_never_includes_validation_text() -> None:
    source = FakeSource(["train text"], ["validation text"])
    assert source.get_tokenizer_text() == "train text"
