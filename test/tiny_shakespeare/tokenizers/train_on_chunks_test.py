from collections.abc import Iterator

import pytest

from src.tiny_shakespeare.tokenizers.bpe_tokenizer import BPETokenizer
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer
from src.tiny_shakespeare.tokenizers.regex_bpe_tokenizer import RegexBPETokenizer
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer

EOT = "<|endoftext|>"
TEXT = "the quick brown fox jumps over the lazy dog, then the dog chases the fox. " * 20


def chunked(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


# --- char: learns as the chunks go by

def test_char_trains_on_a_stream() -> None:
    assert CharTokenizer.trains_on_stream is True


def test_char_vocab_from_chunks_is_the_same_as_from_the_joined_text() -> None:
    streamed = CharTokenizer()
    streamed.train_on_chunks(chunked(TEXT, 7))
    joined = CharTokenizer()
    joined.train(TEXT)

    assert streamed.vocab == joined.vocab
    assert streamed.get_vocab() == joined.get_vocab()
    assert streamed.encode_string(TEXT) == joined.encode_string(TEXT)


def test_char_train_on_chunks_consumes_a_generator_once_without_holding_the_chunks() -> None:
    seen: list[int] = []

    def generator() -> Iterator[str]:
        for chunk in chunked(TEXT, 50):
            seen.append(len(chunk))
            yield chunk

    tokenizer = CharTokenizer()
    tokenizer.train_on_chunks(generator())

    assert sum(seen) == len(TEXT)
    assert tokenizer.vocab == sorted(set(TEXT))


def test_char_train_on_chunks_learns_a_character_that_only_a_later_chunk_has() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train_on_chunks(["abc", "abc", "abcz"])
    assert tokenizer.vocab == ["a", "b", "c", "z"]


def test_char_train_on_chunks_ignores_the_special_token_text() -> None:
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens([EOT])

    tokenizer.train_on_chunks(["ab" + EOT, "ba" + EOT + "ab"])

    assert tokenizer.vocab == ["a", "b"]
    assert tokenizer.special_token_ids == {EOT: 2}
    assert tokenizer.encode_string("ab" + EOT) == [0, 1, 2]


def test_char_train_on_chunks_handles_empty_chunks_and_an_empty_stream() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train_on_chunks(["", "ab", ""])
    assert tokenizer.vocab == ["a", "b"]

    empty = CharTokenizer()
    empty.train_on_chunks([])
    assert empty.vocab == []
    assert empty.get_vocab() == {}


def test_a_char_tokenizer_trained_on_chunks_is_trained() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train_on_chunks(["abc"])
    with pytest.raises(ValueError, match="before train"):
        tokenizer.register_special_tokens([EOT])


def test_char_train_on_chunks_resets_a_vocab_that_was_already_asked_for() -> None:
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens([EOT])
    assert tokenizer.get_vocab() == {0: EOT.encode()}  # cached before training

    tokenizer.train_on_chunks(["ab"])

    assert tokenizer.get_vocab() == {0: b"a", 1: b"b", 2: EOT.encode()}


# --- the others: joined, as train() does

@pytest.mark.parametrize("cls", [BPETokenizer, RegexBPETokenizer])
def test_bpe_tokenizers_do_not_train_on_a_stream(cls: type[Tokenizer]) -> None:
    assert cls.trains_on_stream is False


@pytest.mark.parametrize("cls", [BPETokenizer, RegexBPETokenizer])
def test_train_on_chunks_for_bpe_equals_training_on_the_joined_text(cls: type[BPETokenizer | RegexBPETokenizer]) -> None:
    streamed = cls(300)
    streamed.train_on_chunks(chunked(TEXT, 37))
    joined = cls(300)
    joined.train(TEXT)

    assert streamed.merges == joined.merges
    assert streamed.get_vocab() == joined.get_vocab()


@pytest.mark.parametrize("cls", [BPETokenizer, RegexBPETokenizer])
def test_train_on_chunks_for_bpe_skips_special_tokens_in_the_chunks(cls: type[BPETokenizer | RegexBPETokenizer]) -> None:
    streamed = cls(300)
    streamed.register_special_tokens([EOT])
    streamed.train_on_chunks([TEXT[:700] + EOT, TEXT[700:1400] + EOT])
    joined = cls(300)
    joined.register_special_tokens([EOT])
    joined.train(TEXT[:700] + EOT + TEXT[700:1400] + EOT)

    assert streamed.merges == joined.merges
