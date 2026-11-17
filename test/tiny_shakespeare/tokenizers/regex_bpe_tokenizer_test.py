import pytest
import regex as re

from src.tiny_shakespeare.data.tiny_shakespeare_source import TinyShakespeareSource
from src.tiny_shakespeare.tokenizers.bpe_utils.bpe_utils import NUM_BASE_TOKENS, IllegalMergeError, Merge, most_common_pair, replace_for_encode
from src.tiny_shakespeare.tokenizers.regex_bpe_tokenizer import GPT4_SPLIT_PATTERN, RegexBPETokenizer
from src.tiny_shakespeare.tokenizers.utils import make_tokenizer
from src.tiny_shakespeare.utils import string_to_bytes

SHAKESPEARE = TinyShakespeareSource().get_tokenizer_text()


def roundtrip(s: str, target_vocab_size: int = 300) -> None:
    tokenizer = RegexBPETokenizer(target_vocab_size)
    tokenizer.train(s)
    assert tokenizer.decode_as_string(tokenizer.encode_string(s)) == s


def test_pattern_split() -> None:
    assert re.findall(GPT4_SPLIT_PATTERN, "hello world you've you're what's this i'd I'd") == [
        "hello", " world", " you", "'ve", " you", "'re", " what", "'s", " this", " i", "'d", " I", "'d",
    ]


def test_roundtrip_contractions() -> None:
    roundtrip("hello world you've you're what's this i'd I'd, hello world you've you're what's this i'd I'd")


def test_roundtrip_whitespace() -> None:
    roundtrip("a  b\n\nc \t d\r\n  e   \n f  \n\n\n g,  h. a  b\n\nc \t d\r\n  e   \n f  \n\n\n g,  h.")


def test_roundtrip_unicode() -> None:
    roundtrip("Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 We all know we ought to “support Unicode” in our software — don’t we? 12345 6789")


def test_roundtrip_shakespeare_slice() -> None:
    roundtrip(SHAKESPEARE[:20000], 400)


def test_merges_do_not_cross_chunk_boundaries() -> None:
    # Chunks are "ab", " ab", " ab", " ab". Joined, "b " would be a candidate pair, but it spans two chunks.
    tokenizer = RegexBPETokenizer(300)
    tokenizer.train("ab ab ab ab")
    merged = [tokenizer.get_vocab()[merge.to_token] for merge in tokenizer.merges]
    assert merged == [b"ab", b" ab"]


def test_training_is_deterministic() -> None:
    first = RegexBPETokenizer(400)
    second = RegexBPETokenizer(400)
    first.train(SHAKESPEARE[:20000])
    second.train(SHAKESPEARE[:20000])
    assert first.merges == second.merges
    assert first.get_vocab() == second.get_vocab()


def test_encode_is_stable_with_cache() -> None:
    tokenizer = RegexBPETokenizer(300)
    text = SHAKESPEARE[:5000]
    tokenizer.train(text)
    first = tokenizer.encode_string(text)
    assert tokenizer.encode_string(text) == first


def test_untrained_raises() -> None:
    tokenizer = RegexBPETokenizer(300)
    with pytest.raises(Exception, match="not been trained"):
        tokenizer.encode_string("hi")
    with pytest.raises(Exception, match="not been trained"):
        tokenizer.decode_as_string([1])


def test_vocab_size_too_small_raises() -> None:
    with pytest.raises(Exception):
        RegexBPETokenizer(256)


def test_make_tokenizer() -> None:
    assert isinstance(make_tokenizer("regex_bpe", 300), RegexBPETokenizer)


def merges_learned_chunk_by_chunk(pieces: list[str], target_vocab_size: int) -> list[Merge]:
    """The straightforward way to train: every occurrence of every chunk is rewritten on every pass, with no counting of distinct chunks."""
    chunks = [(string_to_bytes(chunk), 1) for piece in pieces for chunk in re.findall(GPT4_SPLIT_PATTERN, piece)]
    merges: list[Merge] = []
    for _ in range(target_vocab_size - NUM_BASE_TOKENS):
        try:
            from_tokens = most_common_pair(chunks)
        except IllegalMergeError:
            break
        merge = Merge(from_tokens, NUM_BASE_TOKENS + len(merges))
        merges.append(merge)
        chunks = [(replace_for_encode(tokens, merge.from_tokens, merge.to_token), occurrences) for tokens, occurrences in chunks]
    return merges


@pytest.mark.parametrize("target_vocab_size", [300, 400, 600])
def test_training_on_distinct_chunks_learns_the_same_merges_as_training_on_every_occurrence(target_vocab_size: int) -> None:
    text = SHAKESPEARE[:40_000]
    tokenizer = RegexBPETokenizer(target_vocab_size)
    tokenizer.train(text)

    assert tokenizer.merges == merges_learned_chunk_by_chunk([text], target_vocab_size)


def test_training_on_distinct_chunks_learns_the_same_merges_with_special_tokens_in_the_text() -> None:
    eot = "<|endoftext|>"
    documents = [SHAKESPEARE[i:i + 700] for i in range(0, 14_000, 700)]
    text = eot.join(documents)
    tokenizer = RegexBPETokenizer(310)
    tokenizer.register_special_tokens([eot])
    tokenizer.train(text)

    # One slot of the target goes to the special token.
    assert tokenizer.merges == merges_learned_chunk_by_chunk(documents, 309)
