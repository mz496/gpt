import pytest

from src.tiny_shakespeare.tokenizers.bpe_utils.bpe_utils import (
    IllegalMergeError,
    Merge,
    apply_merges,
    build_vocab,
    count_pairs,
    most_common_pair,
    replace_for_decode,
    replace_for_encode,
    unapply_merges,
)

X = 256


def test_replace_for_encode_runs_go_forward() -> None:
    assert replace_for_encode([1, 1, 1, 1, 1], (1, 1), X) == [X, X, 1]


def test_replace_for_decode_inverts_encode() -> None:
    tokens = [1, 1, 1, 1, 1, 2]
    encoded = replace_for_encode(tokens, (1, 1), X)
    assert replace_for_decode(encoded, X, (1, 1)) == tokens


def test_count_pairs_does_not_span_chunks() -> None:
    counts = count_pairs([[1, 2], [2, 3]])
    assert counts == {(1, 2): 1, (2, 3): 1}
    assert (2, 2) not in counts


def test_count_pairs_sums_across_chunks() -> None:
    assert count_pairs([[1, 2, 1, 2], [1, 2]])[(1, 2)] == 3


def test_most_common_pair() -> None:
    assert most_common_pair(count_pairs([[1, 2, 1, 2, 3, 4]])) == (1, 2)


def test_most_common_pair_ties_go_to_first_seen() -> None:
    assert most_common_pair(count_pairs([[5, 6, 5, 6, 1, 2, 1, 2]])) == (5, 6)


def test_most_common_pair_halts_when_only_singletons() -> None:
    with pytest.raises(IllegalMergeError):
        most_common_pair(count_pairs([[1, 2, 3]]))


def test_most_common_pair_halts_when_empty() -> None:
    with pytest.raises(IllegalMergeError):
        most_common_pair(count_pairs([[1], []]))


def test_apply_and_unapply_merges_roundtrip() -> None:
    merges = [Merge((1, 2), 256), Merge((256, 1), 257)]
    tokens = [1, 2, 1, 3]
    encoded = apply_merges(tokens, merges)
    assert encoded == [257, 3]
    assert unapply_merges(encoded, merges) == tokens


def test_build_vocab_unrolls_merges() -> None:
    vocab = build_vocab([Merge((1, 2), 256), Merge((256, 1), 257)])
    assert len(vocab) == 258
    assert vocab[65] == b"A"
    assert vocab[256] == bytes([1, 2])
    assert vocab[257] == bytes([1, 2, 1])
