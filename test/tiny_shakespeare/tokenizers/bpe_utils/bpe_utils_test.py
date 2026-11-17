import pytest

from src.tiny_shakespeare.tokenizers.bpe_utils.bpe_utils import (
    IllegalMergeError,
    Merge,
    apply_merges,
    build_vocab,
    most_common_pair,
    replace_for_decode,
    replace_for_encode,
    unapply_merges,
)

X = 256


def once(*chunks: list[int]) -> list[tuple[list[int], int]]:
    """Chunks that each occur once."""
    return [(chunk, 1) for chunk in chunks]


def test_replace_for_encode_runs_go_forward() -> None:
    assert replace_for_encode([1, 1, 1, 1, 1], (1, 1), X) == [X, X, 1]


def test_replace_for_decode_inverts_encode() -> None:
    tokens = [1, 1, 1, 1, 1, 2]
    encoded = replace_for_encode(tokens, (1, 1), X)
    assert replace_for_decode(encoded, X, (1, 1)) == tokens


def test_most_common_pair_does_not_span_chunks() -> None:
    # Joined, [1, 2, 2, 3] would have the pair (2, 2); kept as separate chunks, every pair occurs once.
    with pytest.raises(IllegalMergeError):
        most_common_pair(once([1, 2], [2, 3]))


def test_most_common_pair_sums_across_chunks() -> None:
    assert most_common_pair(once([3, 4, 5, 6], [1, 2], [1, 2], [1, 2], [5, 6])) == (1, 2)


def test_most_common_pair() -> None:
    assert most_common_pair(once([1, 2, 1, 2, 3, 4])) == (1, 2)


def test_most_common_pair_ties_go_to_first_seen() -> None:
    assert most_common_pair(once([5, 6, 5, 6, 1, 2, 1, 2])) == (5, 6)


def test_most_common_pair_halts_when_only_singletons() -> None:
    with pytest.raises(IllegalMergeError):
        most_common_pair(once([1, 2, 3]))


def test_most_common_pair_halts_when_empty() -> None:
    with pytest.raises(IllegalMergeError):
        most_common_pair(once([1], []))


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


def test_most_common_pair_counts_a_chunk_as_many_times_as_it_occurs() -> None:
    # (3, 4) is in a chunk that occurs 5 times, which beats (1, 2) in a chunk that occurs twice.
    assert most_common_pair([([3, 4], 5), ([1, 2], 2)]) == (3, 4)
    assert most_common_pair([([3, 4], 2), ([1, 2], 5)]) == (1, 2)


def test_most_common_pair_with_occurrences_matches_repeating_the_chunks() -> None:
    chunks = [([1, 2, 1, 2, 3], 3), ([3, 4, 3], 1), ([9, 9, 9], 4), ([1, 2], 2)]
    repeated = once(*[tokens for tokens, occurrences in chunks for _ in range(occurrences)])

    assert most_common_pair(chunks) == most_common_pair(repeated)


def test_most_common_pair_counts_a_pair_that_repeats_inside_a_chunk_each_time() -> None:
    # (1, 2) is in the chunk twice and the chunk occurs 3 times: 6, against 5 for (7, 8).
    assert most_common_pair([([1, 2, 1, 2], 3), ([7, 8], 5)]) == (1, 2)


def test_most_common_pair_with_occurrences_breaks_ties_towards_the_pair_seen_first() -> None:
    assert most_common_pair([([5, 6], 4), ([1, 2], 4)]) == (5, 6)
    assert most_common_pair([([1, 2], 4), ([5, 6], 4)]) == (1, 2)


def test_most_common_pair_with_occurrences_still_raises_when_nothing_repeats() -> None:
    with pytest.raises(IllegalMergeError):
        most_common_pair(once([1, 2], [3, 4]))
    most_common_pair([([1, 2], 2)])  # a pair in one chunk that occurs twice is worth merging
