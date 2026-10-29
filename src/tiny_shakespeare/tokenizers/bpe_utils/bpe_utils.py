"""
Pure helpers shared by the byte-pair-encoding tokenizers.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# Due to our UTF-8 encoding, the count of distinct byte tokens is 2**8 = 256.
# Merged tokens start at this value and up, so they won't ever conflict with raw bytes, and we can use them as embedding table indices later.
NUM_BASE_TOKENS = 256


@dataclass
class Merge:
    from_tokens: tuple[int, int]
    to_token: int

    def __repr__(self):
        return f"Merge({self.from_tokens} -> {self.to_token})"


class IllegalMergeError(ValueError):
    """Raised when a merge can't be applied to the given tokens."""


def most_common_pair(chunks: Iterable[list[int]]) -> tuple[int, int]:
    """
    Returns the most frequent adjacent token pair across the chunks; ties go to the pair seen first.
    Pairs never span two chunks.
    Note we're slightly off in case of long runs like aaa - this counts 2 occurrences of aa, but we only can compress it once.
    Raises IllegalMergeError if there is nothing worth merging (no pairs, or the best pair occurs only once).
    """
    counts: Counter[tuple[int, int]] = Counter()
    for tokens in chunks:
        # Align every token with its next token, dropping the last.
        counts.update(zip(tokens, tokens[1:]))
    if not counts:
        raise IllegalMergeError("no pairs to merge")
    pair, freq = counts.most_common(1)[0]
    if freq == 1:
        raise IllegalMergeError(f"no more merges possible, most common pair only occurs once: {(pair, freq)}")
    return pair


def replace_for_encode(tokens: list[int], from_tokens: tuple[int, int], to_token: int) -> list[int]:
    """
    Replaces from_tokens with to_token, going forward only, thus replacing the first occurrence(s) of from_tokens with to_token.
    e.g. 1,1,1,1,1 where to_token is X becomes X,X,1.
    Pure.
    """
    out: list[int] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == from_tokens:
            out.append(to_token)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def replace_for_decode(tokens: list[int], to_token: int, from_tokens: tuple[int, int]) -> list[int]:
    """
    Replaces to_token with from_tokens, in reverse of replace_for_encode.
    """
    out: list[int] = []
    for t in tokens:
        if t == to_token:
            out.extend(from_tokens)
        else:
            out.append(t)
    return out


def apply_merges(tokens: list[int], merges: list[Merge]) -> list[int]:
    """Applies every merge in the order it was learned."""
    for merge in merges:
        tokens = replace_for_encode(tokens, merge.from_tokens, merge.to_token)
    return tokens


def unapply_merges(tokens: list[int], merges: list[Merge]) -> list[int]:
    """Undoes apply_merges, back to raw bytes (0-255)."""
    for merge in reversed(merges):
        tokens = replace_for_decode(tokens, merge.to_token, merge.from_tokens)
    return tokens


def build_vocab(merges: list[Merge]) -> dict[int, bytes]:
    """
    Maps every token id to the full sequence of bytes (0-255) it stands for.
    Merges are unrolled in creation order, so e.g. 1,2 -> 257 and 257,1 -> 258 gives 258 -> bytes([1, 2, 1]).
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(NUM_BASE_TOKENS)}
    for merge in merges:
        a, b = merge.from_tokens
        vocab[merge.to_token] = vocab[a] + vocab[b]
    return vocab
