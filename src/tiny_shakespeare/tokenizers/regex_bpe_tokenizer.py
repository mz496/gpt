"""
Byte-pair encoding that first splits the text into chunks with the GPT-4 regex, so merges never cross a chunk boundary.
"""

# Note, this is not the same as python's re.
import regex as re
from collections import Counter
from typing import Any, override

from ..utils import bytes_to_string, string_to_bytes
from .bpe_utils.bpe_utils import (
    NUM_BASE_TOKENS,
    IllegalMergeError,
    Merge,
    apply_merges,
    build_vocab,
    merges_from_state,
    merges_to_state,
    most_common_pair,
    replace_for_encode,
    unapply_merges,
)
from .tokenizer import Tokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexBPETokenizer(Tokenizer):
    """
    Byte-pair encoding tokenizer that splits the text with GPT4_SPLIT_PATTERN and learns merges within each chunk.
    A pair of tokens is only a merge candidate if they sit next to each other inside the same chunk.
    """

    name = "regex_bpe"

    def __init__(self, target_vocab_size: int):
        """
        Initialized with the target vocab size, which includes any special tokens registered before train().
        """
        super().__init__()
        self.merges: list[Merge] = []

        self._merged_token_counter: int = NUM_BASE_TOKENS

        # Encoded chunks by chunk text, since the same chunks (" the", ",") repeat a lot.
        self._chunk_cache: dict[str, list[int]] = {}

        # The vocab size increases by 1 every time we merge, so this (less the special tokens) puts an upper bound on the number of passes done.
        if target_vocab_size <= NUM_BASE_TOKENS:
            raise Exception(f"cannot set a target vocab size <= the number of distinct tokens in UTF-8 ({NUM_BASE_TOKENS})")
        self.target_vocab_size = target_vocab_size

    def _get_next_merged_token_counter(self):
        """
        Returns the next free merged-token-counter and then advances it for the next caller.
        """
        next_counter = self._merged_token_counter
        self._merged_token_counter += 1
        return next_counter

    @override
    def _train_on_pieces(self, pieces: list[str]) -> None:
        """
        Learns the merges based on the data frequencies.

        The text is split into chunks, and each pass finds the best merge to do based on the highest pair frequency across all chunks and records it.
        Pairs never span two chunks (and so never two pieces, the text on either side of a special token).

        Inputs:
        - pieces: the data strings to train the tokenizer on
        """
        # The vocab holds the raw bytes, the merges, and the special tokens.
        num_merge_passes = self.target_vocab_size - NUM_BASE_TOKENS - len(self._special_tokens)
        if num_merge_passes <= 0:
            raise ValueError(f"target vocab size {self.target_vocab_size} leaves no room for merges after {NUM_BASE_TOKENS} bytes and {len(self._special_tokens)} special tokens")
        # The same chunks (" the", ",") repeat a lot, so keep each distinct chunk once with how many times it occurs, in the order they first occur.
        # That makes every pass cost the size of the distinct chunks instead of the whole text, and gives the same merges.
        chunk_counts: Counter[str] = Counter()
        for piece in pieces:
            chunk_counts.update(re.findall(GPT4_SPLIT_PATTERN, piece))
        chunks = [(string_to_bytes(chunk), occurrences) for chunk, occurrences in chunk_counts.items()]

        for _ in range(num_merge_passes):
            try:
                from_tokens = most_common_pair(chunks)
                merge = Merge(from_tokens, self._get_next_merged_token_counter())
                # Record the merge, then apply it so we keep shrinking the chunks.
                self.merges.append(merge)
                chunks = [(replace_for_encode(tokens, merge.from_tokens, merge.to_token), occurrences) for tokens, occurrences in chunks]
            except IllegalMergeError as e:
                print(f"merges halted: {e}")
                break

        self._chunk_cache = {}

    @override
    def _encode_plain(self, s: str) -> list[int]:
        """
        Encode the string as a sequence of tokens, one regex chunk at a time.

        Inputs:
        - s: the string to encode

        Outputs:
        - tokens: the int array of tokens
        """
        if len(self.merges) == 0:
            raise Exception("RegexBPETokenizer has not been trained yet; call train() first")
        tokens: list[int] = []
        for chunk in re.findall(GPT4_SPLIT_PATTERN, s):
            if chunk not in self._chunk_cache:
                self._chunk_cache[chunk] = apply_merges(string_to_bytes(chunk), self.merges)
            tokens.extend(self._chunk_cache[chunk])
        return tokens

    @override
    def _decode_plain(self, tokens: list[int]) -> str:
        """
        Decodes the tokens back to a string. Chunk boundaries don't matter here, since everything expands back to bytes.

        Inputs:
        - tokens: the int array of tokens

        Outputs:
        - the decoded string
        """
        if len(self.merges) == 0:
            raise Exception("RegexBPETokenizer has not been trained yet; call train() first")
        return bytes_to_string(unapply_merges(tokens, self.merges))

    @override
    def __str__(self) -> str:
        return f"RegexBPETokenizer<merges={self.merges}>"

    @override
    def state(self) -> dict[str, Any]:
        # The target is saved rather than derived from the merges, since training can halt before reaching it.
        # The pattern is saved so that a later change to it can't silently reinterpret old merges.
        return {"target_vocab_size": self.target_vocab_size, "pattern": GPT4_SPLIT_PATTERN, "merges": merges_to_state(self.merges)}

    @override
    def _num_learned_tokens(self) -> int:
        return NUM_BASE_TOKENS + len(self.merges)

    @override
    def _learned_vocab(self) -> dict[int, bytes]:
        return build_vocab(self.merges)

    @classmethod
    @override
    def from_state(cls, state: dict[str, Any]) -> "RegexBPETokenizer":
        if state["pattern"] != GPT4_SPLIT_PATTERN:
            raise ValueError("the saved tokenizer was trained with a different split pattern than GPT4_SPLIT_PATTERN")
        tokenizer = cls(state["target_vocab_size"])
        tokenizer.register_special_tokens(state["special_tokens"])
        tokenizer.merges = merges_from_state(state["merges"])
        tokenizer._merged_token_counter = NUM_BASE_TOKENS + len(tokenizer.merges)
        tokenizer._trained = True
        return tokenizer
