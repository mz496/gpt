"""
Byte-pair encoding implementation

From https://www.reedbeta.com/blog/programmers-intro-to-unicode/
From Karpathy's "Let's build the GPT Tokenizer", ~27:46

"""

from pathlib import Path
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

__all__ = ["BPETokenizer", "IllegalMergeError", "Merge"]

# 1:40pm


class BPETokenizer(Tokenizer):
    """
    Implements a simple byte-pair encoding tokenizer.
    Does passes over the input, merging the most commonly seen byte pairs.
    """
    name = "bpe"
    NUM_DISTINCT_TOKENS = NUM_BASE_TOKENS

    def __init__(self, target_vocab_size: int):
        """
        Initialized with the target vocab size, which includes any special tokens registered before train().
        """
        super().__init__()

        # Init the merges: empty to start
        # A merge is: int substring -> merged.
        # In decode, we convert this to a mapping.
        self.merges: list[Merge] = []

        self._merged_token_counter: int = self.NUM_DISTINCT_TOKENS

        # The vocab size increases by 1 every time we merge, so this (less the special tokens) puts an upper bound on the number of passes done.
        if target_vocab_size <= self.NUM_DISTINCT_TOKENS:
            raise Exception(f"cannot set a target vocab size <= the number of distinct tokens in UTF-8 ({self.NUM_DISTINCT_TOKENS})")
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

        Each pass finds the best merge to do based on the highest pair frequency and records it.
        e.g. aaabaa -> ZabZ (but using a token that's outside the vocab)
        Pairs never span two pieces (the text on either side of a special token).

        Inputs:
        - pieces: the data strings to train the tokenizer on
        """
        # The vocab holds the raw bytes, the merges, and the special tokens.
        num_merge_passes = self.target_vocab_size - self.NUM_DISTINCT_TOKENS - len(self._special_tokens)
        if num_merge_passes <= 0:
            raise ValueError(f"target vocab size {self.target_vocab_size} leaves no room for merges after {self.NUM_DISTINCT_TOKENS} bytes and {len(self._special_tokens)} special tokens")
        chunks = [string_to_bytes(piece) for piece in pieces]

        for _ in range(num_merge_passes):
            try:
                if sum(len(chunk) for chunk in chunks) <= 2:
                    raise IllegalMergeError("tokens must be length > 2")
                from_tokens = most_common_pair(chunks)
                merge = Merge(from_tokens, self._get_next_merged_token_counter())
                # Record the merge, then apply it so we keep shrinking the pieces.
                self.merges.append(merge)
                chunks = [replace_for_encode(chunk, merge.from_tokens, merge.to_token) for chunk in chunks]
            except IllegalMergeError as e:
                print(f"merges halted: {e}")
                break

    @override
    def _encode_plain(self, s: str) -> list[int]:
        """
        Encode the string as a sequence of tokens.

        Writes state:
        - the merges used during the byte-pair encoding process

        Inputs:
        - tokens: the int array of bytes (0-255)

        Outputs:
        - tokens: the int array of tokens
        - vocab_size: the size of the vocab
        """
        if len(self.merges) == 0:
            raise Exception("BPE has not been trained yet; call train() first")
        return apply_merges(string_to_bytes(s), self.merges)

    @override
    def _decode_plain(self, tokens: list[int]) -> str:
        """
        Decodes the tokens using the state written during encode.

        Inputs:
        - tokens: the int array of tokens

        Outputs:
        - the int array of bytes (0-255)
        """
        if len(self.merges) == 0:
            raise Exception("BPE has not been trained yet; call train() first")
        return bytes_to_string(unapply_merges(tokens, self.merges))

    @override
    def __str__(self) -> str:
        return f"BPETokenizer<merges={self.merges}>"

    @override
    def _num_learned_tokens(self) -> int:
        return self.NUM_DISTINCT_TOKENS + len(self.merges)

    @override
    def _learned_vocab(self) -> dict[int, bytes]:
        return build_vocab(self.merges)

    @override
    def state(self) -> dict[str, Any]:
        # The target is saved rather than derived from the merges, since training can halt before reaching it.
        return {"target_vocab_size": self.target_vocab_size, "merges": merges_to_state(self.merges)}

    @classmethod
    @override
    def from_state(cls, state: dict[str, Any]) -> "BPETokenizer":
        tokenizer = cls(state["target_vocab_size"])
        tokenizer.register_special_tokens(state["special_tokens"])
        tokenizer.merges = merges_from_state(state["merges"])
        tokenizer._merged_token_counter = cls.NUM_DISTINCT_TOKENS + len(tokenizer.merges)
        tokenizer._trained = True
        return tokenizer
