"""
Byte-pair encoding implementation

From https://www.reedbeta.com/blog/programmers-intro-to-unicode/
From Karpathy's "Let's build the GPT Tokenizer", ~27:46

"""

from pathlib import Path
from typing import override

from ..utils import bytes_to_string, string_to_bytes
from .bpe_utils.bpe_utils import (
    NUM_BASE_TOKENS,
    IllegalMergeError,
    Merge,
    apply_merges,
    build_vocab,
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
    NUM_DISTINCT_TOKENS = NUM_BASE_TOKENS

    def __init__(self, target_vocab_size: int):
        """
        Initialized with the plaintext and the target vocab size.
        """

        # Init the merges: empty to start
        # A merge is: int substring -> merged.
        # In decode, we convert this to a mapping.
        self.merges: list[Merge] = []

        self._merged_token_counter: int = self.NUM_DISTINCT_TOKENS

        self._vocab: dict[int, bytes] = build_vocab(self.merges)

        # The vocab size increases by 1 every time we merge, so this puts an upper bound on the number of passes done.
        if target_vocab_size <= self.NUM_DISTINCT_TOKENS:
            raise Exception(f"cannot set a target vocab size <= the number of distinct tokens in UTF-8 ({self.NUM_DISTINCT_TOKENS})")
        self._num_merge_passes = target_vocab_size - self.NUM_DISTINCT_TOKENS

    def _get_next_merged_token_counter(self):
        """
        Returns the next free merged-token-counter and then advances it for the next caller.
        """
        next_counter = self._merged_token_counter
        self._merged_token_counter += 1
        return next_counter

    @override
    def get_vocab(self) -> dict[int, bytes]:
        """Cached by train(). Note that before train() is called, it's just all raw bytes, unmerged."""
        return self._vocab

    @override
    def train(self, s: str) -> None:
        """
        Learns the merges based on the data frequencies.

        Each pass finds the best merge to do based on the highest pair frequency and records it.
        e.g. aaabaa -> ZabZ (but using a token that's outside the vocab)

        Inputs:
        - s: the data string to train the tokenizer on
        """
        tokens = string_to_bytes(s)

        for _ in range(self._num_merge_passes):
            try:
                if len(tokens) <= 2:
                    raise IllegalMergeError("tokens must be length > 2")
                from_tokens = most_common_pair([tokens])
                merge = Merge(from_tokens, self._get_next_merged_token_counter())
                # Record the merge, then apply it so we keep shrinking the string.
                self.merges.append(merge)
                tokens = replace_for_encode(tokens, merge.from_tokens, merge.to_token)
            except IllegalMergeError as e:
                print(f"merges halted: {e}")
                break

        self._vocab = build_vocab(self.merges)

    @override
    def encode_string(self, s: str) -> list[int]:
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
    def decode_as_string(self, tokens: list[int]) -> str:
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