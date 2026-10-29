"""
Byte-pair encoding implementation

From https://www.reedbeta.com/blog/programmers-intro-to-unicode/
From Karpathy's "Let's build the GPT Tokenizer", ~27:46

"""

from pathlib import Path

from ..utils import bytes_to_string, string_to_bytes
from .bpe_utils.bpe_utils import (
    NUM_BASE_TOKENS,
    IllegalMergeError,
    Merge,
    apply_merges,
    build_vocab,
    count_pairs,
    most_common_pair,
    replace_for_encode,
    unapply_merges,
)
from .tokenizer import Tokenizer

__all__ = ["BytePairEncoding", "IllegalMergeError", "Merge"]

# 1:40pm


class BytePairEncoding(Tokenizer):
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

    def get_vocab(self) -> dict[int, bytes]:
        """Cached by train(). Note that before train() is called, it's just all raw bytes, unmerged."""
        return self._vocab

    def train(self, s: str):
        """Learns the merges based on the data frequencies."""
        tokens = string_to_bytes(s)

        for _ in range(self._num_merge_passes):
            try:
                merge = self._merge(tokens)
                # Record the merge, then apply it so we keep shrinking the string.
                self.merges.append(merge)
                tokens = replace_for_encode(tokens, merge.from_tokens, merge.to_token)
            except IllegalMergeError as e:
                print(f"merges halted: {e}")
                break

        self._vocab = build_vocab(self.merges)

    def encode_string(self, s: str) -> list[int]:
        return self._encode(string_to_bytes(s))

    def _encode(self, tokens: list[int]) -> list[int]:
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
        return apply_merges(tokens, self.merges)

    def _merge(self, tokens: list[int]) -> Merge:
        """
        Find the best merge to do based on the highest pair frequency and return it.
        aaabaa -> ZabZ (but using a token that's outside the vocab)
        Pure.

        Inputs:
        - tokens: input tokens

        Outputs:
        - merge: the Merge that was done
        """
        if len(tokens) <= 2:
            raise IllegalMergeError("tokens must be length > 2")
        from_tokens = most_common_pair(count_pairs([tokens]))
        to_token = self._get_next_merged_token_counter()

        return Merge(from_tokens, to_token)

    def decode_as_string(self, tokens: list[int]) -> str:
        return bytes_to_string(self._decode(tokens))

    def _decode(self, tokens: list[int]) -> list[int]:
        """
        Decodes the tokens using the state written during encode.

        Inputs:
        - tokens: the int array of tokens

        Outputs:
        - the int array of bytes (0-255)
        """
        if len(self.merges) == 0:
            raise Exception("BPE has not been trained yet; call train() first")
        return unapply_merges(tokens, self.merges)

    def save(self, path: Path):
        # Saves state to the given path as JSON.
        pass

    @classmethod
    def load(cls, path: Path):
        # Loads state from the given path as JSON.
        pass

    def __str__(self) -> str:
        return f"BytePairEncoding<merges={self.merges}>"