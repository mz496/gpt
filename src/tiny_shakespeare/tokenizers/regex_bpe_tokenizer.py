"""
Byte-pair encoding that first splits the text into chunks with the GPT-4 regex, so merges never cross a chunk boundary.
"""

# Note, this is not the same as python's re.
import regex as re
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

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexBPETokenizer(Tokenizer):
    """
    Byte-pair encoding tokenizer that splits the text with GPT4_SPLIT_PATTERN and learns merges within each chunk.
    A pair of tokens is only a merge candidate if they sit next to each other inside the same chunk.
    """

    def __init__(self, target_vocab_size: int):
        """
        Initialized with the target vocab size.
        """
        self.merges: list[Merge] = []

        self._merged_token_counter: int = NUM_BASE_TOKENS

        self._vocab: dict[int, bytes] = build_vocab(self.merges)

        # Encoded chunks by chunk text, since the same chunks (" the", ",") repeat a lot.
        self._chunk_cache: dict[str, list[int]] = {}

        # The vocab size increases by 1 every time we merge, so this puts an upper bound on the number of passes done.
        if target_vocab_size <= NUM_BASE_TOKENS:
            raise Exception(f"cannot set a target vocab size <= the number of distinct tokens in UTF-8 ({NUM_BASE_TOKENS})")
        self._num_merge_passes = target_vocab_size - NUM_BASE_TOKENS

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

        The text is split into chunks, and each pass finds the best merge to do based on the highest pair frequency across all chunks and records it.
        Pairs never span two chunks.

        Inputs:
        - s: the data string to train the tokenizer on
        """
        chunks = [string_to_bytes(chunk) for chunk in re.findall(GPT4_SPLIT_PATTERN, s)]

        for _ in range(self._num_merge_passes):
            try:
                from_tokens = most_common_pair(chunks)
                merge = Merge(from_tokens, self._get_next_merged_token_counter())
                # Record the merge, then apply it so we keep shrinking the chunks.
                self.merges.append(merge)
                chunks = [replace_for_encode(chunk, merge.from_tokens, merge.to_token) for chunk in chunks]
            except IllegalMergeError as e:
                print(f"merges halted: {e}")
                break

        self._vocab = build_vocab(self.merges)
        self._chunk_cache = {}

    @override
    def encode_string(self, s: str) -> list[int]:
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
    def decode_as_string(self, tokens: list[int]) -> str:
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
