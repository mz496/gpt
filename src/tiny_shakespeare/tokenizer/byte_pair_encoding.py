"""
Byte-pair encoding implementation

From https://www.reedbeta.com/blog/programmers-intro-to-unicode/
From Karpathy's "Let's build the GPT Tokenizer", ~27:46

"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# 1:40pm

@dataclass
class Merge:
    from_tokens: tuple[int, int]
    to_token: int

    def __repr__(self):
        return f"Merge({self.from_tokens} -> {self.to_token})"

class IllegalMergeError(ValueError):
    """Raised when a merge can't be applied to the given tokens."""

class BytePairEncoding:
    """
    Implements a simple byte-pair encoding tokenizer.
    Does passes over the input, merging the most commonly seen byte pairs.
    """

    def __init__(self, target_vocab_size: int):
        """
        Initialized with the plaintext and the target vocab size.
        """

        # Init the merges: empty to start
        # A merge is: int substring -> merged.
        # In decode, we convert this to a mapping.
        self.merges: list[Merge] = []

        # Init the vocab: all distinct tokens. Due to our UTF-8 encoding, the count of distinct tokens is 2**8 = 256.
        # So if we use values to represent merged tokens, they should start at that range and up, so they won't ever conflict with known chars, and we can use them as embedding table indices later.
        num_distinct_tokens = 256

        self.merged_token_counter: int = num_distinct_tokens

        # The vocab size increases by 1 every time we merge, so this puts an upper bound on the number of passes done.
        if target_vocab_size <= num_distinct_tokens:
            raise Exception(f"cannot set a target vocab size <= the number of distinct tokens in UTF-8 ({num_distinct_tokens})")
        self.num_merge_passes = target_vocab_size - num_distinct_tokens

    def get_next_merged_token_counter(self):
        """
        Returns the next free merged-token-counter and then advances it for the next caller.
        """
        next_counter = self.merged_token_counter
        self.merged_token_counter += 1
        return next_counter

    def get_vocab_size(self):
        """
        The vocab size is exactly the merge counter since it starts at the number of distinct tokens and increases for every merge.
        """
        return self.merged_token_counter

    def train(self, s: str):
        """Learns the merges based on the data frequencies."""
        tokens = self.string_to_bytes(s)

        for _ in range(self.num_merge_passes):
            try:
                merge = self.merge(tokens)
                self.merges.append(merge)
            except IllegalMergeError as e:
                print(f"merges halted: {e}")
                break

    def string_to_bytes(self, plain: str) -> list[int]:
        """plain -> list of 0-255 ints"""
        tokens = plain.encode("utf-8")
        return list(map(int, tokens))

    def bytes_to_string(self, byte_values: list[int]) -> str:
        """list of 0-255 ints -> plain"""
        return bytes(byte_values).decode("utf-8", errors="replace")

    def encode_string(self, s: str) -> list[int]:
        return self.encode(self.string_to_bytes(s))

    def encode(self, tokens: list[int]) -> list[int]:
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
        for merge in self.merges:
            # Do the substitution, going forward
            tokens = self.replace_for_encode(tokens, merge.from_tokens, merge.to_token)
        return tokens

    def merge(self, tokens: list[int]) -> Merge:
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
        # Look for the most common byte-pair across the tokens
        # Note we're slightly off here in case of long runs like aaa - this counts 2 occurrences of aa, but we only can compress it once.
        # Align every token with its next token, dropping the last.
        counter = Counter(zip(tokens, tokens[1:]))
        counter_most_common: tuple[tuple[int, int], int] = counter.most_common(1)[0]
        # The tuple[int, int] pair.
        from_tokens, from_tokens_freq = counter_most_common
        if from_tokens_freq == 1:
            raise IllegalMergeError(f"no more merges possible, most common pair only occurs once: {counter_most_common}")
        to_token = self.get_next_merged_token_counter()

        return Merge(from_tokens, to_token)
        
    def replace_for_encode(self, tokens: list[int], from_tokens: tuple[int, int], to_token: int) -> list[int]:
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
    
    def decode_as_string(self, tokens: list[int]) -> str:
        return self.bytes_to_string(self.decode(tokens))

    def decode(self, tokens: list[int]) -> list[int]:
        """
        Decodes the tokens using the state written during encode.

        Inputs:
        - tokens: the int array of tokens

        Outputs:
        - the int array of bytes (0-255)
        """
        if len(self.merges) == 0:
            raise Exception("BPE has not been trained yet; call train() first")
        for merge in reversed(self.merges):
            tokens = self.replace_for_decode(tokens, merge.to_token, merge.from_tokens)
        return tokens

    def replace_for_decode(self, tokens: list[int], to_token: int, from_tokens: tuple[int, int]) -> list[int]:
        out: list[int] = []
        for t in tokens:
            if t == to_token:
                out.extend(from_tokens)
            else:
                out.append(t)
        return out


    def save(self, path: Path):
        # Saves state to the given path as JSON.
        pass

    @classmethod
    def load(cls, path: Path):
        # Loads state from the given path as JSON.
        pass

    def __str__(self) -> str:
        return f"BytePairEncoding<merges={self.merges}>"