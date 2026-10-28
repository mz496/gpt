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

    def __str__(self):
        return f"Merge({self.from_tokens} -> {self.to_token})"

class IllegalMergeError(ValueError):
    """Raised when a merge can't be applied to the given tokens."""

class BytePairEncoding:
    """
    Implements a simple byte-pair encoding tokenizer.
    Does passes over the input, merging the most commonly seen byte pairs.
    """

    def __init__(self, plain: str):
        """
        Initialized with the plaintext.
        """

        # Init the merges: empty to start
        # A merge is: int substring -> merged.
        # In decode, we convert this to a mapping.
        self.merges: list[Merge] = []

        # Init the vocab: all distinct tokens
        tokens = plain.encode("utf-8")
        tokens = list(map(int, tokens))
        self.tokens: list[int] = tokens

        # Because we're using UTF-8, our tokens will all be in the range [0, 255].
        # So if we use values to represent merged tokens, they should be outside that range, and all positive, so we can use them as embedding table indices later.
        self.merged_token_counter: int = 256

    def get_next_merged_token_counter(self):
        """
        Returns the next free merged-token-counter and then advances it for the next caller.
        """
        next_counter = self.merged_token_counter
        self.merged_token_counter += 1
        return next_counter

    def train(self):
        # TODO: Not training the tokenizer yet.
        pass

    def encode(self) -> tuple[list[int], int]:
        """
        Encode the string as a sequence of tokens.

        Writes state:
        - the merges used during the byte-pair encoding process
        - the vocab

        Inputs:
        - plain: the string

        Outputs:
        - tokens: the int array of tokens
        - vocab_size: the size of the vocab
        """
        
        # Do N merge passes
        merge_passes = 4
        for i in range(merge_passes):
            print(f"before pass {i}:")
            print(self.tokens)

            try:
                merge, tokens_post_merge = self.merge(self.tokens)
                self.merges.append(merge)
                self.tokens = tokens_post_merge
            except IllegalMergeError as e:
                print(f"merges halted: {e}")
                break

            print(f"after pass {i}:")
            print(f"merge: {merge}")
            print(self.tokens)

        # Return the final tokens and vocab size
        vocab_size = len(set(self.tokens))
        return self.tokens, vocab_size

    def merge(self, tokens: list[int]) -> tuple[Merge, list[int]]:
        """
        One merge pass. Pure.
        aaab -> Zab (but using a token that's outside the vocab)

        Inputs:
        - tokens: input tokens

        Outputs:
        - merge: the Merge that was done
        - tokens: the tokens post-merge
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

        # Do the substitution, going forward
        tokens = self.replace_pair(tokens, from_tokens, to_token)

        merge = Merge(from_tokens, to_token)
        return merge, tokens
        
    def replace_pair(self, tokens: list[int], from_tokens: tuple[int, int], to_token: int) -> list[int]:
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
    

    def decode(self, tokens: list[int], vocab: dict[str, str]):
        """
        Decodes the tokens using the state written during encode.

        Inputs:
        - tokens: the int array of tokens

        Outputs:
        - plain: the string
        """
        pass

    def save(self, path: Path):
        # Saves state to the given path as JSON.
        pass

    @classmethod
    def load(cls, path: Path):
        # Loads state from the given path as JSON.
        pass

    def __str__(self) -> str:
        return f"BytePairEncoding<>"