# From https://www.reedbeta.com/blog/programmers-intro-to-unicode/
# From Karpathy's "Let's build the GPT Tokenizer", ~27:46
from dataclasses import dataclass
from pathlib import Path


# 1:40pm

@dataclass
class Merge:
    from_tokens: list[int]
    to_token: int

    def __str__(self):
        return f"Merge({self.from_tokens} -> {self.to_token})"


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
        # So if we use values to represent merged tokens, they should be outside that range.
        # We use negatives to make it easy to see while keeping them small.
        self.merged_token_counter: int = -1

    def get_next_merged_token_counter(self):
        """
        Returns the next free merged-token-counter and then advances it for the next caller.
        """
        next_counter = self.merged_token_counter
        self.merged_token_counter -= 1
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
        merge_passes = 5
        for i in range(merge_passes):
            print(f"before pass {i}:")
            print(self.tokens)
            merge, self.tokens = self.merge(self.tokens)
            # Add to merges
            print(f"after pass {i}:")
            print(f"merge: {merge}")
            print(self.tokens)
            self.merges.append(merge)
            break

        # Return the final tokens and vocab size
        vocab_size = len(set(self.tokens))
        return self.tokens, vocab_size

    def merge(self, tokens: list[int]) -> tuple[Merge, list[int]]:
        """
        One merge pass.
        aaab -> Zab (but using a token that's outside the vocab)

        Inputs:
        - tokens: input tokens

        Outputs:
        - merge: the Merge that was done
        - tokens: the tokens post-merge
        """
        # Look for the most common byte-pair across the tokens

        # Get the next merged token counter and assign it: <byte pair> -> merged token counter

        # Do the substitution

        # Return the post-merge tokens
        # Placeholder
        merge = Merge([1,2], -1)
        return merge, tokens
        

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