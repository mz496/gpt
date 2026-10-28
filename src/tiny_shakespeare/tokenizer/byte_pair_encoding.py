# From https://www.reedbeta.com/blog/programmers-intro-to-unicode/
# From Karpathy's "Let's build the GPT Tokenizer", ~27:46
from pathlib import Path



# 1:40pm

class BytePairEncoding:
    """
    Implements a simple byte-pair encoding tokenizer.
    Does passes over the input, merging the most commonly seen byte pairs.
    """

    def train(self):
        # TODO: Not training the tokenizer yet.
        pass

    def encode(self, plain: str):
        """
        Encode the string as a sequence of tokens.
        Writes state: the merges used during the byte-pair encoding process.

        Inputs:
        - plain: the string

        Outputs:
        - tokens: the int array of tokens
        - vocab_size
        """
        tokens = plain.encode("utf-8")
        tokens = list(map(int, tokens))
        pass

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