from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Interface for turning text into integer tokens and back.
    """

    @abstractmethod
    def train(self, s: str) -> None:
        """Learn the vocab from the given text."""

    @abstractmethod
    def encode_string(self, s: str) -> list[int]:
        """Text -> tokens."""

    @abstractmethod
    def decode_as_string(self, tokens: list[int]) -> str:
        """Tokens -> text."""

    @abstractmethod
    def get_vocab(self) -> dict[int, bytes]:
        """
        Maps every token id to the UTF-8 bytes it decodes to. Its length is the vocab size.

        Callers (e.g. training loops) may call this often, so implementations must make it inexpensive:
        build the vocab once in train() and return the cached result here. Callers must not mutate it.
        """
