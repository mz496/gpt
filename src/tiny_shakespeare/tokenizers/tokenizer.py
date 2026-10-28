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
    def get_vocab_size(self) -> int:
        """Number of distinct tokens."""
