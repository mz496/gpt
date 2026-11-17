import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

# Bump when the saved layout changes, so old files fail loudly instead of loading wrong.
SAVE_FORMAT = 1


class Tokenizer(ABC):
    """
    Interface for turning text into integer tokens and back.
    """

    # Identifies the kind of tokenizer in saved files, and matches the names make_tokenizer takes.
    name: ClassVar[str]

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

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Everything learned in train(), as JSON-serializable values, so that from_state can rebuild an identical tokenizer."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        """The inverse of state(): a trained tokenizer, without needing the training text."""

    def save(self, path: Path) -> None:
        """Writes the trained tokenizer to a JSON file; load it back with load_tokenizer (in tokenizers/utils.py)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": SAVE_FORMAT, "name": self.name, **self.state()}))
