from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from src.tiny_shakespeare.data.special_tokens import IM_START

from ..data_source import Split

@dataclass(frozen=True)
class SftExample:
    """One supervised example. Does not assume anything about text format."""

    input: str
    output: str

    def to_prompt_string(self):
        """Prompt formatted as ChatML"""
        return f"{IM_START}user\n{self.input}\n"
    def to_response_string(self):
        """Response formatted as ChatML"""
        return f"{IM_START}assistant\n{self.input}\n"

class SFTDataSource(ABC):
    """
    A supervised finetuning dataset with a train and a validation split, read as a stream of examples.
    """

    @abstractmethod
    def iter_examples(self, split: Split) -> Iterator[SftExample]:
        """The examples of one split, one SftExample each."""
