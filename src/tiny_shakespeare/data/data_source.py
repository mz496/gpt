from abc import ABC, abstractmethod
from typing import ClassVar, Iterable, Iterator, Literal

Split = Literal["train", "validation"]
# Cap on the text the tokenizer trains on; BPE training is pure Python, so it needs a sample of a big dataset, not all of it.
TOKENIZER_TEXT_MAX_CHARS = 100_000

class DataSource(ABC):
    """
    A text dataset with a train and a validation split, read as a stream of chunks so that big datasets never have to be held in memory as one string.
    """

    name: str

    # Strings that appear literally in the text and mark something rather than being content, e.g. "<|endoftext|>" between documents.
    # Callers register these with the tokenizer (Tokenizer.register_special_tokens) before training it. Don't mutate.
    special_tokens: ClassVar[list[str]] = []

    @abstractmethod
    def iter_text(self, split: Split) -> Iterator[str]:
        """
        The text of one split as consecutive chunks of raw text. Concatenating the chunks gives the whole split.
        The chunks contain the special tokens as they appear in the data.

        A chunk may end in the middle of a document, but never in the middle of a special token.
        """

    def get_tokenizer_text(self, max_chars: int = TOKENIZER_TEXT_MAX_CHARS) -> str:
        """
        The text to train the tokenizer on: the start of the train split, capped at (about) max_chars.
        Only the train split, so the tokenizer learns nothing from the validation text.
        """
        return _take_chars(self.iter_text("train"), max_chars)

def _take_chars(chunks: Iterable[str], max_chars: int) -> str:
    """
    Concatenates chunks until at least max_chars chars have been collected; the chunk that crosses the cap is kept whole.
    """
    if max_chars <= 0:
        return ""
    taken: list[str] = []
    total = 0
    for chunk in chunks:
        taken.append(chunk)
        total += len(chunk)
        # Check here rather than at the top of the loop, so we never pull a chunk we won't use.
        if total >= max_chars:
            break
    return "".join(taken)
