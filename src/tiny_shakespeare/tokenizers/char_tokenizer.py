from collections.abc import Iterable
from typing import Any, ClassVar, override

from .tokenizer import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Tokenizes by character: the vocab is the sorted set of distinct characters seen in the dataset.
    Roughly a trimmed down version of a byte-level tokenizer, but without all 256 bytes.
    """

    name = "char"
    # The vocab is just the set of characters, which can be collected a chunk at a time.
    trains_on_stream: ClassVar[bool] = True

    def __init__(self):
        super().__init__()
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @override
    def _train_on_pieces(self, pieces: list[str]) -> None:
        self._set_vocab(sorted(set("".join(pieces))))

    @override
    def train_on_chunks(self, chunks: Iterable[str]) -> None:
        """Collects the characters chunk by chunk, so only the set of distinct characters is ever held, never all the text."""
        chars: set[str] = set()
        for chunk in chunks:
            for piece in self._split_on_special_tokens(chunk):
                chars.update(piece)
        self._set_vocab(sorted(chars))
        self._trained = True

    def _set_vocab(self, chars: list[str]) -> None:
        self.vocab = chars
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self._vocab_cache = None

    @override
    def _encode_plain(self, s: str) -> list[int]:
        return [self.token_to_id[tok] for tok in s]

    @override
    def _decode_plain(self, tokens: list[int]) -> str:
        return "".join(self.id_to_token[id] for id in tokens)

    @override
    def _num_learned_tokens(self) -> int:
        return len(self.vocab)

    @override
    def _learned_vocab(self) -> dict[int, bytes]:
        return {i: tok.encode("utf-8") for i, tok in self.id_to_token.items()}

    @override
    def state(self) -> dict[str, Any]:
        return {"vocab": self.vocab}

    @classmethod
    @override
    def from_state(cls, state: dict[str, Any]) -> "CharTokenizer":
        tokenizer = cls()
        tokenizer.register_special_tokens(state["special_tokens"])
        tokenizer._set_vocab(state["vocab"])
        tokenizer._trained = True
        return tokenizer
