from typing import Any, override

from .tokenizer import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Tokenizes by character: the vocab is the sorted set of distinct characters seen in the dataset.
    Roughly a trimmed down version of a byte-level tokenizer, but without all 256 bytes.
    """

    name = "char"

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._vocab: dict[int, bytes] = {}

    @override
    def train(self, s: str) -> None:
        self._set_vocab(sorted(set(s)))

    def _set_vocab(self, chars: list[str]) -> None:
        self.vocab = chars
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self._vocab = {i: tok.encode("utf-8") for i, tok in self.id_to_token.items()}

    @override
    def encode_string(self, s: str) -> list[int]:
        return [self.token_to_id[tok] for tok in s]

    @override
    def decode_as_string(self, tokens: list[int]) -> str:
        return "".join(self.id_to_token[id] for id in tokens)

    @override
    def get_vocab(self) -> dict[int, bytes]:
        return self._vocab

    @override
    def state(self) -> dict[str, Any]:
        return {"vocab": self.vocab}

    @classmethod
    @override
    def from_state(cls, state: dict[str, Any]) -> "CharTokenizer":
        tokenizer = cls()
        tokenizer._set_vocab(state["vocab"])
        return tokenizer
