from typing import override

from .tokenizer import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Tokenizes by character: the vocab is the sorted set of distinct characters seen in the dataset.
    Roughly a trimmed down version of a byte-level tokenizer, but without all 256 bytes.
    """

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._vocab: dict[int, bytes] = {}

    @override
    def train(self, s: str) -> None:
        self.vocab = sorted(set(s))
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
