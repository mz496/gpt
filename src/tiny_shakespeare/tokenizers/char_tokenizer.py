from .tokenizer import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Tokenizes by character: the vocab is the sorted set of distinct characters seen in train.
    """

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._vocab: dict[int, bytes] = {}

    def train(self, s: str) -> None:
        self.vocab = sorted(set(s))
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self._vocab = {i: tok.encode("utf-8") for i, tok in self.id_to_token.items()}

    def encode_string(self, s: str) -> list[int]:
        return [self.token_to_id[tok] for tok in s]

    def decode_as_string(self, tokens: list[int]) -> str:
        return "".join(self.id_to_token[id] for id in tokens)

    def get_vocab(self) -> dict[int, bytes]:
        return self._vocab
