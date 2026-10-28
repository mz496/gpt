from .tokenizer import Tokenizer


class CharEncoding(Tokenizer):
    """
    Tokenizes by character: the vocab is the sorted set of distinct characters seen in train.
    """

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def train(self, s: str) -> None:
        self.vocab = sorted(set(s))
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

    def encode_string(self, s: str) -> list[int]:
        return [self.token_to_id[tok] for tok in s]

    def decode_as_string(self, tokens: list[int]) -> str:
        return "".join(self.id_to_token[id] for id in tokens)

    def get_vocab_size(self) -> int:
        return len(self.vocab)
