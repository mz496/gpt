from .byte_pair_encoding import BytePairEncoding
from .char_encoding import CharEncoding
from .tokenizer import Tokenizer

TOKENIZER_NAMES = ["char", "bpe"]

def make_tokenizer(name: str, bpe_vocab_size: int) -> Tokenizer:
    if name == "char":
        return CharEncoding()
    if name == "bpe":
        return BytePairEncoding(bpe_vocab_size)
    raise ValueError(f"unknown tokenizer: {name}")
