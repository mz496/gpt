from .bpe_tokenizer import BPETokenizer
from .char_tokenizer import CharTokenizer
from .regex_bpe_tokenizer import RegexBPETokenizer
from .tokenizer import Tokenizer

TOKENIZER_NAMES = ["char", "bpe", "regex_bpe"]

def make_tokenizer(name: str, bpe_vocab_size: int) -> Tokenizer:
    if name == "char":
        return CharTokenizer()
    if name == "bpe":
        return BPETokenizer(bpe_vocab_size)
    if name == "regex_bpe":
        return RegexBPETokenizer(bpe_vocab_size)
    raise ValueError(f"unknown tokenizer: {name}")
