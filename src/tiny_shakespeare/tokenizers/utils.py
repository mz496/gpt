import json
from pathlib import Path

from .bpe_tokenizer import BPETokenizer
from .char_tokenizer import CharTokenizer
from .regex_bpe_tokenizer import RegexBPETokenizer
from .tokenizer import SAVE_FORMAT, Tokenizer

TOKENIZER_NAMES = ["char", "bpe", "regex_bpe"]

def make_tokenizer(name: str, bpe_vocab_size: int) -> Tokenizer:
    if name == "char":
        return CharTokenizer()
    if name == "bpe":
        return BPETokenizer(bpe_vocab_size)
    if name == "regex_bpe":
        return RegexBPETokenizer(bpe_vocab_size)
    raise ValueError(f"unknown tokenizer: {name}")

def load_tokenizer(path: Path) -> Tokenizer:
    """Reads a tokenizer written by Tokenizer.save."""
    saved = json.loads(path.read_text())
    if saved.get("format") != SAVE_FORMAT:
        raise ValueError(f"unsupported tokenizer file format {saved.get('format')!r} in {path}; expected {SAVE_FORMAT}")
    classes: dict[str, type[Tokenizer]] = {cls.name: cls for cls in (CharTokenizer, BPETokenizer, RegexBPETokenizer)}
    if saved["name"] not in classes:
        raise ValueError(f"unknown tokenizer {saved['name']!r} in {path}; known: {sorted(classes)}")
    return classes[saved["name"]].from_state(saved)
