from scripts.compare_tokenizers_scaleup import CONFIG, build_specs
from src.tiny_shakespeare.config import CONFIG_NAMES


def test_config_exists():
    assert CONFIG in CONFIG_NAMES


def test_build_specs_is_char_and_regex_bpe_2048():
    char, regex = build_specs()
    assert (char.tokenizer, char.bpe_vocab_size) == ("char", None)
    assert (regex.tokenizer, regex.bpe_vocab_size) == ("regex_bpe", 2048)
    assert char.label != regex.label
