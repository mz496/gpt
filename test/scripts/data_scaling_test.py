from scripts.compare_tokenizers import AMBER_600, INDIGO_600, LIME_600
from scripts.data_scaling import DEFAULT_FRACTIONS, build_specs


def test_build_specs_is_every_tokenizer_at_every_fraction():
    specs = build_specs(DEFAULT_FRACTIONS)
    assert len(specs) == 12
    assert len({s.label for s in specs}) == 12
    assert {(s.tokenizer, s.bpe_vocab_size) for s in specs} == {("char", None), ("bpe", 2048), ("regex_bpe", 2048)}


def test_colors_are_per_tokenizer():
    colors = {s.tokenizer: s.color for s in build_specs(DEFAULT_FRACTIONS)}
    assert colors == {"char": AMBER_600, "bpe": LIME_600, "regex_bpe": INDIGO_600}


def test_fraction_is_passed_to_gpt_train_and_named():
    spec = next(s for s in build_specs(DEFAULT_FRACTIONS) if s.label == "regex_bpe_0.125")
    assert spec.extra_args == ["--train-fraction", "0.125"]
    assert spec.display_name == "tokenizer=regex_bpe | train_fraction=0.125"
