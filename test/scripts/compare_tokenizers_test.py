import pytest

from scripts.compare_tokenizers import BLUE_300_TO_600, LIME_600, TEAL_300_TO_600, build_specs, parse_wandb_run_id, shade_for


def test_shade_for_spans_the_palette():
    assert [shade_for(i, 4, TEAL_300_TO_600) for i in range(4)] == TEAL_300_TO_600
    assert shade_for(0, 1, TEAL_300_TO_600) == TEAL_300_TO_600[0]
    assert shade_for(0, 2, TEAL_300_TO_600) == TEAL_300_TO_600[0]
    assert shade_for(1, 2, TEAL_300_TO_600) == TEAL_300_TO_600[-1]


def test_build_specs_default_sizes():
    specs = build_specs([300, 512, 1024, 2048])
    assert len(specs) == 9
    assert specs[0].tokenizer == "char" and specs[0].bpe_vocab_size is None and specs[0].color == LIME_600
    by_label = {s.label: s for s in specs}
    assert by_label["bpe_300"].color == TEAL_300_TO_600[0]
    assert by_label["bpe_2048"].color == TEAL_300_TO_600[-1]
    assert by_label["regex_bpe_1024"].color == BLUE_300_TO_600[2]
    assert by_label["regex_bpe_512"].display_name == "tokenizer=regex_bpe | bpe_vocab_size=512"
    assert len({s.label for s in specs}) == len(specs)


def test_parse_wandb_run_id():
    log = "step=0000 loss=...\nsaved checkpoint to x\nrun_id=2026-10-29T12:34:45_8u06ouw5\n"
    assert parse_wandb_run_id(log) == "8u06ouw5"


def test_parse_wandb_run_id_missing():
    with pytest.raises(ValueError):
        parse_wandb_run_id("no id here")
