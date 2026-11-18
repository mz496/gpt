import pytest

from scripts.compare_tokenizers import (
    AMBER_600,
    INDIGO_300_TO_600,
    LIME_300_TO_600,
    build_specs,
    parse_args,
    parse_wandb_run_id,
    shade_for,
    training_command,
)


def test_shade_for_spans_the_palette():
    assert [shade_for(i, 4, LIME_300_TO_600) for i in range(4)] == LIME_300_TO_600
    assert shade_for(0, 1, LIME_300_TO_600) == LIME_300_TO_600[0]
    assert shade_for(0, 2, LIME_300_TO_600) == LIME_300_TO_600[0]
    assert shade_for(1, 2, LIME_300_TO_600) == LIME_300_TO_600[-1]


def test_build_specs_default_sizes():
    specs = build_specs([300, 512, 1024, 2048])
    assert len(specs) == 9
    assert specs[0].tokenizer == "char" and specs[0].bpe_vocab_size is None and specs[0].color == AMBER_600
    by_label = {s.label: s for s in specs}
    assert by_label["bpe_300"].color == LIME_300_TO_600[0]
    assert by_label["bpe_2048"].color == LIME_300_TO_600[-1]
    assert by_label["regex_bpe_1024"].color == INDIGO_300_TO_600[2]
    assert by_label["regex_bpe_512"].display_name == "tokenizer=regex_bpe | bpe_vocab_size=512"
    assert len({s.label for s in specs}) == len(specs)


def test_parse_wandb_run_id():
    log = "step=0000 loss=...\nsaved checkpoint to x\nrun_id=2026-10-29T12:34:45_8u06ouw5\n"
    assert parse_wandb_run_id(log) == "8u06ouw5"


def test_parse_wandb_run_id_missing():
    with pytest.raises(ValueError):
        parse_wandb_run_id("no id here")


def test_build_specs_has_no_extra_training_args_by_default():
    assert all(spec.extra_args == [] for spec in build_specs([300, 512]))


def test_build_specs_gives_every_run_the_train_args_without_sharing_the_list():
    specs = build_specs([300, 512], ["--source", "tiny-stories"])
    assert len(specs) == 5
    assert all(spec.extra_args == ["--source", "tiny-stories"] for spec in specs)
    assert len({id(spec.extra_args) for spec in specs}) == len(specs)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.modal is False
    assert args.train_args == []
    assert args.sizes == [300, 512, 1024, 2048]
    assert args.max_parallel == 3


def test_parse_args_train_args_takes_everything_after_it_including_flags():
    args = parse_args(["--modal", "--sizes", "512", "2048", "--max-parallel", "2", "--train-args", "--source", "tiny-stories", "--tokenizer-text-only"])
    assert args.modal is True
    assert args.sizes == [512, 2048]
    assert args.max_parallel == 2
    assert args.train_args == ["--source", "tiny-stories", "--tokenizer-text-only"]


@pytest.mark.parametrize("train_args", [
    ["--tokenizer", "bpe"],
    ["--tokenizer=bpe"],
    ["--bpe-vocab-size", "512"],
    ["--bpe", "512"],  # an abbreviation that argparse would accept for --bpe-vocab-size
    ["--group", "x"],
    ["--config", "scaleup"],
    ["--c", "scaleup"],  # an abbreviation of --config
    ["--source", "tiny-stories", "--tokenizer", "bpe"],
])
def test_parse_args_rejects_the_flags_the_script_sets_per_run(train_args: list[str], capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--train-args", *train_args])

    assert exit_info.value.code == 2
    assert "is set per run by this script" in capsys.readouterr().err


@pytest.mark.parametrize("train_args", [
    ["--tokenizer-text-only"],
    ["--source", "tiny-stories"],
    ["--precision", "fp32", "--profile"],
    ["--suffix", "x"],
])
def test_parse_args_accepts_other_training_flags(train_args: list[str]):
    assert parse_args(["--train-args", *train_args]).train_args == train_args


def test_the_training_command_has_the_extra_args_last_and_works_locally_or_on_modal():
    char, regex_bpe = build_specs([2048], ["--source", "tiny-stories"])[0], build_specs([2048], ["--source", "tiny-stories"])[2]

    assert training_command(char, "g") == ["just", "gpt-train", "--tokenizer", "char", "--group", "g", "--source", "tiny-stories"]
    assert training_command(regex_bpe, "g", modal=True) == [
        "just", "modal-train", "--tokenizer", "regex_bpe", "--group", "g", "--bpe-vocab-size", "2048", "--source", "tiny-stories",
    ]


def test_build_specs_defaults_to_every_tokenizer():
    assert build_specs([300, 512], tokenizers=None) == build_specs([300, 512], tokenizers=["char", "bpe", "regex_bpe"])


def test_build_specs_with_char_and_regex_bpe_leaves_out_bpe():
    specs = build_specs([300, 512, 1024, 2048], tokenizers=["char", "regex_bpe"])
    assert [spec.label for spec in specs] == ["char", "regex_bpe_300", "regex_bpe_512", "regex_bpe_1024", "regex_bpe_2048"]


def test_build_specs_with_only_regex_bpe_has_no_char():
    assert [spec.label for spec in build_specs([512], tokenizers=["regex_bpe"])] == ["regex_bpe_512"]


def test_the_colors_do_not_depend_on_which_tokenizers_are_picked():
    everything = {spec.label: (spec.color, spec.display_name) for spec in build_specs([300, 512, 1024, 2048])}
    subset = {spec.label: (spec.color, spec.display_name) for spec in build_specs([300, 512, 1024, 2048], tokenizers=["char", "regex_bpe"])}

    assert subset and all(everything[label] == value for label, value in subset.items())
    assert subset["char"][0] == AMBER_600
    assert subset["regex_bpe_2048"][0] == INDIGO_300_TO_600[-1]


def test_parse_args_tokenizers_defaults_to_all_and_takes_several():
    assert parse_args([]).tokenizers == ["char", "bpe", "regex_bpe"]
    assert parse_args(["--tokenizers", "char", "regex_bpe", "--max-parallel", "5"]).tokenizers == ["char", "regex_bpe"]


def test_parse_args_rejects_an_unknown_tokenizer(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--tokenizers", "wordpiece"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err

