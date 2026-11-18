import pytest

from scripts.compare_tokenizers import build_specs, training_command
from scripts.compare_tokenizers_scaleup import CONFIG, DEFAULT_NAME, parse_args
from src.tiny_shakespeare.config import CONFIG_NAMES


def test_config_exists():
    assert CONFIG in CONFIG_NAMES


def test_the_arguments_are_compare_tokenizers_with_a_different_default_view_name():
    args = parse_args([])
    assert args.name == DEFAULT_NAME == "tokenizers-scaleup"
    assert args.sizes == [300, 512, 1024, 2048]
    assert args.tokenizers == ["char", "bpe", "regex_bpe"]
    assert args.max_parallel == 3
    assert args.modal is False
    assert args.train_args == []


def test_the_view_name_can_still_be_chosen():
    assert parse_args(["--name", "x"]).name == "x"


def test_the_runs_are_all_the_tokenizers_at_all_sizes_trained_on_the_scaleup_config():
    args = parse_args(["--train-args", "--source", "tiny-stories"])
    specs = build_specs(args.sizes, args.train_args, args.tokenizers)

    assert [spec.label for spec in specs] == ["char"] + [f"{tokenizer}_{size}" for size in args.sizes for tokenizer in ["bpe", "regex_bpe"]]
    assert len(specs) == 9
    for spec in specs:
        command = training_command(spec, "g", CONFIG)
        assert command[command.index("--config") + 1] == "scaleup"
        assert command[-2:] == ["--source", "tiny-stories"]


def test_char_and_every_regex_bpe_is_five_runs():
    args = parse_args(["--tokenizers", "char", "regex_bpe", "--max-parallel", "5", "--train-args", "--source", "tiny-stories"])
    specs = build_specs(args.sizes, args.train_args, args.tokenizers)

    assert args.max_parallel == 5
    assert [spec.label for spec in specs] == ["char", "regex_bpe_300", "regex_bpe_512", "regex_bpe_1024", "regex_bpe_2048"]
    assert all(spec.extra_args == ["--source", "tiny-stories"] for spec in specs)


def test_parse_args_rejects_the_flags_the_script_sets_per_run(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--modal", "--train-args", "--tokenizer", "bpe"])

    assert exit_info.value.code == 2
    assert "--tokenizer is set per run by this script" in capsys.readouterr().err


def test_parse_args_accepts_the_tinystories_flags():
    assert parse_args(["--train-args", "--source", "tiny-stories", "--tokenizer-text-only"]).train_args == ["--source", "tiny-stories", "--tokenizer-text-only"]
