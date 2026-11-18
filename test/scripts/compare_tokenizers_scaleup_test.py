import pytest

from scripts.compare_tokenizers import training_command
from scripts.compare_tokenizers_scaleup import CONFIG, build_specs, parse_args
from src.tiny_shakespeare.config import CONFIG_NAMES


def test_config_exists():
    assert CONFIG in CONFIG_NAMES


def test_build_specs_is_char_and_regex_bpe_2048():
    char, regex = build_specs()
    assert (char.tokenizer, char.bpe_vocab_size) == ("char", None)
    assert (regex.tokenizer, regex.bpe_vocab_size) == ("regex_bpe", 2048)
    assert char.label != regex.label


def test_build_specs_has_no_extra_training_args_by_default():
    assert all(spec.extra_args == [] for spec in build_specs())


def test_build_specs_gives_both_runs_the_train_args_without_sharing_the_list():
    char, regex = build_specs(["--source", "tiny-stories", "--tokenizer-text-only"])
    assert char.extra_args == regex.extra_args == ["--source", "tiny-stories", "--tokenizer-text-only"]
    assert char.extra_args is not regex.extra_args


def test_parse_args_train_args_takes_everything_after_it_including_flags():
    args = parse_args(["--modal", "--name", "x", "--train-args", "--source", "tiny-stories", "--tokenizer-text-only"])
    assert args.modal is True
    assert args.name == "x"
    assert args.train_args == ["--source", "tiny-stories", "--tokenizer-text-only"]


def test_parse_args_train_args_defaults_to_none_given():
    assert parse_args([]).train_args == []
    assert parse_args(["--modal"]).train_args == []


def test_the_training_commands_end_with_the_train_args_on_modal():
    char, regex = build_specs(["--source", "tiny-stories", "--tokenizer-text-only"])

    char_command = training_command(char, "group", CONFIG, modal=True)
    regex_command = training_command(regex, "group", CONFIG, modal=True)

    assert char_command[:2] == regex_command[:2] == ["just", "modal-train"]
    assert char_command == ["just", "modal-train", "--tokenizer", "char", "--group", "group", "--config", "scaleup", "--source", "tiny-stories", "--tokenizer-text-only"]
    assert regex_command == [
        "just", "modal-train", "--tokenizer", "regex_bpe", "--group", "group", "--config", "scaleup",
        "--bpe-vocab-size", "2048", "--source", "tiny-stories", "--tokenizer-text-only",
    ]


def test_the_training_commands_run_locally_without_modal():
    (char, _regex) = build_specs(["--source", "tiny-stories"])
    assert training_command(char, "g")[:2] == ["just", "gpt-train"]


def test_parse_args_rejects_the_flags_the_script_sets_per_run(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--modal", "--train-args", "--tokenizer", "bpe"])

    assert exit_info.value.code == 2
    assert "--tokenizer is set per run by this script" in capsys.readouterr().err


def test_parse_args_accepts_the_tinystories_flags():
    assert parse_args(["--train-args", "--source", "tiny-stories", "--tokenizer-text-only"]).train_args == ["--source", "tiny-stories", "--tokenizer-text-only"]

