from collections.abc import Callable
from pathlib import Path

import pytest

from src.tiny_shakespeare import config as config_module
from src.tiny_shakespeare.config import CONFIG_NAMES, Config, FlagConfig, ModelConfig, TrainConfig, load_config, load_train_config

VALID = """
train:
  train_steps: 10
  train_seed: 1
  batch_size: 4
  learning_rate: 1.0e-3
  estimate_loss_seed: 2
  estimate_loss_iters: 5
  estimate_loss_chunk_size: 5
model:
  n_embd: 8
  n_head: 2
  n_layer: 1
  block_size: 16
  dropout: 0.1
flags:
  flag_use_parallel_heads: true
  flag_use_pytorch_attention_impl: false
"""


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], str]:
    """Makes load_config read from a temp dir, and returns a function that writes configs/<name>.yaml there."""
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    def write(text: str, name: str = "tmp") -> str:
        (tmp_path / f"{name}.yaml").write_text(text)
        return name

    return write


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_every_shipped_config_has_a_loadable_train_section(name: str) -> None:
    assert isinstance(load_train_config(name), TrainConfig)


# sft.yaml only has a train section, since a finetuned model comes from its checkpoint.
@pytest.mark.parametrize("name", ["default", "scaleup"])
def test_every_full_config_loads_into_its_three_sections(name: str) -> None:
    config = load_config(name)
    assert isinstance(config, Config)
    assert isinstance(config.train, TrainConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.flags, FlagConfig)


def test_default_and_scaleup_keep_the_values_they_had_before_the_split() -> None:
    default, scaleup = load_config("default"), load_config("scaleup")
    assert (default.train.batch_size, default.train.learning_rate, default.model.n_embd, default.model.block_size) == (32, 1.0e-3, 32, 8)
    assert (scaleup.train.batch_size, scaleup.train.learning_rate, scaleup.model.n_embd, scaleup.model.block_size) == (64, 3.0e-4, 384, 256)
    assert default.flags.flag_use_parallel_heads and default.flags.flag_use_pytorch_attention_impl
    assert scaleup.flags.flag_use_parallel_heads and scaleup.flags.flag_use_pytorch_attention_impl


def test_load_config_puts_each_key_in_its_section(write_config: Callable[[str], str]) -> None:
    config = load_config(write_config(VALID))
    assert config.train == TrainConfig(train_steps=10, train_seed=1, estimate_loss_seed=2, batch_size=4, learning_rate=1.0e-3, estimate_loss_iters=5, estimate_loss_chunk_size=5)
    assert config.model == ModelConfig(n_embd=8, n_head=2, n_layer=1, block_size=16, dropout=0.1)
    assert config.flags == FlagConfig(flag_use_parallel_heads=True, flag_use_pytorch_attention_impl=False)


def test_load_train_config_reads_only_the_train_section(write_config: Callable[[str], str]) -> None:
    train_only = VALID.split("model:")[0]
    assert load_train_config(write_config(train_only)) == load_config(write_config(VALID)).train
    with pytest.raises(TypeError, match="no train section"):
        load_train_config(write_config("model:\n  n_embd: 8\n"))


def test_a_missing_or_extra_section_raises(write_config: Callable[[str], str]) -> None:
    without_flags = VALID.split("flags:")[0]
    with pytest.raises(TypeError, match="sections"):
        load_config(write_config(without_flags))
    with pytest.raises(TypeError, match="sections"):
        load_config(write_config(VALID + "extra:\n  x: 1\n"))


def test_a_key_in_the_wrong_section_or_missing_raises(write_config: Callable[[str], str]) -> None:
    # block_size belongs to model, not train.
    with pytest.raises(TypeError):
        load_config(write_config(VALID.replace("  estimate_loss_chunk_size: 5\n", "  estimate_loss_chunk_size: 5\n  block_size: 16\n")))
    with pytest.raises(TypeError):
        load_config(write_config(VALID.replace("  n_layer: 1\n", "")))


def train_config(estimate_loss_chunk_size: int) -> TrainConfig:
    return TrainConfig(train_steps=1, train_seed=1, estimate_loss_seed=1, batch_size=1, learning_rate=1.0, estimate_loss_iters=5, estimate_loss_chunk_size=estimate_loss_chunk_size)


def test_train_config_rejects_a_chunk_size_that_makes_no_sense() -> None:
    with pytest.raises(ValueError, match="estimate_loss_chunk_size"):
        train_config(0)
    with pytest.raises(ValueError, match="estimate_loss_chunk_size"):
        train_config(6)
    train_config(5)
