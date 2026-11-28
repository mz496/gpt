from pathlib import Path

import pytest

from src.tiny_shakespeare.data.sft import reverse_source
from src.tiny_shakespeare.data.sft.reverse_gen import CATEGORIES, PER_SLICE
from src.tiny_shakespeare.data.sft.reverse_source import SFT_DIR, SLICES_DIRNAME, TRAIN_FILENAME, ReverseSource
from src.tiny_shakespeare.data.sft.sft_data_source import SftExample


@pytest.fixture
def sft_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the source at an empty temp dir, in place of the real data, and returns it."""
    monkeypatch.setattr(reverse_source, "SFT_DIR", tmp_path)
    return tmp_path


def write_slice(sft_dir: Path, name: str, text: str) -> None:
    (sft_dir / SLICES_DIRNAME).mkdir(exist_ok=True)
    (sft_dir / SLICES_DIRNAME / f"{name}.txt").write_text(text)


# --- the committed data

def test_the_train_split_is_the_train_file() -> None:
    on_disk = [tuple(line.split("\t")) for line in (SFT_DIR / TRAIN_FILENAME).read_text().splitlines()]
    assert [(e.input, e.output) for e in ReverseSource().iter_examples("train")] == on_disk


def test_the_validation_slices_are_the_slice_files_by_name() -> None:
    slices = ReverseSource().get_validation_slices()
    assert list(slices) == sorted(CATEGORIES)
    for name, examples in slices.items():
        on_disk = [tuple(line.split("\t")) for line in (SFT_DIR / SLICES_DIRNAME / f"{name}.txt").read_text().splitlines()]
        assert [(e.input, e.output) for e in examples] == on_disk
        assert len(examples) == PER_SLICE


def test_the_validation_split_is_the_pool_of_every_slice() -> None:
    source = ReverseSource()
    pooled = list(source.iter_examples("validation"))
    by_slice = [example for examples in source.get_validation_slices().values() for example in examples]
    assert pooled == by_slice
    assert len(pooled) == PER_SLICE * len(CATEGORIES)
    # An even mix of the subtasks, which is not the mix the model trains on.
    assert not {e.input for e in pooled} & {e.input for e in source.iter_examples("train")}


# --- the loader's rules, on small files

def test_every_file_in_the_slices_directory_is_a_slice_named_after_it(sft_dir: Path) -> None:
    write_slice(sft_dir, "tricky", "ab\tba\nxyzzy\tyzzyx\n")
    write_slice(sft_dir, "easy", "a\ta\n")
    (sft_dir / SLICES_DIRNAME / "notes.md").write_text("not a slice")  # only .txt files count

    slices = ReverseSource().get_validation_slices()

    assert list(slices) == ["easy", "tricky"]
    assert slices["tricky"] == [SftExample(input="ab", output="ba"), SftExample(input="xyzzy", output="yzzyx")]
    # A slice of your own joins the validation pool too.
    assert [e.input for e in ReverseSource().iter_examples("validation")] == ["a", "ab", "xyzzy"]


def test_a_malformed_line_or_a_wrong_reversal_raises_naming_the_file_and_line(sft_dir: Path) -> None:
    write_slice(sft_dir, "bad_format", "ab\tba\nabc\n")
    with pytest.raises(ValueError, match=r"bad_format\.txt:2: expected"):
        ReverseSource().get_validation_slices()

    (sft_dir / SLICES_DIRNAME / "bad_format.txt").unlink()
    write_slice(sft_dir, "bad_reverse", "ab\tba\nabc\tabc\n")
    with pytest.raises(ValueError, match=r"bad_reverse\.txt:2: 'abc' is not the reverse of 'abc'"):
        ReverseSource().get_validation_slices()


def test_a_missing_directory_an_empty_directory_and_an_empty_slice_all_raise(sft_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reverse_gen"):
        ReverseSource().get_validation_slices()

    (sft_dir / SLICES_DIRNAME).mkdir()
    with pytest.raises(ValueError, match="no slice files"):
        ReverseSource().get_validation_slices()

    write_slice(sft_dir, "empty", "")
    with pytest.raises(ValueError, match=r"empty\.txt has no examples"):
        ReverseSource().get_validation_slices()


def test_a_malformed_train_line_raises_naming_the_file_and_line(sft_dir: Path) -> None:
    (sft_dir / TRAIN_FILENAME).write_text("ab\tba\nabc\tcab\n")
    with pytest.raises(ValueError, match=r"reverse-train\.txt:2: 'cab' is not the reverse of 'abc'"):
        list(ReverseSource().iter_examples("train"))
