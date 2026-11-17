from pathlib import Path

from src.tiny_shakespeare.data.tiny_shakespeare_source import TinyShakespeareSource

RAW = (Path(__file__).parent.parent.parent.parent / "src" / "tiny_shakespeare" / "data" / "tiny_shakespeare.txt").read_text()
SPLIT = int(len(RAW) * 0.9)


def test_train_is_the_first_90_percent_of_the_file() -> None:
    assert "".join(TinyShakespeareSource().iter_text("train")) == RAW[:SPLIT]


def test_validation_is_the_rest_of_the_file_minus_one_char() -> None:
    assert "".join(TinyShakespeareSource().iter_text("validation")) == RAW[SPLIT + 1:]


def test_each_split_is_a_single_chunk() -> None:
    source = TinyShakespeareSource()
    assert len(list(source.iter_text("train"))) == len(list(source.iter_text("validation"))) == 1


def test_tokenizer_text_is_the_train_split_and_never_the_validation_text() -> None:
    text = TinyShakespeareSource().get_tokenizer_text()
    assert text == RAW[:SPLIT]
    assert RAW[SPLIT + 1:] not in text


def test_name() -> None:
    assert TinyShakespeareSource().name == "tiny-shakespeare"
