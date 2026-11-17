from pathlib import Path

import numpy as np
import pytest

from src.tiny_shakespeare.data import tiny_stories_source as module
from src.tiny_shakespeare.data.prepare import prepare
from src.tiny_shakespeare.data.tiny_stories_source import END_OF_TEXT, TinyStoriesSource
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer

TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"
# Stories of different lengths, with non-ASCII characters, and a "\n" after each END_OF_TEXT as in the real files.
STORIES = [
    "Once upon a time, there was a little cat named Tom.",
    "Sam had a red ball. He said “hello” to his friend â€™s dog.\nThe dog barked.",
    "A.",
    "One day, a girl named Lily went to the park with her mom and her big brother, and they played all day long \U0001f600.",
    "The end.",
]
TRAIN_TEXT = "".join(story + END_OF_TEXT + "\n" for story in STORIES)
VALID_TEXT = "".join(story + END_OF_TEXT + "\n" for story in STORIES[:2])


class FakeDownload:
    """Stands in for hf_hub_download: serves the given files from tmp_path, and records what was asked for."""

    def __init__(self, tmp_path: Path, files: dict[str, str]):
        self.dir = tmp_path
        self.calls: list[tuple[str, str, str]] = []
        for filename, text in files.items():
            (tmp_path / filename).write_text(text, encoding="utf-8", newline="")

    def __call__(self, repo_id: str, filename: str, repo_type: str) -> str:
        self.calls.append((repo_id, filename, repo_type))
        return str(self.dir / filename)


@pytest.fixture
def download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDownload:
    fake = FakeDownload(tmp_path, {TRAIN_FILE: TRAIN_TEXT, VALID_FILE: VALID_TEXT})
    monkeypatch.setattr(module, "hf_hub_download", fake)
    return fake


def chunks(split: str = "train", chunk_chars: int = 100) -> list[str]:
    return list(TinyStoriesSource(chunk_chars).iter_text(split))  # pyright: ignore[reportArgumentType]


def set_file(download: FakeDownload, text: str) -> None:
    (download.dir / TRAIN_FILE).write_text(text, encoding="utf-8", newline="")


# --- what it is

def test_name_and_special_tokens() -> None:
    source = TinyStoriesSource()
    assert source.name == "tiny-stories"
    assert source.special_tokens == ["<|endoftext|>"]


@pytest.mark.parametrize("chunk_chars", [0, -1])
def test_a_chunk_size_that_is_not_positive_raises(chunk_chars: int) -> None:
    with pytest.raises(ValueError, match="chunk_chars"):
        TinyStoriesSource(chunk_chars)


# --- downloading

def test_nothing_is_downloaded_until_the_text_is_read(download: FakeDownload) -> None:
    source = TinyStoriesSource()
    iterator = source.iter_text("train")
    assert download.calls == []

    next(iterator)

    assert download.calls == [("roneneldan/TinyStories", TRAIN_FILE, "dataset")]


def test_each_split_reads_its_own_file(download: FakeDownload) -> None:
    assert "".join(chunks("train")) == TRAIN_TEXT
    assert "".join(chunks("validation")) == VALID_TEXT
    assert [filename for _repo, filename, _type in download.calls] == [TRAIN_FILE, VALID_FILE]


def test_the_tokenizer_text_never_downloads_the_validation_file(download: FakeDownload) -> None:
    TinyStoriesSource(100).get_tokenizer_text(1)
    assert [filename for _repo, filename, _type in download.calls] == [TRAIN_FILE]


# --- the chunks (that they end between stories is how TinyStoriesSource does it, not part of the DataSource contract)

def test_the_chunks_rejoin_to_exactly_the_file(download: FakeDownload) -> None:
    assert "".join(chunks()) == TRAIN_TEXT


def test_every_chunk_but_the_last_ends_right_after_an_end_of_text(download: FakeDownload) -> None:
    result = chunks(chunk_chars=60)
    assert len(result) > 1
    assert all(chunk.endswith(END_OF_TEXT) for chunk in result[:-1])


def test_every_chunk_but_the_last_is_at_least_chunk_chars_and_overshoots_by_at_most_a_story(download: FakeDownload) -> None:
    chunk_chars = 60
    longest_story = max(len(story) for story in STORIES) + len(END_OF_TEXT) + 1
    result = chunks(chunk_chars=chunk_chars)

    assert all(len(chunk) >= chunk_chars for chunk in result[:-1])
    assert all(len(chunk) < chunk_chars + longest_story for chunk in result)


def test_a_chunk_size_bigger_than_the_file_gives_one_chunk(download: FakeDownload) -> None:
    assert chunks(chunk_chars=10**9) == [TRAIN_TEXT]


def test_a_chunk_size_of_one_gives_one_story_per_chunk(download: FakeDownload) -> None:
    result = chunks(chunk_chars=1)
    # The "\n" after an END_OF_TEXT belongs to the start of the next chunk, and the file's final "\n" is the text after the last END_OF_TEXT.
    assert result[0] == STORIES[0] + END_OF_TEXT
    assert result[1] == "\n" + STORIES[1] + END_OF_TEXT
    assert result[-1] == "\n"
    assert len(result) == len(STORIES) + 1


def test_no_end_of_text_is_ever_split_between_chunks(download: FakeDownload) -> None:
    result = chunks(chunk_chars=7)
    assert sum(chunk.count(END_OF_TEXT) for chunk in result) == TRAIN_TEXT.count(END_OF_TEXT) == len(STORIES)


@pytest.mark.parametrize("chars_per_read", range(1, 45))
def test_the_chunks_dont_depend_on_how_much_is_read_at_a_time(chars_per_read: int, download: FakeDownload, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = chunks(chunk_chars=60)

    # Small reads cut the file in the middle of stories, of END_OF_TEXT, and of multi-byte characters.
    monkeypatch.setattr(module, "_CHARS_PER_RAW_READ", chars_per_read)

    assert chunks(chunk_chars=60) == expected


@pytest.mark.parametrize("chars_per_read", range(1, 121))
def test_a_delimiter_split_between_reads_is_found_whole(chars_per_read: int, download: FakeDownload, monkeypatch: pytest.MonkeyPatch) -> None:
    text = "x" * 95 + END_OF_TEXT + "y" * 50 + END_OF_TEXT + "z"  # the first END_OF_TEXT occupies [95, 108)
    set_file(download, text)
    expected = chunks(chunk_chars=100)
    monkeypatch.setattr(module, "_CHARS_PER_RAW_READ", chars_per_read)  # so a read can end before, inside, or after an END_OF_TEXT

    assert chunks(chunk_chars=100) == expected
    assert expected == ["x" * 95 + END_OF_TEXT, "y" * 50 + END_OF_TEXT + "z"]


def test_the_text_comes_out_exactly_as_in_the_file_including_carriage_returns(download: FakeDownload) -> None:
    text = "First story\r\nstill the first." + END_OF_TEXT + "\r\nSecond\rstory" + END_OF_TEXT + "\r\n"
    set_file(download, text)
    assert "".join(chunks(chunk_chars=5)) == text


# --- edge cases for the file

def test_a_file_without_any_end_of_text_is_one_chunk(download: FakeDownload) -> None:
    set_file(download, "no delimiter here, just text. " * 50)
    assert chunks(chunk_chars=10) == ["no delimiter here, just text. " * 50]


def test_text_after_the_last_end_of_text_is_its_own_last_chunk(download: FakeDownload) -> None:
    set_file(download, "one" + END_OF_TEXT + "\nan unfinished sto")
    assert chunks(chunk_chars=1) == ["one" + END_OF_TEXT, "\nan unfinished sto"]


def test_a_file_without_a_trailing_newline_ends_with_the_last_end_of_text(download: FakeDownload) -> None:
    set_file(download, "one" + END_OF_TEXT + "\ntwo" + END_OF_TEXT)
    assert chunks(chunk_chars=1) == ["one" + END_OF_TEXT, "\ntwo" + END_OF_TEXT]


def test_an_empty_file_has_no_chunks(download: FakeDownload) -> None:
    set_file(download, "")
    assert chunks() == []


def test_consecutive_end_of_texts_are_kept(download: FakeDownload) -> None:
    text = "a" + END_OF_TEXT + END_OF_TEXT + "\nb" + END_OF_TEXT
    set_file(download, text)
    result = chunks(chunk_chars=1)
    assert "".join(result) == text
    assert sum(chunk.count(END_OF_TEXT) for chunk in result) == 3


# --- used with the rest of the pipeline

def test_the_tokenizer_text_is_the_start_of_the_train_file_up_to_a_chunk_boundary(download: FakeDownload) -> None:
    text = TinyStoriesSource(100).get_tokenizer_text(max_chars=150)

    assert TRAIN_TEXT.startswith(text)
    assert text.endswith(END_OF_TEXT)
    assert len(text) >= 150


def test_a_tokenizer_with_the_sources_special_tokens_turns_each_end_of_text_into_one_id(download: FakeDownload) -> None:
    source = TinyStoriesSource(chunk_chars=100)
    tokenizer = CharTokenizer()
    tokenizer.register_special_tokens(source.special_tokens)
    tokenizer.train(source.get_tokenizer_text(max_chars=10**9))

    prepared = prepare(source, tokenizer)

    end_of_text = tokenizer.special_token_ids[END_OF_TEXT]
    assert int((prepared.train == end_of_text).sum()) == len(STORIES)
    assert int((prepared.validation == end_of_text).sum()) == 2
    assert tokenizer.decode_as_string(prepared.train.tolist()) == TRAIN_TEXT
    assert prepared.train_chars == len(TRAIN_TEXT)
    assert prepared.train.dtype == np.uint16
