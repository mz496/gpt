from pathlib import Path
from typing import Iterator, override

from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType] Partially unknown overload types.

from .data_source import DataSource, Split

REPO_ID = "roneneldan/TinyStories"
FILENAMES: dict[Split, str] = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "validation": "TinyStoriesV2-GPT4-valid.txt",
}
# The files hold one story after another, separated by this.
END_OF_TEXT = "<|endoftext|>"
DEFAULT_CHUNK_CHARS = 10_000
# How many characters we pull from the file per read.
_CHARS_PER_RAW_READ = 1 << 24

class TinyStoriesSource(DataSource):
    """
    TinyStories V2 (GPT-4 stories, ~2-3GB of train text): https://huggingface.co/datasets/roneneldan/TinyStories
    The train and validation splits are separate files, downloaded (and cached by huggingface_hub) the first time they're read.

    The stories are separated by "<|endoftext|>", which is a special token: the text keeps it, and tokenizers register it.
    """

    name = "tiny-stories"
    special_tokens = [END_OF_TEXT]

    def __init__(self, chunk_chars: int = DEFAULT_CHUNK_CHARS):
        if chunk_chars <= 0:
            raise ValueError(f"chunk_chars must be positive, got {chunk_chars}")
        self.chunk_chars = chunk_chars

    @override
    def iter_text(self, split: Split) -> Iterator[str]:
        """
        The raw text of the split in chunks, exactly as in the file.

        Implementation detail, not part of the DataSource contract: each chunk is at least chunk_chars chars (except the last) and ends right after an "<|endoftext|>",
        so chunks hold whole stories, and run past chunk_chars by at most the rest of one story. The text after the last "<|endoftext|>" is the last chunk.

        The file is read _CHARS_PER_RAW_READ chars at a time, so only about that much text is in memory (more if there is a long stretch without an "<|endoftext|>").
        A file without any "<|endoftext|>" comes out as one chunk.
        """
        path = Path(hf_hub_download(REPO_ID, FILENAMES[split], repo_type="dataset"))
        # Everything read but not yielded yet. It always starts at a chunk boundary.
        buffer = ""
        # newline="" so that the text is exactly what's in the file, with no translation of "\r\n".
        with open(path, encoding="utf-8", newline="") as f:
            while text := f.read(_CHARS_PER_RAW_READ):
                buffer += text
                start = 0
                while (end := _chunk_end(buffer, start, self.chunk_chars)) != -1:
                    yield buffer[start:end]
                    start = end
                buffer = buffer[start:]
        if buffer:
            yield buffer

def _chunk_end(buffer: str, start: int, chunk_chars: int) -> int:
    """
    Where the chunk starting at buffer[start] ends: right after the first END_OF_TEXT that makes it at least chunk_chars long.
    -1 if there isn't one in the buffer yet. An END_OF_TEXT cut off by the end of the buffer isn't found, so it can't be split between two chunks.
    """
    found = buffer.find(END_OF_TEXT, start + max(0, chunk_chars - len(END_OF_TEXT)))
    return -1 if found == -1 else found + len(END_OF_TEXT)
