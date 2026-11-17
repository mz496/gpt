from pathlib import Path
from typing import Iterator, override

from .data_source import DataSource, Split

class TinyShakespeareSource(DataSource):
    """
    Tiny shakespeare (~1MB), small enough to read whole. The first 90% of the file is the train split and the rest is validation.
    """

    name = "tiny-shakespeare"

    @override
    def iter_text(self, split: Split) -> Iterator[str]:
        data = (Path(__file__).parent / "tiny_shakespeare.txt").read_text()
        train_split_len = int(len(data) * 0.9)
        match split:
            case "train":
                yield data[:train_split_len]
            case "validation":
                yield data[train_split_len+1:]
