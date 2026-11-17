import random
from pathlib import Path

def get_tiny_shakespeare_data() -> tuple[str, str]:
    """
    Fetch tiny Shakespeare data.
    """
    # url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    # with urlopen(url) as f:
    #     data: str = f.read().decode("utf-8")
    with open(Path(__file__).parent.parent / "input_data" / "tiny_shakespeare.txt") as f:
        data: str = f.read()
    n = len(data)
    train_split_len = int(n * 0.9)
    train = data[:train_split_len]
    validation = data[train_split_len+1:]
    return train, validation

def take_fraction(text: str, fraction: float, chunk_size: int = 1000, seed: int = 0) -> str:
    """
    A representative subsample of `text`: `fraction` of its chunk_size-character chunks, picked at random and kept in their original order.

    The pick order depends only on the seed, so the subsamples are nested (the 25% sample contains the 12.5% sample), and fraction=1.0 returns the text unchanged.
    Sampling chunks (not taking a prefix) keeps the sample from being closer to or further from the validation split's part of the file as it grows.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)
    keep = sorted(order[:max(1, round(len(chunks) * fraction))])
    return "".join(chunks[i] for i in keep)
