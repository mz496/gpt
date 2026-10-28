"""
https://www.youtube.com/watch?v=kCc8FmEb1nY up to ~38:00

Grab shakespeare input
Use last 1000 as validation holdout; the first part is the train set

Tokenize input by characters, let's start with per-character with a vocab size C (roughly A-za-z plus punctuation so ~65)

Training examples (with some parameter T(time) for length, and B(batch_size) for how many)
- sample B random slices of the validation set of length T
- every example is then (input=slice[:i], output=slice[i]) - starting from length 1 up to B, so we can predict any length sequence

Build bigram model:
- init: an embedding of C*C - which is the char that most likely follows C?
- forward pass on an input of size (B, T) gives logits, loss. You want a distribution over tokens C, which is the softmax logit among all of C?
- generation is: given some (B, T) input, generate max_new_tokens by continually extending the sequence that many times

generate with a (1, 1) zero tensor to get, say, 100 new chars in the time dimension.
now you have a (1, 100) tensor, decode it.
"""

from datetime import datetime
from pathlib import Path

def get_data() -> tuple[str, str, list[str]]:
    """
    Fetch tiny Shakespeare data.
    """
    # url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    # with urlopen(url) as f:
    #     data: str = f.read().decode("utf-8")
    with open(Path(__file__).parent / "input.txt") as f:
        data: str = f.read()
    n = len(data)
    train_split_len = int(n * 0.9)
    train = data[:train_split_len]
    validation = data[train_split_len+1:]
    vocab = sorted(list(set(data)))
    return train, validation, vocab

def encode(plain: str, vocab: list[str]) -> list[int]:
    """
    Tokenize by character.
    """
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    return [token_to_id[tok] for tok in list(plain)]

def decode(tokens: list[int], vocab: list[str]) -> str:
    """
    Detokenize by character.
    """
    id_to_token = {i: tok for i, tok in enumerate(vocab)}
    return "".join([id_to_token[id] for id in tokens])

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
