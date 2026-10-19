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

from dataclasses import dataclass
from urllib.request import urlopen
import numpy as np
from numpy.random import Generator
import torch
from torch.nn import functional as F

def get_data() -> tuple[str, str, list[str]]:
    """
    Fetch tiny Shakespeare data.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urlopen(url) as f:
        data: str = f.read().decode("utf-8")
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

@dataclass
class Example:
    context: list[int]
    target: int

def get_batch(tokenized: list[int], rng: Generator, batch_size: int, context_size: int) -> list[Example]:
    """
    Pull `batch_size` windows of length `context_size` starting at random points in `tokenized`.
    NOTE: This is only used after bigram.
    """
    # Get `batch_size` random ints from [0,n-context_size)
    starts: list[int] = rng.integers(0, len(tokenized) - context_size, size=batch_size).tolist()
    return [Example(context=tokenized[i:i+context_size], target=tokenized[i+context_size]) for i in starts]

def bigram_batch(tokenized: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull `batch_size` windows of length `block_size` starting at random points in `tokenized`.

    Returns two tensors (contexts, targets) of size [B, T]
    - B = batch_size
    - T = block_size
    where the token that follows context[i][j] is targets[i][j].
    """
    # [B] random numbers drawn from [0, n-block_size)
    # Do we need to prevent wraparound here...?
    starts = torch.randint(0, len(tokenized) - block_size, (batch_size,))
    x = torch.stack([tokenized[i:i+block_size] for i in starts])
    y = torch.stack([tokenized[i+1:i+block_size+1] for i in starts])
    return x, y

def main():
    rng = np.random.default_rng(1337)
    torch.manual_seed(1337) # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.

    # Grab shakespeare input
    # Split by train, validation
    train, validation, vocab = get_data()
    
    print(len(train))
    print(len(validation))
    print(vocab)

    # Tokenize input by characters, let's start with per-character with a vocab size C (roughly A-za-z plus punctuation so ~65)
    tokenized = encode(train, vocab)

    # Training examples (with some parameter T(time) for length, and B(batch_size) for how many)
    # - sample B random slices of the validation set of length T
    # - every example is then (input=slice[:i], output=slice[i]) - starting from length 1 up to B, so we can predict any length sequence
    contexts, targets = bigram_batch(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]), batch_size=4, block_size=8)

    # Build bigram model:
    # - init: a simple embedding of C*C - for each char, which is the char that most likely follows?
    # - forward pass on an input of size (B, T) gives logits, loss. You want a distribution over tokens C, which is the softmax logit among all of C?
    # - generation is: given some (B, T) input, generate max_new_tokens by continually extending the sequence that many times
    class BigramModel(torch.nn.Module):
        def __init__(self, vocab: list[str]):
            super().__init__()
            # A simple embedding of C*C learnable params - for each char, which is the char that most likely follows?
            self.embedding = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=len(vocab))
        def forward(self, contexts: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Input:
            - contexts: [B, T] of tokens where
              - B is the number of batches
              - T is the context length per-sample
            - targets: [B, T]
            
            Returns:
            - logits (a distribution over vocab of size C)
            - loss (float)
            """
            # (B, T, C) - the |C| size embedding for every element of the input (B, T)
            logits = self.embedding(contexts)

            # cross_entropy expects input of size [|batch|, C] and targets of size [|batch|] (where the value is then treated as a 1-hot)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T,))
            return logits, loss

    model = BigramModel(vocab)
    logits, loss = model.forward(contexts, targets)
    print(logits, loss)



    # generate with a (1, 1) zero tensor to get, say, 100 new chars in the time dimension.
    # now you have a (1, 100) tensor, decode it.

if __name__ == "__main__":
    main()
