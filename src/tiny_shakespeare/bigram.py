import math

import torch
from torch.nn import functional as F

from .main import encode, get_data


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

    def generate(self, contexts: torch.Tensor, max_tokens: int):
        """
        TODO: Using the model's weights, generate up to `max_tokens` chars per batch.
        """
        pass


def main():
    torch.manual_seed(1337)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.

    train, _validation, vocab = get_data()
    tokenized = torch.tensor(encode(train, vocab))

    contexts, targets = bigram_batch(tokenized, batch_size=4, block_size=8)
    model = BigramModel(vocab)
    logits, loss = model(contexts, targets)

    print(logits.shape)  # expect [B, T, C] = [4, 8, 65]
    # An untrained model is near-uniform over C tokens, so loss should be ~ln(C).
    print(f"loss={loss.item():.3f} expected~{math.log(len(vocab)):.3f}")

def train():
    torch.manual_seed(1337)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.

    train, _validation, vocab = get_data()
    tokenized = torch.tensor(encode(train, vocab))

    model = BigramModel(vocab)
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step_count = 10000
    for step in range(step_count):
        # Zero out gradients for this iteration.
        model.zero_grad()

        contexts, targets = bigram_batch(tokenized, batch_size=4, block_size=8)
        logits, loss = model(contexts, targets)
        # expect logits shape [B, T, C] = [4, 8, 65]
        # An untrained model is near-uniform over C tokens, so loss should start near ~ln(C)
        if step % 10 == 0:
            print(f"step={step:03d} loss={loss.item():.3f} expected init ~{math.log(len(vocab)):.3f}")

        # Write in gradients based on how bad this iteration was.
        loss.backward()

        # Update weights with those gradients.
        optimizer.step() # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.


if __name__ == "__main__":
    train()
