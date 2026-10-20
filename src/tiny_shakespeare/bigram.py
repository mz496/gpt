import math
import torch
from torch.nn import functional as F

from .main import decode, encode, get_data


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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        - idx: [B, T] of tokens where
            - B is the number of batches
            - T is the context length per-sample
        - targets: [B, T]
        
        Returns:
        - logits ([B, T, C] a distribution over vocab of size C)
        - loss (float)
        """
        # [B, T, C] - the |C| size embedding for every element of the input [B, T]
        logits = self.embedding(idx)

        # cross_entropy expects input of size [|batch|, C] and targets of size [|batch|] (where the value is then treated as a 1-hot)
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T,))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_tokens: int):
        """
        Using the model's learned weights, generate up to `max_tokens` chars per batch.
        Inputs
        - idx: [B, T] "indices into the vocab".

        Outputs
        - [B, T+max_tokens]
        """
        if max_tokens <= 0:
            raise Exception("max_tokens must be > 0")

        output = idx
        for _ in range(max_tokens):
            # Build up predictions for each context from the last char on each.
            # [B, T, C] - the |C| size embedding for every element of the input [B, T]
            # Note that the exact value of T increases as we go.
            logits = self.embedding(output)

            # Extract the T-th col into a [B, C] - for each batch, the distribution that the embedding gives over the |C| sized vocab.
            # Convert those into a prob distribution over the vocab dimension (since logits may be negative).
            # Draw from this T-th col to get the (T+1)-th.
            # https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html
            # "The rows of input do not need to sum to one (in which case we use the values as weights), but must be non-negative, finite and have a non-zero sum."
            # "If input is a matrix with m rows, out is an matrix of shape (m * num_samples)"
            last_col_logits = logits[:, -1, :]
            last_col_probs = F.softmax(last_col_logits, dim=-1)
            next_tokens = torch.multinomial(input=last_col_probs, num_samples=1)

            # Append to each batch along the time dimension.
            # torch.cat allocates a new tensor, so no need to .clone().
            output = torch.cat([output, next_tokens], dim=1)

        return output

def train(step_count: int, train_data: str, vocab: list[str]) -> BigramModel:
    tokenized = torch.tensor(encode(train_data, vocab))

    model = BigramModel(vocab)
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
    return model

if __name__ == "__main__":
    torch.manual_seed(1337)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    train_data, _validation_data, vocab = get_data()

    model = train(10000, train_data, vocab)

    idx = torch.zeros((1, 1), dtype=torch.long)
    result_tokens = model.generate(idx, max_tokens=1000)
    result = decode(result_tokens[0].tolist(), vocab)
    print(result)
