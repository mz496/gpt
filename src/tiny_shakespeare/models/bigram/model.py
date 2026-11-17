import torch
from torch.nn import functional as F

# Build bigram model:
# - init: a simple embedding of C*C - for each char, which is the char that most likely follows?
# - forward pass on an input of size (B, T) gives logits, loss. You want a distribution over tokens C, which is the softmax logit among all of C?
# - generation is: given some (B, T) input, generate max_new_tokens by continually extending the sequence that many times
class BigramModel(torch.nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        # A simple lookup table of C*C learnable params - for each char, which is the char that most likely follows?
        # Each token directly reads off the logits for the next token from this.
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        - idx: [B, T] of tokens where
            - B is the number of batches
            - T is the context length per-sample
        - targets: [B, T]

        Note that we ignore everything except the final char of each context for predictions,
        i.e. our examples are of the form (idx[i][j], targets[i][j]).
        With self-attention, the way we build our training examples is exactly the same,
        but the prediction at position j becomes a mixture of all of idx[i][:j+1] instead.
        
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
