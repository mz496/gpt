import math
import torch
from torch.nn import functional as F

from .utils import decode, encode, get_data

# --- hyperparameters ---

# Training
TRAIN_STEPS = 1000
# how many independent sequences will we look at in parallel?
BATCH_SIZE = 4
# what is the max context length for predictions?
BLOCK_SIZE = 8
LEARNING_RATE = 1e-3
# for loss estimation
EVAL_ITERS = 200

# Model
# number of embedding dimensions
N_EMBD = 32
# ---

def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

def get_batch(tokenized: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull `batch_size` windows of length `block_size` starting at random points in `tokenized`.

    Returns two tensors (contexts, targets) of size [B, T]
    - B = batch_size
    - T = block_size
    where targets[i][j] is the token that follows the prefix context[i][:j+1].

    NOTE: In the bigram model we ignore everything except the final char for predictions.
    i.e. our examples are of the form (context[i][j], targets[i][j])
    It sets us up to start using the whole prefix (i.e. is no longer a bigram model).
    """
    # [B] random numbers drawn from [0, n-block_size)
    # Do we need to prevent wraparound here...?
    starts = torch.randint(0, len(tokenized) - block_size, (batch_size,))
    x = torch.stack([tokenized[i:i+block_size] for i in starts])
    y = torch.stack([tokenized[i+1:i+block_size+1] for i in starts])
    return x, y

@torch.no_grad
def estimate_loss(model: torch.nn.Module, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor) -> dict[str, float]:
    """
    Estimate loss over train, validation based on EVAL_ITERS batches.
    """
    model.eval()
    result: dict[str, float] = {}
    for dataset_type, data in [("train", tokenized_train_data), ("validation", tokenized_val_data)]:
        # Compute the mean per dataset type.
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            contexts, targets = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
            _logits, loss = model(contexts, targets)
            losses[k] = loss
        result[dataset_type] = losses.mean().item()
    model.train()
    return result


# Build a decoder-only gpt:
# Roughly up until 38:00
# Starting from the bigram model:
# - init: a simple embedding of C*C - for each char, which is the char that most likely follows?
# - forward pass on an input of size (B, T) gives logits, loss. You want a distribution over tokens C, which is the softmax logit among all of C?
# - generation is: given some (B, T) input, generate max_new_tokens by continually extending the sequence that many times
#
# Roughly 58:00 - 1:19:00
# Add:
# - "number of embedding dimensions": the token embedding goes from vocab_size -> n_embd. Thus every token in the vocab indexes to an embedding of size n_embd.
# - then via an lm_head "language model head" Linear layer of size n_embd back to vocab_size
# - position embedding table that also goes from block_size -> n_embd, so every position gets its own embedding of size n_embd too.
# 
# Self-attention heads
# - query, key, value of size (B, T, head_size) - independently produced by every pos in the (B, T) inputs
#   - query: "what am I looking for"
#   - key: "what do I contain". s.t. query @ key = weights
#   - value: "here is what I will communicate to you"
# Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
# (this is per-batch)
# i.e. scale down by the head_size so that the softmax does not sharpen towards the max too much, since Q*K^T is on the order of head_size.

class Head(torch.nn.Module):

    def __init__(self, head_size: int):
        super().__init__()
        self.head_size = head_size
        # Each of the [B, T] will emit one of these.
        self.query = torch.nn.Linear(N_EMBD, head_size)
        self.key = torch.nn.Linear(N_EMBD, head_size)
        self.value = torch.nn.Linear(N_EMBD, head_size)

    def forward(self, idx: torch.Tensor):
        """
        Input size [B, T, N_EMBD]
        Output size [B, T, N_EMBD]
        
        Implements self-attention: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        """
        device = get_device()

        # [B, T, C] -> [B, T, head_size]
        _B, T, _C = idx.shape
        q: torch.Tensor = self.query(idx)
        k: torch.Tensor = self.key(idx)
        v: torch.Tensor = self.value(idx)

        # [B, T, head_size] * [B, head_size, T] = [B, T, T]
        wei = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        # Masked-fill lower-triangular with -inf
        mask = torch.tril(torch.ones(T, T)).to(device)
        wei = wei.masked_fill(mask == 0, float("-inf"))
        # Softmax per-row, ie the final dimension.
        wei = F.softmax(wei, dim=-1)

        # [B, T, T] * [B, T, head_size] = [B, T, head_size]
        out = wei @ v
        return out

class GPT(torch.nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=N_EMBD)
        self.position_embedding = torch.nn.Embedding(num_embeddings=BLOCK_SIZE, embedding_dim=N_EMBD)
        # Self-attention head.
        # For now just one head of size N_EMBD.
        # The intuition is that we are letting attention run in this dimensional space
        # - with multi-head self-attention we will subdivide the space later
        self.sa_head = Head(N_EMBD)
        # "Encodes token and position into N_EMBD dimensions and back out"
        self.lm_head = torch.nn.Linear(N_EMBD, vocab_size)
    
    def _embed(self, idx: torch.Tensor):
        device = get_device()

        _B, T = idx.shape
        # C := N_EMBD
        # token embedding:
        # [B, T] in
        # [B, T, C] out: the N_EMBD-size embeddings for every element of the input [B, T]
        tok_emb = self.token_embedding(idx)
        # pos embedding:
        # [T] in: just [0..T-1] to index into position embedding.
        # [T, C] out: the N_EMBD-size embeddings for every position from 0 to T
        pos_emb = self.position_embedding(torch.arange(0, T).to(device))
        # (broadcast these across the B batches)
        # [B, T, C]
        x = tok_emb + pos_emb
        # apply one head of self-attention
        # [B, T, C] in
        # [B, T, C] out
        x = self.sa_head(x)
        # language model head:
        # [B, T, C] in, as a combination of both "what is the token" and "what is the position" at every element
        # [B, T, vocab size] out
        logits = self.lm_head(x)
        return logits

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
        logits = self._embed(idx)
        B, T, vocab_size = logits.shape

        # cross_entropy expects input of size [|batch|, num_classes] and targets of size [|batch|] (where the value is then treated as a 1-hot)
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html
        # explicitly calling it vocab_size as that's what it is (instead of reusing "C")
        _, _, vocab_size = logits.shape
        loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T,))
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
            # `output` grows from [B, T] -> [B, T+max_tokens]
            # So we need to look at only the last BLOCK_SIZE.
            output_last_block = output[:, -BLOCK_SIZE:]
            logits = self._embed(output_last_block)

            # Extract the T-th col into a [B, vocab_size] - for each batch, the distribution that the embedding gives over the vocab.
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

def train(step_count: int, train_data: str, validation_data: str, vocab: list[str], device: str) -> GPT:
    model = GPT(len(vocab)).to(device)
    tokenized_train_data = torch.tensor(encode(train_data, vocab)).to(device)
    tokenized_val_data = torch.tensor(encode(validation_data, vocab)).to(device)
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(step_count):
        # Zero out gradients for this iteration.
        model.zero_grad()

        contexts, targets = get_batch(tokenized_train_data, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
        _logits, loss = model(contexts, targets)
        # expect logits shape [B, T, C]
        # An untrained model is near-uniform over C tokens, so loss should start near ~ln(C)
        if step % 100 == 0 or step == step_count-1:
            print(f"step={step:04d} loss={estimate_loss(model, tokenized_train_data, tokenized_val_data)}")

        # Write in gradients based on how bad this iteration was.
        loss.backward()

        # Update weights with those gradients.
        optimizer.step() # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return model

def main():
    torch.manual_seed(1337)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    train_data, validation_data, vocab = get_data()
    device = get_device()
    print(f"available device: {device}")

    model = train(TRAIN_STEPS, train_data, validation_data, vocab, device)

    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    result_tokens = model.generate(idx, max_tokens=1000)
    result = decode(result_tokens[0].tolist(), vocab)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType] Bad torch type stubs.
    print(result)

if __name__ == "__main__":
    main()