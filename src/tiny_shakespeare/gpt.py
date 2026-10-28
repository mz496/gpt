"""
Build a decoder-only gpt:
Roughly up until 38:00
Starting from the bigram model:
- init: a simple embedding of C*C - for each char, which is the char that most likely follows?
- forward pass on an input of size (B, T) gives logits, loss. You want a distribution over tokens C, which is the softmax logit among all of C?
- generation is: given some (B, T) input, generate max_new_tokens by continually extending the sequence that many times

Roughly 58:00 - 1:19:00
Add:
- "number of embedding dimensions": the token embedding goes from vocab_size -> n_embd. Thus every token in the vocab indexes to an embedding of size n_embd.
- then via an lm_head "language model head" Linear layer of size n_embd back to vocab_size
- position embedding table that also goes from block_size -> n_embd, so every position gets its own embedding of size n_embd too.

Self-attention heads
- query, key, value of size (B, T, head_size) - independently produced by every pos in the (B, T) inputs
  - query: "what am I looking for"
  - key: "what do I contain". s.t. query @ key = weights
  - value: "here is what I will communicate to you"
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
(this is per-batch)
i.e. scale down by the head_size so that the softmax does not sharpen towards the max too much, since Q*K^T is on the order of head_size.

From here you can get validation loss down to 2.4-2.5 ish.

1:22:00
- Multi-head attention. Validation loss down to 2.26 ish

1:24:00
- Feed-forward step. Validation loss down to 2.24 ish

1:26:30
- Multiple blocks of this. Validation loss creeps up again to 2.38 ish

1:28:00
- Residual stream/skip connections

1:33:00
- LayerNorm
"""

import torch
from torch.nn import LayerNorm, functional as F

from .utils import decode, encode, get_data

# --- hyperparameters ---

# Training
TRAIN_STEPS = 5000
# how many independent sequences will we look at in parallel?
BATCH_SIZE = 64
# what is the max context length for predictions?
BLOCK_SIZE = 256
LEARNING_RATE = 3e-4
# for loss estimation
EVAL_ITERS = 200

# Model
# number of embedding dimensions. In the paper, this is d_model.
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
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

    NOTE:
    In the bigram model, we ignore everything except the final char for predictions.
      i.e. our examples are of the form (context[i][j], targets[i][j])
    During self-attention, the way we build our training examples is exactly the same,
      but the way we read off `context` changes to make the input a mixture of all of context[i][:j+1] instead.
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

class FeedForward(torch.nn.Module):
    """
    Feedforward linearity to 'make the model think on its attention steps' (? - wording from the video)
    FFN(x) = max(0, xW1 + b1)W2 + b2

    In the paper the authors go up to d_ff = 4 times the d_model (which are larger than we use here).

    Karpathy justifies the second linearity as the projection for the residual stream.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            # Dropout right before outputting, so we preserve the residual pathway happening downstream
            torch.nn.Dropout(DROPOUT),
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
        

class Head(torch.nn.Module):
    """
    One head of self-attention.

    Intuition: In a head_size-dimensional space, let the model learn relationships between tokens in the training examples.
    """

    # Placed by register_buffer in init, written here to satisfy typechecking on this field.
    tril: torch.Tensor

    def __init__(self, head_size: int):
        super().__init__()
        self.head_size = head_size
        # Each of the [B, T] will emit one of these.
        # In the paper, the 1th dim (head_size) is d_k. Query and key must be the same d_k because they get multiplied together as the 'dot product' attention.
        self.query = torch.nn.Linear(N_EMBD, head_size, bias=False)
        self.key = torch.nn.Linear(N_EMBD, head_size, bias=False)
        # In the paper, the 1th dim (head_size) is d_v.
        self.value = torch.nn.Linear(N_EMBD, head_size, bias=False)
        # BLOCK_SIZE is the max value along the T dimension; which we then slice according to what the actual input is.
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, idx: torch.Tensor):
        """
        Input size [B, T, N_EMBD]
        Output size [B, T, head_size]
        
        Implements self-attention: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

        The input is [B, T] where each training example is `tokens := idx[b]`
        Then row(batch) b, col(time) t, holds the N_EMBD-length vector: tok_emb(tokens[t]) + pos_emb(t)
        Without attention, we'd just be learning "what follows tokens[t] at time position t?"
        With masked self-attention, "what follows tokens[t] at time t" means some mixture of information from [0,t)
        - The mask is to ensure we only get information from [0,t) and not from the future
        - Q*K^T describes the fraction of how much each of [0,t) contributes
        - V describes how much each of the [0:t] reaches the final vector

        Intuition:
        Query: "What am I looking for"
        Key: "What do I contain"
        Value: "What will I transmit"

        """
        # [B, T, N_EMBD] -> [B, T, head_size]
        _B, T, C = idx.shape
        q: torch.Tensor = self.query(idx)
        k: torch.Tensor = self.key(idx)
        v: torch.Tensor = self.value(idx)

        # [B, T, head_size] * [B, head_size, T] = [B, T, T]
        # d_k here is C, or N_EMBD.
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        # [T, T] masked-fill lower-triangular with -inf
        # This avoids tokens from being influenced by future tokens when we compute the output
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # Softmax per-row, ie the final dimension.
        wei = F.softmax(wei, dim=-1)
        # At this point, wei has 1-summing rows, with row t holding t+1 values:
        # [1.0, 0.0, ...,          0.0]
        # [0.3, 0.7, 0.0, ...      0.0]
        # [0.2, 0.5, 0.3, 0.0, ... 0.0]
        # ...
        # wei row i, col j: "how much did Q,K overlap re: 'what token i is looking for' with 'what token j has'"

        # Randomly dropout some weights, i.e. stop some positions from communicating with one another.
        wei = self.dropout(wei)

        # [B, T, T] * [B, T, head_size] = [B, T, head_size]
        # Per batch:
        # For time t, a head_size-dimensional representation that represents a mixture of all of [:t+1]'s v vectors.
        # e.g. out[2] = 0.2*v[0] + 0.5*v[1] + 0.3*v[2]
        # This is how we get all of the prefixes [0:i] for i=1..T+1 at once.
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    """
    Multiple heads of self-attention running in parallel, concatenated together along the channel dimension.

    Translations to the paper:
    - num_heads is the paper's `h`.
    - head_size is the paper's `d_k`.

    In the paper:

    MultiHead(Q, K, V) = Concat(head_1, ... head_h) * W^O
    where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor):
        # [B, T, C] in
        # Head(head_size): [B, T, C] -> [B, T, head_size].
        # num_heads of these concatted: [B, T, head_size * num_heads]
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # [B, T, head_size * num_heads] in
        # [B, T, C] out
        # Since num_heads * head_size happens to equal N_EMBD the way we've done it, we could just plainly concat,
        # but instead we let the heads' outputs interact via this projection.
        # W^O in the paper
        out = self.proj(out)
        # Dropout right before outputting here, so the later residual pathway is not affected.
        out = self.dropout(out)
        return out

class Block(torch.nn.Module):
    """
    Transformer block.
    """
    def __init__(self, n_embd: int, n_heads: int):
        super().__init__()
        # Masked Multi-Head Attention
        # The intuition is that we are letting attention run in this dimensional space
        # With multi-head self-attention we subdivide into several smaller-dimensional spaces.
        self.sa_heads = MultiHeadAttention(num_heads=n_heads, head_size=n_embd // n_heads)

        # Feed Forward
        # Note that these embedding dimensions are independently looking at one another, after the self-attention has done the cross-communication.
        self.ffwd = FeedForward(N_EMBD)

        # Add and Norm
        # We put the layer norms before the self-attention and feed-forward layers, in contrast to the paper, but in keeping with the video.
        self.ln1 = LayerNorm(N_EMBD)
        self.ln2 = LayerNorm(N_EMBD)

    def forward(self, x: torch.Tensor):
        # Relative to 1:26:00:
        # Only adding these skip-connections with +, we can drop validation loss from the plateauing 2.38 to 2.16.
        # With the residual stream projections, we can drop validation loss further to 2.04 (and 1.91 training loss which starts to signal overfitting).
        # [B, T, C] in and out
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(torch.nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        # From Figure 1: The Transformer https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf:
        # Output Embedding + Positional Encoding
        # An array of num_embeddings rows; embedding_dim cols; initialized to N(0,1) (https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)
        self.token_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=N_EMBD)
        self.position_embedding = torch.nn.Embedding(num_embeddings=BLOCK_SIZE, embedding_dim=N_EMBD)

        # Transformer blocks
        # Learn longer-range dependencies by passing the enriched vectors along in sequence.
        # e.g. first block is token<>token dependencies; second block might be word<>word dependencies or something
        self.blocks = torch.nn.Sequential(*[Block(N_EMBD, n_heads=N_HEAD) for _ in range(N_LAYER)])
        self.ln_final = torch.nn.LayerNorm(N_EMBD)

        # Final linear layer
        # "Encodes token and position into N_EMBD dimensions and back out"
        self.lm_head = torch.nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Input:
        - idx: [B, T] of tokens where
            - B is the number of batches
            - T is the context length per-sample
        - targets: [B, T]
            - None if we are in generation mode in which case we do not calculate losses.
        
        Returns:
        - logits ([B, T, C] a distribution over vocab of size C)
        - loss (float)
        """
        device = get_device()
        B, T = idx.shape
        # C := N_EMBD
        # token embedding:
        # [B, T] in
        # [B, T, C] out: the N_EMBD-size embeddings for every element of the input [B, T]
        tok_emb = self.token_embedding(idx)
        # pos embedding:
        # [T] in: just [0..T-1] to index into position embedding.
        # [T, C] out: the N_EMBD-size embeddings for every position from 0 to T
        pos_emb = self.position_embedding(torch.arange(0, T).to(device))
        # (broadcast position embeddings across the B batches)
        x = tok_emb + pos_emb # [B, T, C]
        # apply transformer blocks
        x = self.blocks(x) # [B, T, C]
        x = self.ln_final(x)

        # language model head:
        # [B, T, C] in
        # [B, T, vocab size] out
        logits = self.lm_head(x)

        _B, _T, vocab_size = logits.shape

        # cross_entropy expects input of size [|batch|, num_classes] and targets of size [|batch|] (where the value is then treated as a 1-hot)
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html
        # explicitly calling it vocab_size as that's what it is (instead of reusing "C")
        _, _, vocab_size = logits.shape
        if targets is None:
            loss = None
        else:
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
            # NOTE: This means we can generate infinitely, but tokens that fall out of the context window are no longer considered.
            output_last_block = output[:, -BLOCK_SIZE:]
            logits, _loss = self(output_last_block, targets=None)

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

def train(step_count: int, model: GPT, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor) -> GPT:
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

    model = GPT(len(vocab)).to(device)
    tokenized_train_data = torch.tensor(encode(train_data, vocab)).to(device)
    tokenized_val_data = torch.tensor(encode(validation_data, vocab)).to(device)
    model = train(TRAIN_STEPS, model, tokenized_train_data, tokenized_val_data)

    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    result_tokens = model.generate(idx, max_tokens=1000)
    result = decode(result_tokens[0].tolist(), vocab)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType] Bad torch type stubs.
    print(result)

if __name__ == "__main__":
    main()