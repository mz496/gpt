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

1:38:00
- Dropout

1:40:00
- Scaling up
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import LayerNorm, functional as F

@dataclass(frozen=True)
class GPTConfig:
    """
    Everything that defines the model's architecture (and so the shapes of its weights).
    Saved inside the checkpoint so that a run can be reloaded without knowing the training constants.
    """
    vocab_size: int
    # number of embedding dimensions. In the paper, this is d_model.
    n_embd: int
    n_head: int
    n_layer: int
    # what is the max context length for predictions?
    block_size: int
    dropout: float

def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

class FeedForward(torch.nn.Module):
    """
    Feedforward linearity to 'make the model think on its attention steps' (? - wording from the video)
    FFN(x) = max(0, xW1 + b1)W2 + b2

    In the paper the authors go up to d_ff = 4 times the d_model (which are larger than we use here).

    Karpathy justifies the second linearity as the projection for the residual stream.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_embd = config.n_embd
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            # Dropout right before outputting, so we preserve the residual pathway happening downstream
            torch.nn.Dropout(config.dropout),
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

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.head_size = head_size
        # Each of the [B, T] will emit one of these.
        # In the paper, the 1th dim (head_size) is d_k. Query and key must be the same d_k because they get multiplied together as the 'dot product' attention.
        self.query = torch.nn.Linear(config.n_embd, head_size, bias=False)
        self.key = torch.nn.Linear(config.n_embd, head_size, bias=False)
        # In the paper, the 1th dim (head_size) is d_v.
        self.value = torch.nn.Linear(config.n_embd, head_size, bias=False)
        # block_size is the max value along the T dimension; which we then slice according to what the actual input is.
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = torch.nn.Dropout(config.dropout)

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
        _B, T, _C = idx.shape
        q: torch.Tensor = self.query(idx)
        k: torch.Tensor = self.key(idx)
        v: torch.Tensor = self.value(idx)
        _, _, d_k = k.shape

        # [B, T, head_size] * [B, head_size, T] = [B, T, T]
        wei = q @ k.transpose(-2, -1) * (d_k ** -0.5)
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

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    """
    Scaled dot-product attention, agnostic to other dimensions (which we use to do attention on multiple heads at once)

    Basically a reimplementation of https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html

    Inputs:
    - q: [... T_q, hs] in
    - k: [... T_k, hs] in
    - v: [... T_k, hs] in
    - attn_mask: [T_q, T_k] of 0/1 where 1 means "participates in attention" and 0 means "does not participate"
    Output:
    [..., T_q, hs]

    TODO: Can also unwind this and optimize it by not operating on the last 2 dims like the stateless helper and always working in the correct shape,
    since this forces a reshape later as is.
    """
    d_k = k.shape[-1] # head size, ie dimensions in k.
    
    # [..., T_q, hs] * [..., hs, T_k] = [..., T_q, T_k]
    wei = q @ k.transpose(-2, -1) * (d_k ** -0.5)
    # Mask broadcast from [T_q, T_k] to [..., T_q, T_k] (matching wei)
    wei = wei.masked_fill(attn_mask == 0, float("-inf"))
    # Softmax per-row, ie the final dimension.
    wei = F.softmax(wei, dim=-1)

    wei = F.dropout(wei, p=dropout_p)

    # [..., T_q, T_k] * [..., T_k, hs] = [..., T_q, hs]
    out = wei @ v
    return out

class MultiHeadAttentionParallelHeads(torch.nn.Module):
    """
    Before, each head was running in series, which is conceptually simpler but slower.
    This is because the heads can run in parallel as they don't interact until we concat at the very end.

    MultiHeadAttention (before)
    in:  [B, T, C]
    ├─ Head × H (in series)    each: [B, T, C] -> [B, T, hs]
    ├─ cat(dim=-1)                   [B, T, H*hs]
    ├─ proj                          [B, T, C]
    └─ dropout                       [B, T, C]
    out: [B, T, C]

    MultiHeadAttentionParallelHeads (after, unfused)
    in:  [B, T, C]
    ├─ query proj   [B, T, C] -> q: [B, T, H*hs]
    ├─ key proj     [B, T, C] -> k: [B, T, H*hs]
    ├─ value proj   [B, T, C] -> v: [B, T, H*hs]
    ├─ split heads  (each of q, k, v)
    │               [B, T, H*hs] -> [B, H, T, hs]
    ├─ attention (per head)
    │               [B, H, T, hs] -> [B, H, T, hs]
    │    internal scores: [B, H, T, T]
    ├─ merge heads  [B, H, T, hs] -> [B, T, H*hs]
    ├─ proj         [B, T, H*hs] -> [B, T, C]
    └─ dropout      [B, T, C]
    out: [B, T, C]

    C = n_embd, H = n_head
    """

    tril: torch.Tensor

    def __init__(self, config: GPTConfig, num_heads: int, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_p = config.dropout # For the attention-internal dropout

        # All heads along the out-dimension layer at once
        self.query = torch.nn.Linear(config.n_embd, num_heads * head_size, bias=False)
        self.key = torch.nn.Linear(config.n_embd, num_heads * head_size, bias=False)
        self.value = torch.nn.Linear(config.n_embd, num_heads * head_size, bias=False)

        self.proj = torch.nn.Linear(num_heads * head_size, config.n_embd)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x: torch.Tensor):
        # [B, T, C] -> [B, T, H*hs]
        q: torch.Tensor = self.query(x)
        k: torch.Tensor = self.key(x)
        v: torch.Tensor = self.value(x)
        # NOTE: We separate out when we actually mean T_q vs T_k for potential later work when T_q != T_k such as KV caching, despite the fact that T_q = T_k = T right now
        H = self.num_heads
        hs = self.head_size
        B, T_q, _ = q.shape
        _, T_k, _ = k.shape

        # Reshape q, k, v for attention (namely we want the time dimension at -2, and head size dimension at -1)
        # [B, T, H*hs] -> [B, T, H, hs] -> [B, H, T, hs]
        q = q.view(B, T_q, H, hs).transpose(1, 2)
        k = k.view(B, T_k, H, hs).transpose(1, 2)
        v = v.view(B, T_k, H, hs).transpose(1, 2)

        T_q = q.shape[-2] # "L" in pytorch - the target sequence length. Recall Q is "what am I looking for"
        T_k = k.shape[-2] # "S" in pytorch - the source sequence length. Recall K is "what do I contain"
        # This avoids tokens from being influenced by future tokens when we compute the output.
        #
        # This later gives the weights [..., T_q, T_k] 1-summing rows where row i has i+1 nonzero vals
        # [1.0, 0.0, ...,          0.0]
        # [0.3, 0.7, 0.0, ...      0.0]
        # [0.2, 0.5, 0.3, 0.0, ... 0.0]
        # ...
        # wei row i, col j: "how much did Q,K overlap re: 'what token i is looking for' with 'what token j has'"
        # 
        # Then multiplying by wei @ v gives:
        # For time t, a head_size-dimensional representation that represents a mixture of all of [:t+1]'s v vectors.
        # e.g. out[2] = 0.2*v[0] + 0.5*v[1] + 0.3*v[2]
        # This is how we get all of the prefixes [0:i] for i=1..T+1 at once.
        attn_mask = self.tril[:T_q, :T_k]

        # [B, H, T (really T_q), hs]
        attention_heads: torch.Tensor = scaled_dot_product_attention(q, k, v, attn_mask, self.dropout_p if self.training else 0)

        # Reshape back
        # [B, H, T, hs] -> [B, T, H, hs] -> [B, T, H*hs]
        # Note: reshape almost always copies here, because the transpose means the strides go from
        # - from (H*T_q*hs, T_q*hs, hs, 1) (contiguous)
        # - to   (H*T_q*hs, hs, T_q*hs, 1) (no longer contiguous)
        # and no copy is necessary when merging dims a, b and stride_a == size_b * stride_b
        # thus stride_H == size_hs * stride_hs
        #        T_q*hs == hs      * 1
        # i.e. only when T_q == 1.
        # But in practice the inefficiency of the body of scaled_dot_product_attention (allocating a new tensor basically on every line) is more important
        attention_heads = attention_heads.transpose(1, 2).reshape(B, T_q, H*hs)

        # [B, T, H*hs] -> [B, T, C]
        out = self.proj(attention_heads)

        out = self.dropout(out)

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

    def __init__(self, config: GPTConfig, num_heads: int, head_size: int):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(config, head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(num_heads * head_size, config.n_embd)
        self.dropout = torch.nn.Dropout(config.dropout)

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
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Masked Multi-Head Attention
        # The intuition is that we are letting attention run in this dimensional space
        # With multi-head self-attention we subdivide into several smaller-dimensional spaces.
        # -- Sequential version kept for posterity/learning
        # self.sa_heads = MultiHeadAttention(config, num_heads=config.n_head, head_size=config.n_embd // config.n_head)
        # -- Batched version:
        self.sa_heads = MultiHeadAttentionParallelHeads(config, num_heads=config.n_head, head_size=config.n_embd // config.n_head)

        # Feed Forward
        # Note that these embedding dimensions are independently looking at one another, after the self-attention has done the cross-communication.
        self.ffwd = FeedForward(config)

        # Add and Norm
        # We put the layer norms before the self-attention and feed-forward layers, in contrast to the paper, but in keeping with the video.
        self.ln1 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor):
        # Relative to 1:26:00:
        # Only adding these skip-connections with +, we can drop validation loss from the plateauing 2.38 to 2.16.
        # With the residual stream projections, we can drop validation loss further to 2.04 (and 1.91 training loss which starts to signal overfitting).
        # [B, T, C] in and out
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(torch.nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # From Figure 1: The Transformer https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf:
        # Output Embedding + Positional Encoding
        # An array of num_embeddings rows; embedding_dim cols; initialized to N(0,1) (https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)
        self.token_embedding = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd)
        self.position_embedding = torch.nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd)

        # Transformer blocks
        # Learn longer-range dependencies by passing the enriched vectors along in sequence.
        # e.g. first block is token<>token dependencies; second block might be word<>word dependencies or something
        self.blocks = torch.nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = torch.nn.LayerNorm(config.n_embd)

        # Final linear layer
        # "Encodes token and position into N_EMBD dimensions and back out"
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size)

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
            # So we need to look at only the last block_size.
            # NOTE: This means we can generate infinitely, but tokens that fall out of the context window are no longer considered.
            output_last_block = output[:, -self.config.block_size:]
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

def checkpoint_path(run_id: str) -> Path:
    return Path("checkpoints") / run_id / "model.pt"

def save_checkpoint(model: GPT, run_id: str, tokenizer_name: str, bpe_vocab_size: int) -> Path:
    """
    torch.save s the model's config, state_dict, and tokenizer settings to checkpoints/<run_id>/model.pt and returns that path.
    """
    path = checkpoint_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "tokenizer": {"name": tokenizer_name, "bpe_vocab_size": bpe_vocab_size},
    }, path)
    return path

def load_checkpoint(run_id: str, device: str) -> tuple[GPT, str, int]:
    """
    Reads the checkpoint written by save_checkpoint into a fresh model on `device`, in eval mode.
    The architecture comes from the config saved in the checkpoint.
    Returns (model, tokenizer_name, bpe_vocab_size). Checkpoints from before tokenizers were saved are treated as "char".
    """
    path = checkpoint_path(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    saved = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    tokenizer = saved.get("tokenizer", {"name": "char", "bpe_vocab_size": 0})
    return model, tokenizer["name"], tokenizer["bpe_vocab_size"]
