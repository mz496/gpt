import argparse
from dataclasses import asdict

import torch
import wandb

from ..config import CONFIG_NAMES, TrainConfig, load_config
from ..tokenizers.utils import TOKENIZER_NAMES, make_tokenizer
from ..utils import get_tiny_shakespeare_data, timestamp
from .model import GPT, GPTConfig, get_device, save_checkpoint

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
def estimate_loss(cfg: TrainConfig, model: torch.nn.Module, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor) -> dict[str, float]:
    """
    Estimate loss over train, validation based on cfg.eval_iters batches.
    """
    model.eval()
    result: dict[str, float] = {}
    for dataset_type, data in [("train", tokenized_train_data), ("validation", tokenized_val_data)]:
        # Compute the mean per dataset type.
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            contexts, targets = get_batch(data, cfg.batch_size, cfg.block_size)
            _logits, loss = model(contexts, targets)
            losses[k] = loss
        result[dataset_type] = losses.mean().item()
    model.train()
    return result

def train(cfg: TrainConfig, step_count: int, model: GPT, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor, run: wandb.Run) -> GPT:
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    for step in range(step_count):
        # Zero out gradients for this iteration.
        model.zero_grad()

        contexts, targets = get_batch(tokenized_train_data, batch_size=cfg.batch_size, block_size=cfg.block_size)
        _logits, loss = model(contexts, targets)
        # expect logits shape [B, T, C]
        # An untrained model is near-uniform over C tokens, so loss should start near ~ln(C)
        if step % 50 == 0 or step == step_count-1:
            losses = estimate_loss(cfg, model, tokenized_train_data, tokenized_val_data)
            print(f"step={step:04d} loss={losses}")
            run.log({"train/loss": losses["train"], "validation/loss": losses["validation"]}, step=step)

        # Write in gradients based on how bad this iteration was.
        loss.backward()

        # Update weights with those gradients.
        optimizer.step() # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a GPT on tiny shakespeare.")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="default", help="which config in configs/ to use")
    parser.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default="char", help="which tokenizer to use")
    parser.add_argument("--bpe-vocab-size", type=int, default=512, help="target vocab size when --tokenizer=bpe")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.train_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    train_data, validation_data = get_tiny_shakespeare_data()
    tokenizer = make_tokenizer(args.tokenizer, args.bpe_vocab_size)
    tokenizer.train(train_data + validation_data)
    device = get_device()
    print(f"available device: {device}")

    config = GPTConfig(
        vocab_size=len(tokenizer.get_vocab()),
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )
    model = GPT(config).to(device)
    tokenized_train_data = torch.tensor(tokenizer.encode_string(train_data)).to(device)
    tokenized_val_data = torch.tensor(tokenizer.encode_string(validation_data)).to(device)

    name = timestamp()
    run = wandb.init(project="gpt-experiments", name=name, config={
        **asdict(config),
        "config_name": args.config,
        "tokenizer": args.tokenizer,
    })
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"

    model = train(cfg, cfg.train_steps, model, tokenized_train_data, tokenized_val_data, run)

    checkpoint_path = save_checkpoint(model, run_id, args.tokenizer, args.bpe_vocab_size)
    print(f"saved checkpoint to {checkpoint_path}")
    run.finish()
    print(f"run_id={run_id}")

if __name__ == "__main__":
    main()
