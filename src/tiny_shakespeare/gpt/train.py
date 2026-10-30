import argparse
from dataclasses import asdict

import torch
import wandb

from ..config import CONFIG_NAMES, TrainConfig, load_config
from ..tokenizers.utils import TOKENIZER_NAMES, make_tokenizer
from ..utils import bits_per_char, get_tiny_shakespeare_data, timestamp
from .model import GPT, GPTConfig, get_device, save_checkpoint

def get_batch(tokenized: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull `batch_size` windows of length `block_size` starting at random points in `tokenized`.

    `generator` pins the choice of window starts. Without one we draw from torch's global RNG, which is shared with weight init and dropout,
    so the batches would depend on how much randomness the model itself happened to consume.

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
    starts = torch.randint(0, len(tokenized) - block_size, (batch_size,), generator=generator)
    x = torch.stack([tokenized[i:i+block_size] for i in starts])
    y = torch.stack([tokenized[i+1:i+block_size+1] for i in starts])
    return x, y

@torch.no_grad
def estimate_loss(cfg: TrainConfig, model: torch.nn.Module, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor) -> dict[str, float]:
    """
    Estimate loss over train, validation based on cfg.eval_iters batches.

    The batches are the same every call (and for every model): each dataset gets a fresh generator seeded with cfg.eval_seed.
    This makes estimates comparable across steps and across model implementations, and independent of dropout/init randomness.
    """
    model.eval()
    result: dict[str, float] = {}
    for dataset_type, data in [("train", tokenized_train_data), ("validation", tokenized_val_data)]:
        generator = torch.Generator().manual_seed(cfg.eval_seed)
        # Compute the mean per dataset type.
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            contexts, targets = get_batch(data, cfg.batch_size, cfg.block_size, generator)
            _logits, loss = model(contexts, targets)
            losses[k] = loss
        result[dataset_type] = losses.mean().item()
    model.train()
    return result

def train(cfg: TrainConfig, step_count: int, model: GPT, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor, num_train_chars: int, num_val_chars: int, run: wandb.Run) -> GPT:
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # Own generator for training batches, so the order of training data doesn't depend on how much randomness dropout consumes.
    train_generator = torch.Generator().manual_seed(cfg.train_seed)
    for step in range(step_count):
        # Zero out gradients for this iteration.
        model.zero_grad()

        contexts, targets = get_batch(tokenized_train_data, batch_size=cfg.batch_size, block_size=cfg.block_size, generator=train_generator)
        _logits, loss = model(contexts, targets)
        # expect logits shape [B, T, C]
        # An untrained model is near-uniform over C tokens, so loss should start near ~ln(C)
        if step % 50 == 0 or step == step_count-1:
            losses = estimate_loss(cfg, model, tokenized_train_data, tokenized_val_data)
            print(f"step={step:04d} loss={losses}")
            # Loss per token isn't comparable across tokenizers, so also log it per character of the set we're computing loss over.
            train_bpc = bits_per_char(losses["train"], len(tokenized_train_data), num_train_chars)
            val_bpc = bits_per_char(losses["validation"], len(tokenized_val_data), num_val_chars)
            run.log({
                "train/loss": losses["train"],
                "train/loss_bpc": train_bpc,
                "validation/loss": losses["validation"],
                "validation/loss_bpc": val_bpc,
            }, step=step)

        # Write in gradients based on how bad this iteration was.
        loss.backward()

        # Update weights with those gradients.
        optimizer.step() # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a GPT on tiny shakespeare.")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="default", help="which config in configs/ to use")
    parser.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default="char", help="which tokenizer to use")
    parser.add_argument("--bpe-vocab-size", type=int, default=512, help="target vocab size when --tokenizer=bpe or regex_bpe")
    parser.add_argument("--suffix", default=None, help="appended to the wandb run name (and checkpoint name), e.g. <timestamp>_abc123")
    parser.add_argument("--group", default=None, help="wandb group; runs passing the same string are grouped together in the UI")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.train_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    train_data, validation_data = get_tiny_shakespeare_data()
    tokenizer = make_tokenizer(args.tokenizer, args.bpe_vocab_size)
    # NOTE: By using only train data for the tokenization:
    # - CharTokenizer technically could fail when asked to encode the validation set, since the tokenizer only knows about chars in the train set.
    #   - In practice, this does not happen on tiny shakespeare (there is nothing in the validation set that is unrepresented in the train set).
    #   - (This is because it's not a true byte-level tokenizer; ie roughly just A-Z + a-z + punctuation and not all 256 bytes).
    # - This is mostly to make BPETokenizer/RegexBPETokenizer losses more accurate since they learn their frequencies/merges only on train.
    tokenizer.train(train_data)
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
    if args.suffix:
        name = f"{name}_{args.suffix}"
    run = wandb.init(project="gpt-experiments", name=name, group=args.group, config={
        **asdict(config),
        "config_name": args.config,
        "tokenizer": args.tokenizer,
    })
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"

    model = train(cfg, cfg.train_steps, model, tokenized_train_data, tokenized_val_data, len(train_data), len(validation_data), run)

    checkpoint_path = save_checkpoint(model, run_id, args.tokenizer, args.bpe_vocab_size)
    print(f"saved checkpoint to {checkpoint_path}")
    run.finish()
    print(f"run_id={run_id}")

if __name__ == "__main__":
    main()
