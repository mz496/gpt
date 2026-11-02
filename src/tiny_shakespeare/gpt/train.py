import argparse
from dataclasses import asdict

import torch
import wandb

from ..config import CONFIG_NAMES, TrainConfig, load_config
from ..timer import PhaseTimer
from ..tokenizers.utils import TOKENIZER_NAMES, make_tokenizer
from ..utils import bits_per_char, get_device, get_tiny_shakespeare_data, take_fraction, timestamp
from .model import GPT, GPTConfig, save_checkpoint

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

EvalBatches = dict[str, list[tuple[torch.Tensor, torch.Tensor]]]

def make_eval_batches(cfg: TrainConfig, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor) -> EvalBatches:
    """
    Build the cfg.eval_iters (contexts, targets) batches used for loss estimation, per dataset.

    The batches are the same for every estimate (and for every model): each dataset gets a fresh generator seeded with cfg.eval_seed.
    This makes estimates comparable across steps and across model implementations, and independent of dropout/init randomness.
    Since they never change, we build them once up front instead of on every estimate.
    """
    result: EvalBatches = {}
    for dataset_type, data in [("train", tokenized_train_data), ("validation", tokenized_val_data)]:
        generator = torch.Generator().manual_seed(cfg.eval_seed)
        result[dataset_type] = [get_batch(data, cfg.batch_size, cfg.block_size, generator) for _ in range(cfg.eval_iters)]
    return result

@torch.no_grad
def estimate_loss(model: torch.nn.Module, eval_batches: EvalBatches) -> dict[str, float]:
    """
    Estimate loss over train, validation as the mean loss over the fixed batches from make_eval_batches.
    """
    model.eval()
    result: dict[str, float] = {}
    for dataset_type, batches in eval_batches.items():
        # Compute the mean per dataset type.
        losses = torch.zeros(len(batches))
        for k, (contexts, targets) in enumerate(batches):
            _logits, loss = model(contexts, targets)
            losses[k] = loss
        result[dataset_type] = losses.mean().item()
    model.train()
    return result

def train(cfg: TrainConfig, step_count: int, model: GPT, tokenized_train_data: torch.Tensor, tokenized_val_data: torch.Tensor, num_train_chars: int, num_val_chars: int, run: wandb.Run, timer: PhaseTimer | None = None) -> GPT:
    timer = timer or PhaseTimer(enabled=False)
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, fused=True)
    # Generator for training batches, so the order of training data doesn't depend on how much randomness dropout consumes.
    train_generator = torch.Generator().manual_seed(cfg.train_seed)
    # Cache the batches (where the count = cfg.eval_iters) of the train/val sets in order to estimate loss over the whole sets, for reuse in the hot loop.
    eval_batches = make_eval_batches(cfg, tokenized_train_data, tokenized_val_data)
    for step in range(step_count):
        # Zero out gradients for this iteration.
        with timer.phase("zero_grad"):
            model.zero_grad()

        with timer.phase("get_batch"):
            contexts, targets = get_batch(tokenized_train_data, batch_size=cfg.batch_size, block_size=cfg.block_size, generator=train_generator)
        with timer.phase("forward"):
            _logits, loss = model(contexts, targets)
        # expect logits shape [B, T, C]
        # An untrained model is near-uniform over C tokens, so loss should start near ~ln(C)
        if step % 50 == 0 or step == step_count-1:
            with timer.phase("estimate_loss"):
                losses = estimate_loss(model, eval_batches)
            print(f"step={step:04d} loss={losses}")
            # Loss per token isn't comparable across tokenizers, so also log it per character of the set we're computing loss over.
            train_bpc = bits_per_char(losses["train"], len(tokenized_train_data), num_train_chars)
            val_bpc = bits_per_char(losses["validation"], len(tokenized_val_data), num_val_chars)
            with timer.phase("wandb_log"):
                run.log({
                    "train/loss": losses["train"],
                    "train/loss_bpc": train_bpc,
                    "validation/loss": losses["validation"],
                    "validation/loss_bpc": val_bpc,
                }, step=step)

        # Write in gradients based on how bad this iteration was.
        with timer.phase("backward"):
            loss.backward()

        # Update weights with those gradients.
        with timer.phase("optimizer_step"):
            optimizer.step() # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a GPT on tiny shakespeare.")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="default", help="which config in configs/ to use")
    parser.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default="char", help="which tokenizer to use")
    parser.add_argument("--bpe-vocab-size", type=int, default=512, help="target vocab size when --tokenizer=bpe or regex_bpe")
    parser.add_argument("--suffix", default=None, help="appended to the wandb run name (and checkpoint name), e.g. <timestamp>_abc123")
    parser.add_argument("--group", default=None, help="wandb group; runs passing the same string are grouped together in the UI")

    # Train fraction is for when the model is large, but we don't want to scale up the model for runtime purposes, and instead scale down the data.
    parser.add_argument("--train-fraction", type=float, default=1.0, help="train on this fraction (0, 1] of the train split, sampled as random 1000-char chunks (nested across fractions), to measure how loss scales with data; the validation split is unchanged")
    # Toggle the timer.
    parser.add_argument("--profile", action="store_true", help="time each phase of the training loop, syncing the device around each so async MPS/CUDA work is attributed correctly (slows training a bit)")

    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in (0, 1]")
    cfg = load_config(args.config)

    torch.manual_seed(cfg.train_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.

    # Coarse timings of everything around the training loop (data, tokenizer, wandb, checkpoint)
    setup_timer = PhaseTimer()
    with setup_timer.phase("load_data"):
        full_train_data, validation_data = get_tiny_shakespeare_data()
    tokenizer = make_tokenizer(args.tokenizer, args.bpe_vocab_size)
    # NOTE: By using only train data for the tokenization:
    # - CharTokenizer technically could fail when asked to encode the validation set, since the tokenizer only knows about chars in the train set.
    #   - In practice, this does not happen on tiny shakespeare (there is nothing in the validation set that is unrepresented in the train set).
    #   - (This is because it's not a true byte-level tokenizer; ie roughly just A-Z + a-z + punctuation and not all 256 bytes).
    # - This is mostly to make BPETokenizer/RegexBPETokenizer losses more accurate since they learn their frequencies/merges only on train.
    with setup_timer.phase("tokenizer_train"):
        tokenizer.train(full_train_data)
    # With --train-fraction, the tokenizer is still fit on the FULL train split and only the data the model trains on is subsampled.
    # This holds the vocab fixed across fractions (and a char vocab fit on a small slice would be missing chars that show up in validation).
    train_data = take_fraction(full_train_data, args.train_fraction)
    device = get_device()
    print(f"available device: {device}")

    config = GPTConfig(
        vocab_size=len(tokenizer.get_vocab()),
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        flag_use_parallel_heads=cfg.flag_use_parallel_heads,
    )
    with setup_timer.phase("build_model"):
        model = GPT(config, flag_use_pytorch_attention_impl=cfg.flag_use_pytorch_attention_impl).to(device)
    with setup_timer.phase("tokenize_data"):
        tokenized_train_data = torch.tensor(tokenizer.encode_string(train_data)).to(device)
        tokenized_val_data = torch.tensor(tokenizer.encode_string(validation_data)).to(device)

    name = timestamp()
    if args.suffix:
        name = f"{name}_{args.suffix}"
    with setup_timer.phase("wandb_init"):
        run = wandb.init(project="gpt-experiments", name=name, group=args.group, config={
            **asdict(config),
            "flag_use_pytorch_attention_impl": cfg.flag_use_pytorch_attention_impl,
            "config_name": args.config,
            "tokenizer": args.tokenizer,
            "train_fraction": args.train_fraction,
            "train_chars": len(train_data),
        })
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"

    train_timer = PhaseTimer(enabled=args.profile, sync=True)
    with setup_timer.phase("train"):
        model = train(cfg, cfg.train_steps, model, tokenized_train_data, tokenized_val_data, len(train_data), len(validation_data), run, train_timer)

    with setup_timer.phase("save_checkpoint"):
        checkpoint_path = save_checkpoint(model, run_id, args.tokenizer, args.bpe_vocab_size)
    print(f"saved checkpoint to {checkpoint_path}")
    
    with setup_timer.phase("wandb_finish"):
        run.finish()
    setup_timer.report("overall")
    train_timer.report("training loop")
    print(f"run_id={run_id}")

if __name__ == "__main__":
    main()
