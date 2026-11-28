import argparse
from collections.abc import Callable
from dataclasses import asdict

import torch
import wandb

from ...config import CONFIG_NAMES, TrainConfig, load_config
from ...data.data_source import TOKENIZER_TEXT_MAX_CHARS, Split
from ...data.loader import get_batch
from ...data.prepare import prepare, train_tokenizer
from ...data.tiny_shakespeare_source import TinyShakespeareSource
from ...data.tiny_stories_source import TinyStoriesSource
from ...timer import PhaseTimer
from ...tokenizers.utils import TOKENIZER_NAMES, make_tokenizer
from ...utils import bits_per_char, get_device, timestamp
from .model import GPT, GPTConfig, save_checkpoint

def autocast_dtype() -> torch.dtype | None:
    """bf16 on CUDA GPUs that support it (Ampere and newer); None (stay fp32) on mps/cpu and on older CUDA GPUs."""
    if get_device() == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None

EstimateLossBatches = dict[Split, list[tuple[torch.Tensor, torch.Tensor]]]

# Draws one (contexts, targets) batch of [B, T] int64 CPU tensors, using the given generator for all of its randomness.
# The data, batch size and block size are already bound in, so that train() doesn't need to know what kind of data it is training on.
BatchFn = Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]]

# Called at each loss estimate with (step, model, the losses from estimate_loss); returns extra metrics to log next to the losses.
MetricsHook = Callable[[int, GPT, dict[Split, float]], dict[str, float]]

def make_estimate_loss_batches(cfg: TrainConfig, batch_fns: dict[Split, BatchFn], device: str) -> EstimateLossBatches:
    """
    Build the cfg.estimate_loss_iters (contexts, targets) batches used for loss estimation, per dataset split (the keys of batch_fns), grouped into chunks.

    Each chunk concatenates up to cfg.estimate_loss_chunk_size consecutive batches along the batch dimension, so that estimate_loss can run a single forward per chunk.
    The last chunk is smaller when estimate_loss_iters isn't a multiple of estimate_loss_chunk_size.

    The batches are the same for every estimate (and for every model): each dataset gets a fresh generator seeded with cfg.estimate_loss_seed.
    This makes estimates comparable across steps and across model implementations, and independent of dropout/init randomness.
    Since they never change, we build them once up front instead of on every estimate.
    """
    result: EstimateLossBatches = {}
    for dataset_type, batch_fn in batch_fns.items():
        generator = torch.Generator().manual_seed(cfg.estimate_loss_seed)
        batches = [batch_fn(generator) for _ in range(cfg.estimate_loss_iters)]
        result[dataset_type] = [
            (torch.cat([contexts for contexts, _ in chunk]).to(device), torch.cat([targets for _, targets in chunk]).to(device))
            for chunk in (batches[i:i + cfg.estimate_loss_chunk_size] for i in range(0, len(batches), cfg.estimate_loss_chunk_size))
        ]
    return result

@torch.no_grad
def estimate_loss(model: torch.nn.Module, batches: EstimateLossBatches, dtype: torch.dtype | None = None) -> dict[Split, float]:
    """
    Estimate loss over train, validation as the mean loss over the fixed batches from make_estimate_loss_batches.

    Each chunk's loss is a mean over that chunk's tokens, so we weight by the chunk's number of samples to get the exact mean over all tokens
    (chunks differ in size only when the last one is smaller due to divisibility).

    If dtype is given, the forwards run under autocast in that dtype (mixed precision); the losses are still fp32.
    """
    model.eval()
    result: dict[Split, float] = {}
    for dataset_type, chunks in batches.items():
        # Compute the mean per dataset type.
        # model(contexts, targets) outputs (logits, loss) where loss is 0-dim float
        # Everything stays on-device (model, contexts, targets, losses) until the single .item() at the end, so there is no sync per chunk.
        # Weights are Python ints so multiplying doesn't copy anything to the device.
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            weighted_losses = [model(contexts, targets)[1] * contexts.shape[0] for contexts, targets in chunks]
        total_samples = sum(contexts.shape[0] for contexts, _ in chunks)
        result[dataset_type] = (torch.stack(weighted_losses).sum() / total_samples).item()
    model.train()
    return result

@torch.no_grad
def run_metrics_hook(hook: MetricsHook, step: int, model: GPT, losses: dict[Split, float], dtype: torch.dtype | None = None) -> dict[str, float]:
    """
    Runs the hook the way estimate_loss runs its forwards, so that a hook can't affect training: no gradients, eval mode (no dropout), and the training dtype.
    The model is back in train mode afterwards, even if the hook raises.
    """
    model.eval()
    try:
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            return hook(step, model, losses)
    finally:
        model.train()

def train(
    cfg: TrainConfig,
    step_count: int,
    model: GPT,
    # One batch function per dataset split.
    batch_fns: dict[Split, BatchFn],
    device: str,
    run: wandb.Run,
    timer: PhaseTimer | None = None,
    dtype: torch.dtype | None = None,
    metrics_hook: MetricsHook | None = None,
) -> GPT:
    timer = timer or PhaseTimer(enabled=False)
    # A reasonable lr for adam is 1e-3 to 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, fused=True)
    # Generator for training batches, so the order of training data doesn't depend on how much randomness dropout consumes.
    train_generator = torch.Generator().manual_seed(cfg.train_seed)
    # Cache the batches (where the batch count = cfg.estimate_loss_iters) of the train/val sets in order to estimate loss over the whole sets, for reuse in the hot loop.
    estimate_loss_batches = make_estimate_loss_batches(cfg, batch_fns, device)
    for step in range(step_count):
        # Zero out gradients for this iteration.
        with timer.phase("zero_grad"):
            model.zero_grad()

        with timer.phase("get_batch"):
            contexts, targets = batch_fns["train"](train_generator)
            contexts, targets = contexts.to(device), targets.to(device)
        with timer.phase("forward"):
            # bf16 has fp32's exponent range, so unlike fp16 no loss scaling is needed. The backward pass needs no autocast of its own.
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                _logits, loss = model(contexts, targets)
        # expect logits shape [B, T, C]
        # An untrained model is near-uniform over C tokens, so loss should start near ~ln(C)
        if step % 50 == 0 or step == step_count-1:
            with timer.phase("estimate_loss"):
                losses = estimate_loss(model, estimate_loss_batches, dtype)
            print(f"step={step:04d} loss={losses}")
            metrics = {
                "train/loss": losses["train"],
                "validation/loss": losses["validation"],
            }
            if metrics_hook is not None:
                with timer.phase("metrics_hook"):
                    extra = run_metrics_hook(metrics_hook, step, model, losses, dtype)
                if clashing := metrics.keys() & extra.keys():
                    raise ValueError(f"metrics hook returned keys that train() already logs: {sorted(clashing)}")
                metrics.update(extra)
            with timer.phase("wandb_log"):
                run.log(metrics, step=step)

        # Write in gradients based on how bad this iteration was.
        with timer.phase("backward"):
            loss.backward()

        # Update weights with those gradients.
        with timer.phase("optimizer_step"):
            optimizer.step() # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a GPT on a text dataset.")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="default", help="which config in configs/ to use")
    parser.add_argument("--source", choices=["tiny-shakespeare", "tiny-stories"], default="tiny-shakespeare", help="which dataset to train on")
    parser.add_argument("--tokenizer-text-only", action="store_true", help="train the model on only the text the tokenizer is trained on (the first part of the train split, see TOKENIZER_TEXT_MAX_CHARS), to try out a big dataset quickly")
    parser.add_argument("--tokenizer", choices=TOKENIZER_NAMES, default="char", help="which tokenizer to use")
    parser.add_argument("--bpe-vocab-size", type=int, default=512, help="target vocab size when --tokenizer=bpe or regex_bpe")
    parser.add_argument("--suffix", default=None, help="appended to the wandb run name (and checkpoint name), e.g. <timestamp>_abc123")
    parser.add_argument("--group", default=None, help="wandb group; runs passing the same string are grouped together in the UI")

    parser.add_argument("--precision", choices=["auto", "fp32"], default="auto", help="auto: bf16 mixed precision on CUDA GPUs that support it, fp32 elsewhere; fp32: never use mixed precision")

    # Toggle the timer.
    parser.add_argument("--profile", action="store_true", help="time each phase of the training loop, syncing the device around each so async MPS/CUDA work is attributed correctly (slows training a bit)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.train.train_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.

    # Coarse timings of everything around the training loop (data, tokenizer, wandb, checkpoint)
    setup_timer = PhaseTimer()

    source = TinyStoriesSource() if args.source == "tiny-stories" else TinyShakespeareSource()
    tokenizer = make_tokenizer(args.tokenizer, args.bpe_vocab_size)
    # NOTE: The tokenizer is fit on train data only (see train_tokenizer), except that the CharTokenizer's vocab is the characters of both splits:
    # - A CharTokenizer can't encode a character it hasn't seen, and the validation split can have characters that the train split doesn't (TinyStories does).
    #   Knowing that such a character exists tells it nothing about the validation text.
    # - BPETokenizer/RegexBPETokenizer are byte-level, so they can encode anything. Fitting them only on train keeps their losses more accurate,
    #   since they learn their frequencies/merges only on train.
    with setup_timer.phase("tokenizer_train"):
        train_tokenizer(source, tokenizer)

    device = get_device()
    print(f"available device: {device}")

    dtype = autocast_dtype() if args.precision == "auto" else None
    print(f"training dtype: {dtype or torch.float32}")

    config = GPTConfig(
        vocab_size=len(tokenizer.get_vocab()),
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        block_size=cfg.model.block_size,
        dropout=cfg.model.dropout,
        flag_use_parallel_heads=cfg.flags.flag_use_parallel_heads,
    )
    with setup_timer.phase("build_model"):
        model = GPT(config, flag_use_pytorch_attention_impl=cfg.flags.flag_use_pytorch_attention_impl).to(device)
    with setup_timer.phase("tokenize_data"):
        data = prepare(source, tokenizer, max_train_chars=TOKENIZER_TEXT_MAX_CHARS if args.tokenizer_text_only else None)

    name = timestamp()
    if args.suffix:
        name = f"{name}_{args.suffix}"
    with setup_timer.phase("wandb_init"):
        run = wandb.init(project="gpt-experiments", name=name, group=args.group, config={
            # Flat, as it was before the config was split into sections, so that existing runs and views line up.
            "main_config": {**asdict(cfg.train), **asdict(cfg.model), **asdict(cfg.flags)},
            "model_config": asdict(config),
            "_args": vars(args),
            "_debug": {"train_chars": data.train_chars, "device": device, "training_dtype_override": dtype},
        })
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"

    def bpc_hook(_step: int, _model: GPT, losses: dict[Split, float]) -> dict[str, float]:
        # Loss per token isn't comparable across tokenizers, so also log it per character of the set we're computing loss over.
        return {
            "train/loss_bpc": bits_per_char(losses["train"], len(data.train), data.train_chars),
            "validation/loss_bpc": bits_per_char(losses["validation"], len(data.validation), data.validation_chars),
        }

    batch_fns: dict[Split, BatchFn] = {
        "train": lambda generator: get_batch(data.train, cfg.train.batch_size, cfg.model.block_size, generator),
        "validation": lambda generator: get_batch(data.validation, cfg.train.batch_size, cfg.model.block_size, generator),
    }

    train_timer = PhaseTimer(enabled=args.profile, sync=True)
    with setup_timer.phase("train"):
        model = train(cfg.train, cfg.train.train_steps, model, batch_fns, device, run, train_timer, dtype, metrics_hook=bpc_hook)

    with setup_timer.phase("save_checkpoint"):
        checkpoint_path = save_checkpoint(model, run_id, tokenizer)
    print(f"saved checkpoint to {checkpoint_path}")
    
    with setup_timer.phase("wandb_finish"):
        run.finish()
    setup_timer.report("overall")
    train_timer.report("training loop")
    print(f"run_id={run_id}")

if __name__ == "__main__":
    main()
