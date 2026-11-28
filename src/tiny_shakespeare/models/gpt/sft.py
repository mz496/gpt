import argparse
from dataclasses import asdict

import torch
import wandb

from src.tiny_shakespeare.config import CONFIG_NAMES, TrainConfig, load_train_config
from src.tiny_shakespeare.data.data_source import Split
from src.tiny_shakespeare.data.loader import get_sft_batch
from src.tiny_shakespeare.data.prepare import prepare_sft, prepare_sft_validation_slices
from src.tiny_shakespeare.data.sft.reverse_source import ReverseSource
from src.tiny_shakespeare.data.sft.sft_data_source import SFTDataSource
from src.tiny_shakespeare.data.special_tokens import IM_END, IM_START
from src.tiny_shakespeare.models.gpt.model import GPT, load_checkpoint, load_checkpoint_tokenizer, save_checkpoint
from src.tiny_shakespeare.models.gpt.train import BatchFn, autocast_dtype, train
from src.tiny_shakespeare.models.gpt.validation_metrics import make_validation_metrics_hook, validation_report
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare.utils import get_device, timestamp

SOURCES: dict[str, type[SFTDataSource]] = {
    "reverse": ReverseSource,
}

def check_checkpoint(model: GPT, tokenizer: Tokenizer) -> None:
    missing = [token for token in (IM_START, IM_END) if token not in tokenizer.special_tokens]
    if missing:
        raise ValueError(
            f"the checkpoint's tokenizer is missing the special tokens {missing} (it has {tokenizer.special_tokens}). "
            "Pretrain with a source that declares them, e.g. --source tiny-stories, and finetune that run."
        )
    if len(tokenizer.get_vocab()) != model.config.vocab_size:
        raise ValueError(f"the tokenizer's vocab size {len(tokenizer.get_vocab())} does not match the model's {model.config.vocab_size}")

def sft(checkpoint_run_id: str, data_source: SFTDataSource, cfg: TrainConfig, suffix: str | None = None) -> str:
    """Finetunes the checkpoint on the data source and saves the result as a new checkpoint. Returns the new run's id."""
    device = get_device()
    print(f"available device: {device}")
    # Use pytorch attention for the finetune train and the validation checks.
    model = load_checkpoint(checkpoint_run_id, device, flag_use_pytorch_attention_impl=True)
    tokenizer = load_checkpoint_tokenizer(checkpoint_run_id)
    check_checkpoint(model, tokenizer)

    # The checkpoint's context length is fixed (its position table has that many rows), so it decides how long an example can be.
    block_size = model.config.block_size
    prepared_data = prepare_sft(data_source, tokenizer, block_size=block_size)
    # The validation split is the pool of the slices, so scoring each slice scores the rows that the validation loss is measured on, with the subtasks apart.
    # A source with no slices gets the pooled score of its validation split alone.
    slice_examples = data_source.get_validation_slices()
    slices = prepare_sft_validation_slices(slice_examples, tokenizer, block_size)
    metrics_hook = make_validation_metrics_hook(slices, device, validation=prepared_data["validation"])
    # For the report at the end: the same rows as the hook scores, and the length of the string of each. Without slices, the report is of the validation split as one.
    report_slices = slices or {"validation": prepared_data["validation"]}
    report_lengths = {name: [len(example.input) for example in examples] for name, examples in slice_examples.items()} or {
        "validation": [len(example.input) for example in data_source.iter_examples("validation")]
    }

    torch.manual_seed(cfg.train_seed)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    dtype = autocast_dtype()
    print(f"training dtype: {dtype or torch.float32}")

    batch_fns: dict[Split, BatchFn] = {
        "train": lambda generator: get_sft_batch(prepared_data["train"], cfg.batch_size, block_size, generator),
        "validation": lambda generator: get_sft_batch(prepared_data["validation"], cfg.batch_size, block_size, generator),
    }

    name = timestamp()
    if suffix:
        name = f"{name}_{suffix}"
    run = wandb.init(project="gpt-experiments", name=name, tags=["sft"], config={
        "train_config": asdict(cfg),
        "model_config": asdict(model.config),
        "_args": {"base_run_id": checkpoint_run_id, "source": type(data_source).__name__},
        "validation_slices": {name: len(data.inputs) for name, data in slices.items()},
        "_debug": {"device": device, "training_dtype_override": dtype},
    })
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"

    # load_checkpoint returns the model in eval mode, which would train without dropout until the first loss estimate flips it.
    model.train()
    model = train(cfg, cfg.train_steps, model, batch_fns, device, run, dtype=dtype, metrics_hook=metrics_hook)

    checkpoint_path = save_checkpoint(model, run_id, tokenizer)
    print(f"saved checkpoint to {checkpoint_path}")
    # Scored the way the hook scored it (the same autocast), so that these counts are the last ones it logged, but by length too.
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        print(validation_report(model, report_slices, report_lengths, device))
    run.finish()
    print(f"run_id={run_id}")
    return run_id

def main():
    parser = argparse.ArgumentParser(description="Finetune a trained run's checkpoint.")
    parser.add_argument("run_id", help="run id of the checkpoint to finetune, e.g. 2026-10-28T11:34:42_abc123")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="sft", help="which config in configs/ to take the training settings from (its train section)")
    parser.add_argument("--source", choices=list(SOURCES), default="reverse", help="which SFT dataset to finetune on")
    parser.add_argument("--suffix", default=None, help="appended to the wandb run name (and checkpoint name), e.g. <timestamp>_abc123")
    args = parser.parse_args()

    sft(args.run_id, SOURCES[args.source](), load_train_config(args.config), args.suffix)

if __name__ == "__main__":
    main()
