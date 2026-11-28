import argparse
from dataclasses import asdict

import torch
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.reports.v2.interface import MetricType

from src.tiny_shakespeare.config import CONFIG_NAMES, TrainConfig, load_train_config
from src.tiny_shakespeare.data.data_source import Split
from src.tiny_shakespeare.data.loader import get_sft_batch
from src.tiny_shakespeare.data.prepare import prepare_sft, prepare_sft_validation_slices
from src.tiny_shakespeare.data.sft.reverse_source import ReverseSource
from src.tiny_shakespeare.data.sft.sft_data_source import SFTDataSource
from src.tiny_shakespeare.data.special_tokens import IM_END, IM_START
from src.tiny_shakespeare.models.gpt.model import GPT, load_checkpoint, load_checkpoint_tokenizer, save_checkpoint
from src.tiny_shakespeare.models.gpt.train import BatchFn, autocast_dtype, train
from src.tiny_shakespeare.models.gpt.validation_metrics import (
    POOLED_METRICS, SLICE_METRICS, make_validation_metrics_hook, pooled_metric_key, slice_metric_key, validation_report,
)
from src.tiny_shakespeare.tokenizers.tokenizer import Tokenizer
from src.tiny_shakespeare import wandb_views
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

# --- The wandb view of the validation slices: one line plot for each metric, with every slice on it, so that the subtasks can be compared as they learn.
# The generic part (making a view, adding runs to it) is in wandb_views.py, which knows nothing about slices, and what is plotted is named by validation_metrics.py.

TITLES = {"exact_match": "exact match", "token_acc": "token accuracy", "loss": "loss"}
# Rates are between 0 and 1, which keeps the plots comparable at a glance.
RATE_METRICS = ("exact_match", "token_acc")
# The colors of the lines, from Open Color (https://yeun.github.io/open-color/), shade 6 of: red, grape, blue, green, and gray for the pool.
# These are the slices of the reversal task; a slice that isn't here keeps the color that wandb gives it.
SLICE_COLORS = {"repeated_char": "#fa5252", "repeated_seq": "#be4bdb", "palindrome": "#228be6", "random": "#40c057"}
POOLED_COLOR = "#868e96"

def slice_line_colors(slice_names: list[str]) -> dict[str, str]:
    """The color of each plotted metric of the slices that have one, and of the pooled ones."""
    colors = {slice_metric_key(name, metric): SLICE_COLORS[name] for name in slice_names if name in SLICE_COLORS for metric in SLICE_METRICS}
    colors.update({pooled_metric_key(metric): POOLED_COLOR for metric in POOLED_METRICS})
    return colors

def slice_sections(slice_names: list[str]) -> list[ws.Section]:
    """
    The panels of the view: a plot for each metric with every slice on it (and the pool, where the hook logs one), and a plot of the losses that train() logs.
    Without slices there's only the pool, so there is no plot of the loss by slice.
    """
    plots: list[wr.LinePlot] = []
    for metric in SLICE_METRICS:
        keys: list[MetricType] = [slice_metric_key(name, metric) for name in slice_names]
        if metric in POOLED_METRICS:
            keys.append(pooled_metric_key(metric))
        if keys:
            plots.append(wr.LinePlot(title=f"{TITLES[metric]}, by slice", x="Step", y=keys, title_y=metric, range_y=(0.0, 1.0) if metric in RATE_METRICS else (None, None)))
    return [
        ws.Section(name="Validation by slice", is_open=True, panels=list(plots)),
        ws.Section(name="Loss", is_open=True, panels=[wr.LinePlot(title="loss", x="Step", y=["train/loss", "validation/loss"])]),
    ]

def show_run_in_view(entity: str, project: str, run_id: str, slice_names: list[str], create_name: str | None, view_id: str | None) -> str:
    """
    Puts a finished run in a view of the slices: a new one called `create_name`, or the existing one `view_id` (exactly one of the two). Returns the view's id.
    The run has to be finished, since wandb doesn't know it before that.
    """
    if (create_name is None) == (view_id is None):
        raise ValueError("give exactly one of a name for a new view, and the id of an existing one")
    if create_name is not None:
        view_id = wandb_views.create_view(entity, project, create_name, slice_sections(slice_names))
    assert view_id is not None
    wandb_views.add_run_to_view(entity, project, view_id, run_id, line_colors=slice_line_colors(slice_names))
    return view_id

def sft(checkpoint_run_id: str, data_source: SFTDataSource, cfg: TrainConfig, suffix: str | None = None, create_view: str | None = None, view_id: str | None = None) -> str:
    """
    Finetunes the checkpoint on the data source and saves the result as a new checkpoint. Returns the new run's id.
    Once the run is done, it can be put in a wandb view of the validation slices, which is made here if `create_view` names one, or is the existing `view_id`.
    """
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
    if create_view is not None or view_id is not None:
        # The run is done and its checkpoint is saved, and a view is only a convenience, so a failure here must not fail the job.
        try:
            shown_in = show_run_in_view(run.entity, run.project, run.id, list(slices), create_view, view_id)
            print(f"the run is in the wandb view {shown_in}: {wandb_views.view_url(run.entity, run.project, shown_in)}")
        except Exception as error:
            print(f"could not put the run in a wandb view: {error!r}")
    print(f"run_id={run_id}")
    return run_id

def main():
    parser = argparse.ArgumentParser(description="Finetune a trained run's checkpoint.")
    parser.add_argument("run_id", help="run id of the checkpoint to finetune, e.g. 2026-10-28T11:34:42_abc123")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="sft", help="which config in configs/ to take the training settings from (its train section)")
    parser.add_argument("--source", choices=list(SOURCES), default="reverse", help="which SFT dataset to finetune on")
    parser.add_argument("--suffix", default=None, help="appended to the wandb run name (and checkpoint name), e.g. <timestamp>_abc123")
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--create-view", default=None, metavar="NAME", help="when the run is done, put it in a new wandb view of the validation slices with this name")
    view.add_argument("--view-id", default=None, help="when the run is done, add it to this existing wandb view of the validation slices (the nw= value of its URL)")
    args = parser.parse_args()

    sft(args.run_id, SOURCES[args.source](), load_train_config(args.config), args.suffix, args.create_view, args.view_id)

if __name__ == "__main__":
    main()
