"""
Scores a model's answers on validation data, row by row, so that they can be reported per slice and per length of string (see SFTDataSource.get_validation_slices).

A row passes if the model's top choice is right at every position of the response, including the end of it. That is measured with one forward pass that has the true
response in its input (teacher forcing), which gives the same pass or fail as generating the answer greedily: the first wrong top choice is exactly where greedy
generation would go wrong, and while every one is right, generation reproduces the response token by token. So there's no generation loop here.

This is not estimate_loss in train.py, which only measures the loss. Here we score the answers (and report the loss of the same pass, per slice).
"""
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ...data.data_source import Split
from ...data.loader import IGNORE_INDEX, SftData
from .model import GPT
from .train import MetricsHook

# Strings are reported in bins of this many lengths: 1-10, 11-20, and so on.
LENGTH_BIN_WIDTH = 10


@dataclass(frozen=True)
class RowScores:
    """What was found for each of N rows. Every field has one entry per row."""

    # Every position of the response got its right token as the top choice.
    passed: torch.Tensor
    # The number of response positions (the ones that aren't IGNORE_INDEX), and how many of them the top choice got right.
    n_tokens: torch.Tensor
    n_right: torch.Tensor
    # Where in the response the first wrong top choice is (0 is its first token), or -1 if there is none.
    first_error: torch.Tensor
    # The summed cross-entropy (in nats) over the response positions.
    loss_sum: torch.Tensor

    def __len__(self) -> int:
        return len(self.passed)

    def select(self, rows: torch.Tensor) -> "RowScores":
        """The scores of some of the rows, given by a boolean mask or a tensor of indices."""
        return RowScores(self.passed[rows], self.n_tokens[rows], self.n_right[rows], self.first_error[rows], self.loss_sum[rows])

    def cpu(self) -> "RowScores":
        return RowScores(self.passed.cpu(), self.n_tokens.cpu(), self.n_right.cpu(), self.first_error.cpu(), self.loss_sum.cpu())


@dataclass(frozen=True)
class Summary:
    """The scores of a group of rows, boiled down."""

    count: int
    # How many of the rows passed, and the fraction.
    n_passed: int
    exact_match: float
    # The fraction of response tokens whose top choice was right, and the mean loss per response token. Both count every response token the same, long answers more.
    token_acc: float
    loss: float


def score_rows(logits: torch.Tensor, targets: torch.Tensor) -> RowScores:
    """
    Compares logits [N, T, V] with targets [N, T], as one forward pass over rows gives them (see SftData). Only the positions whose target isn't IGNORE_INDEX count.
    Raises if a row has no such position, since there'd be nothing to score.
    """
    trained = targets != IGNORE_INDEX
    n_tokens = trained.sum(dim=1)
    if bool((n_tokens == 0).any()):
        raise ValueError("a row has no response positions to score")
    right = (logits.argmax(dim=-1) == targets) & trained
    wrong = trained & ~right

    # The index of every position within its row's response, so that the first wrong one can be found: the smallest index among the wrong positions.
    position_in_response = trained.cumsum(dim=1) - 1
    none = torch.full_like(position_in_response, targets.shape[1])
    first_error = torch.where(wrong, position_in_response, none).min(dim=1).values
    first_error = torch.where(first_error == targets.shape[1], torch.full_like(first_error, -1), first_error)

    # Ignored positions have a loss of 0 here, so the sum is over the response alone.
    per_position = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none").reshape(targets.shape)
    return RowScores(passed=~wrong.any(dim=1), n_tokens=n_tokens, n_right=right.sum(dim=1), first_error=first_error, loss_sum=per_position.sum(dim=1))


def pool(parts: Iterable[RowScores]) -> RowScores:
    """The scores of all the rows of every part, one after the other."""
    parts = list(parts)
    if not parts:
        raise ValueError("there are no scores to pool")
    return RowScores(
        passed=torch.cat([p.passed for p in parts]),
        n_tokens=torch.cat([p.n_tokens for p in parts]),
        n_right=torch.cat([p.n_right for p in parts]),
        first_error=torch.cat([p.first_error for p in parts]),
        loss_sum=torch.cat([p.loss_sum for p in parts]),
    )


def score_validation(model: GPT, data: SftData, device: str, chunk_size: int = 256) -> RowScores:
    """
    Scores every row of `data` with the model, `chunk_size` rows to a forward pass (so as to bound the memory of the logits). Returns the scores on the CPU.
    The model is put in eval mode, with no gradients, and back as it was. It runs under whatever autocast the caller has set.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    was_training = model.training
    model.eval()
    parts: list[RowScores] = []
    try:
        with torch.no_grad():
            for start in range(0, len(data.inputs), chunk_size):
                inputs = torch.from_numpy(data.inputs[start:start + chunk_size].astype(np.int64)).to(device)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
                targets = torch.from_numpy(data.targets[start:start + chunk_size]).to(device)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
                logits, _ = model(inputs, None)
                parts.append(score_rows(logits, targets).cpu())
    finally:
        model.train(was_training)
    return pool(parts)


def summarize(scores: RowScores) -> Summary:
    """Boils the scores of a group of rows down. Raises if there are none, since no rate can be given for them."""
    if len(scores) == 0:
        raise ValueError("there are no rows to summarize")
    n_tokens = float(scores.n_tokens.sum())
    return Summary(
        count=len(scores),
        n_passed=int(scores.passed.sum()),
        exact_match=float(scores.passed.float().mean()),
        token_acc=float(scores.n_right.sum()) / n_tokens,
        loss=float(scores.loss_sum.sum()) / n_tokens,
    )


def length_bin(length: int) -> str:
    """The bin that a string of this length is reported in, e.g. "21-30"."""
    if length < 1:
        raise ValueError(f"a string has a length of at least 1, got {length}")
    low = (length - 1) // LENGTH_BIN_WIDTH * LENGTH_BIN_WIDTH + 1
    return f"{low}-{low + LENGTH_BIN_WIDTH - 1}"


def summarize_by_length_bin(scores: RowScores, lengths: Sequence[int]) -> dict[str, Summary]:
    """
    A Summary for each bin of string lengths that has rows, shortest bin first. `lengths` is the length of the string of each row, in the order of the rows.
    A row's length isn't in SftData, so it comes from the examples the rows were made from.
    """
    if len(lengths) != len(scores):
        raise ValueError(f"{len(lengths)} lengths for {len(scores)} rows")
    labels = [length_bin(length) for length in lengths]
    ordered = sorted(set(labels), key=lambda label: int(label.split("-")[0]))
    return {label: summarize(scores.select(torch.tensor([row_label == label for row_label in labels]))) for label in ordered}


def make_validation_metrics_hook(slices: dict[str, SftData], device: str, validation: SftData | None = None, chunk_size: int = 256) -> MetricsHook:
    """
    A hook for train() that scores the model on the validation slices each time the loss is estimated, so that each subtask can be watched on its own.

    Per slice it returns validation/<slice>/exact_match, validation/<slice>/token_acc and validation/<slice>/loss, and over all of their rows validation/exact_match
    and validation/token_acc. (There is no pooled loss: train() already logs validation/loss, and a hook can't return a key that it logs.)
    A source with no slices passes `validation`, the whole validation split, which is then scored for the pooled numbers alone.
    It prints one line per call, so that progress shows in the terminal, and not just in wandb.
    """
    if not slices and validation is None:
        raise ValueError("there is nothing to score: no slices, and no validation split")

    def hook(step: int, model: GPT, _losses: dict[Split, float]) -> dict[str, float]:
        if slices:
            by_slice = {name: score_validation(model, data, device, chunk_size) for name, data in slices.items()}
            # The rows of the slices are the rows of the pooled validation split, so pooling their scores is the score of the split, without another pass.
            pooled = summarize(pool(by_slice.values()))
        else:
            assert validation is not None
            by_slice = {}
            pooled = summarize(score_validation(model, validation, device, chunk_size))

        metrics: dict[str, float] = {}
        for name, scores in by_slice.items():
            summary = summarize(scores)
            metrics[f"validation/{name}/exact_match"] = summary.exact_match
            metrics[f"validation/{name}/token_acc"] = summary.token_acc
            metrics[f"validation/{name}/loss"] = summary.loss
        metrics["validation/exact_match"] = pooled.exact_match
        metrics["validation/token_acc"] = pooled.token_acc

        per_slice = "".join(f"{name} {metrics[f'validation/{name}/exact_match']:.3f} | " for name in by_slice)
        print(f"step={step:04d} exact_match: {per_slice}all {pooled.exact_match:.3f}")
        return metrics

    return hook


def format_report(scores: Mapping[str, RowScores], lengths: Mapping[str, Sequence[int]]) -> str:
    """
    How many rows passed out of how many, for each slice, then within it for each bin of string lengths, and for all the rows together.
    Counts alone, so that nothing in it hides how few rows a figure is made of. `lengths[name]` is the length of the string of each row of that slice.
    """
    name_width = max(len(name) for name in [*scores, "all"])
    lines = ["validation, rows passed / rows:"]
    for name, slice_scores in scores.items():
        total = summarize(slice_scores)
        lines.append(f"  {name:<{name_width}}  {total.n_passed}/{total.count}")
        for label, summary in summarize_by_length_bin(slice_scores, lengths[name]).items():
            lines.append(f"    {label:<{name_width}}  {summary.n_passed}/{summary.count}")
    overall = summarize(pool(scores.values()))
    lines.append(f"  {'all':<{name_width}}  {overall.n_passed}/{overall.count}")
    return "\n".join(lines)


def validation_report(model: GPT, slices: Mapping[str, SftData], lengths: Mapping[str, Sequence[int]], device: str, chunk_size: int = 256) -> str:
    """Scores the model on every slice, and gives the format_report of it. Runs under whatever autocast the caller has set."""
    return format_report({name: score_validation(model, data, device, chunk_size) for name, data in slices.items()}, lengths)
