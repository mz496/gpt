from pathlib import Path
from typing import Iterator

import pytest
import torch
import wandb

from src.tiny_shakespeare.config import TrainConfig
from src.tiny_shakespeare.data.data_source import Split
from src.tiny_shakespeare.data.sft.sft_data_source import SFTDataSource, SftExample
from src.tiny_shakespeare.data.special_tokens import END_OF_TEXT, IM_END, IM_START
from src.tiny_shakespeare.models.gpt import sft as sft_module
from src.tiny_shakespeare.models.gpt.model import GPT, GPTConfig, MultiHeadAttentionParallelHeads, load_checkpoint, save_checkpoint
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer

TRAIN = [SftExample(s, s[::-1]) for s in ["abc", "cab", "bca", "abba", "cbbc", "aabb", "bbaa", "cacb"]]
SLICES = {
    "easy": [SftExample(s, s[::-1]) for s in ["ab", "ba", "ca"]],
    "hard": [SftExample(s, s[::-1]) for s in ["abcab", "cbacb"]],
}
# Few steps: loss is estimated at step 0 and at the last step (see train()).
CONFIG = TrainConfig(train_steps=3, train_seed=1, batch_size=4, learning_rate=1e-3, estimate_loss_seed=2, estimate_loss_iters=2, estimate_loss_chunk_size=2)


class SlicedSource(SFTDataSource):
    """Train examples, a validation split that is the pool of the slices, and the slices."""

    def iter_examples(self, split: Split) -> Iterator[SftExample]:
        if split == "train":
            yield from TRAIN
        else:
            for examples in SLICES.values():
                yield from examples

    def get_validation_slices(self) -> dict[str, list[SftExample]]:
        return SLICES


class UnslicedSource(SFTDataSource):
    """A plain train and validation split, with no slices (get_validation_slices keeps its empty default)."""

    def iter_examples(self, split: Split) -> Iterator[SftExample]:
        yield from TRAIN if split == "train" else SLICES["easy"] + SLICES["hard"]


class FakeRun:
    id = "fake"
    entity = "fake-entity"
    project = "fake-project"

    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, float]]] = []
        self.finished = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((step, dict(metrics)))

    def finish(self) -> None:
        self.finished = True


def report_lines(output: str) -> list[str]:
    """The lines of the report at the end of sft()'s output, without the header, or the run id that is printed after it."""
    after_header = output.split("validation, rows passed / rows:\n")[1].splitlines()
    return [line for line in after_header if not line.startswith("run_id=")]


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """A base checkpoint in a temp working directory (where checkpoints are saved), and wandb and the device faked. Returns the fake wandb run."""
    monkeypatch.chdir(tmp_path)
    tokenizer = CharTokenizer.from_state({"vocab": list("\nabcdefghijklmnopqrstuvwxyz"), "special_tokens": [END_OF_TEXT, IM_START, IM_END]})
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    save_checkpoint(GPT(GPTConfig(vocab_size=len(tokenizer.get_vocab()), n_embd=16, n_head=2, n_layer=1, block_size=64, dropout=0.0)), "base", tokenizer)

    fake = FakeRun()

    def fake_init(*args: object, **kwargs: object) -> FakeRun:
        return fake

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(sft_module, "get_device", lambda: "cpu")
    return fake


def test_sft_logs_each_slice_next_to_the_losses_at_every_loss_estimate(run: FakeRun, capsys: pytest.CaptureFixture[str]) -> None:
    run_id = sft_module.sft("base", SlicedSource(), CONFIG, suffix="test")

    assert [step for step, _ in run.logged] == [0, 2]
    for _, metrics in run.logged:
        assert {"train/loss", "validation/loss"} <= set(metrics)
        for name in SLICES:
            assert {f"validation/{name}/exact_match", f"validation/{name}/token_acc", f"validation/{name}/loss"} <= set(metrics)
        assert {"validation/exact_match", "validation/token_acc"} <= set(metrics)
    # It saved the finetuned model as a new checkpoint, and said so.
    assert Path("checkpoints", run_id, "model.pt").is_file() and run.finished

    # And it ends with the counts of rows passed, per slice, by length bin, and overall.
    report = report_lines(capsys.readouterr().out)
    assert [line.split()[0] for line in report] == ["easy", "1-10", "hard", "1-10", "all"]
    assert [line.split()[1].split("/")[1] for line in report] == ["3", "3", "2", "2", "5"]  # the rows in each, whatever the untrained model got right


def test_a_source_with_no_slices_still_gets_the_pooled_score_of_its_validation_split(run: FakeRun, capsys: pytest.CaptureFixture[str]) -> None:
    sft_module.sft("base", UnslicedSource(), CONFIG)

    for _, metrics in run.logged:
        assert {"validation/exact_match", "validation/token_acc"} <= set(metrics)
        assert not any(key.startswith("validation/") and key.count("/") == 2 for key in metrics), "there are no slices to report"
    # The report is of the validation split as a whole.
    report = report_lines(capsys.readouterr().out)
    assert [line.split()[0] for line in report] == ["validation", "1-10", "all"]
    assert report[0].split()[1].endswith("/5") and report[-1].split()[1].endswith("/5")


def test_sft_finetunes_and_scores_with_the_builtin_attention(run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[GPT] = []

    def spy(run_id: str, device: str, flag_use_pytorch_attention_impl: bool = False) -> GPT:
        model = load_checkpoint(run_id, device, flag_use_pytorch_attention_impl=flag_use_pytorch_attention_impl)
        loaded.append(model)
        return model

    monkeypatch.setattr(sft_module, "load_checkpoint", spy)
    sft_module.sft("base", SlicedSource(), CONFIG)

    (model,) = loaded
    assert [m.flag_use_pytorch_attention_impl for m in model.modules() if isinstance(m, MultiHeadAttentionParallelHeads)] == [True]


# --- putting the run in a wandb view

class ViewCalls:
    """Replaces show_run_in_view in sft(): records how it was called, and whether the wandb run was finished by then. Raises `error` if given one."""

    def __init__(self, run: FakeRun, error: Exception | None = None):
        self.run, self.error = run, error
        self.calls: list[tuple[object, ...]] = []
        self.finished_at_call: list[bool] = []

    def __call__(self, entity: str, project: str, run_id: str, slice_names: list[str], create_name: str | None, view_id: str | None) -> str:
        self.calls.append((entity, project, run_id, slice_names, create_name, view_id))
        self.finished_at_call.append(self.run.finished)
        if self.error is not None:
            raise self.error
        return view_id or "new-view"


def test_with_a_name_for_a_view_the_finished_run_is_put_in_a_new_view_of_the_slices(run: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    view_calls = ViewCalls(run)
    monkeypatch.setattr(sft_module, "show_run_in_view", view_calls)

    sft_module.sft("base", SlicedSource(), CONFIG, create_view="my-view")

    assert view_calls.calls == [("fake-entity", "fake-project", "fake", ["easy", "hard"], "my-view", None)]
    assert view_calls.finished_at_call == [True], "wandb doesn't know the run until it is finished"
    assert "https://wandb.ai/fake-entity/fake-project?nw=new-view" in capsys.readouterr().out


def test_with_the_id_of_a_view_the_run_is_added_to_it(run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    view_calls = ViewCalls(run)
    monkeypatch.setattr(sft_module, "show_run_in_view", view_calls)

    sft_module.sft("base", SlicedSource(), CONFIG, view_id="v7")

    assert view_calls.calls == [("fake-entity", "fake-project", "fake", ["easy", "hard"], None, "v7")]


def test_a_source_with_no_slices_makes_a_view_of_the_pool_alone(run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    view_calls = ViewCalls(run)
    monkeypatch.setattr(sft_module, "show_run_in_view", view_calls)

    sft_module.sft("base", UnslicedSource(), CONFIG, create_view="my-view")

    assert view_calls.calls[0][3] == []


def test_without_a_view_flag_wandb_views_are_left_alone(run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    view_calls = ViewCalls(run)
    monkeypatch.setattr(sft_module, "show_run_in_view", view_calls)

    sft_module.sft("base", SlicedSource(), CONFIG)

    assert view_calls.calls == []


def test_a_failure_to_make_the_view_does_not_fail_the_run(run: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sft_module, "show_run_in_view", ViewCalls(run, error=RuntimeError("wandb is down")))

    run_id = sft_module.sft("base", SlicedSource(), CONFIG, create_view="my-view")

    output = capsys.readouterr().out
    assert Path("checkpoints", run_id, "model.pt").is_file(), "the checkpoint was saved before the view"
    assert "could not put the run in a wandb view" in output and "wandb is down" in output
    assert output.strip().splitlines()[-1] == f"run_id={run_id}"
