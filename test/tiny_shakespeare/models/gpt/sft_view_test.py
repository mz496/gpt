import pytest
import torch
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

from src.tiny_shakespeare import wandb_views
from src.tiny_shakespeare.data.prepare import prepare_sft_validation_slices
from src.tiny_shakespeare.data.sft.reverse_source import ReverseSource
from src.tiny_shakespeare.data.sft.sft_data_source import SftExample
from src.tiny_shakespeare.data.special_tokens import END_OF_TEXT, IM_END, IM_START
from src.tiny_shakespeare.models.gpt.model import GPT, GPTConfig
from src.tiny_shakespeare.models.gpt.sft import POOLED_COLOR, SLICE_COLORS, show_run_in_view, slice_line_colors, slice_sections
from src.tiny_shakespeare.models.gpt.validation_metrics import make_validation_metrics_hook
from src.tiny_shakespeare.tokenizers.char_tokenizer import CharTokenizer

SLICES = ["easy", "hard"]


def line_plots(sections: list[ws.Section]) -> list[wr.LinePlot]:
    return [panel for section in sections for panel in section.panels if isinstance(panel, wr.LinePlot)]


def test_there_is_one_plot_for_each_slice_metric_with_every_slice_on_it() -> None:
    by_slice, _ = slice_sections(SLICES)

    assert by_slice.name == "Validation by slice"
    exact_match, token_acc, loss = line_plots([by_slice])
    assert [plot.title for plot in (exact_match, token_acc, loss)] == ["exact match, by slice", "token accuracy, by slice", "loss, by slice"]
    # The pool (the score of all the slices together) is on the plots that it exists for, and the loss has none, as the hook logs none.
    assert exact_match.y == ["validation/easy/exact_match", "validation/hard/exact_match", "validation/exact_match"]
    assert token_acc.y == ["validation/easy/token_acc", "validation/hard/token_acc", "validation/token_acc"]
    assert loss.y == ["validation/easy/loss", "validation/hard/loss"]
    # Rates are between 0 and 1, so that the plots can be compared at a glance.
    assert exact_match.range_y == token_acc.range_y == (0.0, 1.0)


def test_the_losses_that_train_logs_are_on_a_plot_of_their_own() -> None:
    (plot,) = line_plots([slice_sections(SLICES)[1]])
    assert plot.y == ["train/loss", "validation/loss"]


def test_without_slices_there_are_only_the_pooled_plots() -> None:
    plots = line_plots([slice_sections([])[0]])
    assert [plot.y for plot in plots] == [["validation/exact_match"], ["validation/token_acc"]]


def test_every_metric_it_plots_is_one_that_the_training_actually_logs() -> None:
    # A typo would give an empty plot and no error, so the keys are checked against what the hook really returns.
    tokenizer = CharTokenizer.from_state({"vocab": list("\nabcdefghijklmnopqrstuvwxyz"), "special_tokens": [END_OF_TEXT, IM_START, IM_END]})
    slices = prepare_sft_validation_slices({name: [SftExample("ab", "ba"), SftExample("abc", "cba")] for name in SLICES}, tokenizer, block_size=64)
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] Bad torch type stubs.
    model = GPT(GPTConfig(vocab_size=len(tokenizer.get_vocab()), n_embd=8, n_head=2, n_layer=1, block_size=64, dropout=0.0))
    logged = set(make_validation_metrics_hook(slices, "cpu")(0, model, {"train": 1.0, "validation": 1.0})) | {"train/loss", "validation/loss"}

    plotted = {key for plot in line_plots(slice_sections(SLICES)) for key in plot.y if isinstance(key, str)}
    assert plotted == logged, (sorted(plotted - logged), sorted(logged - plotted))


def test_the_sections_make_a_valid_workspace_for_the_slices_the_data_has() -> None:
    names = sorted(ReverseSource().get_validation_slices())
    workspace = ws.Workspace(entity="e", project="p", name="n", sections=slice_sections(names), auto_generate_panels=False)
    assert len(workspace.sections) == 2 and len(line_plots(workspace.sections)[0].y) == len(names) + 1


# --- putting a run in a view

@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[object, ...]]]:
    """Replaces the calls to wandb, and records them."""
    recorded: list[tuple[str, tuple[object, ...]]] = []

    def fake_create_view(entity: str, project: str, name: str, sections: list[ws.Section]) -> str:
        recorded.append(("create_view", (entity, project, name, [panel.y for panel in line_plots(sections)][0])))
        return "new-view"

    def fake_add_run_to_view(entity: str, project: str, view_id: str, run_id: str, line_colors: dict[str, str] | None = None) -> int:
        recorded.append(("add_run_to_view", (entity, project, view_id, run_id, line_colors)))
        return 1

    monkeypatch.setattr(wandb_views, "create_view", fake_create_view)
    monkeypatch.setattr(wandb_views, "add_run_to_view", fake_add_run_to_view)
    return recorded


def test_a_run_is_put_in_a_new_view_of_the_slices(calls: list[tuple[str, tuple[object, ...]]]) -> None:
    assert show_run_in_view("e", "p", "r1", SLICES, create_name="my-view", view_id=None) == "new-view"
    assert calls == [
        ("create_view", ("e", "p", "my-view", ["validation/easy/exact_match", "validation/hard/exact_match", "validation/exact_match"])),
        ("add_run_to_view", ("e", "p", "new-view", "r1", slice_line_colors(SLICES))),
    ]


def test_a_run_is_put_in_an_existing_view_without_making_one(calls: list[tuple[str, tuple[object, ...]]]) -> None:
    assert show_run_in_view("e", "p", "r1", SLICES, create_name=None, view_id="v7") == "v7"
    assert calls == [("add_run_to_view", ("e", "p", "v7", "r1", slice_line_colors(SLICES)))]


@pytest.mark.parametrize("create_name, view_id", [(None, None), ("name", "v7")])
def test_a_new_view_or_an_existing_one_must_be_given_not_neither_and_not_both(calls: list[tuple[str, tuple[object, ...]]], create_name: str | None, view_id: str | None) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        show_run_in_view("e", "p", "r1", SLICES, create_name=create_name, view_id=view_id)
    assert calls == []


# --- the colors of the lines

def test_each_built_in_slice_has_its_color_on_all_of_its_metrics_and_the_pool_is_gray() -> None:
    colors = slice_line_colors(["palindrome", "random", "repeated_char", "repeated_seq"])

    for name, color in {"palindrome": "#228be6", "random": "#40c057", "repeated_char": "#fa5252", "repeated_seq": "#be4bdb"}.items():
        assert {metric: colors[f"validation/{name}/{metric}"] for metric in ("exact_match", "token_acc", "loss")} == dict.fromkeys(("exact_match", "token_acc", "loss"), color)
    assert colors["validation/exact_match"] == colors["validation/token_acc"] == POOLED_COLOR == "#868e96"
    assert len(colors) == 4 * 3 + 2


def test_a_slice_with_no_color_keeps_the_one_wandb_gives_it_and_the_pool_is_colored_anyway() -> None:
    assert slice_line_colors(["custom"]) == {"validation/exact_match": POOLED_COLOR, "validation/token_acc": POOLED_COLOR}
    assert slice_line_colors([]) == slice_line_colors(["custom"])


def test_every_color_is_a_hex_color_and_every_colored_metric_is_one_that_is_plotted() -> None:
    names = sorted(SLICE_COLORS)
    colors = slice_line_colors(names)
    assert all(len(color) == 7 and color.startswith("#") and int(color[1:], 16) >= 0 for color in colors.values())
    plotted = {key for plot in line_plots(slice_sections(names)) for key in plot.y if isinstance(key, str)}
    assert set(colors) <= plotted, sorted(set(colors) - plotted)
