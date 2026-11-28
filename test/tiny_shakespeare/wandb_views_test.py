from typing import Any

import pytest
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

from src.tiny_shakespeare import wandb_views
from src.tiny_shakespeare.wandb_views import add_run, add_run_to_view, create_view, normalize_hex, set_line_colors, set_run_color, view_id_from_internal_name, view_url


def make_view() -> ws.Workspace:
    return ws.Workspace(entity="e", project="p", name="n")


@pytest.mark.parametrize("raw, expected", [("ff0000", "#ff0000"), ("#FF0000", "#ff0000"), ("afafaf", "#afafaf")])
def test_normalize_hex(raw: str, expected: str):
    assert normalize_hex(raw) == expected


@pytest.mark.parametrize("raw", ["red", "#fff", "gggggg", "ff00", ""])
def test_normalize_hex_rejects_non_hex(raw: str):
    with pytest.raises(ValueError):
        normalize_hex(raw)


def test_view_id_from_internal_name():
    assert view_id_from_internal_name("nw-mgg03pctxen-v") == "mgg03pctxen"
    with pytest.raises(ValueError):
        view_id_from_internal_name("mgg03pctxen")


def test_add_run_tracks_runs_and_filters_to_them():
    view = make_view()
    add_run(view, "b2")
    add_run(view, "a1")
    assert set(view.runset_settings.run_settings) == {"a1", "b2"}
    assert view.runset_settings.filters == "ID in ['a1', 'b2']"


def test_add_run_twice_keeps_color():
    view = make_view()
    add_run(view, "a1")
    set_run_color(view, "a1", "ff0000")
    add_run(view, "a1")
    assert view.runset_settings.run_settings["a1"].color == "#ff0000"


def test_set_run_color_requires_run_in_view():
    with pytest.raises(KeyError):
        set_run_color(make_view(), "a1", "ff0000")


def test_set_run_color_normalizes():
    view = make_view()
    add_run(view, "a1")
    set_run_color(view, "a1", "FF0000")
    assert view.runset_settings.run_settings["a1"].color == "#ff0000"


# --- making a view, and adding a run to one, without calling wandb

class RecordingWorkspace:
    """Stands in for ws.Workspace, so that creating a view doesn't call wandb."""

    created: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any):
        self.created.append(kwargs)
        self._internal_name = "nw-abc123-v"

    def save(self) -> None:
        pass


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    RecordingWorkspace.created = []
    monkeypatch.setattr(wandb_views.ws, "Workspace", RecordingWorkspace)
    return RecordingWorkspace.created


def test_a_view_with_sections_has_just_those_panels(recorded: list[dict[str, Any]]) -> None:
    sections = [ws.Section(name="s")]
    assert create_view("e", "p", "n", sections) == "abc123"
    assert recorded[0]["sections"] is sections and recorded[0]["auto_generate_panels"] is False


def test_a_view_with_no_sections_is_empty_and_gets_a_panel_for_every_key(recorded: list[dict[str, Any]]) -> None:
    assert create_view("e", "p", "n") == "abc123"
    assert recorded[0]["sections"] == [] and recorded[0]["auto_generate_panels"] is True


class FakeApi:
    """Stands in for wandb.Api, which has the run if it's one of `existing`."""

    existing: set[str] = set()
    asked: list[str] = []

    def run(self, path: str) -> None:
        self.asked.append(path)
        if path not in self.existing:
            raise ValueError(f"no run {path}")


def test_add_run_to_view_adds_the_run_saves_the_view_and_counts_its_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeApi.existing, FakeApi.asked = {"e/p/r2"}, []
    monkeypatch.setattr(wandb_views.wandb, "Api", FakeApi)
    view = make_view()
    saved: list[ws.Workspace] = []
    add_run(view, "r1")

    def load_view(entity: str, project: str, view_id: str) -> ws.Workspace:
        assert (entity, project, view_id) == ("e", "p", "v")
        return view

    def save(self: ws.Workspace) -> None:
        saved.append(self)

    monkeypatch.setattr(wandb_views, "load_view", load_view)
    monkeypatch.setattr(ws.Workspace, "save", save)

    assert add_run_to_view("e", "p", "v", "r2") == 2
    assert FakeApi.asked == ["e/p/r2"] and saved == [view]
    assert sorted(view.runset_settings.run_settings) == ["r1", "r2"]


def test_add_run_to_view_raises_for_a_run_that_does_not_exist_before_touching_the_view(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeApi.existing, FakeApi.asked = set(), []
    monkeypatch.setattr(wandb_views.wandb, "Api", FakeApi)

    def load_view(*args: object) -> None:
        pytest.fail("the view must not be loaded")

    monkeypatch.setattr(wandb_views, "load_view", load_view)
    with pytest.raises(ValueError, match="no run e/p/nope"):
        add_run_to_view("e", "p", "v", "nope")


def test_view_url() -> None:
    assert view_url("me", "gpt-experiments", "abc123") == "https://wandb.ai/me/gpt-experiments?nw=abc123"


# --- the colors of the lines of a run

def view_with_plots(*plots: wr.LinePlot) -> ws.Workspace:
    return ws.Workspace(entity="e", project="p", name="n", sections=[ws.Section(name="s", panels=list(plots))])


def line_colors(view: ws.Workspace, plot_index: int) -> dict[str, object]:
    panel = view.sections[0].panels[plot_index]
    assert isinstance(panel, wr.LinePlot)
    return panel.line_colors or {}


def test_a_runs_lines_get_the_colors_of_their_metrics_in_every_plot_that_has_them() -> None:
    view = view_with_plots(wr.LinePlot(y=["m1", "m2"]), wr.LinePlot(y=["m1", "other"]))

    set_line_colors(view, "r1", {"m1": "#111111", "m2": "#222222", "unused": "#333333"})

    # A line is named by its run and its metric. A metric without a color keeps wandb's, and a color for a metric that isn't plotted does nothing.
    assert line_colors(view, 0) == {"r1:m1": "#111111", "r1:m2": "#222222"}
    assert line_colors(view, 1) == {"r1:m1": "#111111"}


def test_coloring_the_lines_of_another_run_keeps_the_colors_of_the_first() -> None:
    view = view_with_plots(wr.LinePlot(y=["m1"], line_colors={"older:m1": "#000000"}))

    set_line_colors(view, "r1", {"m1": "#111111"})
    set_line_colors(view, "r2", {"m1": "#222222"})

    assert line_colors(view, 0) == {"older:m1": "#000000", "r1:m1": "#111111", "r2:m1": "#222222"}


def test_a_plot_with_none_of_the_metrics_is_left_alone() -> None:
    view = view_with_plots(wr.LinePlot(y=["other"]))
    set_line_colors(view, "r1", {"m1": "#111111"})
    assert view.sections[0].panels[0].line_colors is None  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


def test_the_colors_reach_the_saved_panel_in_the_form_that_wandb_expects() -> None:
    view = view_with_plots(wr.LinePlot(y=["m1"]))
    set_line_colors(view, "r1", {"m1": "#fa5252"})
    panel = view.sections[0].panels[0]
    assert isinstance(panel, wr.LinePlot)
    assert panel._to_model().config.override_colors == {"r1:m1": {"color": "#fa5252", "transparentColor": "#fa525219"}}  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]


def test_add_run_to_view_colors_the_runs_lines_in_the_same_save(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeApi.existing, FakeApi.asked = {"e/p/r1"}, []
    monkeypatch.setattr(wandb_views.wandb, "Api", FakeApi)
    view = view_with_plots(wr.LinePlot(y=["m1"]))
    saved: list[ws.Workspace] = []

    def load_view(entity: str, project: str, view_id: str) -> ws.Workspace:
        return view

    def save(self: ws.Workspace) -> None:
        saved.append(self)

    monkeypatch.setattr(wandb_views, "load_view", load_view)
    monkeypatch.setattr(ws.Workspace, "save", save)

    add_run_to_view("e", "p", "v", "r1", line_colors={"m1": "#111111"})

    assert saved == [view], "one save, with the run and its colors"
    assert line_colors(view, 0) == {"r1:m1": "#111111"}
