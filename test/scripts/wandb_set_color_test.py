import pytest
import wandb_workspaces.workspaces as ws

from scripts.wandb_views import add_run, normalize_hex, set_run_color, view_id_from_internal_name


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
