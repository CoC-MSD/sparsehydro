"""Tests for sparsehydro.viewer: data helpers, CLI argv, and a Streamlit smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparsehydro.models.unithydrograph import PeakTailUH
from sparsehydro.viewer import cli
from sparsehydro.viewer.data import (
    ViewerResult,
    content_hash,
    events_in_window,
    filter_sort_events,
    fit_details,
    fit_index,
    fitted_model,
    global_event_table,
    kernel_display_steps,
    member_table,
    read_rain_stormflow,
    run_analysis,
    season_of,
    slice_savgol,
    span_nse,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO / "data" / "TC-NB-001" / "rain_stormflow.csv"
APP_PATH = REPO / "sparsehydro" / "viewer" / "app.py"

pytestmark = pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample dataset not present")


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Two weeks of the in-repo TC-NB-001 record (a handful of events; fits in ~1 s)."""
    full = read_rain_stormflow(SAMPLE_CSV)
    return full[(full["datetime"] >= "2022-03-01") & (full["datetime"] < "2022-03-15")].reset_index(drop=True)


@pytest.fixture(scope="module")
def result(sample_df) -> ViewerResult:
    return run_analysis(sample_df, PeakTailUH)


@pytest.fixture(scope="module")
def sample_csv(sample_df, tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("viewer") / "rain_stormflow.csv"
    sample_df.to_csv(p, index=False)
    return p


class TestReadAndHash:
    def test_read_csv_path_and_bytes(self, sample_df, tmp_path):
        p = tmp_path / "rs.csv"
        sample_df.assign(extra=1).to_csv(p, index=False)
        from_path = read_rain_stormflow(p)
        pd.testing.assert_frame_equal(from_path, read_rain_stormflow(p.read_bytes()))
        assert list(from_path.columns) == ["datetime", "rain", "stormflow"]
        assert pd.api.types.is_datetime64_any_dtype(from_path["datetime"])

    def test_read_missing_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"datetime": ["2024-01-01"], "rain": [0.0]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="stormflow"):
            read_rain_stormflow(p)

    def test_content_hash_stable(self):
        assert content_hash(b"abc") == content_hash(b"abc") != content_hash(b"abd")

    @pytest.mark.parametrize("month,season", [(1, "Winter"), (4, "Spring"), (7, "Summer"), (10, "Fall"), (12, "Winter")])
    def test_season_of(self, month, season):
        assert season_of(pd.Timestamp(2024, month, 15)) == season


class TestRunAnalysis:
    def test_detects_and_fits(self, result):
        assert result.global_events and result.sub_events
        assert result.summary.events
        assert result.dt_hours == pytest.approx(5 / 60)

    def test_fitted_events_are_global_events(self, result):
        ids = {g.global_id for g in result.global_events}
        assert {g.global_id for g in result.summary.events} <= ids


class TestTables:
    def test_global_event_table(self, result):
        table = global_event_table(result)
        assert set(table["global_id"]) == {g.global_id for g in result.global_events}
        assert table["n_sub"].sum() == len(result.sub_events)
        assert table["start"].is_monotonic_increasing
        assert (table["window_start"] <= table["start"]).all() and (table["end"] <= table["window_end"]).all()
        assert table.loc[table["fitted"], "nse"].notna().all()

    def test_filter_sort(self, result):
        table = global_event_table(result)
        assert filter_sort_events(table, sort_by="total_rain", descending=True)["total_rain"].is_monotonic_decreasing
        assert filter_sort_events(table, seasons=["Summer"]).empty  # sample is in March
        assert len(filter_sort_events(table, seasons=["Spring"])) == len(table)
        assert filter_sort_events(table, min_rain=table["total_rain"].max() + 1).empty

    def test_member_table(self, result):
        gid = result.global_events[0].global_id
        expected = [s for s in result.sub_events if s.global_id == gid]
        assert len(member_table(result, gid)) == len(expected)

    def test_fit_details(self, result):
        gid = result.summary.events[0].global_id
        details = fit_details(result, gid)
        assert len(details) == 1 and details.loc[0, "global_id"] == gid
        assert {"nash_sutcliffe", "A", "w"} <= set(details.columns)
        assert fit_details(result, -1).empty

    def test_span_nse_perfect_and_constant(self, result):
        g = result.summary.events[0]
        pred = result.summary.global_predicted.copy()
        obs = result.summary.global_observed
        perfect = ViewerResult(result.data, result.global_events, result.sub_events, result.savgol,
                               type(result.summary)(**{**result.summary.__dict__,
                                                       "global_predicted": pred.assign(Q_pred=obs["stormflow"].values)}),
                               result.dt_hours)
        assert span_nse(perfect, g.start_datetime, g.end_datetime) == pytest.approx(1.0)
        assert np.isnan(span_nse(result, pd.Timestamp("1990-01-01"), pd.Timestamp("1990-01-02")))


class TestModelsAndWindows:
    def test_fit_index(self, result):
        g = result.summary.events[-1]
        assert result.summary.events[fit_index(result, g.global_id)] is g
        assert fit_index(result, -1) is None

    def test_fitted_model_reproduces_parameters(self, result):
        model = fitted_model(result.summary, 0, PeakTailUH)
        cal = result.summary.calibration_results[0]
        for name, val in zip(cal.param_names, cal.pareto_X[0]):
            assert model.read_flat_parameter(name) == pytest.approx(val)
        kernel = model.get_kernel(result.dt_hours)
        assert kernel.size and np.all(np.isfinite(kernel))

    def test_slice_savgol(self, result):
        g = result.global_events[0]
        sl = slice_savgol(result.savgol, g.window_start_datetime, g.window_end_datetime)
        t = pd.to_datetime(sl.datetime)
        assert t.min() >= g.window_start_datetime and t.max() <= g.window_end_datetime
        assert len(sl.smoothed) == len(sl.datetime) == len(sl.curvature) == len(sl.windows)
        assert np.all((sl.peak_idxs >= 0) & (sl.peak_idxs < len(sl.datetime)))

    def test_events_in_window(self, result):
        g = result.global_events[0]
        subs = events_in_window(result.sub_events, g.start_datetime, g.end_datetime)
        assert {s.sub_id for s in result.sub_events if s.global_id == g.global_id} <= {s.sub_id for s in subs}

    def test_kernel_display_steps(self):
        fast = np.r_[np.ones(10), np.zeros(90)]
        slow = np.r_[np.ones(50), np.zeros(50)]
        assert kernel_display_steps([fast, slow], mass=1.0) == 50
        assert kernel_display_steps([np.zeros(30)], min_steps=12) == 12


class TestCli:
    def test_no_args(self):
        assert cli.build_argv([]) == ["streamlit", "run", str(cli.APP_PATH)]

    def test_csv_forwarded_after_double_dash(self, tmp_path):
        p = tmp_path / "x.csv"
        argv = cli.build_argv([str(p), "--server.port", "9000"])
        assert argv == ["streamlit", "run", str(cli.APP_PATH), "--server.port", "9000", "--", str(p.resolve())]

    def test_app_path_exists(self):
        assert cli.APP_PATH.is_file()


class TestAppSmoke:
    def test_app_renders_and_navigates(self, sample_csv, monkeypatch):
        testing = pytest.importorskip("streamlit.testing.v1")
        monkeypatch.setenv("SPARSEHYDRO_VIEWER_CSV", str(sample_csv))
        at = testing.AppTest.from_file(str(APP_PATH), default_timeout=120)
        at.run()
        assert not at.exception, at.exception
        assert " — G" in at.title[0].value
        assert len(at.tabs) == 4
        assert at.dataframe

        options = list(at.sidebar.selectbox(key="gev").options)
        if len(options) > 1:
            next_btn = [b for b in at.sidebar.button if b.label == "▶"][0]
            next_btn.click().run()
            assert not at.exception, at.exception
            assert at.sidebar.selectbox(key="gev").value != options[0]

    def test_bad_path_shows_error(self, tmp_path, monkeypatch):
        testing = pytest.importorskip("streamlit.testing.v1")
        monkeypatch.setenv("SPARSEHYDRO_VIEWER_CSV", str(tmp_path / "missing.csv"))
        at = testing.AppTest.from_file(str(APP_PATH), default_timeout=60)
        at.run()
        assert not at.exception
        assert at.sidebar.error
