"""Tests for sparsehydro.visualization.display (inline-display thinning + val shading).

Covers:
- thin_for_display bounds the point count of long scatter / scattergl traces
- the global max / min of every thinned trace survive (envelope, not stride)
- the original figure is not modified
- traces sharing x are thinned at identical indices (stacked fills stay aligned)
- NaN gaps survive; short traces and non-scatter traces (bar, parcoords) are untouched
- max_points=None returns an unthinned copy
- per-point attributes (customdata, marker.color) are subsampled with x / y
- figures re-loaded from JSON (base64 typed arrays) are thinned too
- add_val_vrects draws one shape per contiguous validation run
"""

import json
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from sparsehydro.visualization import add_val_vrects as add_val_vrects_pkg
from sparsehydro.visualization import thin_for_display as thin_pkg
from sparsehydro.visualization.display import add_val_vrects, thin_for_display


def _series(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2022-01-01", periods=n, freq="15min")
    y = np.cumsum(rng.normal(size=n))
    y[12_345] = y.max() + 50.0      # isolated spike
    y[23_456] = y.min() - 50.0      # isolated trough
    return t, y


class TestThinForDisplay(unittest.TestCase):
    def test_exported_from_package(self):
        self.assertIs(thin_pkg, thin_for_display)
        self.assertIs(add_val_vrects_pkg, add_val_vrects)

    def test_bounds_points_and_keeps_extremes(self):
        t, y = _series()
        for kind in (go.Scatter, go.Scattergl):
            with self.subTest(kind=kind.__name__):
                fig = go.Figure(kind(x=t, y=y, mode="lines"))
                thin = thin_for_display(fig, max_points=2000)
                ty = np.asarray(thin.data[0].y, dtype=float)
                self.assertLessEqual(ty.size, 2000)
                self.assertGreater(ty.size, 500)
                self.assertEqual(ty.max(), y.max())
                self.assertEqual(ty.min(), y.min())
                self.assertEqual(len(thin.data[0].x), ty.size)
                self.assertEqual(pd.Timestamp(thin.data[0].x[0]), t[0])
                self.assertEqual(pd.Timestamp(thin.data[0].x[-1]), t[-1])

    def test_original_not_modified(self):
        t, y = _series()
        fig = go.Figure(go.Scattergl(x=t, y=y))
        thin_for_display(fig, max_points=1000)
        self.assertEqual(len(fig.data[0].y), y.size)
        np.testing.assert_array_equal(np.asarray(fig.data[0].y, dtype=float), y)

    def test_shared_x_traces_stay_aligned(self):
        t, y = _series()
        y2 = np.roll(y, 5000)
        fig = go.Figure([go.Scatter(x=t, y=y, fill="tozeroy"),
                         go.Scatter(x=t, y=y2, fill="tonexty")])
        thin = thin_for_display(fig, max_points=3000)
        x0 = pd.to_datetime(np.asarray(thin.data[0].x))
        x1 = pd.to_datetime(np.asarray(thin.data[1].x))
        self.assertTrue((x0 == x1).all())
        self.assertLessEqual(len(x0), 3000)
        self.assertEqual(np.max(np.asarray(thin.data[1].y, float)), y2.max())

    def test_index_x_and_nan_gaps(self):
        _, y = _series()
        y[30_000:30_500] = np.nan
        fig = go.Figure(go.Scatter(y=y))
        thin = thin_for_display(fig, max_points=2000)
        ty = np.asarray(thin.data[0].y, dtype=float)
        self.assertTrue(np.isnan(ty).any())
        self.assertEqual(np.nanmax(ty), np.nanmax(y))

    def test_short_and_non_scatter_untouched(self):
        t, y = _series()
        fig = go.Figure([go.Scatter(x=t[:100], y=y[:100]),
                         go.Bar(x=t, y=np.abs(y)),
                         go.Scatter(x=[0, 1], y=[0, 1])])
        thin = thin_for_display(fig, max_points=1000)
        self.assertEqual(len(thin.data[0].y), 100)
        self.assertEqual(len(thin.data[1].y), y.size)
        self.assertEqual(len(thin.data[2].y), 2)

    def test_parcoords_passes_through(self):
        fig = go.Figure(go.Parcoords(dimensions=[dict(values=[1, 2, 3], label="a")]))
        thin = thin_for_display(fig)
        self.assertEqual(thin.data[0].type, "parcoords")

    def test_per_point_attributes_follow(self):
        t, y = _series()
        cd = np.arange(y.size)
        fig = go.Figure(go.Scattergl(x=t, y=y, mode="markers", customdata=cd,
                                     marker=dict(color=y)))
        thin = thin_for_display(fig, max_points=1000)
        tr = thin.data[0]
        kept = np.asarray(tr.customdata).ravel().astype(int)
        np.testing.assert_array_equal(np.asarray(tr.y, float), y[kept])
        np.testing.assert_array_equal(np.asarray(tr.marker.color, float), y[kept])

    def test_layout_and_shapes_kept(self):
        t, y = _series()
        fig = go.Figure(go.Scatter(x=t, y=y))
        fig.add_vrect(x0=t[10], x1=t[20])
        fig.update_layout(title="hello")
        thin = thin_for_display(fig, max_points=1000)
        self.assertEqual(len(thin.layout.shapes), 1)
        self.assertEqual(thin.layout.title.text, "hello")

    def test_bad_max_points(self):
        with self.assertRaises(ValueError):
            thin_for_display(go.Figure(), max_points=2)

    def test_none_disables_thinning(self):
        t, y = _series()
        fig = go.Figure(go.Scattergl(x=t, y=y))
        full = thin_for_display(fig, max_points=None)
        self.assertIsNot(full, fig)
        self.assertEqual(len(full.data[0].y), y.size)

    def test_figure_reloaded_from_json(self):
        # pio.to_json stores numpy data as base64 typed arrays ({"dtype", "bdata"}).
        t, y = _series()
        fig = go.Figure([go.Scattergl(x=t, y=y), go.Scattergl(x=t, y=-y)])
        reloaded = go.Figure(json.loads(pio.to_json(fig)))
        thin = thin_for_display(reloaded, max_points=2000)
        for tr, ref in zip(thin.data, (y, -y)):
            ty = np.asarray(tr.y, dtype=float)
            self.assertLessEqual(ty.size, 2000)
            self.assertEqual(ty.max(), ref.max())
            self.assertEqual(ty.min(), ref.min())
            self.assertEqual(len(tr.x), ty.size)
        self.assertLess(len(pio.to_json(thin)), len(pio.to_json(reloaded)) / 5)


class TestAddValVrects(unittest.TestCase):
    def test_one_shape_per_run(self):
        t = pd.date_range("2022-01-01", periods=20, freq="h")
        m = np.zeros(20, bool)
        m[2:5] = True
        m[9:10] = True
        m[15:] = True
        fig = add_val_vrects(go.Figure(), t, m)
        self.assertEqual(len(fig.layout.shapes), 3)
        s = fig.layout.shapes[0]
        self.assertEqual(pd.Timestamp(s.x0), t[2])
        self.assertEqual(pd.Timestamp(s.x1), t[4])
        self.assertEqual(s.fillcolor, "rgba(255,127,14,0.07)")

    def test_kwargs_override_and_length_check(self):
        t = pd.date_range("2022-01-01", periods=5, freq="h")
        fig = add_val_vrects(go.Figure(), t, [0, 1, 1, 0, 0], fillcolor="red")
        self.assertEqual(fig.layout.shapes[0].fillcolor, "red")
        with self.assertRaises(ValueError):
            add_val_vrects(go.Figure(), t, [True, False])


if __name__ == "__main__":
    unittest.main()
