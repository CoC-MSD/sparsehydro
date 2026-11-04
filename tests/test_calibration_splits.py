"""Tests for sparsehydro.calibration.splits (spin-up warmup + train/validation split).

Covers:
- SplitConfig validation (bad modes / knobs raise) and step conversion
- prepend_spinup: wrap copies the tail, repeat_head the head, none is a no-op;
  prepended timestamps continue the grid; warmup >= record falls back with a warning
- wrap_seasonal_slip_days: whole-year records slip ~0 (incl. the 4-year case the old
  ``365 - record_days`` formula reported as -1096 d), short records slip their length
- build_masks: masks partition the record; continuous fraction; stratified val share
  ~ 1/val_every; dry-snapped edges have no rain in the look-back
- label_events_by_split: majority labels and straddle count
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from sparsehydro.calibration import splits as sp
from sparsehydro.calibration.splits import (
    SplitConfig,
    build_masks,
    dry_edge_flags,
    label_events_by_split,
    prepend_spinup,
    wrap_seasonal_slip_days,
)


def _frame(days, dt_hours=1.0, start="2022-01-01", seed=0):
    n = int(round(days * 24 / dt_hours))
    rng = np.random.default_rng(seed)
    rain = np.where(rng.random(n) < 0.03, rng.random(n), 0.0)
    return pd.DataFrame({
        "datetime": pd.date_range(start, periods=n, freq=pd.Timedelta(hours=dt_hours)),
        "rainfall_in": rain,
        "flow_cfs": np.arange(n, dtype=float),
    })


class TestSplitConfig(unittest.TestCase):
    def test_bad_modes_raise(self):
        with self.assertRaises(ValueError):
            SplitConfig(dt_hours=1.0, spinup_mode="tail")
        with self.assertRaises(ValueError):
            SplitConfig(dt_hours=1.0, split_mode="random")

    def test_bad_knobs_raise(self):
        for kw in ({"dt_hours": 0.0}, {"dt_hours": 1.0, "train_frac": 0.0},
                   {"dt_hours": 1.0, "val_every": 1}, {"dt_hours": 1.0, "block_days": 0},
                   {"dt_hours": 1.0, "warmup_days": -1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                SplitConfig(**kw)

    def test_steps_follow_timestep(self):
        self.assertEqual(SplitConfig(dt_hours=1.0, warmup_days=91).warmup_steps, 91 * 24)
        self.assertEqual(SplitConfig(dt_hours=0.25, warmup_days=91).warmup_steps, 91 * 96)
        self.assertEqual(SplitConfig(dt_hours=0.25).steps(hours=48), 192)


class TestPrependSpinup(unittest.TestCase):
    def test_wrap_copies_tail_and_continues_grid(self):
        df = _frame(400)
        cfg = SplitConfig(dt_hours=1.0, warmup_days=28, spinup_mode="wrap")
        out, n_spin = prepend_spinup(df, cfg, verbose=False)
        self.assertEqual(n_spin, 28 * 24)
        self.assertEqual(len(out), len(df) + n_spin)
        np.testing.assert_array_equal(out["flow_cfs"].iloc[:n_spin], df["flow_cfs"].iloc[-n_spin:])
        np.testing.assert_array_equal(out["flow_cfs"].iloc[n_spin:], df["flow_cfs"])
        steps = out["datetime"].diff().dropna().unique()
        self.assertEqual(list(steps), [pd.Timedelta(hours=1)])
        self.assertEqual(out["datetime"].iloc[n_spin], df["datetime"].iloc[0])

    def test_repeat_head_copies_head(self):
        df = _frame(200, dt_hours=0.25)
        cfg = SplitConfig(dt_hours=0.25, warmup_days=14, spinup_mode="repeat_head")
        out, n_spin = prepend_spinup(df, cfg, verbose=False)
        self.assertEqual(n_spin, 14 * 96)
        np.testing.assert_array_equal(out["flow_cfs"].iloc[:n_spin], df["flow_cfs"].iloc[:n_spin])
        self.assertTrue((out["datetime"].diff().dropna() == pd.Timedelta(minutes=15)).all())

    def test_none_and_zero_warmup_are_noops(self):
        df = _frame(100)
        for cfg in (SplitConfig(dt_hours=1.0, spinup_mode="none"),
                    SplitConfig(dt_hours=1.0, warmup_days=0)):
            out, n_spin = prepend_spinup(df, cfg, verbose=False)
            self.assertEqual(n_spin, 0)
            pd.testing.assert_frame_equal(out, df)

    def test_spinup_longer_than_record_falls_back(self):
        df = _frame(30)
        cfg = SplitConfig(dt_hours=1.0, warmup_days=60)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out, n_spin = prepend_spinup(df, cfg, verbose=False)
        self.assertEqual(n_spin, 0)
        self.assertEqual(len(out), len(df))
        self.assertTrue(any("falling back" in str(x.message) for x in w))

    def test_input_not_modified(self):
        df = _frame(100)
        before = df.copy()
        prepend_spinup(df, SplitConfig(dt_hours=1.0, warmup_days=7), verbose=False)
        pd.testing.assert_frame_equal(df, before)


class TestSeasonalSlip(unittest.TestCase):
    def test_whole_years_slip_zero(self):
        self.assertAlmostEqual(wrap_seasonal_slip_days(365.25), 0.0)
        self.assertAlmostEqual(wrap_seasonal_slip_days(1461.0), 0.0)   # old formula: -1096
        self.assertAlmostEqual(wrap_seasonal_slip_days(365.0), -0.25)

    def test_partial_years(self):
        self.assertAlmostEqual(wrap_seasonal_slip_days(300.0), -65.25)
        self.assertAlmostEqual(wrap_seasonal_slip_days(400.0), 34.75)
        self.assertAlmostEqual(wrap_seasonal_slip_days(100.0), 100.0)


class TestBuildMasks(unittest.TestCase):
    def _assert_partition(self, m, n):
        self.assertEqual(len(m.warmup), n)
        total = m.warmup.astype(int) + m.train.astype(int) + m.val.astype(int)
        np.testing.assert_array_equal(total, np.ones(n, dtype=int))
        np.testing.assert_array_equal(m.scored, m.train | m.val)

    def test_unpacks_as_four_tuple(self):
        df = _frame(200)
        warmup, train, val, w = build_masks(df["datetime"], SplitConfig(dt_hours=1.0, warmup_days=7),
                                            verbose=False)
        self.assertEqual(w, 7 * 24)
        self.assertTrue(warmup[: w].all() and not warmup[w:].any())

    def test_continuous_fraction(self):
        df = _frame(300)
        cfg = SplitConfig(dt_hours=1.0, warmup_days=10, split_mode="continuous", train_frac=0.7)
        m = build_masks(df["datetime"], cfg, warmup_steps=240, verbose=False)
        self._assert_partition(m, len(df))
        n_post = len(df) - 240
        self.assertEqual(int(m.train.sum()), round(0.7 * n_post))
        first_val = int(np.flatnonzero(m.val)[0])
        self.assertTrue(m.val[first_val:].all())   # one contiguous tail

    def test_stratified_share_without_snapping(self):
        df = _frame(4 * 365)
        cfg = SplitConfig(dt_hours=1.0, warmup_days=91, block_days=21, val_every=4,
                          snap_edges_to_dry=False)
        m = build_masks(df["datetime"], cfg, verbose=False)
        self._assert_partition(m, len(df))
        share = m.val.sum() / m.scored.sum()
        self.assertAlmostEqual(share, 0.25, delta=0.02)
        # Validation blocks are spread across every calendar quarter.
        q = df["datetime"].dt.quarter.to_numpy()
        self.assertEqual(set(q[m.val]), {1, 2, 3, 4})

    def test_dry_snapped_edges_have_dry_lookback(self):
        df = _frame(2 * 365, seed=3)
        cfg = SplitConfig(dt_hours=1.0, warmup_days=28, block_days=21, val_every=4,
                          edge_dry_hours=24.0)
        rain = df["rainfall_in"].to_numpy()
        m = build_masks(df["datetime"], cfg, rain=rain, verbose=False)
        self._assert_partition(m, len(df))
        dry = dry_edge_flags(rain, cfg)
        scored_lab = np.where(m.val, 2, np.where(m.train, 1, 0))
        starts = np.flatnonzero(np.diff(scored_lab) != 0) + 1
        starts = starts[starts > m.warmup_steps]
        self.assertGreater(starts.size, 10)
        self.assertGreater(dry[starts].mean(), 0.9)   # snapping can fail only past 5 d
        self.assertAlmostEqual(m.val.sum() / m.scored.sum(), 0.25, delta=0.05)

    def test_dry_edge_flags_lookback(self):
        cfg = SplitConfig(dt_hours=1.0, edge_dry_hours=3.0)
        rain = np.zeros(10)
        rain[4] = 0.5
        np.testing.assert_array_equal(
            dry_edge_flags(rain, cfg),
            [True, True, True, True, False, False, False, True, True, True])

    def test_rain_length_mismatch_raises(self):
        df = _frame(100)
        with self.assertRaises(ValueError):
            build_masks(df["datetime"], SplitConfig(dt_hours=1.0, warmup_days=7),
                        rain=np.zeros(10), verbose=False)

    def test_large_warmup_warns(self):
        df = _frame(60)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = build_masks(df["datetime"], SplitConfig(dt_hours=1.0, warmup_days=30), verbose=False)
        self._assert_partition(m, len(df))
        self.assertTrue(any("over 1/3" in str(x.message) for x in w))


class TestLabelEvents(unittest.TestCase):
    def test_labels_and_straddle(self):
        n = 100
        warmup = np.zeros(n, bool)
        warmup[:10] = True
        val = np.zeros(n, bool)
        val[50:70] = True
        train = ~warmup & ~val

        class Ev:
            def __init__(self, s, e):
                self.start_idx, self.end_idx = s, e

        events = [Ev(2, 6), Ev(20, 25), Ev(52, 60), Ev(42, 52), (90, 95)]
        labels, n_straddle = label_events_by_split(events, warmup, train, val,
                                                   pre_steps=1, tail_steps=2)
        self.assertEqual(list(labels), ["warmup", "train", "val", "train", "train"])
        self.assertEqual(labels.dtype, object)
        self.assertEqual(n_straddle, 1)

    def test_split_labels_constant(self):
        self.assertEqual(sp.SPLIT_LABELS, ("warmup", "train", "val"))


if __name__ == "__main__":
    unittest.main()
