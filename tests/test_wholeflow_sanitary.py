"""Tests for the sanitary_mode switch in sparsehydro.models.wholeflow.

Covers:
  - WholeFlowModel requires sanitary_mode ("offset" | "profiled"); no frozen mode
  - parameter counts (offset 18 calibratable, profiled 17) and floor_offset_cfs bounds
  - profiled sanitary recovers the true slot-of-week level and respects its clip / mask
  - GwiStoreModelB: same switch; omitted mode = legacy frozen floor
  - G_at_cap saturation flag and pickle round-trip
"""

from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd

from sparsehydro.models.wholeflow import (
    GwiStoreModelB,
    J_INIT_RTK,
    WholeFlowModel,
    _slot_of_week,
)

DT_H = 1.0
AREA = 100.0


def _synthetic(n_days=120, seed=3):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-01-04", periods=n_days * 24, freq="h")   # starts on a Monday
    doy = t.dayofyear.to_numpy() + t.hour.to_numpy() / 24.0
    temp = 12.0 + 12.0 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.0)
    rain = (rng.random(t.size) < 0.02) * rng.exponential(0.3, t.size)
    h = t.hour.to_numpy()
    san = 0.4 * (1.0 + 0.5 * np.sin(2.0 * np.pi * (h - 9) / 24.0))
    san = san * np.where(t.dayofweek.to_numpy() >= 5, 0.9, 1.0)
    df = pd.DataFrame({"datetime": t, "rainfall_in": rain, "temperature_c": temp})
    return df, san


def _joint(df, floor, mode, mask=None):
    m = WholeFlowModel(floor, AREA, DT_H, J_INIT_RTK,
                       sanitary_mode=mode, sanitary_fit_mask=mask)
    m.initialize()
    return m


def _n_cal(model):
    return sum(model.get_scalar_parameter(n).calibrate for n in model.scalar_parameter_names)


class TestSlotOfWeek(unittest.TestCase):
    def test_hourly_slots(self):
        t = pd.date_range("2021-01-04", periods=24 * 14, freq="h")
        slot, n = _slot_of_week(t.values, 1.0)
        self.assertEqual(n, 168)
        np.testing.assert_array_equal(slot, t.dayofweek * 24 + t.hour)

    def test_15min_slots(self):
        t = pd.date_range("2021-01-04", periods=96 * 7, freq="15min")
        slot, n = _slot_of_week(t.values, 0.25)
        self.assertEqual(n, 7 * 96)
        np.testing.assert_array_equal(slot, np.arange(7 * 96))


class TestWholeFlowSanitaryMode(unittest.TestCase):
    def setUp(self):
        self.df, self.san = _synthetic()
        self.n = len(self.df)
        self.mask = np.ones(self.n, dtype=bool)

    def _truth_obs(self, san_true):
        # Truth GWI + RDII from the model itself, with a zero-level offset sanitary.
        m = _joint(self.df, np.zeros(self.n), "offset")
        m.prepare(self.df)
        p = m.predict()
        return san_true + p["gwi_cfs"].to_numpy() + p["rdii_cfs"].to_numpy()

    def test_mode_is_required(self):
        with self.assertRaises(TypeError):
            WholeFlowModel(self.san, AREA, DT_H, J_INIT_RTK)
        with self.assertRaises(ValueError):
            WholeFlowModel(self.san, AREA, DT_H, J_INIT_RTK, sanitary_mode=None)
        with self.assertRaises(ValueError):
            WholeFlowModel(self.san, AREA, DT_H, J_INIT_RTK, sanitary_mode="frozen")
        with self.assertRaises(ValueError):
            WholeFlowModel(self.san, AREA, DT_H, J_INIT_RTK, sanitary_mode="profiled")

    def test_parameter_counts_and_offset_bounds(self):
        off = _joint(self.df, self.san, "offset")
        prof = _joint(self.df, self.san, "profiled", self.mask)
        self.assertEqual(_n_cal(off), 18)
        self.assertEqual(_n_cal(prof), 17)
        self.assertNotIn("floor_offset_cfs", prof.scalar_parameter_names)
        p = off.get_scalar_parameter("floor_offset_cfs")
        self.assertTrue(p.calibrate)
        self.assertAlmostEqual(p.lower_bound, -float(self.san.min()))
        self.assertEqual(p.upper_bound, 0.0)
        self.assertTrue(off.validate() and prof.validate())

    def test_offset_shifts_level_keeps_shape(self):
        m = _joint(self.df, self.san, "offset")
        m.get_scalar_parameter("floor_offset_cfs").value = -0.1
        m.prepare(self.df)
        p = m.predict()
        np.testing.assert_allclose(p["sanitary_cfs"], self.san - 0.1)
        np.testing.assert_allclose(p["total_cfs"],
                                   p["sanitary_cfs"] + p["gwi_cfs"] + p["rdii_cfs"])

    def test_profiled_recovers_true_sanitary(self):
        df = self.df.assign(flow_cfs=self._truth_obs(self.san))
        m = _joint(df, 1.3 * self.san, "profiled", self.mask)     # cap not binding
        m.prepare(df)
        p = m.predict()
        np.testing.assert_allclose(p["sanitary_cfs"], self.san, atol=1e-9)
        np.testing.assert_allclose(p["total_cfs"], df["flow_cfs"], atol=1e-9)
        self.assertEqual(m.sanitary.limit_hits["at_cap"], 0.0)
        self.assertEqual(m.sanitary.limit_hits["empty_slots"], 0)

    def test_profiled_clipped_to_fall_level(self):
        df = self.df.assign(flow_cfs=self._truth_obs(self.san))
        floor = 0.8 * self.san                                     # true level above the cap
        m = _joint(df, floor, "profiled", self.mask)
        m.prepare(df)
        np.testing.assert_allclose(m.predict()["sanitary_cfs"], floor, atol=1e-9)
        self.assertEqual(m.sanitary.limit_hits["at_cap"], 1.0)

    def test_profiled_uses_only_fit_mask(self):
        obs = self._truth_obs(self.san)
        mask = np.zeros(self.n, dtype=bool)
        mask[: self.n // 2] = True
        bumped = obs.copy()
        bumped[~mask] += 5.0                                       # outside the fit window
        outs = []
        for y in (obs, bumped):
            df = self.df.assign(flow_cfs=y)
            m = _joint(df, 1.3 * self.san, "profiled", mask)
            m.prepare(df)
            outs.append(m.predict()["sanitary_cfs"].to_numpy())
        np.testing.assert_allclose(outs[0], outs[1])
        np.testing.assert_allclose(outs[0], self.san, atol=1e-9)

    def test_profiled_missing_flow_column(self):
        m = _joint(self.df, self.san, "profiled", self.mask)
        with self.assertRaises(KeyError):
            m.prepare(self.df)

    def test_g_at_cap(self):
        m = _joint(self.df, self.san, "offset")
        m.prepare(self.df)
        m.get_scalar_parameter("g_max").value = 1e6                # unbounded G never spills
        self.assertEqual(m.predict()["G_at_cap"].sum(), 0.0)
        m.get_scalar_parameter("g_max").value = 0.01
        m.get_scalar_parameter("k_div").value = 0.05
        cap = m.predict()["G_at_cap"].to_numpy()
        self.assertGreater(cap.sum(), 0.0)
        self.assertTrue(set(np.unique(cap)) <= {0.0, 1.0})

    def test_pickle_round_trip(self):
        df = self.df.assign(flow_cfs=self._truth_obs(self.san))
        m = _joint(df, 1.3 * self.san, "profiled", self.mask)
        m.prepare(df)
        m2 = pickle.loads(pickle.dumps(m))
        pd.testing.assert_frame_equal(m.predict(), m2.predict())


class TestGwiStoreModelBSanitaryMode(unittest.TestCase):
    def setUp(self):
        self.df, self.san = _synthetic()
        self.n = len(self.df)

    def _model(self, **kw):
        m = GwiStoreModelB(self.san if "floor" not in kw else kw.pop("floor"),
                           AREA, DT_H, **kw)
        m.initialize()
        return m

    def test_legacy_frozen_when_omitted(self):
        m = self._model()
        self.assertNotIn("floor_offset_cfs", m.scalar_parameter_names)
        self.assertEqual(_n_cal(m), 8)
        m.prepare(self.df)
        p = m.predict()
        np.testing.assert_allclose(p["sanitary_cfs"], self.san)
        np.testing.assert_allclose(p["model_baseflow_cfs"], self.san + p["gwi_cfs"])

    def test_offset_mode(self):
        m = self._model(sanitary_mode="offset")
        self.assertEqual(_n_cal(m), 9)
        m.get_scalar_parameter("floor_offset_cfs").value = -0.05
        m.prepare(self.df)
        np.testing.assert_allclose(m.predict()["sanitary_cfs"], self.san - 0.05)

    def test_profiled_recovers_true_sanitary(self):
        legacy = self._model(floor=np.zeros(self.n))
        legacy.prepare(self.df)
        gwi = legacy.predict()["gwi_cfs"].to_numpy()
        df = self.df.assign(baseflow_cfs=self.san + gwi)
        m = self._model(floor=1.3 * self.san, sanitary_mode="profiled",
                        sanitary_fit_mask=np.ones(self.n, dtype=bool))
        self.assertEqual(_n_cal(m), 8)
        m.prepare(df)
        p = m.predict()
        np.testing.assert_allclose(p["sanitary_cfs"], self.san, atol=1e-9)
        np.testing.assert_allclose(p["model_baseflow_cfs"], df["baseflow_cfs"], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
