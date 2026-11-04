"""Tests for sparsehydro.data.utilities — unittest.TestCase."""

from __future__ import annotations

import unittest
import unittest.mock
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sparsehydro.data.utilities import add_time_features, read_tsf


_STORM_TSF = (
    "IDs:\tTC-NB-001 Flow Edited Revised\n"
    "Date/Time\tStormFlow\n"
    "M/d/yyyy\tMGD\n"
    "1/25/2022 12:40:00 PM\t0.10\n"
    "1/25/2022 12:45:00 PM\t0.20\n"
    "1/26/2022 12:40:00 PM\t0.30\n"
    "1/27/2022 12:40:00 PM\t0.40\n"
)


_MULTI_TSF = (
    "IDs:\tTC-NB-001 Flow Edited Revised\n"
    "Date/Time\tStormFlow\tBaseFlow\tRainfall\n"
    "M/d/yyyy\tMGD\tMGD\tin\n"
    "1/25/2022 12:40:00 PM\t0.10\t1.10\t0.01\n"
    "1/25/2022 12:45:00 PM\t0.20\t1.20\t0.02\n"
)


# ===========================================================================
# read_tsf
# ===========================================================================

class TestReadTsf(unittest.TestCase):

    def test_read_from_text(self):
        df = read_tsf(_STORM_TSF)
        self.assertEqual(len(df), 4)
        self.assertEqual(df.index.name, "Date/Time")
        self.assertEqual(list(df.columns), ["StormFlow"])

    def test_read_from_bytes(self):
        df = read_tsf(_STORM_TSF.encode("utf-8"))
        self.assertEqual(len(df), 4)

    def test_read_from_path(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "storm.tsf"
            p.write_text(_STORM_TSF, encoding="utf-8")
            df = read_tsf(p)
            self.assertEqual(len(df), 4)

    def test_date_filtering(self):
        df = read_tsf(_STORM_TSF, start="2022-01-26")
        self.assertEqual(len(df), 2)

    def test_sorted_index(self):
        df = read_tsf(_STORM_TSF)
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_too_few_lines_raises(self):
        with self.assertRaises(ValueError):
            read_tsf("IDs:\ttitle\nDate/Time\tX")

    # -- usecols ------------------------------------------------------------

    def test_usecols_none_reads_every_column(self):
        df = read_tsf(_MULTI_TSF)
        self.assertEqual(list(df.columns), ["StormFlow", "BaseFlow", "Rainfall"])

    def test_usecols_by_name(self):
        df = read_tsf(_MULTI_TSF, usecols=["BaseFlow"])
        self.assertEqual(list(df.columns), ["BaseFlow"])
        self.assertEqual(df.index.name, "Date/Time")
        self.assertAlmostEqual(df["BaseFlow"].iloc[0], 1.10)

    def test_usecols_by_position(self):
        df = read_tsf(_MULTI_TSF, usecols=[1, 3])
        self.assertEqual(list(df.columns), ["StormFlow", "Rainfall"])

    def test_usecols_callable(self):
        df = read_tsf(_MULTI_TSF, usecols=lambda c: c.endswith("Flow"))
        self.assertEqual(list(df.columns), ["StormFlow", "BaseFlow"])

    def test_usecols_keeps_file_order_not_usecols_order(self):
        df = read_tsf(_MULTI_TSF, usecols=["Rainfall", "StormFlow"])
        self.assertEqual(list(df.columns), ["StormFlow", "Rainfall"])

    def test_usecols_time_column_is_implicit(self):
        # Naming the time column explicitly must not duplicate or shift it.
        df = read_tsf(_MULTI_TSF, usecols=["Date/Time", "BaseFlow"])
        self.assertEqual(list(df.columns), ["BaseFlow"])
        self.assertEqual(df.index.name, "Date/Time")

    def test_usecols_empty_keeps_only_the_index(self):
        df = read_tsf(_MULTI_TSF, usecols=[])
        self.assertEqual(list(df.columns), [])
        self.assertEqual(len(df), 2)

    def test_usecols_combines_with_date_filtering(self):
        df = read_tsf(_MULTI_TSF, usecols=["BaseFlow"], start="2022-01-25 12:45")
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), ["BaseFlow"])

    def test_usecols_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            read_tsf(_MULTI_TSF, usecols=["NotAColumn"])

    def test_usecols_out_of_range_position_raises(self):
        with self.assertRaises(ValueError):
            read_tsf(_MULTI_TSF, usecols=[9])

    def test_usecols_bare_string_raises(self):
        with self.assertRaises(ValueError):
            read_tsf(_MULTI_TSF, usecols="BaseFlow")

    def test_usecols_mixed_types_raises(self):
        with self.assertRaises(ValueError):
            read_tsf(_MULTI_TSF, usecols=["BaseFlow", 2])

    # -- datetime parsing ---------------------------------------------------

    def _tsf_with_times(self, *stamps: str) -> str:
        rows = "".join(f"{s}\t{i}.0\n" for i, s in enumerate(stamps))
        return "IDs:\ttitle\nDate/Time\tStormFlow\nM/d/yyyy\tMGD\n" + rows

    def test_parses_us_12_hour_timestamps(self):
        df = read_tsf(self._tsf_with_times("1/25/2022 12:40:00 PM"))
        self.assertEqual(df.index[0], pd.Timestamp("2022-01-25 12:40:00"))

    def test_parses_us_24_hour_timestamps(self):
        df = read_tsf(self._tsf_with_times("1/25/2022 13:40:00"))
        self.assertEqual(df.index[0], pd.Timestamp("2022-01-25 13:40:00"))

    def test_parses_date_only_timestamps(self):
        df = read_tsf(self._tsf_with_times("1/25/2022", "1/26/2022"))
        self.assertEqual(df.index[0], pd.Timestamp("2022-01-25"))
        self.assertEqual(len(df), 2)

    def test_month_day_order_is_us_not_day_first(self):
        # 3/4/2022 must be 4 March, not 3 April.
        df = read_tsf(self._tsf_with_times("3/4/2022 01:00:00"))
        self.assertEqual(df.index[0], pd.Timestamp("2022-03-04 01:00:00"))

    def test_falls_back_for_non_us_timestamps(self):
        df = read_tsf(self._tsf_with_times("2022-01-25 12:40:00"))
        self.assertEqual(df.index[0], pd.Timestamp("2022-01-25 12:40:00"))

    def test_unparseable_rows_are_dropped(self):
        df = read_tsf(
            self._tsf_with_times("1/25/2022 12:40:00 PM", "not a date")
        )
        self.assertEqual(len(df), 1)

    def test_explicit_format_path_is_used_for_us_timestamps(self):
        # The whole point of the explicit-format attempt is to avoid pandas'
        # per-element dateutil fallback, so assert a format= was supplied and
        # that no formatless parse of the time column was needed.
        import sparsehydro.data.utilities as utilities

        real = pd.to_datetime
        formats: list[str | None] = []

        def spy(arg, *args, **kwargs):
            formats.append(kwargs.get("format"))
            return real(arg, *args, **kwargs)

        with unittest.mock.patch.object(utilities.pd, "to_datetime", spy):
            read_tsf(_STORM_TSF)

        self.assertEqual(formats, ["%m/%d/%Y %I:%M:%S %p"])

    def test_fallback_path_is_used_for_iso_timestamps(self):
        import sparsehydro.data.utilities as utilities

        real = pd.to_datetime
        formats: list[str | None] = []

        def spy(arg, *args, **kwargs):
            formats.append(kwargs.get("format"))
            return real(arg, *args, **kwargs)

        with unittest.mock.patch.object(utilities.pd, "to_datetime", spy):
            read_tsf(self._tsf_with_times("2022-01-25 12:40:00"))

        # All three explicit formats miss, then the formatless fallback runs.
        self.assertIsNone(formats[-1])

    def test_explicit_format_path_emits_no_inference_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            read_tsf(_STORM_TSF)


# ===========================================================================
# add_time_features
# ===========================================================================

class TestAddTimeFeatures(unittest.TestCase):

    def _indexed_frame(self) -> pd.DataFrame:
        # 2022-01-03 is a Monday (dayofweek == 0).
        idx = pd.DatetimeIndex(
            [
                "2022-01-03 00:00:00",  # Monday midnight
                "2022-01-03 06:00:00",  # Monday 06:00
                "2022-01-04 12:30:00",  # Tuesday 12:30
            ],
            name="Date/Time",
        )
        return pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=idx)

    def test_no_flags_leaves_columns_unchanged(self):
        df = self._indexed_frame()
        out = add_time_features(df)
        self.assertEqual(list(out.columns), ["value"])

    def test_does_not_mutate_input(self):
        df = self._indexed_frame()
        add_time_features(df, hour_of_day=True)
        self.assertEqual(list(df.columns), ["value"])

    def test_hour_of_day_from_index(self):
        out = add_time_features(self._indexed_frame(), hour_of_day=True)
        self.assertIn("hour_of_day", out.columns)
        self.assertAlmostEqual(out["hour_of_day"].iloc[0], 0.0)
        self.assertAlmostEqual(out["hour_of_day"].iloc[1], 6.0)
        self.assertAlmostEqual(out["hour_of_day"].iloc[2], 12.5)

    def test_day_of_week_from_index(self):
        out = add_time_features(self._indexed_frame(), day_of_week=True)
        self.assertAlmostEqual(out["day_of_week"].iloc[0], 0.0)
        self.assertAlmostEqual(out["day_of_week"].iloc[1], 6.0 / 24.0)
        self.assertAlmostEqual(out["day_of_week"].iloc[2], 1.0 + 12.5 / 24.0)

    def test_day_of_year_from_index(self):
        out = add_time_features(self._indexed_frame(), day_of_year=True)
        self.assertAlmostEqual(out["day_of_year"].iloc[0], 3.0)
        self.assertAlmostEqual(out["day_of_year"].iloc[2], 4.0 + 12.5 / 24.0)

    def test_all_flags_together(self):
        out = add_time_features(
            self._indexed_frame(),
            hour_of_day=True,
            day_of_week=True,
            day_of_year=True,
        )
        for col in ("hour_of_day", "day_of_week", "day_of_year"):
            self.assertIn(col, out.columns)

    def test_named_datetime_column(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2022-01-03 06:00:00", "2022-01-04 12:30:00"]
                ),
                "value": [1.0, 2.0],
            }
        )
        out = add_time_features(df, hour_of_day=True, datetime_col="datetime")
        self.assertAlmostEqual(out["hour_of_day"].iloc[0], 6.0)
        self.assertAlmostEqual(out["hour_of_day"].iloc[1], 12.5)

    def test_default_datetime_column(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2022-01-03 06:00:00"]),
                "value": [1.0],
            }
        )
        out = add_time_features(df, hour_of_day=True)
        self.assertAlmostEqual(out["hour_of_day"].iloc[0], 6.0)

    def test_missing_named_column_raises(self):
        df = self._indexed_frame()
        with self.assertRaises(KeyError):
            add_time_features(df, hour_of_day=True, datetime_col="nope")

    def test_no_datetime_source_raises(self):
        df = pd.DataFrame({"value": [1.0, 2.0]})
        with self.assertRaises(KeyError):
            add_time_features(df, hour_of_day=True)


if __name__ == "__main__":
    unittest.main()
