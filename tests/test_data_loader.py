"""Loader tests. Most of them are about the checks, not the tidying.

There is almost nothing to clean in this dataset. The work is proving the target is what we
think it is, that the daily index has no holes, and that the loader refuses to proceed if
either is untrue. A loader that silently repairs a broken series produces a model that
trains on nonsense and reports confident numbers.
"""
import unittest

import numpy as np
import pandas as pd

from src import config, data_loader


def _frame(n=10, cnt=None, start="2012-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    cnt = cnt if cnt is not None else np.arange(100, 100 + n, dtype=float)
    return pd.DataFrame(
        {
            "instant": np.arange(1, n + 1),
            "dteday": idx.strftime("%Y-%m-%d"),
            "season": 1,
            "yr": 0,
            "mnth": 1,
            "holiday": 0,
            "weekday": idx.dayofweek,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.2,
            "casual": 40,
            "registered": 60,
            "cnt": cnt,
        }
    )


class TestClean(unittest.TestCase):
    def test_input_is_not_mutated(self):
        raw = _frame()
        before = raw.copy()
        data_loader.clean(raw)
        pd.testing.assert_frame_equal(raw, before)

    def test_date_column_becomes_a_sorted_datetime_index(self):
        shuffled = _frame().sample(frac=1.0, random_state=3)
        out = data_loader.clean(shuffled)
        self.assertIsInstance(out.index, pd.DatetimeIndex)
        self.assertTrue(out.index.is_monotonic_increasing)

    def test_index_name_is_the_date_column(self):
        out = data_loader.clean(_frame())
        self.assertEqual(out.index.name, config.DATE_COL)

    def test_real_data_shape_and_span(self):
        out = data_loader.load_clean()
        self.assertEqual(len(out), 731)
        self.assertEqual(out.index[0].strftime("%Y-%m-%d"), "2011-01-01")
        self.assertEqual(out.index[-1].strftime("%Y-%m-%d"), "2012-12-31")


class TestContiguity(unittest.TestCase):
    def test_a_missing_day_is_refused(self):
        """The failure this catches is silent. A hole makes lag-7 wrong while every
        computation still runs and still returns a plausible number."""
        raw = _frame(10).drop(index=5).reset_index(drop=True)
        with self.assertRaises(ValueError):
            data_loader.assert_contiguous(data_loader.clean(raw))

    def test_real_data_is_contiguous(self):
        data_loader.assert_contiguous(data_loader.load_clean())

    def test_duplicate_dates_are_refused(self):
        raw = pd.concat([_frame(5), _frame(5)], ignore_index=True)
        with self.assertRaises(ValueError):
            data_loader.assert_contiguous(data_loader.clean(raw))


class TestTargetIntegrity(unittest.TestCase):
    def test_identity_is_verified_on_real_data(self):
        data_loader.assert_target_integrity(data_loader.load_clean())

    def test_broken_identity_is_refused(self):
        raw = _frame(5, cnt=np.array([1.0, 2.0, 3.0, 4.0, 999.0]))
        with self.assertRaises(ValueError):
            data_loader.assert_target_integrity(data_loader.clean(raw))

    def test_components_must_still_be_declared_leaky(self):
        """If someone drops casual/registered from LEAKY_COLUMNS the danger is undocumented."""
        for col in ("casual", "registered"):
            self.assertIn(col, config.LEAKY_COLUMNS)

    def test_the_remaining_forbidden_columns_are_declared(self):
        """instant is a row counter and season is determined by the date; neither may reach
        the model, and neither is a target component, so they are checked separately from the
        casual/registered identity above."""
        for col in ("instant", "season"):
            self.assertIn(col, config.LEAKY_COLUMNS)

    def test_the_forbidden_column_policy_is_actually_enforced(self):
        """The list was documentation until this existed. A design matrix built with any
        forbidden column must now raise, rather than being caught by nobody."""
        with self.assertRaises(ValueError):
            data_loader.assert_design_is_clean(["dow", "casual"])
        with self.assertRaises(ValueError):
            data_loader.assert_design_is_clean(["dow", "registered", "instant"])
        data_loader.assert_design_is_clean(["dow", "lag_0", "temp"])


class TestQualityReport(unittest.TestCase):
    def test_report_matches_the_real_dataset(self):
        raw = data_loader.load_raw()
        clean_df = data_loader.load_clean()
        r = data_loader.data_quality_report(raw, clean_df)
        self.assertEqual(r["raw_rows"], 731)
        self.assertEqual(r["clean_rows"], 731)
        self.assertEqual(r["rows_dropped"], 0)
        self.assertEqual(r["span_days"], 731)
        self.assertEqual(r["null_cells"], 0)
        self.assertEqual(r["duplicate_dates"], 0)
        self.assertEqual(r["non_daily_gaps"], 0)
        self.assertEqual(r["target_min"], 22)
        self.assertEqual(r["target_max"], 8714)
        self.assertEqual(r["target_zeros"], 0)

    def test_report_flags_a_broken_series(self):
        raw = _frame(10).drop(index=3).reset_index(drop=True)
        r = data_loader.data_quality_report(raw, data_loader.clean(raw))
        self.assertEqual(r["non_daily_gaps"], 1)
