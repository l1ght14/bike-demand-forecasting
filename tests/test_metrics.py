"""Metric tests. Every definition is pinned against a hand-computed fixture.

These are written rather than imported, so the tests are the specification. If a metric is
later swapped for a library call, these fixtures say whether the definition changed.
"""
import unittest

import numpy as np

from src import config, metrics


class TestErrors(unittest.TestCase):
    def test_mae_and_rmse_on_a_known_fixture(self):
        y = np.array([10.0, 20.0, 30.0])
        p = np.array([12.0, 18.0, 33.0])
        self.assertAlmostEqual(metrics.mae(y, p), (2 + 2 + 3) / 3)
        self.assertAlmostEqual(metrics.rmse(y, p), np.sqrt((4 + 4 + 9) / 3))

    def test_perfect_forecast_scores_zero(self):
        y = np.array([1.0, 5.0, 9.0])
        self.assertEqual(metrics.mae(y, y), 0.0)
        self.assertEqual(metrics.rmse(y, y), 0.0)
        self.assertEqual(metrics.smape(y, y), 0.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.mae(np.array([1.0, 2.0]), np.array([1.0]))

    def test_rmse_penalises_large_mistakes_more_than_mae(self):
        y = np.array([100.0] * 10)
        severe = np.array([100.0] * 9 + [400.0])
        self.assertGreater(metrics.rmse(y, severe), metrics.mae(y, severe))
        # One 300-rental miss among ten good days: MAE absorbs it, RMSE does not.
        self.assertAlmostEqual(metrics.mae(y, severe), 30.0)


class TestSmape(unittest.TestCase):
    def test_symmetric_when_both_sides_equal(self):
        """sMAPE is bounded at 200%, unlike MAPE which is unbounded."""
        y = np.array([100.0, 100.0])
        p = np.array([100.0, 300.0])
        self.assertAlmostEqual(metrics.smape(y, p), 50.0)

    def test_stays_finite_when_actual_is_zero(self):
        """The reason sMAPE is used here rather than MAPE. MAPE divides by the actual."""
        y = np.array([0.0, 100.0])
        p = np.array([5.0, 100.0])
        self.assertTrue(np.isfinite(metrics.smape(y, p)))

    def test_is_bounded_at_two_hundred_percent(self):
        y = np.array([1.0])
        self.assertLessEqual(metrics.smape(y, np.array([1000.0])), 200.0)

    def test_larger_percentage_error_gives_larger_smape(self):
        y = np.array([100.0])
        self.assertGreater(metrics.smape(y, np.array([200.0])), metrics.smape(y, np.array([120.0])))


class TestMase(unittest.TestCase):
    def test_below_one_means_better_than_the_naive_it_is_scaled_against(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        naive = np.array([12.0, 22.0, 32.0, 42.0])  # MAE 2
        good = y + 1.0  # MAE 1
        self.assertAlmostEqual(metrics.mase(y, good, metrics.naive_mae(y, naive)), 0.5)
        worse = y + 4.0  # MAE 4
        self.assertAlmostEqual(metrics.mase(y, worse, metrics.naive_mae(y, naive)), 2.0)

    def test_mase_of_the_naive_forecast_itself_is_one(self):
        y = np.array([10.0, 20.0, 30.0])
        naive = np.array([11.0, 19.0, 33.0])
        self.assertAlmostEqual(
            metrics.mase(y, naive, metrics.naive_mae(y, naive)), 1.0, places=10
        )

    def test_a_fixed_scale_is_the_whole_point(self):
        """Two different periods, one fixed scale, comparable numbers."""
        scale = 100.0
        easy = metrics.accuracy_report(np.array([100.0] * 5), np.array([90.0] * 5), scale)
        hard = metrics.accuracy_report(np.array([100.0] * 5), np.array([200.0] * 5), scale)
        self.assertAlmostEqual(easy["mase"], 0.1, places=10)
        self.assertAlmostEqual(hard["mase"], 1.0, places=10)
        self.assertEqual(easy["mase_scale"], scale)
        self.assertEqual(hard["mase_scale"], scale)

    def test_zero_naive_error_cannot_be_scaled_by(self):
        y = np.array([10.0, 20.0])
        with self.assertRaises(ValueError):
            metrics.mase(y, y, 0.0)

    def test_naive_mae_ignores_rows_with_no_naive_value(self):
        """The warmup rows have no seasonal naive; including them would understate it."""
        y = np.array([10.0, 20.0, 30.0])
        naive = np.array([np.nan, 18.0, 28.0])
        self.assertAlmostEqual(metrics.naive_mae(y, naive), 2.0)

    def test_naive_mae_refuses_when_nothing_is_available(self):
        with self.assertRaises(ValueError):
            metrics.naive_mae(np.array([1.0, 2.0]), np.array([np.nan, np.nan]))


class TestIntervals(unittest.TestCase):
    def test_coverage_counts_actuals_inside_the_band(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.0, 3.0, 0.0, 0.0])
        hi = np.array([2.0, 4.0, 2.0, 5.0])
        self.assertAlmostEqual(metrics.interval_coverage(y, lo, hi), 0.5)

    def test_band_boundaries_are_inclusive(self):
        y = np.array([1.0, 2.0])
        self.assertAlmostEqual(
            metrics.interval_coverage(y, np.array([1.0, 0.0]), np.array([1.0, 1.0])), 0.5
        )

    def test_wider_band_never_reduces_coverage(self):
        rng = np.random.default_rng(1)
        y = rng.normal(100, 20, 200)
        point = y + rng.normal(0, 10, 200)
        narrow = metrics.interval_coverage(y, point - 10, point + 10)
        wide = metrics.interval_coverage(y, point - 60, point + 60)
        self.assertGreaterEqual(wide, narrow)

    def test_mean_width(self):
        self.assertAlmostEqual(
            metrics.mean_interval_width(np.array([0.0, 0.0]), np.array([10.0, 20.0])), 15.0
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.interval_coverage(np.array([1.0, 2.0]), np.array([0.0]), np.array([1.0, 2.0]))


class TestPerHorizon(unittest.TestCase):
    def test_rows_are_split_by_position_within_the_week(self):
        """Day 1 of each fold block is the first element, not the seventh."""
        actual = np.arange(1, config.HORIZON * 3 + 1, dtype=float)
        pred = actual + 1.0  # MAE 1 against a scale of 2, so MASE is 0.5
        rows = metrics.per_horizon_report(actual, pred, 2.0)
        self.assertEqual([r["horizon"] for r in rows], list(range(1, config.HORIZON + 1)))
        self.assertEqual([r["n"] for r in rows], [3] * config.HORIZON)
        for r in rows:
            self.assertAlmostEqual(r["mase"], 0.5, places=10)
            self.assertAlmostEqual(r["mae"], 1.0, places=10)
            self.assertAlmostEqual(r["mase_scale"], 2.0, places=10)

    def test_later_horizons_can_be_worse_and_the_table_shows_it(self):
        """The table can express a bad day 7. Whether the pipeline's copy is confounded with
        weekday is a separate question, tested in test_pipeline."""
        actual = np.array([100.0] * config.HORIZON)
        pred = actual.copy()
        pred[0] = 100.0  # day 1 perfect
        pred[config.HORIZON - 1] = 200.0  # day 7 badly wrong
        rows = metrics.per_horizon_report(actual, pred, 2.0)
        self.assertAlmostEqual(rows[0]["mase"], 0.0, places=10)
        self.assertAlmostEqual(rows[-1]["mase"], 50.0, places=10)
        self.assertGreater(rows[-1]["mase"], rows[0]["mase"])


class TestMaseScale(unittest.TestCase):
    """The scale convention, which is the single most consequential number in the project."""

    _IRREGULAR = None

    def _series(self):
        # A linear ramp, or any series whose HORIZON-day differences are constant, makes the
        # scale identical at every cut and the test below passes without testing anything.
        # Noise guarantees the differences vary row to row.
        if TestMaseScale._IRREGULAR is None:
            rng = np.random.default_rng(0)
            TestMaseScale._IRREGULAR = 500 + rng.normal(0, 120, 200)
        return TestMaseScale._IRREGULAR

    def test_scale_is_computed_from_the_rows_up_to_the_cut_only(self):
        y = self._series()
        full = metrics.naive_scale(y, horizon=7)
        cut = metrics.naive_scale(y, horizon=7, upto=120)
        self.assertNotAlmostEqual(full, cut, places=6)

    def test_scale_excludes_everything_after_the_cut(self):
        y = self._series()
        cut = 120
        baseline = metrics.naive_scale(y, horizon=7, upto=cut)
        corrupted = y.copy()
        corrupted[cut:] += 10_000
        self.assertAlmostEqual(
            metrics.naive_scale(corrupted, horizon=7, upto=cut), baseline, places=10
        )

    def test_using_the_scored_rows_as_the_denominator_changes_the_verdict(self):
        """The mistake, pinned so it cannot be reintroduced by accident.

        A holdout on a steep seasonal decline makes the naive look bad, so scaling by its
        error on those same rows shrinks MASE and manufactures skill. On this project's real
        holdout the two conventions differ by a factor of 1.62 and straddle 1.0.
        """
        actual = np.array([100.0, 100.0, 100.0, 100.0])
        predicted = np.array([150.0, 150.0, 150.0, 150.0])  # MAE 50
        naive = np.array([200.0, 200.0, 200.0, 200.0])  # MAE 100, a hard period
        in_sample_scale = 50.0
        self.assertAlmostEqual(metrics.mase(actual, predicted, in_sample_scale), 1.0, places=10)
        self.assertLess(
            metrics.mase(actual, predicted, metrics.naive_mae(actual, naive)),
            1.0,
            "scaling by the scored rows understates the model on a hard period",
        )
