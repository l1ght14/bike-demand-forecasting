"""The leakage guard. This is the most important test file in the project.

Everything else checks that the code does what it says. This file checks the one property
that, if violated, would make every number in the project a lie: that a feature describing a
target day was built only from demand that was already known on the day the forecast was made.

The test is a perturbation, not an inspection. It does not read the feature-building code and
conclude that it looks correct. It corrupts the target on specific days and watches to see
which feature rows move. Code that leaks is caught by behaviour, so it cannot be satisfied by
a plausible-looking implementation.
"""
import unittest

import numpy as np
import pandas as pd

from src import config, data_loader, features


def _small_frame(n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-01", periods=n_days, freq="D")
    weekly = 1000 + 300 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
    return pd.DataFrame(
        {
            "season": 1,
            "yr": 0,
            "mnth": 1,
            "holiday": (idx.dayofweek >= 5).astype(int),
            "weekday": idx.dayofweek,
            "workingday": (idx.dayofweek < 5).astype(int),
            "weathersit": 1,
            "temp": rng.uniform(0.3, 0.8, n_days),
            "atemp": rng.uniform(0.3, 0.8, n_days),
            "hum": rng.uniform(0.3, 0.8, n_days),
            "windspeed": rng.uniform(0.0, 0.4, n_days),
            "casual": 100,
            "registered": 200,
            "cnt": weekly + rng.normal(0, 20, n_days),
        },
        index=idx,
    )


class TestNoFutureLeakage(unittest.TestCase):
    def test_features_ignore_the_days_between_origin_and_target(self):
        """Corrupting y[i-6 .. i] must not change the feature row for day i.

        The forecast for day i is made standing at day i-7. Days i-6 through i are in the
        future at that moment, so no feature may depend on them.
        """
        base = _small_frame()
        target_i = 120
        target_date = base.index[target_i]

        corrupted = base.copy()
        future = corrupted.columns.get_loc(config.TARGET)
        corrupted.iloc[target_i - config.HORIZON + 1 : target_i + 1, future] += 50_000

        row_a = features.build_features(base, use_weather=True).loc[target_date]
        row_b = features.build_features(corrupted, use_weather=True).loc[target_date]

        design = [c for c in row_a.index if c != config.TARGET]
        np.testing.assert_allclose(
            row_a[design].to_numpy(float),
            row_b[design].to_numpy(float),
            err_msg="features for the target day changed when only future demand was altered",
        )

    def test_features_do_respond_to_history_within_the_horizon(self):
        """The mirror image of the test above, and the one that stops a false pass.

        A feature builder that ignored the target entirely would also satisfy the first test.
        Corrupting day i-7 - the origin itself, which is legitimately knowable - must change
        the feature row for day i. Without this, the leakage guard would pass on a model that
        had simply thrown the signal away.
        """
        base = _small_frame()
        target_i = 120
        target_date = base.index[target_i]

        corrupted = base.copy()
        corrupted.iloc[target_i - config.HORIZON, corrupted.columns.get_loc(config.TARGET)] += 50_000

        row_a = features.build_features(base, use_weather=True).loc[target_date]
        row_b = features.build_features(corrupted, use_weather=True).loc[target_date]

        design = [c for c in row_a.index if c != config.TARGET]
        self.assertFalse(
            np.allclose(row_a[design].to_numpy(float), row_b[design].to_numpy(float)),
            "feature row ignored the origin day, so the previous test proves nothing",
        )

    def test_lag_offsets_resolve_to_the_horizon_shifted_target(self):
        """lag_0 must be y[i-HORIZON] exactly - that is the whole no-leakage argument."""
        df = _small_frame()
        feat = features.build_features(df)
        y = df[config.TARGET]
        for k in features.LAG_OFFSETS:
            col = feat[f"lag_{k}"]
            expected = y.shift(config.HORIZON).shift(k).loc[col.index]
            np.testing.assert_allclose(col.to_numpy(float), expected.to_numpy(float))

    def test_no_rolling_window_reaches_past_the_origin(self):
        """Recompute the rolling means by hand and compare, so a shifted rolling is caught."""
        df = _small_frame()
        feat = features.build_features(df)
        known = df[config.TARGET].shift(config.HORIZON)
        for w in features.ROLLING_WINDOWS:
            expected = known.rolling(w).mean().loc[feat.index]
            np.testing.assert_allclose(
                feat[f"roll_mean_{w}"].to_numpy(float),
                expected.to_numpy(float),
                err_msg=f"roll_mean_{w} does not match a plain rolling window on the shifted target",
            )

    def test_target_components_never_appear_as_features(self):
        df = _small_frame()
        for use_weather in (False, True):
            feat = features.build_features(df, use_weather=use_weather)
            for leaky in ("casual", "registered", "instant", "season", "cnt_x"):
                self.assertNotIn(leaky, features.design_columns(feat))


class TestHorizonHonesty(unittest.TestCase):
    def test_lag_one_of_the_raw_target_is_not_a_feature(self):
        """The most predictive value available is excluded on purpose, and that must be visible.

        y[i-1] at a seven-day lead time is unknowable. The design expresses this as: the
        smallest offset into the shifted target is 0, so the nearest history feature any row
        carries is y[i-7]. A raw lag_1 would be the unknowable one, so it cannot be expressed
        in this feature set at all without changing the horizon.
        """
        self.assertEqual(min(features.LAG_OFFSETS), 0)
        self.assertIn(0, features.LAG_OFFSETS)
        # Offset 1 is y[i-8], which is legitimate. What must not exist is a feature equal to
        # y[i-1]; asserting that directly is the real check.
        df = _small_frame()
        feat = features.build_features(df)
        y = df[config.TARGET]
        for col in features.design_columns(feat):
            if not col.startswith("lag_"):
                continue
            offset = int(col.split("_")[1])
            np.testing.assert_allclose(
                feat[col].to_numpy(float),
                y.shift(config.HORIZON + offset).loc[feat.index].to_numpy(float),
                err_msg=f"{col} is not y[i-{config.HORIZON + offset}]",
            )

    def test_weather_is_opt_in(self):
        df = _small_frame()
        calendar_only = features.build_features(df, use_weather=False)
        with_weather = features.build_features(df, use_weather=True)
        for col in config.WEATHER_COLUMNS:
            self.assertNotIn(col, calendar_only.columns)
            self.assertIn(col, with_weather.columns)

    def test_warmup_rows_are_dropped_not_imputed(self):
        """The first unusable rows are dropped. Nothing is filled in to hide the gap.

        The warmup is exactly HORIZON + max(lag offset): the shifted target needs HORIZON
        rows before it has a value, and the deepest lag needs max(LAG_OFFSETS) more. Rolling
        windows are satisfied by then, because rolling(28) over the shifted target finishes
        one row before the deepest lag does.
        """
        df = _small_frame(200)
        feat = features.build_features(df)
        expected_warmup = config.HORIZON + max(features.LAG_OFFSETS)
        self.assertEqual(len(df) - len(feat), expected_warmup)
        self.assertFalse(feat.isna().any().any())
        self.assertEqual(
            feat.index[0], df.index[expected_warmup], "warmup is not contiguous from the start"
        )


class TestRealData(unittest.TestCase):
    def test_real_features_have_expected_shape(self):
        df = data_loader.load_clean()
        feat = features.build_features(df, use_weather=True)
        self.assertEqual(len(feat.columns), len(features.design_columns(feat)) + 1)
        self.assertEqual(len(feat), 696)
        self.assertEqual(list(feat.columns[-1:]), [config.TARGET])

    def test_target_identity_holds_on_real_data(self):
        df = data_loader.load_clean()
        np.testing.assert_array_equal(
            df["casual"].to_numpy() + df["registered"].to_numpy(),
            df[config.TARGET].to_numpy(),
        )
