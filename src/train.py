"""Model comparison under walk-forward validation, and the selection that follows.

The seasonal naive forecast is not a special case in this module. It is read out of the
feature matrix as `lag_0`, which is by construction y[t-7] - last week's same weekday. Using
the same column for both the baseline and the feature means the two cannot drift apart.

The MASE *scale*, by contrast, is deliberately computed elsewhere, from development rows only.
See metrics.mase for why using the naive's error on the rows being scored would invert the
headline result.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features, metrics

# n_jobs=1 everywhere, deliberately. A forest over dozens of walk-forward folds is the peak
# memory cost in this project, the parallel version was slower in wall clock on a series this
# small, and it made the last float bits of the committed forecast move between runs. Same
# trade-off as the churn project, for the same reason: a committed artefact that changes hash
# on rebuild is not an artefact anyone can trust.
_SHARED = {"random_state": config.RANDOM_STATE}

def _scaled(estimator) -> Pipeline:
    """Wrap a linear model in a scaler, fitted inside every fold.

    Not optional here. `t_index` runs 0..700 and the target averages ~4,500, so an unscaled
    Poisson fit hits invalid values in its Newton step and converges in a single iteration to
    a constant equal to that fold's training-window target mean - across this project's folds
    a flat forecast somewhere between 3,600 and 4,700 rentals, with no error raised. The
    failure is silent and looks like a working model. Scaling inside the fold rather than
    before the loop is also what keeps the scaler's mean and standard deviation out of the
    test window.
    """
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def candidate_models() -> dict:
    """Four models spanning the interpretability/accuracy range for a count series.

    Ridge and Poisson are scaled; the tree ensembles are not, because a split rule is
    invariant to monotone rescaling and paying for it would buy nothing.
    """
    return {
        "ridge": _scaled(Ridge(alpha=1.0, **_SHARED)),
        # Poisson assumes a count whose variance grows with the mean, which is the right
        # likelihood here, and its log link keeps predictions positive. The clip in the
        # walk-forward loop is belt-and-braces, not the mechanism.
        "poisson": _scaled(PoissonRegressor(alpha=1e-4, max_iter=1000)),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=1, **_SHARED
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, loss="huber", **_SHARED
        ),
    }


def _slice(feat, start, end):
    return feat.iloc[start:end]


def walk_forward_predictions(model, feat, folds) -> pd.DataFrame:
    """Fit on each fold's training window, forecast its test block. Returns one row per test day.

    Test blocks are non-overlapping, so the returned frame is a contiguous out-of-sample
    forecast series and every day in it was predicted exactly once, by a model that had not
    seen it. Metrics are computed on this concatenation rather than averaged per fold, so
    every day carries equal weight and the per-horizon breakdown stays meaningful.
    """
    cols = features.design_columns(feat)
    records = []
    for train_end, test_start, test_end in folds:
        train = feat.iloc[:train_end]
        test = feat.iloc[test_start:test_end]
        if train.empty or test.empty:
            continue
        est = clone(model)
        est.fit(train[cols], train[config.TARGET])
        pred = np.clip(est.predict(test[cols]), 0.0, None)
        for date, actual, p, naive in zip(
            test.index, test[config.TARGET], pred, test["lag_0"]
        ):
            records.append(
                {
                    "date": date,
                    "actual": float(actual),
                    "predicted": float(p),
                    "naive": float(naive),
                }
            )
    return pd.DataFrame.from_records(records)


def score_predictions(preds: pd.DataFrame, scale: float) -> dict:
    """Fold a prediction frame into the metric dict, scored against the fixed MASE scale."""
    actual = preds["actual"].to_numpy()
    report = metrics.accuracy_report(actual, preds["predicted"].to_numpy(), scale)
    report["per_horizon"] = metrics.per_horizon_report(actual, preds["predicted"].to_numpy(), scale)
    # Kept alongside for reference. It is the seasonal naive's error on the same rows, which
    # is deliberately NOT the MASE denominator - see metrics.mase.
    report["naive_mae_same_rows"] = round(
        metrics.naive_mae(actual, preds["naive"].to_numpy()), 4
    )
    return report


def select(clean_df, folds, scale: float) -> dict:
    """Score every model on every feature variant across the development folds.

    Selection key is development MASE and nothing else. Holdout rows are not in `folds` when
    this is called, so there is no path by which a holdout metric can influence the choice.
    """
    results = []
    for variant_name, variant_df in features.feature_sets(clean_df).items():
        if variant_df is None:
            continue
        for model_name, model in candidate_models().items():
            preds = walk_forward_predictions(model, variant_df, folds)
            if preds.empty:
                continue
            results.append(
                {
                    "feature_set": variant_name,
                    "model": model_name,
                    "dev": score_predictions(preds, scale),
                }
            )
    if not results:
        raise ValueError("no candidate produced predictions; check the fold list")
    best = min(results, key=lambda r: r["dev"]["mase"])
    return {
        "results": sorted(results, key=lambda r: r["dev"]["mase"]),
        "best_feature_set": best["feature_set"],
        "best_model": best["model"],
    }
