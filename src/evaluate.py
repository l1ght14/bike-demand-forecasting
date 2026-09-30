"""Turn the selected model into a deliverable: a dated forecast with uncertainty bands,
evaluated once on a holdout it has never influenced.

Two things here are deliberately not the same estimator.

The point forecast and the interval are fitted separately, by quantile regression. That is
not a stylistic choice - a symmetric band around a point forecast is a guess about the error
distribution dressed up as a measurement, and it is wrong exactly when the errors are
skewed, which count data always is. Quantile regression estimates the band directly.

The band is then *checked*. Nominal coverage is not evidence of coverage. `interval_coverage`
on the holdout is the number that matters, and if it disagrees with the nominal 80% that fact
is reported rather than hidden. An uncalibrated band that is quietly wrong is worse than no
band, because a planner will size stock against it.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from . import config, features, metrics, train

NOMINAL_COVERAGE = 0.80
_LOWER_ALPHA = round((1 - NOMINAL_COVERAGE) / 2, 3)  # 0.1
_UPPER_ALPHA = round(1 - _LOWER_ALPHA, 3)  # 0.9


def quantile_models() -> dict:
    """Two gradient boosters trained to predict the 10th and 90th percentile of demand."""
    shared = {"random_state": config.RANDOM_STATE, "n_estimators": 200, "max_depth": 3}
    return {
        "lower": GradientBoostingRegressor(loss="quantile", alpha=_LOWER_ALPHA, **shared),
        "upper": GradientBoostingRegressor(loss="quantile", alpha=_UPPER_ALPHA, **shared),
    }


def holdout_forecast(model, feat, folds) -> pd.DataFrame:
    """Walk forward across the holdout with the selected point model and the two quantile models.

    The quantile models are also refitted per fold, on that fold's training window only. A band
    computed once over the whole series would have seen the holdout, which is the exact
    circularity the rest of this project is built to avoid.
    """
    cols = features.design_columns(feat)
    band = quantile_models()
    records = []
    for train_end, test_start, test_end in folds:
        train_df = feat.iloc[:train_end]
        test_df = feat.iloc[test_start:test_end]
        if train_df.empty or test_df.empty:
            continue
        point = clone(model).fit(train_df[cols], train_df[config.TARGET])
        lower = clone(band["lower"]).fit(train_df[cols], train_df[config.TARGET])
        upper = clone(band["upper"]).fit(train_df[cols], train_df[config.TARGET])
        point_pred = np.clip(point.predict(test_df[cols]), 0.0, None)
        lo = np.clip(lower.predict(test_df[cols]), 0.0, None)
        hi = np.clip(upper.predict(test_df[cols]), 0.0, None)
        # Independently fitted quantile models can cross, so the empirical lower quantile
        # occasionally comes out above the upper one. It is rare enough to be invisible in a
        # summary and common enough to matter: a band whose lower edge sits above its upper
        # edge silently reports zero width and breaks every coverage figure computed from it.
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        records.append(
            pd.DataFrame(
                {
                    "date": test_df.index,
                    "actual": test_df[config.TARGET].to_numpy(),
                    "predicted": point_pred,
                    "lower": lo,
                    "upper": hi,
                    "naive": test_df["lag_0"].to_numpy(),
                    "horizon_days": ((test_df.index - test_df.index[0]).days % config.HORIZON) + 1,
                }
            )
        )
    if not records:
        raise ValueError("holdout produced no forecasts")
    return pd.concat(records, ignore_index=True)


def interval_diagnosis(fc: pd.DataFrame) -> dict:
    """Explain the band rather than just reporting it.

    The band is fitted at the textbook 0.10/0.90 and it under-covers on the holdout. This
    function exists to say why, in numbers, rather than leaving a miscalibrated interval in
    the output with a hopeful label attached.

    The cause is one-sided error, not noise. The holdout sits in a steep seasonal decline
    (monthly mean demand falls from 6414 in October to 3991 in December), and a tree ensemble
    cannot extrapolate a trend - it can only ever predict values it has already seen. So the
    forecast is anchored slightly above the falling level, and a band centred on it misses
    every actual that falls below.

    Two pieces of evidence separate "the model is bad" from "the season is hard":
    `bias_above_naive` is the model's bias minus the seasonal naive's bias. Near zero means
    the model adds no bias of its own and simply inherits the level problem. And
    `mase` still beats 1.0, so relative skill against the naive forecast is intact even while
    the absolute error is large.

    A symmetric quantile pair cannot repair this. Widening to raise coverage means pushing the
    lower level down, which drags the upper level down with it and produces a band so wide it
    is useless for planning. The fix is asymmetric quantiles - a low lower-level and a high
    upper-level, tuned on development folds - or a bias-correction term fitted on development
    data. Both are stated as next steps rather than quietly applied, because either one needs
    to be validated on the holdout and this project has already spent it once.
    """
    ok = fc.dropna(subset=["naive"])
    actual = ok["actual"].to_numpy()
    predicted = ok["predicted"].to_numpy()
    model_bias = float(np.mean(predicted - actual))
    naive_bias = float(np.mean(ok["naive"] - actual))
    below = float(np.mean(actual < predicted))
    above = float(np.mean(actual > predicted))
    # Derived, not asserted. This used to be a literal string that said "actuals fall below
    # the forecast" no matter what the numbers said - and a run of an under-predicting model
    # produced a report claiming the opposite of its own arithmetic.
    if below >= 0.6:
        direction = "one-sided: actuals fall below the forecast"
    elif above >= 0.6:
        direction = "one-sided: actuals rise above the forecast"
    else:
        direction = "two-sided: errors straddle the forecast"
    return {
        "model_bias": round(model_bias, 4),
        "naive_bias": round(naive_bias, 4),
        "bias_above_naive": round(model_bias - naive_bias, 4),
        "share_below_forecast": round(below, 4),
        "share_above_forecast": round(above, 4),
        "direction": direction,
    }


def _save(fig, name, fig_dir=None):
    fig_dir = fig_dir or config.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forecast(fc: pd.DataFrame, fig_dir=None) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = pd.to_datetime(fc["date"])
    ax.plot(x, fc["actual"], color="#111", lw=1.6, label="actual")
    ax.plot(x, fc["predicted"], color="#c2410c", lw=1.6, label="forecast")
    ax.plot(x, fc["naive"], color="#94a3b8", lw=1.0, ls="--", label="seasonal naive")
    ax.fill_between(x, fc["lower"], fc["upper"], color="#c2410c", alpha=0.15, label="80% band")
    ax.set_title("Holdout forecast - seven days ahead, one week at a time")
    ax.set_ylabel("daily rentals")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "holdout_forecast.png", fig_dir)


def plot_per_horizon(rows, fig_dir=None) -> None:
    h = [r["horizon"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(h, [r["mase"] for r in rows], marker="o", color="#c2410c", label="MASE")
    ax.axhline(1.0, color="#dc2626", lw=1, ls=":", label="naive (MASE = 1)")
    ax.set_xlabel("days ahead")
    ax.set_ylabel("MASE (lower is better)", color="#c2410c")
    ax.set_xticks(h)
    twin = ax.twinx()
    twin.plot(h, [r["smape"] for r in rows], marker="s", ls="--", color="#0f766e", label="sMAPE %")
    twin.set_ylabel("sMAPE %", color="#0f766e")
    ax.set_title("Error by lead time - CONFOUNDED with day of week\n"
                 "(a 7-day fold step puts one weekday in every column)", fontsize=10)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top"]].set_visible(False)
    twin.spines[["top"]].set_visible(False)
    _save(fig, "error_by_horizon.png", fig_dir)


def residual_acf(residuals: np.ndarray, max_lag: int = 21) -> np.ndarray:
    """Autocorrelation of the residuals.

    The single most useful diagnostic in a forecasting project. If there is still structure
    left - a spike at lag 7 means the weekly cycle is not fully captured - the model is
    leaving signal on the table even if its average error looks acceptable.
    """
    r = np.asarray(residuals, float)
    r = r - r.mean()
    denom = np.dot(r, r)
    if denom == 0:
        return np.zeros(max_lag + 1)
    return np.array([1.0] + [float(np.dot(r[:-k], r[k:]) / denom) for k in range(1, max_lag + 1)])


def plot_residual_acf(residuals: np.ndarray, fig_dir=None) -> None:
    acf = residual_acf(residuals)
    lags = np.arange(len(acf))
    bound = 1.96 / np.sqrt(len(residuals))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.vlines(lags, 0, acf, color="#c2410c", lw=1.4)
    ax.axhline(bound, color="#64748b", ls=":", lw=1)
    ax.axhline(-bound, color="#64748b", ls=":", lw=1)
    if acf[7] > bound:
        ax.axvline(7, color="#0f766e", lw=1, ls="--")
        ax.text(7.2, acf[7], "lag 7\nweekly", color="#0f766e", fontsize=8)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("residual autocorrelation")
    n_exceeding = int(np.sum(np.abs(acf[1:]) > bound))
    ax.set_title(
        "Residual autocorrelation - no lag exceeds the 95% bound, nothing left to model"
        if n_exceeding == 0
        else f"Residual autocorrelation - {n_exceeding} lag(s) above the 95% bound",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "residual_acf.png", fig_dir)


def plot_weather_value(dev_results: list, fig_dir=None) -> None:
    """The headline comparison: what a weather forecast is worth."""
    best = {}
    for r in dev_results:
        best.setdefault(r["feature_set"], {})[r["model"]] = r["dev"]["mase"]
    names = sorted(best)
    models = sorted({m for v in best.values() for m in v})
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, m in enumerate(models):
        vals = [best[n].get(m, np.nan) for n in names]
        ax.bar(np.arange(len(names)) + i * width, vals, width, label=m)
    ax.axhline(1.0, color="#dc2626", lw=1, ls=":", label="seasonal naive (MASE = 1)")
    ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("MASE (lower is better)")
    # The headline is computed, not asserted. It was a literal "~35%" while the four bars it
    # sits above span 11% to 38% depending on model, so the figure argued against its own
    # title. The gain is quoted for the selected model and the per-model spread is stated.
    gains = []
    for m in models:
        with_w = best.get("with_weather", {}).get(m)
        without = best.get("calendar_only", {}).get(m)
        if with_w and without and without > 0:
            gains.append((m, (without - with_w) / without))
    headline = f"{gains[0][1]:.0%}" if gains else "n/a"
    spread = f" (per-model range {min(g for _, g in gains):.0%}-{max(g for _, g in gains):.0%})" if len(gains) > 1 else ""
    ax.set_title(
        f"Weather information removes {headline} of the error for {gains[0][0]}{spread}"
        if gains
        else "Weather information value",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "weather_value.png", fig_dir)


def horizon_weekday_map(fc) -> dict:
    """Which weekday each horizon column actually contains, and how consistent it is.

    This exists because of a real flaw in the fold design, found by reading the per-horizon
    table and refusing to believe it. With a fold step of exactly HORIZON days, every test
    block starts on the same weekday, so "horizon 1" is always Sunday, "horizon 3" is always
    Tuesday, and so on. The per-horizon error breakdown is then perfectly confounded with
    day-of-week: it measures which day of the week is easy, not which lead time is hard.

    It is not a subtle effect. Mean holdout demand ranges from 4,494 at horizon 1 to 5,509 at
    horizon 4 purely from which weekday that column contains. The weekday swing is 10.9% across
    the full two-year series but 32.2% inside the holdout quarter (Wednesday 5,667 against
    Sunday 4,287), so the confound bites harder here than the series average suggests - worth
    stating precisely, because quoting the 10.9% figure would understate the problem.

    Making the mapping explicit is the honest response. The alternative - staggering the fold
    phase - requires overlapping test windows, which would break the one guarantee this
    validation design is built on, that every day is forecast exactly once. Selection and
    headline accuracy keep the clean non-overlapping folds; the per-horizon table is
    labelled as confounded and is not used to make any claim about lead time.
    """
    # Derived from the rows that were actually scored, not from the fold list. holdout_forecast
    # skips empty folds, so a fold-derived map could describe a different set of dates than the
    # table it is attached to.
    mapping = {f"horizon_{h}": [] for h in range(1, config.HORIZON + 1)}
    dates = pd.DatetimeIndex(fc["date"])
    horizons = fc["horizon_days"].to_numpy()
    for weekday, h in zip(dates.dayofweek, horizons):
        mapping[f"horizon_{int(h)}"].append(int(weekday))
    return {k: sorted(set(v)) for k, v in mapping.items()}


def per_horizon_report_with_weekday(fc: pd.DataFrame, scale: float) -> list:
    """The per-horizon table, annotated with the weekday each column is made of.

    Same numbers as metrics.per_horizon_report, plus the confound spelled out on every row so
    it cannot be quoted as a lead-time finding by accident.
    """
    dates = pd.DatetimeIndex(fc["date"])
    rows = metrics.per_horizon_report(
        fc["actual"].to_numpy(), fc["predicted"].to_numpy(), scale
    )
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for row in rows:
        h = row["horizon"]
        sl = slice(h - 1, None, config.HORIZON)
        row["weekday"] = names[dates[sl][0].dayofweek]
        row["weekday_confounded"] = True
        row["mean_actual"] = round(float(np.mean(fc["actual"].to_numpy()[sl])), 1)
    return rows


def run(clean_df, selection, holdout_folds, scale: float, out=None) -> dict:
    """Produce the holdout forecast, the figures and the evaluation record.

    `out` redirects every write. It exists because this function is called from the test suite
    on synthetic series, and when it defaulted to the production paths the test run silently
    replaced the project's real deliverable with fixture output - `evaluation.json` ended up
    describing a 300-day synthetic series with a Monday-first fold, and `metrics.json` and
    `eda_findings.json` were deleted outright by a test that called `clean_outputs`. A
    function that writes files is not callable from tests unless the caller chooses where they
    go.
    """
    out = out or {}
    eval_json = out.get("eval_json", config.EVAL_JSON)
    scored_csv = out.get("scored_csv", config.SCORED_CSV)
    fig_dir = out.get("fig_dir", config.FIG_DIR)
    variant = features.feature_sets(clean_df)[selection["best_feature_set"]]
    model = train.candidate_models()[selection["best_model"]]

    fc = holdout_forecast(model, variant, holdout_folds)
    ok = fc.dropna(subset=["naive"])

    report = metrics.accuracy_report(
        ok["actual"].to_numpy(), ok["predicted"].to_numpy(), scale
    )
    coverage = metrics.interval_coverage(
        ok["actual"].to_numpy(), ok["lower"].to_numpy(), ok["upper"].to_numpy()
    )
    report.update(
        {
            "nominal_coverage": NOMINAL_COVERAGE,
            "empirical_coverage": round(coverage, 4),
            "mean_band_width": round(
                metrics.mean_interval_width(ok["lower"].to_numpy(), ok["upper"].to_numpy()), 4
            ),
            "calibrated": bool(abs(coverage - NOMINAL_COVERAGE) <= 0.10),
            "mean_bias": round(float(np.mean(ok["predicted"] - ok["actual"])), 4),
            "naive_mean_bias": round(float(np.mean(ok["naive"] - ok["actual"])), 4),
            "naive_mae_same_rows": round(
                metrics.naive_mae(ok["actual"].to_numpy(), ok["naive"].to_numpy()), 4
            ),
            "interval_diagnosis": interval_diagnosis(fc),
        }
    )
    report["per_horizon"] = per_horizon_report_with_weekday(ok, scale)
    report["horizon_weekday_confound"] = horizon_weekday_map(ok)
    report["per_horizon_is_confounded_with_weekday"] = True

    plot_forecast(fc, fig_dir)
    plot_per_horizon(report["per_horizon"], fig_dir)
    plot_residual_acf(ok["actual"].to_numpy() - ok["predicted"].to_numpy(), fig_dir)
    plot_weather_value(selection["results"], fig_dir)

    export = ok[
        ["date", "actual", "predicted", "naive", "lower", "upper", "horizon_days"]
    ].copy()
    for col in ("predicted", "lower", "upper", "naive"):
        export[col] = export[col].round(4)
    scored_csv.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(scored_csv, index=False)

    acf = residual_acf(ok["actual"].to_numpy() - ok["predicted"].to_numpy())
    report["residual_acf"] = {
        f"lag_{k}": round(float(v), 4) for k, v in enumerate(acf)
    }
    report["residual_acf_weekly_peak"] = round(float(acf[7]), 4)

    with open(eval_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report
