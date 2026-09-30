"""Forecast accuracy metrics, written out rather than imported so the definitions are
inspectable and unit-testable.

The headline is MASE, not MAPE and not R2. Two reasons, both practical:

- R2 on a trending series rewards predicting the trend and says nothing about whether the
  number you hand to a scheduler is right. It can go up while the forecast gets worse.
- MAPE divides by the actual value, so it explodes on small counts and is undefined at zero.
  This series happens to have no zeros, which is a gift, not a guarantee - sMAPE is used
  instead because it stays finite when either side approaches zero.

MASE divides the model's MAE by the MAE of the seasonal naive forecast. Below 1 means the
model beat last week's same weekday. It is scale-free, so it stays comparable if the operator
doubles every bike in the fleet, and it needs no business judgement about what counts as an
acceptable error in rentals.
"""
import numpy as np

from . import config


def _align(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE as a percentage. Bounded to [0, 200], finite when either side is zero."""
    y_true, y_pred = _align(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    terms = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(100.0 * np.mean(terms))


def mase(y_true, y_pred, scale: float) -> float:
    """Mean absolute scaled error. Below 1 beats the reference forecast it is scaled against.

    `scale` is the seasonal naive's MAE, and it MUST be computed on training data, not on the
    rows being scored. This is the single most consequential convention in the project and
    getting it wrong inverts the headline result.

    Hyndman & Koehler define the denominator over the in-sample series. If instead you divide
    by the naive's MAE *on the evaluation rows*, the denominator inflates exactly when the
    period is hard, the ratio shrinks, and a model that does not beat the naive appears to.
    Measured on this project's holdout: the test-set denominator is 1,513.9 against the
    development-only 856.9 - a factor of 1.77 - which turned a true MASE of 1.098, worse than
    the naive, into a reported 0.622. The scale is therefore computed once, from data that
    ends before the holdout, and passed in as a constant.

    Both denominators here are the ones this project actually used. A third figure, 933.5,
    is the whole-series naive scale; it was superseded along with the rest of the metric and
    should not be quoted.
    """
    if scale <= 0:
        raise ValueError("scale must be positive to scale against")
    return mae(y_true, y_pred) / scale


def naive_mae(y_true, naive_pred) -> float:
    """The scale for MASE: the seasonal naive forecast's own MAE on the given rows.

    Call this on training rows only, once, and reuse the value. Calling it on the rows being
    scored is the mistake described in mase().
    """
    y_true = np.asarray(y_true, dtype=float)
    naive_pred = np.asarray(naive_pred, dtype=float)
    mask = ~np.isnan(naive_pred)
    if not mask.any():
        raise ValueError("no non-null naive predictions to scale against")
    return float(np.mean(np.abs(y_true[mask] - naive_pred[mask])))


def naive_scale(y, horizon=None, upto: int = None) -> float:
    """The MASE scale, computed once on data that ends before the holdout.

    `upto` is a positional cut. Everything from there on is the holdout and is excluded, so
    the scale is a constant derived only from development data and can be applied unchanged to
    development folds and to the holdout without contaminating either.

    Using the whole series here would be a smaller sin - the denominator is a single
    aggregate, not a fitted parameter - but it would still let holdout demand influence a
    reported number, and there is no reason to accept that when excluding it costs one
    argument.
    """
    horizon = horizon or config.HORIZON
    y = np.asarray(y, dtype=float)
    if upto is not None:
        y = y[:upto]
    return naive_mae(y[horizon:], y[:-horizon])


def interval_coverage(y_true, lo, hi) -> float:
    """Fraction of actuals inside the predicted band. Should match the nominal level."""
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if not (y_true.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch: {y_true.shape}, {lo.shape}, {hi.shape}")
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())


def mean_interval_width(lo, hi) -> float:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return float(np.mean(hi - lo))


def accuracy_report(y_true, y_pred, scale: float) -> dict:
    """The full set for one (actual, forecast) pair, scored against a fixed MASE scale."""
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "smape": round(smape(y_true, y_pred), 4),
        "mase": round(mase(y_true, y_pred, scale), 4),
        "mase_scale": round(float(scale), 4),
        "n": int(len(y_true)),
    }


def per_horizon_report(y_true, y_pred, scale: float) -> list:
    """Error by lead time.

    Kept for completeness, but read evaluate.horizon_weekday_map before using it. With a
    7-day fold step each column of this table is made of a single day of the week, so it
    measures which weekday is easy rather than which lead time is hard. The pipeline labels
    the confound on every row; this docstring records why the function itself does not fix it.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rows = []
    for h in range(1, config.HORIZON + 1):
        sl = slice(h - 1, None, config.HORIZON)
        yt, yp = y_true[sl], y_pred[sl]
        report = accuracy_report(yt, yp, scale)
        report["horizon"] = h
        rows.append(report)
    return rows
