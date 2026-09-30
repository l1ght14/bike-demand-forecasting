"""Feature construction. The whole module turns on one decision.

For a target day i with a HORIZON-day lead time, the forecaster stands at origin o = i - 7
and knows demand up to and including day o. So every history feature for row i must be built
from y[0 .. i-7] and nothing later.

The way to guarantee that without hand-checking each column is to shift the target first:

    known = y.shift(horizon)   # known[i] = y[i-7], the last value knowable at origin

Then a lag or rolling window of `known` is automatically a function of y[<= i-7]. Building
them off the raw target instead would put y[i-1] in the row being predicted, which is the
most common way a forecasting pipeline cheats and produces a beautiful useless model.

The cost of this choice is real and worth stating: y[i-1] is by far the most predictive
single value available, and we are throwing it away, because at a seven-day lead time nobody
knows it. That is not a modelling compromise, it is the honest description of the problem.
"""
import numpy as np
import pandas as pd

from . import config, data_loader

# Lags are offsets into `known`, i.e. into y[i-7-k]. Offset 0 is last week's same weekday,
# which is the seasonal-naive forecast expressed as a feature.
LAG_OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 28]

# Built from the date index of the target day itself. A calendar is always knowable, however
# far ahead you are asked, so none of these can leak.
CALENDAR_FEATURES = [
    "dow",
    "is_weekend",
    "day_of_month",
    "day_of_year",
    "month",
    "year",
    "is_holiday",
    "t_index",
]


def build_features(df: pd.DataFrame, use_weather: bool = False) -> pd.DataFrame:
    """Return a leak-free design matrix, or None if no rows survive.

    `use_weather=False` builds a calendar-only matrix. That is the variant you would ship if
    you had no weather forecast. Both are reported so the price of assuming one is measured.
    """
    y = df[config.TARGET].astype(float)
    horizon = config.HORIZON
    known = y.shift(horizon)  # the entire no-leakage argument

    out = pd.DataFrame(index=df.index)

    dow = df.index.dayofweek
    out["dow"] = dow
    out["is_weekend"] = (dow >= 5).astype(int)
    out["day_of_month"] = df.index.day
    out["day_of_year"] = df.index.dayofyear
    out["month"] = df.index.month
    out["year"] = df.index.year
    out["is_holiday"] = df["holiday"].astype(int)
    out["t_index"] = np.arange(len(df))

    for k in LAG_OFFSETS:
        out[f"lag_{k}"] = known.shift(k)
    for w in ROLLING_WINDOWS:
        rolled = known.rolling(w)
        out[f"roll_mean_{w}"] = rolled.mean()
        out[f"roll_std_{w}"] = rolled.std()

    if use_weather:
        for col in config.WEATHER_COLUMNS:
            out[col] = df[col].astype(float)

    out[config.TARGET] = y

    # Rows before HORIZON + max(LAG_OFFSETS) cannot be featurised. That is 7 + 28 = 35, and
    # the rolling windows add nothing to it: rolling(28) over the already-shifted target
    # finishes one row before the deepest lag does. An earlier version of this comment added
    # max(ROLLING_WINDOWS) as well and put the figure at 63, which was wrong.
    # Dropping the rows is not an inconvenience to work around, it is the correct answer:
    # there genuinely is no history to build those features from yet.
    data_loader.assert_design_is_clean(design_columns(out))
    valid = out.dropna()
    if valid.empty:
        return None
    return valid


def feature_sets(df: pd.DataFrame) -> dict:
    """Both variants, keyed by name. Built once, compared later."""
    return {
        "calendar_only": build_features(df, use_weather=False),
        "with_weather": build_features(df, use_weather=True),
    }


def design_columns(feat: pd.DataFrame) -> list:
    """Predictor names, in a fixed order, target excluded."""
    return [c for c in feat.columns if c != config.TARGET]
