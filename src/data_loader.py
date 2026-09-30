"""Loading, cleaning and - the part that matters for a forecasting project - proving the
target is what we think it is.

The cleaning here is deliberately thin. There is almost nothing to clean: the file has no
nulls, no duplicates, no whitespace damage. The work in this module is verification, not
tidying. A time-series project is easy to get wrong quietly, so the checks are the content.
"""
import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Read the file exactly as it sits on disk. No cleaning. No type coercion."""
    path = path or config.RAW_CSV
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, date-indexed copy. Input is never mutated.

    The only real transformation is turning `dteday` into a proper datetime index, because
    every downstream time-aware operation depends on it. Everything else is asserted rather
    than silently repaired: assert_contiguous for the index and assert_target_integrity for
    the target. The leaky-column policy in config.LEAKY_COLUMNS is enforced by
    assert_design_is_clean, not by this function, because it applies to the design matrix
    rather than to the loaded frame.
    """
    out = df.copy()
    out[config.DATE_COL] = pd.to_datetime(out[config.DATE_COL])
    out = out.sort_values(config.DATE_COL).reset_index(drop=True)
    out = out.set_index(config.DATE_COL, drop=True)
    return out


def assert_contiguous(df: pd.DataFrame) -> None:
    """Fail if the daily index has a hole.

    A missing day is not a cosmetic problem here. Lag-7 and lag-14 features silently become
    wrong if row n is not actually 7 days after row n-1: the arithmetic still runs, it just
    stops meaning what the column name says. Nothing downstream would notice.
    """
    gaps = df.index.to_series().diff().dropna()
    bad = gaps[gaps != pd.Timedelta(days=1)]
    if len(bad):
        raise ValueError(f"non-daily gaps in index: {bad.to_dict()}")


def assert_target_integrity(df: pd.DataFrame) -> None:
    """Prove `cnt` is the total and that its components are recognised as leakage.

    cnt == casual + registered exactly, on all 731 rows. That identity is the single most
    dangerous fact in this dataset: keeping either component as a feature hands the model the
    target. It is checked rather than assumed so that a future version of the file, where the
    relationship might drift, fails loudly instead of quietly producing a perfect score.
    """
    if "casual" not in df.columns or "registered" not in df.columns:
        raise ValueError("expected casual/registered components to be present for the check")
    reconstructed = df["casual"] + df["registered"]
    mismatch = (reconstructed != df[config.TARGET]).sum()
    if mismatch:
        raise ValueError(f"cnt != casual + registered on {mismatch} rows")
    for col in ("casual", "registered"):
        if col not in config.LEAKY_COLUMNS:
            raise ValueError(f"{col} is a target component but is not marked leaky")


def data_quality_report(raw: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Counts an interview can quote, and a doc test can pin."""
    target = clean_df[config.TARGET]
    gaps = clean_df.index.to_series().diff().dropna()
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean_df)),
        "rows_dropped": int(len(raw) - len(clean_df)),
        "start_date": clean_df.index[0].strftime("%Y-%m-%d"),
        "end_date": clean_df.index[-1].strftime("%Y-%m-%d"),
        "span_days": int((clean_df.index[-1] - clean_df.index[0]).days) + 1,
        "duplicate_dates": int(clean_df.index.duplicated().sum()),
        "null_cells": int(clean_df.isna().sum().sum()),
        "non_daily_gaps": int((gaps != pd.Timedelta(days=1)).sum()),
        "target_min": int(target.min()),
        "target_max": int(target.max()),
        "target_mean": round(float(target.mean()), 4),
        "target_std": round(float(target.std(ddof=1)), 4),
        "target_zeros": int((target == 0).sum()),
        "leaky_columns": list(config.LEAKY_COLUMNS),
    }


def assert_design_is_clean(design_columns) -> None:
    """Fail if a forbidden column reached the design matrix.

    The LEAKY_COLUMNS list was originally documentation with nothing enforcing it: the design
    columns were clean only because features.py happened not to include them, and a future
    edit adding df["casual"] would have passed the entire runtime suite. This closes the gap.
    """
    present = [c for c in config.LEAKY_COLUMNS if c in set(design_columns)]
    if present:
        raise ValueError(f"forbidden columns in the design matrix: {present}")


def load_clean(path=None) -> pd.DataFrame:
    """raw -> verified clean, in one call."""
    out = clean(load_raw(path))
    assert_contiguous(out)
    assert_target_integrity(out)
    return out
