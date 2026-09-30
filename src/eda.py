"""Exploratory analysis whose only job is to justify the choices made in features.py.

Each finding here maps to a specific modelling decision. Weekly seasonality is why the
seasonal naive baseline exists and why lag-7 is offset 0 rather than lag-1. The weather
sensitivity is why a second feature set is built without weather, so the cost of not having
a forecast can be measured instead of assumed. The autocorrelation structure is why the lag
set looks the way it does.

EDA that does not change a decision is decoration, so this module returns a small dict of
findings that reports/eda_findings.json records and the documentation quotes.

These run over the whole series, holdout period included. That is deliberate and it is a
different thing from the model: nothing here is selected on, so a descriptive statistic over
all 731 days cannot bias a forecast. It does mean the segment and autocorrelation figures
should not be read as evidence about holdout performance.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, data_loader


def _save(fig, name, fig_dir=None):
    fig_dir = fig_dir or config.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def demand_by_weekday(df: pd.DataFrame) -> dict:
    g = df.groupby(df.index.dayofweek)[config.TARGET].agg(["mean", "count"])
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "by_weekday": {names[i]: round(float(g["mean"].iloc[i]), 1) for i in range(7)},
        "busiest": names[int(g["mean"].idxmax())],
        "quietest": names[int(g["mean"].idxmin())],
        "weekend_over_weekday": round(
            float(g["mean"].iloc[5:].mean() / g["mean"].iloc[:5].mean()), 4
        ),
    }


def demand_by_year(df: pd.DataFrame) -> dict:
    g = df.groupby(df.index.year)[config.TARGET].mean()
    out = {str(k): round(float(v), 1) for k, v in g.items()}
    years = sorted(g.index)
    growth = None
    if len(years) == 2:
        growth = round(float(g.iloc[1] / g.iloc[0] - 1), 4)
    return {"mean_by_year": out, "growth_second_year": growth}


def demand_by_weather(df: pd.DataFrame) -> dict:
    """Clear, mist and heavy rain are the three documented `weathersit` levels."""
    labels = {1: "clear", 2: "mist/cloudy", 3: "light rain/snow"}
    g = df.groupby("weathersit")[config.TARGET].mean()
    out = {labels.get(int(k), str(k)): round(float(v), 1) for k, v in g.items()}
    drop = None
    if 1 in g.index and 3 in g.index:
        drop = round(float(1 - g.loc[3] / g.loc[1]), 4)
    return {"mean_by_weather": out, "rain_drop_vs_clear": drop}


def demand_by_temperature_band(df: pd.DataFrame) -> dict:
    temp = df["temp"]  # normalised: 0.0 = freezing, 1.0 = boiling point
    bands = pd.cut(temp, bins=[0, 0.3, 0.5, 0.7, 1.0], labels=["<0.3", "0.3-0.5", "0.5-0.7", ">0.7"])
    g = df.groupby(bands, observed=True)[config.TARGET].mean()
    return {
        "mean_by_temp_band": {str(k): round(float(v), 1) for k, v in g.items()},
        "best_band": str(g.idxmax()),
        "worst_band": str(g.idxmin()),
    }


def holiday_effect(df: pd.DataFrame) -> dict:
    """Demand on holidays against demand on every other day.

    The comparison group is all non-holiday days, weekends included - not working days. The
    key used to say `mean_on_workday` while computing exactly that, and the difference is not
    cosmetic: the non-holiday mean is 4,527.1 and the true working-day mean is 4,584.8.
    """
    g = df.groupby("holiday")[config.TARGET].mean()
    working = df[df["workingday"] == 1][config.TARGET].mean()
    return {
        "mean_on_holiday": round(float(g.loc[1]), 1),
        "mean_on_nonholiday": round(float(g.loc[0]), 1),
        "mean_on_workingday": round(float(working), 1),
        "drop_vs_nonholiday": round(float(1 - g.loc[1] / g.loc[0]), 4),
        "n_holidays": int((df["holiday"] == 1).sum()),
    }


def target_autocorrelation(df: pd.DataFrame, max_lag: int = 21) -> dict:
    y = df[config.TARGET].to_numpy(float)
    y = y - y.mean()
    denom = float(np.dot(y, y))
    acf = {1: float(np.dot(y[:-1], y[1:]) / denom)}
    for lag in (7, 14, 21):
        if len(y) > lag:
            acf[lag] = float(np.dot(y[:-lag], y[lag:]) / denom)
    return {
        "acf_lag1": round(acf[1], 4),
        "acf_lag7": round(acf[7], 4),
        "acf_lag14": round(acf[14], 4),
        "acf_lag21": round(acf[21], 4),
        "strongest": f"lag_{max(acf, key=acf.get)}",
    }


def _plot_weekday(df, fig_dir=None):
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    g = df.groupby(df.index.dayofweek)[config.TARGET].mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#c2410c" if i >= 5 else "#475569" for i in range(7)]
    ax.bar(names, g.to_numpy(), color=colors)
    ax.set_ylabel("mean daily rentals")
    ax.set_title("Weekly cycle - the reason last week's same weekday is a real baseline")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "demand_by_weekday.png", fig_dir)


def _plot_weather(df, fig_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    labels = {1: "clear", 2: "mist", 3: "rain/snow"}
    g = df.groupby("weathersit")[config.TARGET].mean()
    axes[0].bar([labels[i] for i in g.index], g.to_numpy(), color="#0f766e")
    axes[0].set_ylabel("mean daily rentals")
    axes[0].set_title("Bad weather costs demand")
    axes[0].spines[["top", "right"]].set_visible(False)

    bands = pd.cut(df["temp"], bins=10)
    profile = df.groupby(bands, observed=True)[config.TARGET].mean()
    axes[1].plot(profile.index.map(lambda b: b.mid), profile.to_numpy(),
                 color="#c2410c", marker="o", ms=3)
    axes[1].set_xlabel("normalised temperature (0 = freezing, 1 = boiling)")
    axes[1].set_ylabel("mean daily rentals")
    axes[1].set_title("Demand peaks in mild weather")
    axes[1].spines[["top", "right"]].set_visible(False)
    _save(fig, "demand_by_weather.png", fig_dir)


def _plot_series(df, fig_dir=None):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(df.index, df[config.TARGET].to_numpy(), color="#111", lw=0.7)
    ax.set_ylabel("daily rentals")
    ax.set_title("Two years of daily demand - weekly ripple inside a clear upward trend")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "demand_series.png", fig_dir)


def run(eda_json=None, fig_dir=None) -> dict:
    """Write the findings. Output paths are injectable so tests never touch production."""
    eda_json = eda_json if eda_json is not None else config.EDA_JSON
    df = data_loader.load_clean()
    findings = {
        "weekday": demand_by_weekday(df),
        "year": demand_by_year(df),
        "weather": demand_by_weather(df),
        "temperature": demand_by_temperature_band(df),
        "holiday": holiday_effect(df),
        "autocorrelation": target_autocorrelation(df),
    }
    _plot_weekday(df, fig_dir)
    _plot_weather(df, fig_dir)
    _plot_series(df, fig_dir)
    eda_json.parent.mkdir(parents=True, exist_ok=True)
    with open(eda_json, "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return findings
