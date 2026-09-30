"""Run the whole thing: fetch data, analyse, select on development folds, report once on the
holdout, write artefacts.

`clean_outputs` deletes by glob rather than by a hard-coded list. An allowlist rots: a newly
written report is not on it, survives the rebuild that should have deleted it, and then a
stale figure or number lingers in the tree looking current. That already happened once in the
churn project, when a leftover figure pushed the documented PNG count out of date.
"""
import json
import sys
import urllib.request
import zipfile

from . import config, data_loader, eda, evaluate, features, metrics, splits, train

# UCI's static mirror. The raw CSV is gitignored, so a fresh clone cannot run without this.
RAW_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW_MEMBER = "day.csv"


def fetch_raw_data() -> str:
    """Download the dataset if it is not already present. Returns the path to day.csv."""
    if config.RAW_CSV.exists():
        return str(config.RAW_CSV)
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    archive = config.RAW_CSV.parent / "_bike_sharing.zip"
    print(f"downloading dataset from {RAW_URL}")
    with urllib.request.urlopen(RAW_URL, timeout=120) as resp:
        archive.write_bytes(resp.read())
    with zipfile.ZipFile(archive) as zf:
        with zf.open(RAW_MEMBER) as src, open(config.RAW_CSV, "wb") as dst:
            dst.write(src.read())
    archive.unlink()
    print(f"wrote {config.RAW_CSV}")
    return str(config.RAW_CSV)


def clean_outputs(out=None) -> None:
    """Delete generated artefacts so a rebuild cannot inherit a stale one.

    Takes a single `out` dict of destinations, same shape evaluate.run uses, and there is no
    argument-by-argument form on purpose. An earlier version took five separate optional path
    arguments, and a test that passed two of them deleted the project's real
    `evaluation.json`, `eda_findings.json` and `daily_forecast.csv` while passing CI - partial
    injection is worse than none, because it looks hermetic. One dict cannot be half-filled.
    """
    out = out or {}
    fig_dir = out.get("fig_dir", config.FIG_DIR)
    targets = [
        out.get("metrics_json", config.METRICS_JSON),
        out.get("eval_json", config.EVAL_JSON),
        out.get("eda_json", config.EDA_JSON),
        out.get("scored_csv", config.SCORED_CSV),
    ]
    for pattern in ("*.json", "*.png"):
        for path in fig_dir.glob(pattern) if fig_dir.exists() else []:
            path.unlink()
    for path in targets:
        if path.exists():
            path.unlink()


_OUTPUTS = {
    "fig_dir": config.FIG_DIR,
    "metrics_json": config.METRICS_JSON,
    "eval_json": config.EVAL_JSON,
    "eda_json": config.EDA_JSON,
    "scored_csv": config.SCORED_CSV,
}


def main() -> int:
    fetch_raw_data()
    clean_outputs(out=_OUTPUTS)

    clean_df = data_loader.load_clean()
    quality = data_loader.data_quality_report(data_loader.load_raw(), clean_df)
    print(
        f"data: {quality['clean_rows']} daily rows, "
        f"{quality['start_date']} to {quality['end_date']}, "
        f"{quality['null_cells']} null cells, {quality['non_daily_gaps']} gaps"
    )

    print("running EDA...")
    eda.run(eda_json=config.EDA_JSON, fig_dir=config.FIG_DIR)

    # Fold geometry is derived from the with_weather frame because that is the variant the
    # winner comes from, and every variant is featurised identically for the leading rows, so
    # all of them share the same warmup and therefore the same fold bounds.
    variant = features.feature_sets(clean_df)["with_weather"]

    # Both fold sets use a 7-day step, so development covers every day between the first fold
    # and the holdout boundary rather than every other week. An earlier version stepped the
    # development folds by 14 days to halve the runtime; that left holes in the development
    # series, which contradicted the claim that every day is forecast exactly once, and it
    # moved the selection metric from 0.625 to 0.665 for no reason that was ever justified.
    all_folds = splits.walk_forward_folds(len(variant), step=config.HORIZON)
    holdout_start = splits.holdout_start_position(variant)
    dev_folds, _ = splits.partition_folds(all_folds, holdout_start)
    holdout_folds = [f for f in all_folds if f[1] >= holdout_start]
    splits.assert_folds_are_chronological(dev_folds + holdout_folds, len(variant))

    # The MASE scale is fixed once, from development rows only, and reused unchanged for the
    # holdout. Deriving it per fold from the rows being scored would inflate the denominator
    # whenever a period is hard and would understate the model.
    # The cut is computed by date rather than by adding the warmup length, because the
    # feature frame's positions are not the clean series' positions.
    holdout_date = variant.index[holdout_start]
    development_rows = int((clean_df.index < holdout_date).sum())
    scale = metrics.naive_scale(clean_df[config.TARGET].to_numpy(), upto=development_rows)
    print(
        f"walk-forward: {len(dev_folds)} development folds, {len(holdout_folds)} holdout "
        f"folds, holdout starts {variant.index[holdout_start].date()}; "
        f"MASE scale {scale:.1f} (from development rows only)"
    )

    print("selecting on development folds only...")
    selection = train.select(clean_df, dev_folds, scale)
    print(f"  winner: {selection['best_model']} / {selection['best_feature_set']}")
    for r in selection["results"]:
        d = r["dev"]
        print(
            f"    {r['feature_set']:14s} {r['model']:20s} "
            f"MAE {d['mae']:8.1f}  MASE {d['mase']:.3f}  sMAPE {d['smape']:5.2f}%"
        )

    print("evaluating once on the holdout...")
    report = evaluate.run(clean_df, selection, holdout_folds, scale)

    metrics_doc = {
        "data_quality": quality,
        "horizon_days": config.HORIZON,
        "mase_scale": round(float(scale), 4),
        "mase_scale_rows": development_rows,
        "n_dev_folds": len(dev_folds),
        "n_holdout_folds": len(holdout_folds),
        "holdout_start": holdout_date.strftime("%Y-%m-%d"),
        "best_model": selection["best_model"],
        "best_feature_set": selection["best_feature_set"],
        "results": selection["results"],
        "holdout": {k: v for k, v in report.items() if k != "per_horizon"},
        "per_horizon": report["per_horizon"],
    }
    with open(config.METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics_doc, fh, indent=2, sort_keys=True)
        fh.write("\n")

    h = report
    print(
        f"\nHOLDOUT  MAE {h['mae']:.1f} rentals  MASE {h['mase']:.3f}  "
        f"sMAPE {h['smape']:.2f}%  bias {h['mean_bias']:+.1f} "
        f"(seasonal naive {h['naive_mean_bias']:+.1f})"
    )
    print(
        f"         nominal {h['nominal_coverage']:.0%} band: empirical coverage "
        f"{h['empirical_coverage']:.1%}, mean width {h['mean_band_width']:.0f} rentals "
        f"({'calibrated' if h['calibrated'] else 'UNDER-COVERING - see interval_diagnosis'})"
    )
    print(f"         forecast written to {config.SCORED_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
