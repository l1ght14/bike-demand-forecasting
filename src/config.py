"""Global configuration. One place to change anything that more than one module needs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "raw" / "bike-sharing-day.csv"
SCORED_CSV = ROOT / "data" / "processed" / "daily_forecast.csv"
METRICS_JSON = ROOT / "reports" / "metrics.json"
EVAL_JSON = ROOT / "reports" / "evaluation.json"
EDA_JSON = ROOT / "reports" / "eda_findings.json"
FIG_DIR = ROOT / "reports" / "figures"

# Target. cnt is the daily rental count. casual and registered are its exact components
# (verified: cnt == casual + registered on all 731 rows), so they are target leakage.
TARGET = "cnt"
DATE_COL = "dteday"

RANDOM_STATE = 42

# Forecast horizon in days. A week is the decision a forecaster actually faces: how many
# bikes and staff to put on the road next week. It also forces the feature design - see
# features.HORIZON - to exclude lags that are unknowable that far out.
HORIZON = 7

# Walk-forward validation. Expanding window: fold k trains on everything strictly before
# its test block, so no fold ever sees a future it could not have seen live.
MIN_TRAIN_DAYS = 365

# Weather columns are contemporaneous actuals. Tomorrow's humidity is not knowable on
# today, so a real forecast can only use these if a weather forecast is supplied. Both
# variants are built and reported so the cost of that assumption is measured, not assumed.
WEATHER_COLUMNS = ["weathersit", "temp", "atemp", "hum", "windspeed"]

# Raw columns that must never reach the model.
LEAKY_COLUMNS = ["casual", "registered", "instant", "season"]
