# Daily Demand Forecasting — Bike Hire (UCI)

Seven-day-ahead daily rental forecasting for a bike-share operator, built so that every
reported number can be defended.

**The headline result is not a win, and that is the point.** On 231 out-of-sample development
days the selected model beats the seasonal naive at **MASE 0.796**. On the 84-day holdout it
scores **MASE 1.098** — it does *not* beat "last week's same weekday". The gap is explained,
measured and reported in section 6 rather than hidden.

---

## 1. What the model is for

A bike operator has to decide, each week, how many bikes to redistribute and how many staff
and rebalancing capacity to commit. That is a **seven-day-ahead** question, and the choice of
horizon drives the entire design.

Seven days is not a default. It is the shortest horizon at which you must commit resources
before you can observe the outcome, and it is long enough to force the most interesting
constraint in the project: **at a seven-day lead time you do not know yesterday's demand.**

## 2. Data

| | |
|---|---|
| Source | UCI Bike Sharing Dataset, `day.csv` (auto-downloaded) |
| Rows | 731 daily observations, 2011-01-01 to 2012-12-31 |
| Target | `cnt`, daily rentals — min 22, max 8,714, mean 4,504.3, sd 1,937.2 |
| Nulls / gaps / duplicates | 0 / 0 / 0 |
| Rows dropped in cleaning | 0 |

There is almost nothing to clean here, so `data_loader.py` spends its effort on **verification**
instead. Three structural traps are checked rather than assumed:

- **`cnt == casual + registered` exactly, on all 731 rows.** The two components sum to the
  target, so either one as a feature is direct target leakage. (They are not a 50/50 split -
  casual is 18.8% of the target series-wide.) `assert_target_integrity`
  fails loudly if the identity ever stops holding.
- **`instant` is a pure row counter** (1…731) and `season` is determined by the date. Neither
  may reach the model. `config.LEAKY_COLUMNS` names them and `assert_design_is_clean` enforces
  that, rather than leaving the policy as documentation.
- **The index must be gapless.** A missing day is silent and severe: `lag_7` still computes,
  it just stops meaning what the name says. `assert_contiguous` checks the day deltas.

## 3. What the data says

The EDA exists to justify design choices, so every finding below maps to a decision in
section 4 or 5. It is written to `reports/eda_findings.json` and quoted in the tests.

**Weather dominates.**

| Condition | Mean daily rentals |
|---|---|
| Clear | 4,876.8 |
| Mist / cloudy | 4,035.9 |
| Light rain or snow | 1,803.3 |

**Rain removes 63.0% of demand.** By normalised temperature: 2,381 rentals below 0.3 versus
5,664 above 0.7 — 5,664 vs 2,381 is a **138%** swing, the largest single effect in the data.

**A strong upward trend.** Mean daily demand was 3,405.8 in 2011 and 5,599.9 in 2012 —
**+64.4%** year over year. A static model would under-forecast from day one.

**A weekly cycle that the naive baseline already captures.** Busiest day Friday (4,690), quietest
Sunday (4,229), and the weekend/weekday ratio is **0.965** — nearly flat, because Saturday is
high and Sunday is low and they largely cancel. That flat ratio is worth stating precisely
because the intuitive guess ("weekends are quiet") is wrong.

**Holidays cost 17.5%** of an average non-holiday day (3,735.0 vs 4,527.1, across 21
holidays). Against a true working day the comparison is 4,584.8.

**Autocorrelation** of the raw target: 0.846 at lag 1, 0.739 at lag 7, 0.707 at lag 14. The
weekly cycle is strong in the raw series — which is why the seasonal naive is a serious
baseline rather than a strawman.

## 4. How the forecast is built

### The one decision everything else follows from

For a target day `i` with a seven-day lead, the forecaster stands at origin `o = i - 7` and
knows demand up to and including day `o`. So **every history feature for row `i` must be built
from `y[0 .. i-7]` and nothing later.**

The way to guarantee that without auditing each column is to shift the target first:

```python
known = y.shift(horizon)   # the entire no-leakage argument
```

Every lag and rolling window of `known` is then automatically a function of `y[<= i-7]`.

**The cost is real and stated.** `y[i-1]` is by far the most predictive single value in the
series and is deliberately discarded, because at a seven-day lead time nobody knows it. That
is not a modelling compromise; it is the honest description of the problem. `LAG_OFFSETS`
starts at 0, so the nearest history any feature carries is `y[i-7]`.

### Features

Calendar (always knowable): day of week, is-weekend, day of month, day of year, month, year,
is-holiday, and a linear trend index.

History, from the shifted target: lags at offsets 0–7, 14, 21, 28 (offset 0 is last week's
same weekday, which *is* the seasonal naive), plus rolling mean and standard deviation over
7- and 28-day windows.

Weather (`weathersit`, `temp`, `atemp`, `hum`, `windspeed`) is **opt-in and built twice** —
see section 4.

35 warmup rows are dropped rather than imputed, because there genuinely is no history yet to
build those features from. 696 feature rows remain, 28 of them design columns plus the target.

## 5. Weather: the assumption, measured rather than assumed

`temp`, `hum` and `weathersit` are **contemporaneous actuals for the target day**. At a
seven-day lead you do not have them — you would need a weather *forecast*.

Rather than quietly depending on an oracle, the pipeline builds two feature sets and reports
both. The cost of not having a forecast is then a number, not an assumption:

| Model | calendar only | with weather | error removed |
|---|---|---|---|
| Gradient boosting | 1.235 | **0.796** | **35.5%** |
| Random forest | 1.186 | 0.961 | 18.9% |
| Ridge | 1.014 | 0.940 | 7.3% |
| Poisson | 1.203 | 1.141 | 5.2% |

*(development MASE; lower is better, 1.0 = the seasonal naive)*

The pattern is the finding: **the more expressive the model, the more it can exploit weather.**
Linear models gain almost nothing; the boosted ensemble gains a third of its error. The
calendar-only gradient booster scores 1.235 — clearly *worse* than the naive baseline.

If you have no weather forecast, the honest conclusion is that a learned model adds little over
last week's number. That is worth more than a headline that quietly assumes otherwise.

## 6. Validation, and the result

**Walk-forward, expanding window, non-overlapping 7-day test blocks.** 33 development folds
(231 forecast days) select the model; the last 84 **scored** days, 2012-10-07 to 2012-12-29
(12 folds), are the holdout.

The scored range is not the final 84 days of the series. The 7-day block 2012-09-30 to
2012-10-06 straddles the holdout boundary, and because development filters on `test_end <=`
while the holdout filters on `test_start >=`, it belongs to neither — along with the 35 warmup
rows and the last two days of the series, those seven days are never scored. Leaving them out
is the conservative choice; folding them into development would spend holdout days during
selection.

Two properties come from the construction rather than from discipline:

- `train_end == test_start - HORIZON`, never `test_start`. A test row's features come from its
  own training window, so the forecaster can never see demand it would not have had on the
  morning it made the call.
- The holdout is not a special case. It is the same generator, partitioned by date.

### Why MASE, and the bug that inverted the headline

MASE divides the model's MAE by the seasonal naive's MAE. Below 1 beats the naive. It is
scale-free, so it stays comparable if the fleet doubles.

**The denominator must come from training data.** An earlier version of this project divided
by the naive's error *on the rows being scored*. Because the holdout is a steep seasonal
decline where the naive itself does badly, that denominator inflates from 857 to 1,514 — a
factor of 1.77 — and reports a true MASE of 1.098 as **0.622**, i.e. it manufactures a 38%
improvement that does not exist. The scale is now computed once from the first 640 rows
(development only, `metrics.naive_scale(..., upto=...)`) and passed in as a constant.

### Development results (33 folds, 231 days)

| Feature set | Model | MAE | MASE | sMAPE |
|---|---|---|---|---|
| with weather | **gradient boosting** | **682.2** | **0.796** | **12.32%** |
| with weather | ridge | 805.6 | 0.940 | 13.89% |
| with weather | random forest | 823.9 | 0.961 | 14.99% |
| calendar only | ridge | 868.7 | 1.014 | 15.40% |
| with weather | poisson | 977.6 | 1.141 | 15.86% |
| calendar only | random forest | 1,016.3 | 1.186 | 18.32% |
| calendar only | poisson | 1,031.0 | 1.203 | 17.44% |
| calendar only | gradient boosting | 1,058.4 | 1.235 | 18.97% |

### Holdout results (12 folds, 84 scored days, 2012-10-07 to 2012-12-29)

| Metric | Value |
|---|---|
| MAE | 941.0 rentals |
| RMSE | 1,284.3 |
| sMAPE | 24.48% |
| **MASE** | **1.098** — does not beat the seasonal naive |
| MASE scale | 856.9 (from development rows only) |
| Mean bias | +472.8 |
| Seasonal naive's bias, same rows | +459.7 |
| 80% band empirical coverage | 69.0% — under-covering |
| Mean band width | 2,377 rentals |

**Development says the model is 20% better than naive. The holdout says it is 10% worse.**
That gap is the most informative number in the project, and section 7 explains it.

## 7. Why the holdout is worse — diagnosed, not speculated

The holdout sits in a steep seasonal decline: mean monthly demand falls 6,414 (Oct) → 5,089
(Nov) → 3,991 (Dec).

| | Bias |
|---|---|
| Model | +472.8 |
| Seasonal naive, same rows | +459.7 |
| **Difference** | **+13.1** |

The model over-predicts by 473 rentals, and so does the naive — by 460. **The model adds
essentially no bias of its own; it inherits the level problem.** The cause is structural: a
tree ensemble cannot extrapolate a trend. It can only ever predict values it has already seen,
so a forecast anchored on a falling level sits slightly above it. 67.9% of actuals fall below
the forecast, and `evaluate.interval_diagnosis` reports that direction as a *derived* field.

The same one-sidedness explains the under-covering band: a band centred on a high point forecast
misses every actual that falls below it. Fixing it properly needs asymmetric quantile levels or
a bias term fitted on development data — both stated as next steps, neither quietly applied,
because each needs validating on a holdout this project has already spent.

**What survives the diagnosis:** the residual autocorrelation at lag 7 is 0.049, well inside the
95% bound of 0.214. There is no weekly structure left in the residuals — the weekly cycle has
been captured. The model's problem is level bias, not missing pattern.

## 8. A diagnostic I had to retract

The per-horizon error table was built as the key operational diagnostic: *how bad is day 7?*

**It cannot answer that.** With a 7-day fold step, every test block starts on the same weekday,
so "horizon 1" is always Sunday, "horizon 3" always Tuesday. The table is **perfectly confounded
with day of week** — it measures which weekday is easy, not which lead time is hard. Mean
demand across the columns ranges 4,494 (Sun) to 5,509 (Wed) purely from that.

It is disclosed rather than removed. `evaluation.json` carries
`per_horizon_is_confounded_with_weekday: true`, every row carries its weekday and
`weekday_confounded: true`, and the figure title says so. The weekday swing is 10.9% across the
full series but 32.2% inside the holdout quarter, so the confound bites harder than the series
average suggests.

Fixing it properly requires staggering the fold phase, which needs overlapping test windows and
would break the guarantee that no day is forecast twice. That trade is stated rather than made
silently.

## 9. Uncertainty

Prediction intervals come from **quantile regression** — two gradient boosters predicting the
10th and 90th percentile — refitted inside every fold, not applied once to the whole series.
Independently fitted quantile models can cross, so the export takes an element-wise
`min`/`max`; a band whose lower edge sits above its upper edge reports zero width and breaks
every coverage figure computed from it.

Coverage is 69.0% against a nominal 80%, for the reason in section 7. An uncalibrated band that
is quietly wrong is worse than no band, because a planner will size stock against it.

## 10. Reproducing

```bash
python -m src.run_all                      # ~3 min: downloads data, rebuilds everything
python -m pytest tests -q                  # 132 tests, ~4-7 min
python -m pyflakes src tests               # clean
```

Two clean rebuilds produce byte-identical `daily_forecast.csv`, `metrics.json` and
`evaluation.json`. `n_jobs=1` everywhere is deliberate: it was the peak memory cost, it was
slower in wall clock on a series this small, and it made the last float bits of a committed
artefact move between runs.

**Tests are hermetic.** Every test that writes an artefact takes an explicit `out=` dict of
destinations. This is not theoretical: an earlier suite called `clean_outputs()` with no
arguments and then with two of five path arguments, and between them they deleted the project's
real `evaluation.json`, `eda_findings.json` and forecast while reporting green. `evaluate.run`
and `clean_outputs` now both take one `out` dict precisely so a call cannot be half-filled, and
a test parses the test sources to enforce it.

## 11. Layout

```
data/raw/          bike-sharing-day.csv (gitignored, auto-downloaded)
data/processed/    daily_forecast.csv   <- the deliverable, committed
src/               config 35 · data_loader 108 · splits 87 · features 101
                   metrics 146 · train 139 · evaluate 379 · eda 167 · run_all 164
tests/             132 tests across 6 files
reports/           figures/ (7 PNGs) · metrics.json · evaluation.json · eda_findings.json
```

| File | Responsibility |
|---|---|
| `config.py` | Paths, seed, horizon, leaky-column policy |
| `data_loader.py` | Load, verify contiguity and target identity, enforce the column policy |
| `splits.py` | Walk-forward folds and the dev/holdout partition — the core of the project |
| `features.py` | The horizon shift that makes leakage structurally impossible |
| `metrics.py` | MAE, RMSE, sMAPE, MASE and the scale convention |
| `train.py` | Four models × two feature sets, walk-forward selection |
| `evaluate.py` | Holdout forecast, intervals, interval diagnosis, figures |
| `eda.py` | Findings that justify the design choices |
| `run_all.py` | Orchestration, data bootstrap, artefact cleanup |

## 12. What I would do next

1. **Asymmetric quantile levels or a bias term**, fitted on development data, to fix the
   under-covering band. This is the highest-value fix.
2. **A seasonal-naive blend.** The model and the naive share the same +470 bias; a convex
   combination often beats both. Must be fitted on development folds and validated on a fresh
   holdout.
3. **A model that can extrapolate.** A linear trend term outside the trees, or a model class
   with genuine trend extrapolation, aimed squarely at the level bias rather than the pattern.
4. **Hourly granularity.** `hour.csv` has 17,000+ rows and would support a much better-behaved
   forecasting problem, at the cost of a harder one.

## 13. Honest limitations

1. **The model does not beat the seasonal naive on the holdout** (MASE 1.098). The development
   result (0.796) does not carry over.
2. **The headline depends on a weather forecast** that does not exist in the data. Without it,
   the best model is worse than naive.
3. **Two years is a short series**, and only one annual cycle, so the model has seen the
   autumn-winter season exactly once.
4. **Out-of-fold is not out-of-time.** Each fold shares one distribution; drift within the
   holdout is not modelled.
5. **The per-horizon table is confounded with day of week** and cannot answer lead-time
   questions (section 8).
6. **The interval is under-covering** (69% vs 80% nominal) and is labelled as such.
7. **The holdout has been spent.** Section 6 diagnoses it, so a further iteration cannot be
   validated on the same 84 days. Any change from here needs new data.
