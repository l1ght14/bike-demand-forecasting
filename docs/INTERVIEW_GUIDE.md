# Interview Guide — Daily Demand Forecasting (UCI Bike Hire)

**Read top to bottom once, then drill section 7 (the walkthrough script) and section 8 (Q&A).**
Every line reference is verified against the code. Line numbers are anchors, not arguments —
if one disagrees with your editor the code moved, the reasoning stands.

---

## 1. The 60-second version

> "I built a seven-day-ahead demand forecaster for a bike-share operator: 731 daily
> observations, four models, two feature sets, walk-forward validation.
>
> The design is driven by one constraint. At a seven-day lead time you do not know yesterday's
> demand, so the most predictive value in the series is unusable. I make that structural by
> shifting the target by the horizon first, so every history feature is *provably* built only
> from data that existed at the forecast moment.
>
> **The model beats the seasonal naive on 231 development days at MASE 0.796 and loses to it on
> the 84-day holdout at MASE 1.098.** I report that rather than the development number.
>
> Three findings I'd defend hardest: weather removes 35% of the error for the boosted model but
> only 5% for the Poisson — and it isn't available seven days out anyway, so I built both
> variants and measured the assumption. The holdout bias is +473 rentals and the *naive's* bias
> is +460, so the model adds no bias of its own; it's a tree failing to extrapolate a seasonal
> decline. And my per-horizon diagnostic was measuring day-of-week, not lead time, so I labelled
> it confounded rather than quoting it."

**If they ask the hardest follow-up** — *"how do you know your evaluation isn't fooling you?"* —
go to section 6. That is where this project is strongest.

## 2. What problem am I actually solving?

A bike operator commits resources weekly: redistribute bikes, schedule staff, plan rebalancing.
So the question is **seven days ahead**, and the horizon is the design.

| Question | Answered by | Not answered by |
|---|---|---|
| What drives demand? | EDA / seasonality analysis | A model |
| What will next week's demand be? | A ranked forecast + interval | A single accuracy number |

Three facts from EDA that set the direction:

- **Rain removes 63.0% of demand** (4,877 rentals on clear days, 1,803 in rain).
- **Cold suppresses it** — 2,381 rentals below 0.3 normalised temperature vs 5,664 above 0.7.
- **Year two grew 64.4%** over year one (3,406 → 5,600 mean daily). A static model would
  under-forecast immediately.

Busiest weekday is Friday, quietest Sunday, and the weekend/weekday ratio is 0.965 — nearly
flat, because Saturday is high and Sunday is low and they cancel. **Be ready for someone to
guess "weekends are quiet". The number says otherwise, and why is interesting.**

## 3. Architecture

Run order is bottom-up; `python -m src.run_all` walks it.

| File | Lines | Responsibility | Why it exists |
|---|---|---|---|
| `config.py` | 35 | Paths, seed, horizon, leaky-column policy | One place to change globals |
| `data_loader.py` | 108 | Load, verify, enforce column policy | Verification, not tidying |
| `splits.py` | 87 | Walk-forward folds, dev/holdout partition | The core of the project |
| `features.py` | 101 | The horizon shift, lags, rolling | Makes leakage structurally impossible |
| `metrics.py` | 146 | MAE, RMSE, sMAPE, MASE, scale convention | Every metric defined in code |
| `train.py` | 139 | 4 models × 2 feature sets, selection | Walk-forward scoring |
| `evaluate.py` | 379 | Holdout forecast, intervals, diagnosis, figures | Model → deliverable |
| `eda.py` | 167 | Findings that justify the design | EDA that changes a decision |
| `run_all.py` | 164 | Orchestration, bootstrap, cleanup | One command, reproducible |

## 4. Line-by-line walkthrough, in execution order

### Stage 0 — `config.py`

```
15  TARGET = "cnt"
23  HORIZON = 7
32  WEATHER_COLUMNS = ["weathersit", "temp", "atemp", "hum", "windspeed"]
35  LEAKY_COLUMNS = ["casual", "registered", "instant", "season"]
```

**Line 23** is the single most consequential constant in the project. **Line 35** is a policy,
and it is *enforced*, not documented — see `data_loader.assert_design_is_clean`.

### Stage 1 — `data_loader.py` — the checks

There is nothing to clean. So the module verifies instead:

**`assert_target_integrity` (49–65).** Proves `cnt == casual + registered` on all 731 rows and
that both are declared leaky.

**Say this:** *"`casual` and `registered` sum exactly to the target — 331 + 654 = 985. Either
one as a feature is direct leakage, and it would produce a near-perfect score that means
nothing. I check the identity rather than assuming it, so if a future version of the file drifts
it fails loudly."*

**`assert_contiguous` (36–46).** Checks the day deltas are all exactly 1 day.

**Say this if they ask why it matters:** *"A missing day is silent and severe. `lag_7` still
computes — it just stops meaning what the name says. Nothing downstream would notice."*

**`assert_design_is_clean` (91–100).** Enforces the leaky-column policy against the design
matrix. This function exists because an earlier version had `LEAKY_COLUMNS` as documentation
with nothing enforcing it, and a reviewer pointed out that a future edit adding
`df["casual"]` would pass the entire runtime suite.

### Stage 2 — `splits.py` — where the project lives

**`walk_forward_folds` (21–46).** The critical line:

```python
43  train_end = test_start - horizon
```

**Read that line aloud.** A test row at `test_start` is forecast from origin
`test_start - 7`, so training must stop there. Training to `test_start` instead would hand the
model the answer for the first day of its own test block, through the rolling features.

**`assert_folds_are_chronological` (74–87).** Re-encodes that invariant at runtime and rejects
a fold whose `train_end` is not exactly one horizon back.

**Be ready for:** *"why not a random split with a time-series cross-validator?"* — a random split
lets the model train on March and test on January, which is a question nobody is asking. And
`TimeSeriesSplit` gives you expanding or rolling windows but not the *alignment* requirement:
every feature of a test row must come from its own training window. That is a stronger property
and I wanted to own it in ten lines I could test.

**`partition_folds` (49–62).** The holdout is not a special case — it is the same fold list,
filtered by date. Filters development on `test_end <= holdout_start`, *not* `test_start`; a
mutation check confirmed the difference matters, because filtering on `test_start` admits a fold
whose test block runs past the boundary and quietly spends part of the holdout during selection.

**Know what the geometry does and does not promise.** It guarantees no day is **forecast
twice** — that is the non-overlapping property, and it is what the tests assert. It does *not*
promise every day is forecast. A block that **straddles** the boundary satisfies neither
filter, so it belongs to neither side: on this series that is 2012-09-30 to 2012-10-06, seven
days that are never scored, alongside the 35 warmup rows and the last two days. Excluding it is
the conservative choice — folding it into development would spend holdout days during selection
— but say it out loud rather than implying full coverage. (An earlier draft of this document
claimed "every day is forecast exactly once". That was false, and a test named
`test_every_test_day_is_forecast_exactly_once` was quietly asserting only uniqueness.)

### Stage 3 — `features.py` — the no-leakage argument

```python
51  known = y.shift(horizon)  # the entire no-leakage argument
```

**One line.** For a target day `i` the forecaster stands at `i - 7` and knows demand through
that day. `known[i] = y[i-7]`. Every lag and rolling window of `known` is therefore
automatically a function of `y[<= i-7]`.

**Say this — it is the best answer in the project:** *"The most predictive value available is
`y[i-1]`, and I throw it away on purpose, because at a seven-day lead time nobody knows it.
That's not a modelling compromise, it's the problem statement. `LAG_OFFSETS` starts at 0, so
the nearest history any feature carries is `y[i-7]`."*

**`build_features` (43–88).** Calendar from the target day's date (always knowable) plus lags
and rollings of `known`. Then `assert_design_is_clean`, then `dropna`.

**`LAG_OFFSETS` (26):** `[0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28]` — offset 0 *is* the seasonal
naive expressed as a feature, so the baseline and the model cannot drift apart.

**35 warmup rows are dropped, not imputed.** An earlier comment put that at 63 by adding the
rolling window length; the test asserted 35 and was right — `rolling(28)` over the
already-shifted target finishes one row before the deepest lag does.

### Stage 4 — `metrics.py` — the convention that inverted my headline

**`mase` (48–65).** `model_MAE / scale`, where `scale` is the seasonal naive's MAE **from
training data**.

**This is the strongest thing in the project and it is a bug I found, not a design I had.**
An earlier version divided by the naive's error *on the rows being scored*. Because the holdout
is a steep seasonal decline where the naive itself does badly, that denominator inflated from
857 to 1,514 — a factor of 1.77 — and reported a true MASE of **1.098** as **0.622**, i.e. it
manufactured a 38% improvement over a model that actually loses to the baseline.

**`naive_scale` (82–98).** Computes it once, from the first 640 rows. The `upto` argument is
what keeps the holdout out of the denominator.

**Be precise if asked why it matters:** *"The denominator is the yardstick. If you measure the
yardstick on the same hard ground you're testing on, it gets longer and you look better. The
harder the period, the more skill you appear to have."*

**`smape` (40–45).** Symmetric MAPE, bounded to 200%, finite when either side is zero. MAPE
divides by the actual and explodes on small counts. This series happens to have no zeros — a
gift, not a guarantee.

**Why not R²:** on a trending series it rewards predicting the trend and says nothing about
whether the number you hand to a scheduler is right. It can rise while the forecast worsens.

### Stage 5 — `train.py` — selection

```python
77  train = feat.iloc[:train_end]
78  test = feat.iloc[test_start:test_end]
81  est = clone(model)
134 best = min(results, key=lambda r: r["dev"]["mase"])
```

**Line 81 —** `clone` per fold. Without it the caller's estimator becomes the fitted step, so
fold 5 inherits fold 4's state. A test asserts the caller's model is never fitted in place.

**Line 134 —** the selection key is development MASE and nothing else. `select` never receives
the holdout folds, so there is no path by which a holdout metric reaches the choice.

**`_scaled` (28–38).** Ridge and Poisson get a `StandardScaler` inside the fold.

**Say this if asked why:** *"`t_index` runs 0–700 and the target averages 4,504, so an unscaled
Poisson fit hits invalid values in its Newton step and converges in one iteration to a constant
equal to that fold's training-window mean — a flat forecast between roughly 3,600 and 4,700
rentals, with no error raised. Silent, and it looks like a working model."*

**The four models (`candidate_models`, 41–59).** Ridge, Poisson, random forest, gradient boosting. Two of them
scale, two don't — a split rule is invariant to monotone rescaling.

### Stage 6 — `evaluate.py` — model to deliverable

**`holdout_forecast` (42–83).** Walks the holdout, refitting the point model **and both
quantile models** per fold on that fold's training window.

**Line 67** — the quantile-crossing guard:

```python
67  lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
```

Independently fitted quantile models occasionally cross. It's rare enough to be invisible in a
summary and common enough to matter: a band whose lower edge sits above its upper edge reports
zero width and breaks every coverage figure derived from it. I found it because a test asserting
`lower <= upper` failed, and the first thing I did was assume the test was wrong.

**`interval_diagnosis` (86–135).** Explains the band rather than just reporting it. Returns
model bias, naive bias, the difference, and a *derived* direction string.

**Be honest about the direction field:** it used to be a literal `"one-sided: actuals fall below
the forecast"` that fired regardless of the arithmetic. A run of an under-predicting model
produced a report asserting the opposite of its own numbers. Now derived from
`share_below_forecast`.

**`residual_acf` (181–193).** The single most useful diagnostic here. Lag-7 autocorrelation of
the holdout residuals is **0.049**, well inside the 95% bound of 0.214 — no weekly structure
left. The model's problem is level bias, not missing pattern. That distinction is the whole
section-6 argument.

**`run` (310–379).** Takes `out=None`. See section 9 — that parameter is itself a lesson.

## 5. Every headline number

| Number | Value | Produced by | Test |
|---|---|---|---|
| Rows / span | 731, 2011-01-01→2012-12-31 | `data_quality_report` | `test_data_loader.py` |
| Target mean / sd | 4,504.3 / 1,937.2 | `data_quality_report` | `test_data_loader.py` |
| `cnt == casual+registered` | all 731 rows | `assert_target_integrity` | `test_data_loader.py` |
| Warmup dropped | 35 rows | `build_features` | `test_features.py` |
| Design columns | 28 | `design_columns` | `test_features.py` |
| MASE scale | 856.92 (from 640 rows) | `naive_scale` | `test_metrics.py` |
| Dev folds / days | 33 / 231 | `walk_forward_folds` | `test_splits.py` |
| Holdout folds / days | 12 / 84 | `partition_folds` | `test_splits.py` |
| Model table | 8 rows | `train.select` | `test_pipeline.py` |
| Dev MASE (winner) | 0.796 | `accuracy_report` | `test_pipeline.py` |
| Holdout MASE | 1.098 | `accuracy_report` | `test_pipeline.py` |
| Bias / naive bias | +472.8 / +459.7 | `interval_diagnosis` | `test_pipeline.py` |
| Coverage | 69.0% vs 80% | `interval_coverage` | `test_pipeline.py` |
| Weather gain | 35.5% (GB) to 5.2% (Poisson) | dev results | `test_pipeline.py` |
| Forecast bytes | byte-identical across two runs | `holdout_forecast` | `test_pipeline.py` |
| Total tests | 132 | — | doc test |

**Coverage gaps I am disclosing rather than papering over:** `test_data_loader.py` pins the
shape and the target identity but not the EDA segment values; and the doc-fidelity test pins
this README's numbers to `metrics.json` but cannot catch a model change that moves a number no
test quotes.

## 6. The evaluation-validity story — your strongest asset

Three real defects, each found by adversarial review, each now pinned by a test.

### Bug 1 — MASE scaled on the rows being scored

Described in section 4. This one **inverted the headline**: 1.098 reported as 0.622. The
general form is worth stating: *the harder the evaluation period, the more skill a model
appears to have if you measure the yardstick on that same period.* My holdout was a steep
autumn decline, the naive did badly on it, and dividing by that inflated denominator did the
rest.

### Bug 2 — the pipeline-level leakage guard corrupted the wrong rows

The feature-level guard was genuine and mutation-tested. But the *pipeline-level* guard
corrupted positions in the raw frame using bounds that were positions in the **feature** frame,
which drops 35 warmup rows. It was corrupting each fold's training tail 35–42 days before the
test block — rows that legitimately feed the test features — and never touching the actuals it
claimed to protect. It passed, and it was measuring the wrong thing.

Mutation testing then showed something worse: a model that literally reads its own answer from
the target column **passed that test unchanged**.

Fixed with two spies that record how many rows each fit saw, checked against each fold's
`train_end`. That catches the mutation — training on the test block — which every test at the time
let through. It was the single most valuable finding of the review.

### Bug 3 — the test suite was deleting the project's own deliverable

Not a modelling bug, but the one that would have embarrassed me in an interview. Two tests
called `clean_outputs()`, which defaults to the production paths. The first passed no
arguments; after that was fixed, a second passed two of five path arguments and deleted the
other three. Net effect: a green suite, and no `evaluation.json`, no `eda_findings.json` and no
forecast on disk. The committed artefact had briefly been synthetic fixture output.

Fixed structurally, not by vigilance: `evaluate.run` and `clean_outputs` each take **one** `out`
dict, so a call cannot be half-filled. A test parses the test sources and fails if any
production-writing call lacks `out=`.

**Say this if asked what you learned:** *"partial injection is worse than none, because it looks
hermetic."*

### The one I found by refusing to believe a table

The per-horizon table showed day 1 *worse* than day 3 — contradicting the premise of the
table. Chasing it found that **every holdout fold starts on Sunday**, so "horizon 1" is always
Sunday and "horizon 3" always Tuesday. The table measured weekday, not lead time. Mean demand
across its columns runs 4,494 (Sun) to 5,509 (Wed).

Labelled confounded on every row, in the JSON, and in the figure title — rather than quoted as
insight or quietly dropped.

## 7. Live code walkthrough script

| # | Open | Say |
|---|---|---|
| 1 | `README.md` §1 | The 60-second version, including that the model loses to naive on holdout. |
| 2 | `src/features.py:51` | The horizon shift. The single line the project rests on. |
| 3 | `src/features.py:26` | `LAG_OFFSETS` starts at 0 — `y[i-1]` is unknowable and excluded. |
| 4 | `src/splits.py:43` | `train_end = test_start - horizon`. Why it isn't `test_start`. |
| 5 | `src/metrics.py:82` | `naive_scale(..., upto=...)` — the convention that inverted my headline. |
| 6 | `src/metrics.py:48` | Why the denominator must come from training data. |
| 7 | `src/train.py:77-81` | The slicing and `clone`. |
| 8 | `src/evaluate.py:67` | Quantile crossing. Found because a test failed and I was wrong. |
| 9 | `src/evaluate.py:86` | `interval_diagnosis` — bias decomposed against the naive. |
| 10 | `src/evaluate.py:258` | `horizon_weekday_map` — the confound I found and disclosed. |
| 11 | `src/run_all.py:38` | `clean_outputs(out=...)` — why tests must be hermetic. |

**If they ask you to run something:**

```bash
python -m src.run_all      # ~3 min
python -m pytest tests -q  # 132 tests, ~4-7 min
```

Then: *"132 tests. The feature guard is a perturbation test, not an inspection — it corrupts
specific days and watches which feature rows move, so a plausible-looking implementation can't
satisfy it. The fold guard records how many rows each fit saw, because a prediction-level
assertion can't tell you a model was trained on its own test block."*

## 8. Question bank

**"Why seven days?"** It's the shortest horizon at which you commit resources before you can
observe the outcome. It also produces the most interesting constraint in the project.

**"Your model loses to the baseline. Why ship it?"** Two honest reasons. First, the loss is
+10% on MASE while development showed −20%, and the gap is fully explained: the holdout is a
steep seasonal decline and a tree ensemble cannot extrapolate a trend, so it sits above the
falling level. The seasonal naive is over-predicting too — by 460 against the model's 473.
Second, what the project actually delivers is the diagnosis and the infrastructure: a provably
leak-free feature pipeline, a correct validation harness, a calibrated-coverage check, and a
quantified answer to "how much is a weather forecast worth?" — 35% of the error for the boosted
model, 5% for Poisson. That's reusable; a 0.8 MASE claim would not be.

**"Would you deploy it?"** Not as the sole forecast. I'd deploy the naive as the operational
default and the model as a challenger in shadow mode, with the monitoring in section 11. The
first production change I'd make is the bias fix, not a new model.

**"How is this different from a random split?"** A random split trains on March and tests on
January. `TimeSeriesSplit` gives you expanding windows but not the alignment property — that
every feature of a test row comes from *its own* training window. I wanted to own and test that
in ten lines.

**"What is MASE and why?"** Mean absolute error divided by the seasonal naive's mean absolute
error. Below 1 beats the naive. Scale-free, so it survives the fleet doubling. The subtlety is
*whose* naive error — and that subtlety cost me my headline once.

**"Why is your band only 69% covered?"** Because the forecast is biased high during the
seasonal decline and a band centred on a high forecast misses everything below it. 67.9% of
actuals fall below the forecast. I could widen the band, but symmetric quantiles would need a
lower edge so far down the band becomes useless for planning. The honest fixes are asymmetric
quantile levels or a bias term, both fitted on development data.

**"You built both weather variants — is that cheating?"** No, it's the opposite. The weather
columns are contemporaneous actuals, so using them assumes a weather forecast you don't have.
Building both turns that assumption into a measured number instead of a hidden dependency. And
the per-model spread is the finding: expressive models exploit weather, linear ones don't.

**"How would you improve it?"** Section 11 of the README, in order: asymmetric intervals or a
bias term first, then a naive blend, then a model class that can extrapolate. All three must be
validated on data this project hasn't seen.

## 9. Traps — what NOT to say

| ❌ Don't say | ✅ Say instead |
|---|---|
| "The model beats the baseline by 38%" | It beats naive on development (0.796) and loses on holdout (1.098). |
| "MASE 0.62" | That was a denominator bug. The number is 1.098. |
| "Error grows with lead time" | My per-horizon table is confounded with day of week and can't answer that. |
| "Weather improves accuracy" | Weather removes 35% of the error for gradient boosting and 5% for Poisson — and it isn't available seven days out. |
| "The interval is an 80% confidence interval" | It was fitted to be 80% and covers 69%. Here's the diagnosis. |
| "The model finds demand drivers" | It forecasts. Any causal reading is mine, not the model's. |
| "Out-of-fold validation" | Out-of-fold, not out-of-time. Each fold shares one distribution. |

## 10. Limitations — say these before they find them

1. The model does not beat the seasonal naive on the holdout.
2. The headline depends on a weather forecast that doesn't exist in the data.
3. Two years, one annual cycle — the model has seen autumn once.
4. Out-of-fold is not out-of-time; within-holdout drift isn't modelled.
5. The per-horizon table is confounded with day of week.
6. The interval under-covers and is labelled so.
7. **The holdout is spent** — section 6 diagnoses it, so further iteration needs new data.

Point 7 is the one to volunteer. It's the difference between a project and a fishing expedition.

## 11. If they ask you to extend it

- **Asymmetric quantile levels / bias term.** The highest-value fix. Must be fitted on
  development folds and validated on fresh data.
- **Naive blend.** Model and naive share the same +470 bias, so a convex combination often
  beats both. Tune the weight on development only.
- **A model that extrapolates.** A linear trend outside the trees, aimed at level bias rather
  than pattern.
- **Hourly data.** `hour.csv` has 17,000+ rows, a much better-behaved problem and a harder one.
- **Stagger the fold phase** to de-confound the per-horizon table, accepting overlapping test
  windows and losing the no-double-counting guarantee.

## 12. Rehearsal checklist

- [ ] The 60-second version, including that the model loses to naive on holdout.
- [ ] All eight model rows; know which column selected the winner.
- [ ] `known = y.shift(horizon)` explained in one sentence, with the `y[i-1]` sacrifice.
- [ ] The MASE denominator bug, with both numbers: 1.098 true, 0.622 reported.
- [ ] Bias decomposition: model +472.8, naive +459.7, difference +13.1.
- [ ] The per-horizon weekday confound, with the Sunday detail.
- [ ] The three evaluation defects, including the one that deleted my own artefacts.
- [ ] Why trees can't extrapolate, and what that predicts about the residual ACF (0.049).
- [ ] Name three limitations before you're asked.
- [ ] Run `python -m src.run_all` and `python -m pytest tests -q` from memory.
