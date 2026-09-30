"""Documentation fidelity. The docs must quote the artefacts, not remember them.

In the churn project, docs drift was not caught by anything: a model change moved PR-AUC and
the prose kept claiming the old figure, and a stale number read exactly like a fresh one in
review. Here the numbers in README.md and docs/INTERVIEW_GUIDE.md are compared against
reports/metrics.json on every run, so a change to the pipeline that moves a number fails the
build instead of quietly making the documentation wrong.
"""
import json
import re
import unittest
from pathlib import Path

from src import config

ROOT = config.ROOT
README = (ROOT / "README.md").read_text(encoding="utf-8")
GUIDE = (ROOT / "docs" / "INTERVIEW_GUIDE.md").read_text(encoding="utf-8")
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
EVAL = json.loads((ROOT / "reports" / "evaluation.json").read_text(encoding="utf-8"))
EDA = json.loads((ROOT / "reports" / "eda_findings.json").read_text(encoding="utf-8"))

HOLD = METRICS["holdout"]
DEV = {r["model"] + "/" + r["feature_set"]: r["dev"] for r in METRICS["results"]}
WINNER_DEV = DEV[METRICS["best_model"] + "/" + METRICS["best_feature_set"]]
DOCS = {"README": README, "GUIDE": GUIDE}


class TestArtefactsExist(unittest.TestCase):
    def test_generated_artefacts_exist(self):
        for path in (
            config.METRICS_JSON,
            config.EVAL_JSON,
            config.EDA_JSON,
            config.SCORED_CSV,
        ):
            self.assertTrue(path.exists(), f"{path.name} missing; run python -m src.run_all")

    def test_the_figure_count_quoted_is_correct(self):
        actual = len(list(config.FIG_DIR.glob("*.png")))
        self.assertEqual(actual, 7, "figure count changed; update the docs")
        for name, doc in DOCS.items():
            self.assertNotIn("6 PNGs", doc)
            self.assertNotIn("8 figures", doc)


class TestHeadlineNumbersMatchArtefacts(unittest.TestCase):
    def test_selected_model_is_named_in_both_docs(self):
        self.assertIn(METRICS["best_model"].replace("_", " "), README.lower().replace("_", " "))
        self.assertIn(METRICS["best_feature_set"].replace("_", " "), README.lower().replace("_", " "))

    def test_development_mase_is_quoted_correctly(self):
        self.assertIn(f"MASE {WINNER_DEV['mase']:.3f}", README)

    def test_holdout_mase_is_quoted_correctly(self):
        """The headline. If this number changes, the docs are wrong."""
        self.assertIn(f"MASE {HOLD['mase']:.3f}", README)
        self.assertIn(f"MASE {HOLD['mase']:.3f}", GUIDE)

    def test_holdout_error_metrics_are_quoted_correctly(self):
        self.assertIn(f"{HOLD['mae']:,.1f}", README)
        self.assertIn(f"{HOLD['rmse']:,.1f}", README)
        self.assertIn(f"{HOLD['smape']:.2f}%", README)

    def test_mase_scale_is_quoted_and_attributed_to_development_rows(self):
        self.assertIn(f"{HOLD['mase_scale']:,.1f}", README)
        self.assertIn(f"{METRICS['mase_scale_rows']}", README)
        # The scale must not be recomputed from the holdout.
        self.assertLess(METRICS["mase_scale_rows"], METRICS["data_quality"]["clean_rows"])

    def test_bias_decomposition_is_quoted_correctly(self):
        diagnosis = HOLD["interval_diagnosis"]
        self.assertIn(f"{diagnosis['model_bias']:+.1f}", README)
        self.assertIn(f"{diagnosis['naive_bias']:+.1f}", README)
        self.assertIn(f"{diagnosis['bias_above_naive']:.1f}", README)
        self.assertIn(f"{diagnosis['share_below_forecast']:.1%}", README)

    def test_interval_coverage_is_quoted_with_its_nominal_level(self):
        self.assertIn(f"{HOLD['empirical_coverage']:.1%}", README)
        self.assertIn(f"{HOLD['nominal_coverage']:.0%}", README)
        self.assertIn(f"{HOLD['mean_band_width']:,.0f}", README)

    def test_residual_acf_weekly_peak_is_quoted(self):
        self.assertIn(f"{HOLD['residual_acf_weekly_peak']:.3f}", README)

    def test_fold_counts_and_forecast_days_are_quoted_correctly(self):
        self.assertIn(f"{METRICS['n_dev_folds']} development folds", README)
        self.assertIn(f"{METRICS['n_holdout_folds']} folds", README)
        self.assertIn(f"{WINNER_DEV['n']} forecast days", README)
        self.assertIn(f"{HOLD['n']}-day holdout", README)

    def test_data_quality_figures_are_quoted_correctly(self):
        q = METRICS["data_quality"]
        self.assertIn(f"{q['clean_rows']} daily observations", README)
        self.assertIn(q["start_date"], README)
        self.assertIn(q["end_date"], README)
        self.assertIn(f"{q['target_mean']:,.1f}", README)
        self.assertIn(f"{q['target_std']:,.1f}", README)
        self.assertIn(str(q["target_min"]), README)
        self.assertIn(f"{q['target_max']:,}", README)

    def test_all_eight_development_rows_appear_in_the_readme(self):
        for row in METRICS["results"]:
            self.assertIn(
                f"{row['dev']['mae']:,.1f}", README, f"MAE for {row} missing from README"
            )
            self.assertIn(
                f"{row['dev']['mase']:.3f}", README, f"MASE for {row} missing from README"
            )


class TestDerivedClaimsAreConsistent(unittest.TestCase):
    def test_weather_gain_matches_the_development_table(self):
        """The README quotes a gain per model; recompute it and compare.

        An earlier version hardcoded "~35%" into a figure title while the four bars it sat above
        spanned 5% to 35%, so the plot argued against its own caption. The gain is now computed
        at render time, and this test pins the same arithmetic in the prose.
        """
        for model in ("gradient_boosting", "random_forest", "ridge", "poisson"):
            with_w = DEV[f"{model}/with_weather"]["mase"]
            without = DEV[f"{model}/calendar_only"]["mase"]
            gain = (without - with_w) / without
            self.assertIn(f"{gain:.1%}", README, f"weather gain for {model} missing")

    def test_the_weather_gain_spread_is_actually_spread(self):
        gains = []
        for model in ("gradient_boosting", "random_forest", "ridge", "poisson"):
            with_w = DEV[f"{model}/with_weather"]["mase"]
            without = DEV[f"{model}/calendar_only"]["mase"]
            gains.append((without - with_w) / without)
        self.assertGreater(max(gains) - min(gains), 0.15, "the per-model spread is the finding")

    def test_calendar_only_winner_is_worse_than_the_naive(self):
        """The claim that without weather the best model loses to the baseline.

        If this ever stops being true the README's central argument changes.
        """
        best_calendar = min(
            r["dev"]["mase"] for r in METRICS["results"] if r["feature_set"] == "calendar_only"
        )
        self.assertGreater(best_calendar, 1.0)
        self.assertIn("*worse* than the naive baseline", README)

    def test_eda_findings_quoted_in_the_readme_exist(self):
        self.assertIn(f"{EDA['weather']['rain_drop_vs_clear']:.1%}", README)
        self.assertIn(f"{EDA['year']['growth_second_year']:.1%}", README)
        self.assertIn(f"{EDA['weekday']['weekend_over_weekday']:.3f}", README)
        self.assertIn(
            f"{EDA['temperature']['mean_by_temp_band']['<0.3']:,.0f}", README
        )
        self.assertIn(f"{EDA['holiday']['mean_on_holiday']:,.1f}", README)
        self.assertIn(f"{EDA['holiday']['mean_on_nonholiday']:,.1f}", README)
        self.assertIn(f"{EDA['holiday']['drop_vs_nonholiday']:.1%}", README)
        # The comparison group is all non-holiday days, not working days, and the two differ.
        self.assertNotEqual(
            EDA["holiday"]["mean_on_nonholiday"], EDA["holiday"]["mean_on_workingday"]
        )

    def test_the_denominator_inflation_factor_is_recomputed_not_asserted(self):
        """1.77 is derived from the artefact, and the docs must quote that.

        A previous version of both documents said 1.62, which came from a third denominator
        the project had already discarded. Any hand-quoted factor can drift out of sync with
        the two numbers it is derived from, so it is recomputed here.
        """
        factor = HOLD["naive_mae_same_rows"] / HOLD["mase_scale"]
        self.assertAlmostEqual(factor, 1.7667, places=3)
        for name, doc in DOCS.items():
            self.assertIn(f"{factor:.2f}", doc, f"{name} quotes the wrong inflation factor")
            self.assertNotIn("1.62", doc, f"{name} still quotes a superseded factor")

    def test_the_7_day_block_belonging_to_neither_side_is_disclosed(self):
        """One fold straddles the holdout boundary and is excluded from both sides.

        That is the conservative choice - folding it into development would spend holdout days
        during selection - but it means seven days are never scored, and the docs have to say
        so rather than implying full coverage.
        """
        for name, doc in DOCS.items():
            self.assertIn("straddle", doc.lower(), f"{name} omits the straddling block")

    def test_coverage_claim_is_no_day_twice_not_every_day_once(self):
        """The fold geometry guarantees uniqueness, not completeness.

        35 warmup rows, the straddling block and the final two days are never scored, so
        "every day is forecast exactly once" would be false. A test that no day is forecast
        twice is the real, and weaker, guarantee.
        """
        for name, doc in DOCS.items():
            # "forecast twice" is split by bold markers and line wrapping in one document, so
            # the substantive claim is matched on "twice" alone.
            self.assertIn("twice", doc, f"{name} does not state the real guarantee")
            # The retracted phrase is allowed only where it is explicitly being corrected,
            # for the same reason the retracted 0.622 figure is allowed in a retraction.
            for match in re.finditer(r"every day is forecast exactly once", doc, re.I):
                window = doc[max(0, match.start() - 300) : match.end() + 200].lower()
                self.assertTrue(
                    any(w in window for w in ("false", "earlier draft", "claimed", "an earlier")),
                    f"{name}: the false coverage claim is asserted rather than retracted",
                )

    def test_per_horizon_confound_is_disclosed_everywhere_it_is_tabulated(self):
        self.assertTrue(HOLD["per_horizon_is_confounded_with_weekday"])
        for name, doc in DOCS.items():
            self.assertIn("confound", doc.lower(), f"{name} does not disclose the confound")
        self.assertIn("per_horizon_is_confounded_with_weekday", README)
        # Every row of the artefact carries the flag and its weekday.
        for row in METRICS["per_horizon"]:
            self.assertTrue(row["weekday_confounded"])
            self.assertIn(row["weekday"], {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"})
        for key, weekdays in EVAL["horizon_weekday_confound"].items():
            self.assertEqual(len(weekdays), 1, f"{key} mixes weekdays {weekdays}")

    def test_the_holdout_worse_than_development_claim_is_backed(self):
        """The README leads with a generalisation gap. Verify there is one."""
        self.assertGreater(HOLD["mase"], WINNER_DEV["mase"])
        self.assertIn("does not beat", README)

    def test_bias_above_naive_is_small_relative_to_the_bias_itself(self):
        """The 'inherits the problem, does not add to it' claim."""
        d = HOLD["interval_diagnosis"]
        self.assertLess(abs(d["bias_above_naive"]), 0.1 * abs(d["model_bias"]))


class TestNoStaleOrFalseFigures(unittest.TestCase):
    def test_superseded_numbers_are_blocked(self):
        # The 0.622 figure is the MASE denominator bug, not a result. It must never reappear.
        stale = [
            "0.6216", "17 development folds", "119 forecast days", "0.8048",
            "2.90x", "about 35 seconds", "69 unit tests", "Segment 4",
            "0.4909", "0.4930", "0.5260", "0.75x", "540 of", "6 PNGs", "5 PNGs",
        ]
        for name, doc in DOCS.items():
            for value in stale:
                self.assertNotIn(value, doc, f"{name} still quotes {value!r}")

    def test_the_beats_naive_claim_is_always_scoped_to_development(self):
        """It is true on development and false on the holdout, so the docs may say it - but
        only when the sentence names development. An unqualified claim is an overclaim."""
        for name, doc in DOCS.items():
            for phrase in ("beats the seasonal naive", "beats the naive"):
                for match in re.finditer(re.escape(phrase), doc):
                    # A window, not a sentence: these documents hard-wrap, so a sentence
                    # splitter cuts "development" off the end of the claim it qualifies.
                    window = doc[max(0, match.start() - 200) : match.end() + 200].lower()
                    if "below 1" in window:
                        # "Below 1 beats the naive" is the definition of MASE, not a claim
                        # about this project. Only claims about the model are in scope.
                        continue
                    self.assertTrue(
                        any(w in window for w in ("development", "holdout", "does not", "loses")),
                        f"{name}: unqualified baseline claim near {phrase!r}",
                    )

    def test_the_retracted_mase_figure_only_ever_appears_as_a_retraction(self):
        """0.622 is quoted in both docs, deliberately, as the number the denominator bug
        produced. It must never appear as a result."""
        for name, doc in DOCS.items():
            for match in re.finditer(r"0\.622", doc):
                window = doc[max(0, match.start() - 260) : match.end() + 260].lower()
                self.assertTrue(
                    any(w in window for w in ("bug", "denominator", "manufactured",
                                               "inverted", "don't say", "retracted")),
                    f"{name}: 0.622 appears without being marked as the retracted figure",
                )

    def test_no_claim_of_a_confidence_level_the_band_does_not_meet(self):
        for name, doc in DOCS.items():
            low = doc.lower()
            self.assertNotIn("calibrated 80%", low)
            if "80%" in doc:
                # Any mention must be qualified.
                self.assertTrue(
                    "under-cover" in low or "nominal" in low or "69" in doc,
                    f"{name} mentions 80% without qualifying the coverage",
                )

    def test_no_causal_claim_from_the_weather_features(self):
        for name, doc in DOCS.items():
            low = doc.lower()
            for phrase in ("weather causes", "rain causes", "proves that weather"):
                self.assertNotIn(phrase, low, f"{name} makes a causal claim: {phrase!r}")

    def test_out_of_time_is_never_claimed(self):
        for name, doc in DOCS.items():
            self.assertIn("out-of-time", doc, f"{name} omits the out-of-fold/out-of-time caveat")

    def test_the_holdout_is_admitted_to_be_spent(self):
        for name, doc in DOCS.items():
            self.assertIn("spent", doc.lower(), f"{name} does not admit the holdout is spent")

    def test_no_feature_importance_claim(self):
        for name, doc in DOCS.items():
            self.assertNotIn("feature importance", doc.lower())


class TestDocMechanics(unittest.TestCase):
    def test_test_count_claim_matches_reality(self):
        """Counted with a line-anchored regex, not a substring.

        The first version used `.count("    def test")` over each test file, which counted
        its own search string as a thirty-fifth test in this file and inflated the total by
        one. The docs were then "correct" against a number that was itself wrong.
        """
        pattern = re.compile(r"^\s*def test_\w+\(self", re.M)
        total = sum(
            len(pattern.findall(path.read_text(encoding="utf-8")))
            for path in (ROOT / "tests").glob("test_*.py")
        )
        for name, doc in DOCS.items():
            self.assertIn(f"{total} tests", doc, f"{name} quotes the wrong test count")

    def test_docs_reference_the_right_paths(self):
        for name, doc in DOCS.items():
            for name_ in ("splits.py", "features.py", "metrics.py"):
                self.assertIn(name_, doc, f"{name} never mentions {name_}")
        self.assertIn("data/processed", README)
        self.assertIn("splits.py", README)

    def test_run_command_is_real(self):
        for name, doc in DOCS.items():
            self.assertIn("python -m src.run_all", doc)
            self.assertIn("python -m pytest tests -q", doc)
        self.assertTrue(Path("src/run_all.py").exists())

    def test_raw_data_bootstrap_is_documented(self):
        self.assertIn("auto-downloaded", README)
        self.assertIn("downloads data", README)
        for name, doc in DOCS.items():
            self.assertIn("python -m src.run_all", doc, f"{name} omits the bootstrap command")

    def test_hermeticity_is_documented_because_it_bit_me(self):
        self.assertIn("hermetic", README.lower())
        self.assertIn("clean_outputs", README)
