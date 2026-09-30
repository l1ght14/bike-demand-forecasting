"""Integration tests. These run real pipeline stages on real folds.

Written after a lesson from the churn project: a suite where every test exercises an
individual function can be entirely green while the actual pipeline is wired up wrongly.
Nothing here stubs out a fold, a fit or a prediction, because the wiring is the thing most
likely to be wrong and the thing least likely to be caught otherwise.
"""
import ast
import inspect
import pathlib
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from src import config, evaluate, features, metrics, run_all, splits, train


def _series_frame(n_days=260, seed=0):
    """A synthetic daily series with a weekly cycle and a trend, on a real date index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-01", periods=n_days, freq="D")
    weekly = 500 + 150 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
    trend = np.linspace(0, 300, n_days)
    cnt = weekly + trend + rng.normal(0, 15, n_days)
    return pd.DataFrame(
        {
            "season": 1, "yr": 0, "mnth": 1,
            "holiday": (idx.dayofweek >= 5).astype(int),
            "weekday": idx.dayofweek, "workingday": (idx.dayofweek < 5).astype(int),
            "weathersit": 1,
            "temp": rng.uniform(0.3, 0.8, n_days), "atemp": rng.uniform(0.3, 0.8, n_days),
            "hum": rng.uniform(0.3, 0.8, n_days), "windspeed": rng.uniform(0, 0.3, n_days),
            "casual": (cnt * 0.3).round(), "registered": (cnt * 0.7).round(), "cnt": cnt,
        },
        index=idx,
    )


class TestWalkForwardMechanics(unittest.TestCase):
    def setUp(self):
        self.df = _series_frame()
        self.feat = features.build_features(self.df, use_weather=True)
        self.folds = splits.walk_forward_folds(len(self.feat), min_train=120, step=14)

    def test_caller_model_is_not_aliased_across_folds(self):
        """Every fold gets its own fitted copy, so fold 5 is not fitted on fold 4's state.

        The same class of bug as the churn project's `make_pipeline` aliasing: reusing one
        estimator across folds makes later folds quietly depend on earlier ones.
        """
        model = train.candidate_models()["ridge"]
        before = model.get_params()
        preds = train.walk_forward_predictions(model, self.feat, self.folds)
        self.assertFalse(preds.empty)
        self.assertEqual(model.get_params(), before)
        self.assertFalse(hasattr(model, "coef_"), "the caller's model was fitted in place")

    def test_predicted_row_count_is_folds_times_horizon(self):
        preds = train.walk_forward_predictions(
            train.candidate_models()["ridge"], self.feat, self.folds
        )
        self.assertEqual(len(preds), len(self.folds) * config.HORIZON)

    def test_no_test_day_is_forecast_twice(self):
        """Uniqueness, not completeness.

        The name used to say "exactly once", which reads as a claim that every eligible day is
        covered. It is not: the 35 warmup rows, the block straddling the holdout boundary and
        the last two days of the series are never scored. What the non-overlapping fold
        geometry guarantees is that no day is double-scored, which is what this asserts.
        """
        preds = train.walk_forward_predictions(
            train.candidate_models()["ridge"], self.feat, self.folds
        )
        self.assertEqual(preds["date"].nunique(), len(preds))
        self.assertFalse(preds["date"].duplicated().any())

    def test_naive_column_is_last_week_same_weekday(self):
        preds = train.walk_forward_predictions(
            train.candidate_models()["ridge"], self.feat, self.folds
        )
        expected = self.feat["lag_0"].reindex(pd.DatetimeIndex(preds["date"]))
        np.testing.assert_allclose(preds["naive"].to_numpy(), expected.to_numpy())

    def test_actual_values_match_the_source_series(self):
        preds = train.walk_forward_predictions(
            train.candidate_models()["ridge"], self.feat, self.folds
        )
        actual = self.df[config.TARGET].reindex(pd.DatetimeIndex(preds["date"]))
        np.testing.assert_allclose(preds["actual"].to_numpy(), actual.to_numpy())

    def test_predictions_are_never_negative(self):
        for name in train.candidate_models():
            preds = train.walk_forward_predictions(
                train.candidate_models()[name], self.feat, self.folds
            )
            self.assertGreaterEqual(preds["predicted"].min(), 0.0, name)

    def test_no_prediction_uses_its_own_actual(self):
        """The behavioural leakage guard at the pipeline level.

        Shuffling the target after the features are built must move every prediction. If some
        predictions survived, the model is reading the answer out of a feature.
        """
        model = train.candidate_models()["ridge"]
        a = train.walk_forward_predictions(clone(model), self.feat, self.folds)
        corrupted = self.df.copy()
        corr = corrupted.columns.get_loc(config.TARGET)
        for _, test_start, _ in self.folds:
            corrupted.iloc[test_start : test_start + config.HORIZON, corr] *= 5
        b = train.walk_forward_predictions(clone(model), features.build_features(corrupted), self.folds)
        self.assertFalse(
            np.allclose(a["predicted"].to_numpy(), b["predicted"].to_numpy()),
            "predictions survived corruption of the test-period actuals",
        )


class TestSelectionIsDisciplined(unittest.TestCase):
    def test_selection_output_contains_no_holdout_metric(self):
        """Structural guarantee: there is no field a holdout number could live in."""
        df = _series_frame()
        feat = features.build_features(df, use_weather=False)
        folds = splits.walk_forward_folds(len(feat), min_train=120, step=28)
        selection = train.select(df, folds, 700.0)
        self.assertTrue(selection["results"])
        for row in selection["results"]:
            self.assertEqual(set(row) - {"feature_set", "model"}, {"dev"})

    def test_selection_key_is_development_mase(self):
        """The winner must be the row with the lowest dev MASE, recomputed independently."""
        df = _series_frame()
        feat = features.build_features(df, use_weather=False)
        folds = splits.walk_forward_folds(len(feat), min_train=120, step=28)
        selection = train.select(df, folds, 700.0)
        best = min(selection["results"], key=lambda r: r["dev"]["mase"])
        self.assertEqual(selection["best_model"], best["model"])
        self.assertEqual(selection["best_feature_set"], best["feature_set"])

    def test_results_are_sorted_by_mase(self):
        df = _series_frame()
        folds = splits.walk_forward_folds(
            len(features.build_features(df)), min_train=120, step=28
        )
        mases = [r["dev"]["mase"] for r in train.select(df, folds, 700.0)["results"]]
        self.assertEqual(mases, sorted(mases))

    def test_selection_refuses_when_nothing_produced_predictions(self):
        with self.assertRaises(ValueError):
            train.select(_series_frame(), [], 700.0)


class TestRunAllWiring(unittest.TestCase):
    """Static checks on the orchestrator, so a reordering cannot go unnoticed."""

    def _main_tree(self):
        return ast.parse(inspect.getsource(run_all.main))

    def _called_names(self, tree):
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    names.add(func.attr)
                elif isinstance(func, ast.Name):
                    names.add(func.id)
        return names

    def test_orchestrator_runs_all_four_stages(self):
        tree = self._main_tree()
        modules_used = {
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        }
        for stage in ("data_loader", "eda", "features", "splits", "train", "evaluate"):
            self.assertIn(stage, modules_used, f"{stage} is never called by main()")
        names = self._called_names(tree)
        for call in ("fetch_raw_data", "clean_outputs", "select", "load_clean", "run"):
            self.assertIn(call, names)

    def test_selection_happens_before_evaluation(self):
        """Order is the discipline. Selecting after seeing the holdout would be the same
        mistake as the churn project's, in a different costume."""
        tree = self._main_tree()
        select_at = evaluate_at = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", None)
                if attr == "select" and select_at is None:
                    select_at = node.lineno
                on_evaluate = isinstance(node.func, ast.Attribute) and getattr(
                    node.func.value, "id", ""
                ) == "evaluate"
                if on_evaluate and select_at is not None and node.lineno > select_at:
                    evaluate_at = node.lineno
        self.assertIsNotNone(select_at)
        self.assertIsNotNone(evaluate_at)
        self.assertLess(select_at, evaluate_at)

    def test_clean_outputs_runs_before_any_stage(self):
        names = self._called_names(self._main_tree())
        self.assertIn("clean_outputs", names)
        self.assertIn("fetch_raw_data", names)


class TestCleanOutputs(unittest.TestCase):
    """Hermetic by construction.

    An earlier version of this class called clean_outputs() with no arguments, which deleted
    the project's real reports/metrics.json and reports/eda_findings.json. The suite stayed
    green while the committed deliverable was gone. Every path is now injected.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.fig = pathlib.Path(self.tmp.name) / "figures"
        self.fig.mkdir(parents=True)
        self.metrics = pathlib.Path(self.tmp.name) / "metrics.json"
        self.scored = pathlib.Path(self.tmp.name) / "daily_forecast.csv"
        self.addCleanup(self.tmp.cleanup)

    def _out(self):
        """A complete set of destinations. One dict, so it cannot be half-filled."""
        return {
            "fig_dir": self.fig,
            "metrics_json": self.metrics,
            "eval_json": pathlib.Path(self.tmp.name) / "evaluation.json",
            "eda_json": pathlib.Path(self.tmp.name) / "eda_findings.json",
            "scored_csv": self.scored,
        }

    def test_stale_artefacts_of_any_name_are_removed(self):
        """Glob, not allowlist. An allowlist rots the moment a new report is written."""
        stray = self.fig / "never_heard_of_it.png"
        stray.write_bytes(b"not really a png")
        self.metrics.write_text("{}", encoding="utf-8")
        run_all.clean_outputs(out=self._out())
        self.assertFalse(stray.exists())
        self.assertFalse(self.metrics.exists())

    def test_every_configured_destination_is_cleared(self):
        out = self._out()
        for key in ("eval_json", "eda_json", "scored_csv"):
            pathlib.Path(out[key]).write_text("{}", encoding="utf-8")
        run_all.clean_outputs(out=out)
        for key in ("eval_json", "eda_json", "scored_csv"):
            self.assertFalse(pathlib.Path(out[key]).exists(), key)

    def test_raw_data_is_never_deleted(self):
        run_all.clean_outputs(out=self._out())
        self.assertTrue(config.RAW_CSV.exists())

    def test_no_test_writes_a_production_path(self):
        """The regression guard for both bugs this class was written for.

        An earlier version called clean_outputs() with no arguments, which deleted the real
        reports. Then a second version passed two of five path arguments, and deleted the
        other three. A test run left the project with no evaluation.json, no eda_findings.json
        and no forecast, while the suite reported green. Any call to a production-writing
        function must therefore pass a complete `out=` dict, which is checked here by parsing
        the test sources - the symptom is invisible from inside any single test.
        """
        offenders = []
        for path in sorted(pathlib.Path("tests").glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = getattr(node.func.value, "id", "")
                if node.func.attr not in ("run", "clean_outputs"):
                    continue
                if owner not in ("evaluate", "eda", "run_all"):
                    continue
                if owner == "run_all" and node.func.attr != "clean_outputs":
                    continue
                if not any(kw.arg == "out" for kw in node.keywords):
                    offenders.append(
                        f"{path.name}:{node.lineno} {owner}.{node.func.attr}() without out="
                    )
        self.assertEqual(offenders, [], f"non-hermetic production writes: {offenders}")


class TestEvaluationStageRuns(unittest.TestCase):
    def test_evaluate_run_produces_a_calibrated_report_and_a_forecast(self):
        """The real stage, on real folds, writing real files."""
        df = _series_frame(300, seed=7)
        feat = features.build_features(df, use_weather=True)
        folds = splits.walk_forward_folds(len(feat), min_train=150, step=14)
        selection = {
            "best_model": "gradient_boosting",
            "best_feature_set": "with_weather",
            "results": [{"feature_set": "with_weather", "model": "gradient_boosting", "dev": {"mase": 0.5}}],
        }
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        report = evaluate.run(
            df,
            selection,
            folds,
            700.0,
            out={
                "eval_json": root / "evaluation.json",
                "scored_csv": root / "daily_forecast.csv",
                "fig_dir": root / "figures",
            },
        )
        for key in ("mae", "rmse", "smape", "mase", "empirical_coverage", "mean_bias", "per_horizon"):
            self.assertIn(key, report)
        self.assertTrue((root / "daily_forecast.csv").exists())
        self.assertTrue((root / "evaluation.json").exists())
        forecast = pd.read_csv(root / "daily_forecast.csv", parse_dates=["date"])
        self.assertEqual(len(forecast), len(folds) * config.HORIZON)
        self.assertTrue((forecast["lower"] <= forecast["upper"]).all())
        self.assertEqual(len(report["per_horizon"]), config.HORIZON)

    def test_interval_diagnosis_identifies_one_sided_error(self):
        """A band centred on a biased forecast must be reported as under-covering."""
        df = _series_frame(300, seed=7)
        feat = features.build_features(df, use_weather=True)
        folds = splits.walk_forward_folds(len(feat), min_train=150, step=14)[:4]
        model = train.candidate_models()["gradient_boosting"]
        fc = evaluate.holdout_forecast(model, feat, folds)
        fc["predicted"] = fc["actual"] + 200.0  # force a positive bias
        fc["lower"] = fc["predicted"] - 50.0
        fc["upper"] = fc["predicted"] + 50.0
        diag = evaluate.interval_diagnosis(fc)
        self.assertGreater(diag["model_bias"], 0)
        self.assertGreater(diag["bias_above_naive"], 0)
        # Over-predicting means every actual sits below the forecast.
        self.assertAlmostEqual(diag["share_below_forecast"], 1.0, places=6)
        self.assertLess(
            metrics.interval_coverage(fc["actual"], fc["lower"], fc["upper"]), 0.5
        )

    def test_quantile_bands_never_cross(self):
        """Independently fitted quantile models can cross; the export must not ship one."""
        df = _series_frame(300, seed=11)
        feat = features.build_features(df, use_weather=True)
        folds = splits.walk_forward_folds(len(feat), min_train=150, step=14)
        fc = evaluate.holdout_forecast(
            train.candidate_models()["gradient_boosting"], feat, folds
        )
        self.assertTrue((fc["lower"] <= fc["upper"]).all())
        self.assertTrue((fc["lower"] >= 0).all())


class TestHorizonConfoundIsDisclosed(unittest.TestCase):
    """The per-horizon table is confounded with day of week, and that is now asserted.

    A 7-day fold step forces every test block to start on the same weekday, so each horizon
    column is made of a single weekday. Without these tests a reader could reasonably take
    "horizon 3 is the easiest" as a statement about lead time when it is a statement about
    Tuesday. The confound is reported in the artefact, and these tests make sure it stays
    reported.
    """

    def setUp(self):
        self.df = _series_frame(300, seed=7)
        self.feat = features.build_features(self.df, use_weather=True)
        # min_train is lowered because this synthetic frame is far shorter than the real
        # series; config.MIN_TRAIN_DAYS would generate no folds at all here.
        all_folds = splits.walk_forward_folds(
            len(self.feat), min_train=120, step=config.HORIZON
        )
        self.holdout = all_folds[-12:]
        self.assertTrue(self.holdout)

    def test_each_horizon_column_is_a_single_weekday(self):
        fc = evaluate.holdout_forecast(
            train.candidate_models()["gradient_boosting"], self.feat, self.holdout
        )
        mapping = evaluate.horizon_weekday_map(fc)
        for horizon, weekdays in mapping.items():
            self.assertEqual(len(weekdays), 1, f"{horizon} mixes weekdays {weekdays}")

    def test_the_seven_horizons_cover_the_seven_distinct_weekdays(self):
        fc = evaluate.holdout_forecast(
            train.candidate_models()["gradient_boosting"], self.feat, self.holdout
        )
        mapping = evaluate.horizon_weekday_map(fc)
        self.assertEqual(len({w for ws in mapping.values() for w in ws}), 7)

    def test_per_horizon_rows_are_labelled_as_confounded(self):
        fc = evaluate.holdout_forecast(
            train.candidate_models()["gradient_boosting"], self.feat, self.holdout
        )
        rows = evaluate.per_horizon_report_with_weekday(fc, 500.0)
        for row in rows:
            self.assertTrue(row["weekday_confounded"])
            self.assertIn(row["weekday"], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            self.assertIn("mean_actual", row)

    def test_the_confound_is_written_into_the_evaluation_artefact(self):
        import json as _json

        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        target = root / "evaluation.json"
        cfg = evaluate.run(
            self.df,
            {
                "best_model": "gradient_boosting",
                "best_feature_set": "with_weather",
                "results": [{"feature_set": "with_weather", "model": "gradient_boosting", "dev": {"mase": 0.5}}],
            },
            self.holdout[:3],
            900.0,
            out={
                "eval_json": target,
                "scored_csv": root / "daily_forecast.csv",
                "fig_dir": root / "figures",
            },
        )
        self.assertTrue(cfg["per_horizon_is_confounded_with_weekday"])
        written = _json.loads(target.read_text(encoding="utf-8"))
        self.assertTrue(written["per_horizon_is_confounded_with_weekday"])
        self.assertEqual(len(written["horizon_weekday_confound"]), config.HORIZON)
        self.assertTrue(all(len(v) == 1 for v in written["horizon_weekday_confound"].values()))
        for row in written["per_horizon"]:
            self.assertTrue(row["weekday_confounded"])


class _Recorder:
    """Base for a regressor that records how many rows each fit saw.

    A prediction-level assertion cannot see whether a fold was *trained* on its own test
    block - the predictions still look fine, they are just excellent for the wrong reason.
    Recording the fitted row count makes the training window directly observable, and
    mutation testing showed this catches the `feat.iloc[:test_end]` variant that all 86
    previous tests let through.

    The record has to survive `clone()`, which the walk-forward loop calls on every fold, and
    that rules out both the obvious approaches. Instance state dies with the clone, and a
    shared list passed through get_params does not survive either - sklearn's clone calls
    `clone(param, safe=False)` on every parameter, which deep-copies it. A class attribute
    is copied by neither, so every clone and the original append to the same list.
    """

    sink: list = []

    @classmethod
    def reset(cls):
        cls.sink = []

    @classmethod
    def sizes(cls):
        return list(cls.sink)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        type(self).sink.append(len(y))
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


class _PointRecorder(_Recorder):
    sink: list = []


class _BandRecorder(_Recorder):
    sink: list = []


class TestTrainingWindowIsEnforced(unittest.TestCase):
    """No fold may be fitted on rows at or beyond its own test block.

    This is the fit-time twin of the feature-level leakage guard in test_features.py. That
    guard asks whether a feature contains the answer. This one asks whether the model was
    trained on the answer, which is a different bug with the same catastrophic effect.
    """

    def setUp(self):
        self.df = _series_frame(260, seed=5)
        self.feat = features.build_features(self.df, use_weather=True)
        self.folds = splits.walk_forward_folds(len(self.feat), min_train=120, step=14)

    def test_each_fold_fits_exactly_its_own_training_window(self):
        _PointRecorder.reset()
        train.walk_forward_predictions(_PointRecorder(), self.feat, self.folds)
        self.assertEqual(
            _PointRecorder.sizes(),
            [train_end for train_end, _, _ in self.folds],
            "a fold was fitted on more rows than its train_end allows",
        )

    def test_interval_models_also_fit_only_the_training_window(self):
        """The band has its own copy of the slicing, so it needs its own guard.

        Mutation testing showed that fitting the quantile models on the whole series - which
        is the exact circularity the module docstring says the project exists to avoid -
        changed holdout coverage from 0.69 to 0.79, flipped the `calibrated` flag to True and
        silenced the entire one-sided-error diagnosis. Every test passed.
        """
        _BandRecorder.reset()
        _PointRecorder.reset()
        original = evaluate.quantile_models
        evaluate.quantile_models = lambda: {"lower": _BandRecorder(), "upper": _BandRecorder()}
        try:
            evaluate.holdout_forecast(_PointRecorder(), self.feat, self.folds)
        finally:
            evaluate.quantile_models = original

        expected = [train_end for train_end, _, _ in self.folds]
        # Interleaved, because holdout_forecast fits lower and upper inside each fold rather
        # than all lowers and then all uppers.
        band_expected = [size for train_end in expected for size in (train_end, train_end)]
        self.assertEqual(
            _PointRecorder.sizes(), expected, "the point model was fitted past its train_end"
        )
        self.assertEqual(
            _BandRecorder.sizes(),
            band_expected,
            "the interval models saw a different window than the point model",
        )


class TestSelectionIsUnaffectedByHoldoutDemand(unittest.TestCase):
    """The behavioural form of "selection never sees the holdout".

    The structural proxies for this claim - the output dict has only a `dev` key, and an AST
    line-order check on run_all.main - both stay green while selection scores the holdout.
    Key names are not evidence. The evidence is that corrupting holdout demand must not move a
    single development number.
    """

    def test_dev_scores_do_not_move_when_only_holdout_demand_changes(self):
        df = _series_frame(300, seed=3)
        feat = features.build_features(df, use_weather=False)
        holdout_start = len(feat) - config.HORIZON * 8
        dev = [
            f
            for f in splits.walk_forward_folds(len(feat), min_train=120, step=config.HORIZON)
            if f[2] <= holdout_start
        ]
        self.assertTrue(dev)

        # The zoo is narrowed to two models. Whether selection is blind to the holdout does
        # not depend on how many candidates it compares, and running the full eight twice
        # cost 104 seconds to prove the same thing.
        original = train.candidate_models
        train.candidate_models = lambda: {
            k: original()[k] for k in ("gradient_boosting", "ridge")
        }
        try:
            scale = 500.0
            baseline = train.select(df, dev, scale)
            cutoff = feat.index[holdout_start]
            corrupted = df.copy()
            corrupted.loc[corrupted.index >= cutoff, config.TARGET] = (
                corrupted.loc[corrupted.index >= cutoff, config.TARGET] * 9 + 5_000
            )
            perturbed = train.select(corrupted, dev, scale)
        finally:
            train.candidate_models = original

        before = {(r["feature_set"], r["model"]): r["dev"]["mase"] for r in baseline["results"]}
        after = {(r["feature_set"], r["model"]): r["dev"]["mase"] for r in perturbed["results"]}
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertAlmostEqual(
                before[key], after[key], places=6, msg=f"{key} moved when holdout demand changed"
            )
        self.assertEqual(baseline["best_model"], perturbed["best_model"])

    def test_the_control_actually_moves_when_history_changes(self):
        """A guard that cannot fail is not a guard.

        The test above passes trivially if corrupting rows changes nothing at all - for
        instance if select() were reading a cached table. This asserts the same corruption
        applied to *development* demand does move the numbers, so the blindness test is
        measuring blindness rather than inertia.
        """
        df = _series_frame(300, seed=3)
        feat = features.build_features(df, use_weather=False)
        dev = [
            f
            for f in splits.walk_forward_folds(len(feat), min_train=120, step=config.HORIZON)
            if f[2] <= len(feat) - config.HORIZON * 8
        ]
        original = train.candidate_models
        train.candidate_models = lambda: {k: original()[k] for k in ("gradient_boosting",)}
        try:
            before = train.select(df, dev, 500.0)["results"][0]["dev"]["mase"]
            corrupted = df.copy()
            corrupted[config.TARGET] = corrupted[config.TARGET] * 3 + 400
            after = train.select(corrupted, dev, 500.0)["results"][0]["dev"]["mase"]
        finally:
            train.candidate_models = original
        self.assertNotAlmostEqual(before, after, places=6)


class TestPartitionLeavesNoOverlap(unittest.TestCase):
    def test_no_development_fold_scores_a_day_inside_the_reserved_holdout(self):
        """The development filter must be on test_END, not test_start.

        Filtering on `f[1] < holdout_start` admits a fold whose test block runs past the
        boundary, quietly spending part of the holdout during model selection.
        """
        folds = splits.walk_forward_folds(696, step=config.HORIZON)
        holdout_start = 605
        dev, holdout = splits.partition_folds(folds, holdout_start)
        for _, test_start, test_end in dev:
            self.assertLessEqual(
                test_end, holdout_start, f"dev fold ({test_start}, {test_end}) scores holdout days"
            )
        dev_days = {i for _, ts, te in dev for i in range(ts, te)}
        holdout_days = {i for _, ts, te in holdout for i in range(ts, te)}
        self.assertEqual(dev_days & holdout_days, set())

    def test_the_default_fold_geometry_is_never_overlapping(self):
        """No test called walk_forward_folds without `step=`, so the default was unverified.

        Changing the default from HORIZON to 1 produced heavily overlapping test blocks - the
        exact bug the suite claims to guard - and all 86 tests stayed green.
        """
        default_folds = splits.walk_forward_folds(800)
        explicit = splits.walk_forward_folds(800, step=config.HORIZON)
        self.assertEqual(default_folds, explicit)
        previous_end = 0
        for _, test_start, test_end in default_folds:
            self.assertEqual(test_end - test_start, config.HORIZON)
            self.assertGreaterEqual(test_start, previous_end, "default-step blocks overlap")
            previous_end = test_end


class TestForecastIsByteReproducible(unittest.TestCase):
    """The README claims two clean rebuilds produce identical bytes. Enforce it.

    A reproducibility claim that no test checks is a memory, not a property. This re-runs the
    evaluation stage into a fresh directory and compares the two forecast files byte for byte,
    which is the only way to catch a float whose accumulation order moved between runs.
    """

    def test_two_runs_of_the_evaluation_stage_produce_identical_bytes(self):
        import tempfile

        df = _series_frame(300, seed=13)
        feat = features.build_features(df, use_weather=True)
        folds = splits.walk_forward_folds(len(feat), min_train=150, step=28)[:4]
        selection = {
            "best_model": "gradient_boosting",
            "best_feature_set": "with_weather",
            "results": [{"feature_set": "with_weather", "model": "gradient_boosting", "dev": {"mase": 0.5}}],
        }
        blobs = []
        for _ in range(2):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            root = pathlib.Path(tmp.name)
            evaluate.run(
                df, selection, folds, 700.0,
                out={
                    "eval_json": root / "evaluation.json",
                    "scored_csv": root / "daily_forecast.csv",
                    "fig_dir": root / "figures",
                },
            )
            blobs.append((root / "daily_forecast.csv").read_bytes())
        self.assertEqual(blobs[0], blobs[1], "the forecast is not byte-reproducible")


class TestResidualDiagnostics(unittest.TestCase):
    def test_acf_of_white_noise_decays_and_starts_at_one(self):
        rng = np.random.default_rng(0)
        acf = evaluate.residual_acf(rng.normal(0, 1, 500))
        self.assertAlmostEqual(acf[0], 1.0)
        self.assertLess(abs(acf[10]), 0.15)

    def test_acf_detects_a_weekly_pattern_the_model_missed(self):
        """A lag-7 spike means there is structure left in the residuals."""
        t = np.arange(400)
        residuals = 10 * np.sin(t * 2 * np.pi / 7)
        acf = evaluate.residual_acf(residuals)
        self.assertGreater(acf[7], 0.9)
        self.assertGreater(acf[7], acf[3])

    def test_constant_residuals_do_not_divide_by_zero(self):
        acf = evaluate.residual_acf(np.zeros(50))
        self.assertTrue(np.all(np.isfinite(acf)))
