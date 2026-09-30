"""Fold-construction tests.

The invariants asserted here are the whole reason this project produces a trustworthy
holdout number. Each one corresponds to a way walk-forward validation is commonly done wrong:
training too close to the test block, letting test blocks overlap so the same day is scored
twice, or folding backwards in time.
"""
import unittest

from src import config, data_loader, features, splits


class TestFoldGeometry(unittest.TestCase):
    def test_train_end_is_exactly_one_horizon_before_test_start(self):
        """The single most important line in the module.

        A test row at test_start is forecast from origin test_start - HORIZON, so training
        must stop there. Training to test_start instead would hand the model the answer for
        the first day of its own test block via the rolling features.
        """
        folds = splits.walk_forward_folds(800, step=14)
        self.assertTrue(folds)
        for train_end, test_start, test_end in folds:
            self.assertEqual(test_start - train_end, config.HORIZON)

    def test_test_blocks_never_overlap_and_move_forward(self):
        folds = splits.walk_forward_folds(800, step=7)
        previous_end = 0
        for _, test_start, test_end in folds:
            self.assertGreaterEqual(test_start, previous_end)
            previous_end = test_end

    def test_every_test_block_is_exactly_one_horizon_long(self):
        for step in (7, 14):
            for _, test_start, test_end in splits.walk_forward_folds(800, step=step):
                self.assertEqual(test_end - test_start, config.HORIZON)

    def test_training_window_grows(self):
        """Expanding, not rolling. A shrinking window throws away the oldest history for no
        reason on a series whose early data is the only record of its first season."""
        folds = splits.walk_forward_folds(800, step=14)
        train_ends = [f[0] for f in folds]
        self.assertEqual(train_ends, sorted(train_ends))
        self.assertGreater(train_ends[-1], train_ends[0])

    def test_folds_are_generated_up_to_the_end_of_the_series(self):
        n = 800
        folds = splits.walk_forward_folds(n, step=7)
        self.assertEqual(folds[-1][2], n - (n - folds[-1][2]) % config.HORIZON)

    def test_invariant_checker_rejects_a_leaky_fold(self):
        n = 800
        good = splits.walk_forward_folds(n, step=14)
        self.assertTrue(splits.assert_folds_are_chronological(good, n))
        leaky = [(te + 3, ts, e) for te, ts, e in good[:1]]
        with self.assertRaises(ValueError):
            splits.assert_folds_are_chronological(leaky, n)

    def test_invariant_checker_rejects_overlapping_blocks(self):
        n = 800
        folds = splits.walk_forward_folds(n, step=7)
        overlapping = folds[:1] + [(folds[1][0], folds[0][2] - 1, folds[0][2] + 6)]
        with self.assertRaises(ValueError):
            splits.assert_folds_are_chronological(overlapping, n)


class TestPartition(unittest.TestCase):
    def test_holdout_is_the_tail_and_never_appears_in_development(self):
        folds = splits.walk_forward_folds(800, step=7)
        holdout_start = 605
        dev, holdout = splits.partition_folds(folds, holdout_start)
        self.assertTrue(dev and holdout)
        for _, test_start, _ in dev:
            self.assertLess(test_start, holdout_start)
        for _, test_start, _ in holdout:
            self.assertGreaterEqual(test_start, holdout_start)

    def test_partition_rejects_an_empty_side(self):
        folds = splits.walk_forward_folds(800, step=7)
        with self.assertRaises(ValueError):
            splits.partition_folds(folds, 5)
        with self.assertRaises(ValueError):
            splits.partition_folds(folds, 795)

    def test_holdout_position_leaves_the_requested_tail(self):
        feat = features.build_features(data_loader.load_clean())
        pos = splits.holdout_start_position(feat)
        self.assertEqual(len(feat) - pos, config.HORIZON * 13)

    def test_holdout_too_large_for_the_series_is_refused(self):
        feat = features.build_features(data_loader.load_clean())
        with self.assertRaises(ValueError):
            splits.holdout_start_position(feat, holdout_days=len(feat) - 10)


class TestSelectionSeesOnlyDevelopment(unittest.TestCase):
    def test_dev_and_holdout_folds_are_disjoint_and_ordered(self):
        """Mirrors run_all: one fold list, partitioned by date. Both sides use a 7-day step,
        so development covers every day up to the boundary rather than alternate weeks."""
        feat = features.build_features(data_loader.load_clean(), use_weather=True)
        all_folds = splits.walk_forward_folds(len(feat), step=config.HORIZON)
        holdout_start = splits.holdout_start_position(feat)
        dev, _ = splits.partition_folds(all_folds, holdout_start)
        holdout = [f for f in all_folds if f[1] >= holdout_start]
        self.assertTrue(dev and holdout)
        dev_test_starts = {f[1] for f in dev}
        holdout_test_starts = {f[1] for f in holdout}
        self.assertEqual(dev_test_starts & holdout_test_starts, set())
        self.assertLess(max(dev_test_starts), min(holdout_test_starts))

    def test_development_covers_every_day_up_to_the_boundary(self):
        """No holes. An earlier design stepped development folds by 14 days, which left
        alternate weeks unforecast and made "every day is predicted exactly once" false."""
        feat = features.build_features(data_loader.load_clean(), use_weather=True)
        all_folds = splits.walk_forward_folds(len(feat), step=config.HORIZON)
        holdout_start = splits.holdout_start_position(feat)
        dev, _ = splits.partition_folds(all_folds, holdout_start)
        covered = {i for _, ts, te in dev for i in range(ts, te)}
        first, last = min(covered), max(covered)
        self.assertEqual(len(covered), last - first + 1, "development has gaps")
        # The last fold must end at or before the boundary. It cannot end exactly on it,
        # because that would need a seven-day block starting inside the holdout.
        self.assertLessEqual(last + 1, holdout_start)
        self.assertLess(holdout_start - (last + 1), config.HORIZON)
