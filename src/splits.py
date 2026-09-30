"""Walk-forward validation. This is the module that makes the project a forecasting project
rather than a classification project with dates on the side.

Every fold is shaped like a real operating decision: train on everything up to some day,
then forecast the next HORIZON days and nothing else. Test blocks never overlap, so no day in
the series is forecast twice, and each is forecast by the model that would have been standing
at that point in time.

What the design does NOT promise is that every day is forecast. A block that straddles the
holdout boundary belongs to neither development nor holdout, because development filters on
test_end <= holdout_start and the holdout on test_start >= holdout_start. On this series that
straddling block is 2012-09-30 to 2012-10-06, so those seven days are never scored, along with
the 35 warmup rows and the last two days of the series. Excluding the straddling block is
the conservative choice - folding it into development would spend holdout days during
selection - but the hole is stated rather than assumed away.

Three properties fall out of the construction rather than out of discipline:

1. No fold's test rows were in its training set, or in any earlier fold's test set.
2. A test row's features come only from its own training window, so the forecaster can never
   see demand it would not have had on the morning it made the call.
3. The holdout is not a special case. It is the same generator, partitioned by date. Holding
   it out is a filter on the fold list, not a separate code path, which means the holdout is
   validated by the same code that validates everything else.
"""
from . import config


def walk_forward_folds(n_rows, horizon=None, min_train=None, step=None):
    # One honest note about the training window. A test row at test_start is forecast from
    # origin test_start - horizon, so demand at that origin day is knowable and is used as a
    # feature. It is still not used as a training *label*, because the fit is sliced as
    # iloc[:train_end] and train_end is the origin itself. The window is therefore one week
    # shorter than "everything up to the origin" suggests. The direction is conservative - it
    # cannot leak - and it costs one week of the most recent labels, which is a small and
    # deliberate price for keeping a single slicing rule in one place.
    """Expanding-window folds as (train_end, test_start, test_end) positional bounds.

    `train_end` is deliberately `test_start - horizon`, not `test_start`. A test row at
    `test_start` is forecast from origin `test_start - horizon`, and it may only use demand up
    to that origin. Training up to `test_start` instead would hand the model the answer for the
    first day of its own test block, through the rolling features.
    """
    horizon = horizon or config.HORIZON
    min_train = min_train or config.MIN_TRAIN_DAYS
    step = step or horizon

    folds = []
    test_start = min_train + horizon
    while test_start + horizon <= n_rows:
        train_end = test_start - horizon
        folds.append((train_end, test_start, test_start + horizon))
        test_start += step
    return folds


def partition_folds(folds, holdout_start):
    """Split a fold list into development and holdout by date position.

    Development folds are the only ones a selection routine may see. The holdout is the tail
    of the same list, so it is exactly what the model would have faced in the final weeks of
    the series.
    """
    dev = [f for f in folds if f[2] <= holdout_start]
    holdout = [f for f in folds if f[1] >= holdout_start]
    if not dev:
        raise ValueError("no development folds; lower min_train or shorten the holdout")
    if not holdout:
        raise ValueError("no holdout folds; holdout is shorter than one horizon")
    return dev, holdout


def holdout_start_position(feat, holdout_days=None):
    """Positional index where the holdout begins, leaving `holdout_days` untouched at the end."""
    holdout_days = holdout_days or (config.HORIZON * 13)  # a full quarter of weeks
    start = len(feat) - holdout_days
    if start <= config.MIN_TRAIN_DAYS:
        raise ValueError("series too short for the requested holdout")
    return start


def assert_folds_are_chronological(folds, n_rows):
    """Guard the invariants the rest of the module depends on. Cheap, and it fails loudly."""
    previous_test_end = 0
    for train_end, test_start, test_end in folds:
        if not 0 < train_end < test_start < test_end <= n_rows:
            raise ValueError(f"malformed fold {(train_end, test_start, test_end)}")
        if test_start - train_end != config.HORIZON:
            raise ValueError(
                f"fold leaks: train_end {train_end} is not HORIZON before test_start {test_start}"
            )
        if test_start < previous_test_end:
            raise ValueError("test blocks overlap or run backwards")
        previous_test_end = test_end
    return True
