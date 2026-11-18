import numpy as np
import pytest

from sparsehydro.calibration import Con15


def test_con15_fraction_rounds_up_and_ranks_observed_not_modeled_peaks():
    observed = np.repeat(np.arange(1, 8, dtype=float), 3)
    predicted = observed.copy()
    predicted[:3] = 1000
    predicted[-6:-3] *= .5
    predicted[-3:] *= 2
    windows = np.arange(21).reshape(7, 3)[:, [0, 2]]
    objective = Con15(windows, top_fraction=.15)
    expected_slope = (6 * 3 + 7 * 14) / (6 ** 2 + 7 ** 2)
    assert objective.slope(observed, predicted) == pytest.approx(expected_slope)
    assert objective.compute(observed, predicted) == pytest.approx(abs(expected_slope - 1))
    assert objective.evaluate(observed, predicted) == objective.compute(observed, predicted)
    assert objective.name == "con15_error" and objective.minimize
    assert Con15(windows, .01).slope(observed, predicted) == 2
    assert Con15(windows, 1).slope(observed, predicted) != pytest.approx(expected_slope)


@pytest.mark.parametrize("fraction", [0, -.1, 1.01, 15, np.nan, np.inf, -np.inf])
def test_con15_rejects_invalid_decimal_fractions(fraction):
    with pytest.raises(ValueError, match="top_fraction"):
        Con15(np.array([[0, 2]]), fraction)


@pytest.mark.parametrize("windows", [
    [], [[0, 1, 2]], [[0., 2.]], [[-1, 2]], [[2, 1]],
    [[0, 2], [2, 4]], [[3, 5], [0, 2]],
])
def test_con15_rejects_invalid_windows(windows):
    with pytest.raises(ValueError, match="window|Event"):
        Con15(np.asarray(windows))


def test_con15_preserves_original_indices_masks_and_complete_event_count():
    observed = np.full(15, np.nan)
    observed[3:6], observed[10:13] = [1, 3, 1], [1, 5, 1]
    predicted = observed.copy()
    predicted[10:13] *= 1.2
    train = np.arange(15) < 8
    objective = Con15(np.array([[3, 5], [10, 12]]), .5, mask=train)
    assert objective.compute(observed, predicted) == 0
    assert objective.compute(observed, predicted, mask=~train) == pytest.approx(.2)
    partial = train.copy()
    partial[3] = False
    with pytest.raises(ValueError, match="complete event"):
        objective.compute(observed, predicted, mask=partial)
    with pytest.raises(ValueError, match="mask shape"):
        objective.compute(observed, predicted, mask=np.ones(2, bool))
    with pytest.raises(TypeError, match="mask"):
        objective.compute(observed, predicted, mask="train")


@pytest.mark.parametrize("validation_count", [1, 40])
@pytest.mark.parametrize("validation_peak", [.01, 1e6])
@pytest.mark.parametrize("fraction,selected_count", [(.15, 2), (.3, 3), (1., 8)])
def test_con15_counts_and_ranks_training_storms_before_applying_fraction(
    validation_count, validation_peak, fraction, selected_count,
):
    observed = np.repeat(np.r_[np.arange(1., 9.), np.full(validation_count, validation_peak)], 3)
    predicted = observed.copy()
    predicted[21:24] *= .8
    predicted[24:] *= 100
    windows = np.arange(len(observed)).reshape(-1, 3)[:, [0, 2]]
    train = np.arange(len(observed)) < 24
    objective = Con15(windows, top_fraction=fraction, mask=train)
    selected = objective.selected_event_indices(observed)
    np.testing.assert_array_equal(selected, np.arange(7, 7 - selected_count, -1))
    assert len(selected) == int(np.ceil(8 * fraction))
    explicit_training = Con15(windows[:8], top_fraction=fraction, mask=train)
    assert objective.compute(observed, predicted) == explicit_training.compute(observed, predicted)
    x = np.arange(8., 8 - selected_count, -1)
    y = x.copy()
    y[0] *= .8
    assert objective.slope(observed, predicted) == pytest.approx(x @ y / (x @ x))
    observed[24:] = np.nan
    assert objective.compute(observed, predicted) == explicit_training.compute(observed, predicted)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_con15_does_not_drop_invalid_samples_or_change_event_ranking(bad):
    objective = Con15(np.array([[0, 2], [3, 5]]))
    observed = np.array([1., 3., 1., 1., 5., 1.])
    corrupted = observed.copy()
    corrupted[0] = bad
    with pytest.raises(ValueError, match="observed events"):
        objective.compute(corrupted, observed)
    corrupted = observed.copy()
    corrupted[3] = bad
    with pytest.raises(ValueError, match="modeled events"):
        objective.compute(observed, corrupted)


def test_con15_rejects_undefined_targets_and_shape_mismatches():
    objective = Con15(np.array([[0, 2]]))
    with pytest.raises(ValueError, match="positive observed"):
        objective.compute(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="observation domain"):
        objective.compute(np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="aligned"):
        objective.compute(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="one-dimensional"):
        objective.compute(np.ones((3, 1)), np.ones((3, 1)))


def test_con15_ties_are_stable_and_opposing_peak_errors_can_cancel():
    objective = Con15(np.array([[0, 0], [1, 1]]))
    assert objective.slope(np.array([10., 10.]), np.array([20., 5.])) == 2
    all_events = Con15(np.array([[0, 0], [1, 1]]), 1)
    assert all_events.compute(np.array([10., 20.]), np.array([14., 18.])) == 0
































