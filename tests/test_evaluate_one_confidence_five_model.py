import numpy as np

from evaluate_one_confidence_five_model import choose_threshold, metrics


def test_error_is_positive_and_threshold_is_inclusive():
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.9, 0.8, 0.8, 0.1])
    result = metrics(labels, scores, 0.8)
    assert (result["TP"], result["FN"], result["FP"], result["TN"]) == (2, 0, 1, 1)


def test_choose_threshold_respects_fpr():
    labels = np.asarray([1] * 10 + [0] * 10)
    scores = np.asarray([0.9] * 8 + [0.05] * 2 + [0.8] + [0.1] * 9)
    result = choose_threshold(labels, scores, max_fpr=0.10)
    assert result["recall"] == 0.8
    assert result["fpr"] <= 0.1
