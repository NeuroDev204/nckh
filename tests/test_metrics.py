import numpy as np
import pytest

from nckh.metrics import classification_metrics, segmentation_metrics


def test_perfect_classification() -> None:
    result = classification_metrics(np.array([0, 1, 2]), np.eye(3))
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "macro_f2", "macro_auroc"):
        assert result[name] == 1
    assert result["brier_score"] == 0
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_imbalanced_classification_and_missing_class() -> None:
    result = classification_metrics(np.array([0, 0, 0, 1, 2]), np.tile([1, 0, 0], (5, 1)))
    assert result["accuracy"] == .6
    assert result["balanced_accuracy"] == pytest.approx(1 / 3)
    missing = classification_metrics(np.array([0, 1]), np.eye(3)[:2])
    assert missing["macro_auroc"] is None
    assert missing["per_class"][2]["auroc"] is None
    assert missing["per_class"][2]["auroc_reason"]
    assert missing["macro_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("probabilities", [
    [[1, 0]], [[np.nan, 0, 0]], [[1.1, -.1, 0]], [[.2, .2, .2]],
])
def test_rejects_invalid_probabilities(probabilities: list) -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.array([0]), np.array(probabilities))


@pytest.mark.parametrize("labels", [[3], [-1], [.5], []])
def test_rejects_invalid_labels(labels: list) -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.array(labels), np.array([[1, 0, 0]]))


def test_segmentation_counts_and_f2() -> None:
    result = segmentation_metrics(np.array([[1, 1], [1, 0]]), np.array([[1, 0], [0, 1]]))
    assert result["dice"] == .4
    assert result["iou"] == .25
    assert result["precision"] == .5
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f2"] == pytest.approx(5 / 14)
    assert result["pixel_accuracy"] == .25


@pytest.mark.parametrize("reference,prediction,dice,precision,recall", [
    (0, 0, 1, 1, 1), (1, 0, 0, 0, 0), (0, 1, 0, 0, 1),
])
def test_empty_masks(reference: int, prediction: int, dice: int, precision: int, recall: int) -> None:
    result = segmentation_metrics(np.full((2, 2), reference), np.full((2, 2), prediction))
    assert result["dice"] == dice
    assert result["iou"] == dice
    assert result["f2"] == dice
    assert result["precision"] == precision
    assert result["recall"] == recall


@pytest.mark.parametrize("reference,prediction", [
    (np.zeros((2, 2)), np.zeros((3, 3))),
    (np.zeros((2, 2)), np.full((2, 2), 2)),
    (np.zeros((0, 0)), np.zeros((0, 0))),
])
def test_rejects_invalid_masks(reference: np.ndarray, prediction: np.ndarray) -> None:
    with pytest.raises(ValueError):
        segmentation_metrics(reference, prediction)
