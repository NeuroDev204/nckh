"""Chỉ số đối chiếu nhãn thật; không phụ thuộc môi trường torch."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from nckh.infer import CLS_LABELS


def validate_probabilities(probabilities: np.ndarray) -> None:
    if (probabilities.ndim != 2 or probabilities.shape[1] != 3 or not len(probabilities)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4, rtol=0)):
        raise ValueError("Cần xác suất hữu hạn N×3 trong [0,1], tổng mỗi dòng bằng 1 (sai số 1e-4)")


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    validate_probabilities(probabilities)
    if labels.shape != (len(probabilities),) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Nhãn phải là vector N phần tử trong {0,1,2}")
    labels = labels.astype(int)
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1, 2], zero_division=0,
    )
    _, _, f2, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1, 2], beta=2, zero_division=0,
    )
    per_class = []
    for i, name in enumerate(CLS_LABELS):
        binary = labels == i
        auc = float(roc_auc_score(binary, probabilities[:, i])) if binary.any() and not binary.all() else None
        per_class.append({
            "label": i, "name": name, "support": int(support[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "f2": float(f2[i]), "auroc": auc,
            "auroc_reason": None if auc is not None else "Thiếu positive hoặc negative trong mẫu",
        })
    aucs = [row["auroc"] for row in per_class]
    return {
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "macro_f2": float(f2.mean()),
        "macro_auroc": float(np.mean(aucs)) if all(a is not None for a in aucs) else None,
        "brier_score": float(np.square(probabilities - np.eye(3)[labels]).sum(axis=1).mean()),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist(),
        "per_class": per_class,
        "missing_classes": [row["name"] for row in per_class if row["support"] == 0],
    }


def segmentation_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    if (reference.ndim != 2 or reference.shape != prediction.shape or not reference.size
            or not np.isin(reference, [0, 1]).all() or not np.isin(prediction, [0, 1]).all()):
        raise ValueError("Mask phải nhị phân, không rỗng kích thước và có cùng shape H×W")
    reference, prediction = reference.astype(bool), prediction.astype(bool)
    tp = int((reference & prediction).sum())
    fp = int((~reference & prediction).sum())
    fn = int((reference & ~prediction).sum())
    tn = int((~reference & ~prediction).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float(fn == 0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "f2": 5 * tp / (5 * tp + fp + 4 * fn) if tp + fp + fn else 1.0,
        "pixel_accuracy": (tp + tn) / reference.size,
        "reference_empty": not reference.any().item(), "prediction_empty": not prediction.any().item(),
    }
