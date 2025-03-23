import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from typing import List


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    recall_targets: List[float] = [0.15, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
) -> dict:
    """
    Compute a variety of evaluation metrics for binary classification.

    This function calculates both threshold-independent metrics (e.g., ROC AUC, AUPRC)
    and threshold-dependent metrics (e.g., F1 score, accuracy, precision, recall).
    Additionally, it computes precision at specific recall thresholds.

    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_score (np.ndarray): Predicted scores or probabilities for the positive class.
        y_pred (np.ndarray): Predicted binary labels (0 or 1).
        recall_targets (List[float], optional): List of recall thresholds at which to
            compute precision. Defaults to [0.15, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99].

    Returns:
        dict: A dictionary containing the computed metrics. Keys include:
            - "roc_auc": Area under the ROC curve.
            - "auprc": Area under the precision-recall curve.
            - "f1": F1 score.
            - "accuracy": Accuracy score.
            - "precision": Precision score.
            - "recall": Recall score.
            - "precision@<X>%recall": Precision at specific recall thresholds, where
              <X> corresponds to the percentage value of the recall target.
    """
    # Threshold-independent metrics
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }

    # Threshold-dependent metrics
    metrics.update(
        {
            "f1": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }
    )

    # Precision at specific recall thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    for target in recall_targets:
        # Find last index where recall >= target
        idx = np.argmin(recall >= target) - 1
        prec_at_recall = precision[idx] if idx < len(precision) else np.nan
        metrics[f"precision@{int(target*100)}%recall"] = prec_at_recall

    return metrics
