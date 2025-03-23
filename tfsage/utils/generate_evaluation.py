import pandas as pd
import numpy as np
import pybedtools
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


def intersect_predictions_with_test_set(
    predictions: pd.DataFrame, test_set: pd.DataFrame
) -> pd.DataFrame:
    """
    Intersect predictions with a test set to evaluate overlaps and extract relevant information.

    Args:
        predictions (pd.DataFrame): A DataFrame containing predicted genomic regions with the following columns:
            - "chrom": Chromosome name.
            - "start": Start position of the region.
            - "end": End position of the region.
            - "score": Prediction score for the region.
        test_set (pd.DataFrame): A DataFrame containing the test set genomic regions with the following columns:
            - "chrom": Chromosome name.
            - "start": Start position of the region.
            - "end": End position of the region.
            - "positive": Boolean indicating if the region is positive.

    Returns:
        pd.DataFrame: A DataFrame containing the intersected regions with the following columns:
            - "chrom": Chromosome name.
            - "start": Start position of the region.
            - "end": End position of the region.
            - "positive": Boolean indicating if the region is positive.
            - "score": Prediction score for the region, with missing scores replaced by -inf.
    """
    # Filter the columns to only include the necessary information
    predictions = predictions.filter(["chrom", "start", "end", "score"], axis=1)
    test_set = test_set.filter(["chrom", "start", "end", "positive"], axis=1)

    # Convert the DataFrames to BedTools objects
    predictions_bed = pybedtools.BedTool.from_dataframe(predictions)
    test_set_bed = pybedtools.BedTool.from_dataframe(test_set)

    # Perform the intersection and extract the relevant columns
    columns = [
        "chrom",
        "start",
        "end",
        "positive",
        "chrom_insilico",
        "start_insilico",
        "end_insilico",
        "score",
        "overlap",
    ]
    df = (
        test_set_bed.intersect(predictions_bed, wao=True)
        .to_dataframe(disable_auto_names=True, names=columns)
        .filter(["chrom", "start", "end", "positive", "score"], axis=1)
        .assign(
            score=lambda x: pd.to_numeric(x["score"], errors="coerce").fillna(-np.inf)
        )
    )
    return df


def sanitize_scores(scores: pd.Series) -> pd.Series:
    """
    Sanitize a pandas Series of scores by handling negative infinity values.

    This function replaces occurrences of negative infinity (-inf) in the input
    Series with the negative of the maximum absolute finite value in the Series.
    If no finite values are present, -inf is replaced with NaN.

    Args:
        scores (pd.Series): A pandas Series containing numeric scores, which may
            include negative infinity (-inf).

    Returns:
        pd.Series: A pandas Series with -inf values replaced by -max_abs_val,
        where max_abs_val is the maximum absolute finite value in the input Series.
    """
    # Get the max absolute value (ignoring -inf first)
    finite_scores = scores.replace(-np.inf, np.nan).abs()
    max_abs_val = finite_scores.max()

    # Default to zero if all values are -inf (edge case)
    if np.isinf(max_abs_val) or np.isnan(max_abs_val):
        max_abs_val = 0.0

    # Replace -inf with -max_abs_val
    scores = scores.replace(-np.inf, -max_abs_val)

    return scores


def compute_metrics_panel(
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
