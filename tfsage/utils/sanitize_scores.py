import pandas as pd
import numpy as np


def sanitize_scores(scores: pd.Series) -> pd.Series:
    """
    Sanitize a pandas Series of scores by handling negative infinity values.

    This function replaces occurrences of negative infinity (-inf) in the input
    Series with the negative of 1 + the maximum absolute finite value in the Series.
    If no finite values are present, -inf is replaced with NaN.

    Args:
        scores (pd.Series): A pandas Series containing numeric scores, which may
            include negative infinity (-inf).

    Returns:
        pd.Series: A pandas Series with -inf values replaced by -max_abs_val,
        where max_abs_val is 1 + the maximum absolute finite value in the input Series.
    """
    # Get the max absolute value (ignoring -inf first)
    finite_scores = scores.replace(-np.inf, np.nan).abs()
    max_abs_val = 1 + finite_scores.max()

    # Default to zero if all values are -inf (edge case)
    if np.isinf(max_abs_val) or np.isnan(max_abs_val):
        max_abs_val = 0.0

    # Replace -inf with -max_abs_val
    scores = scores.replace(-np.inf, -max_abs_val)

    return scores
