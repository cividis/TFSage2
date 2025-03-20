import pandas as pd
from sklearn.metrics import pairwise_distances


def compute_distances(df: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    """
    Compute pairwise distances between rows of a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame where each row represents a data point.
                           The DataFrame should contain numerical values only.
        metric (str, optional): The distance metric to use. Default is "cosine".
                                Other options include "euclidean", "manhattan", "chebyshev", etc.
                                Refer to sklearn.metrics.pairwise_distances documentation for more options.

    Returns:
        pd.DataFrame: A DataFrame where the element at [i, j] represents the distance
                      between the i-th and j-th rows of the input DataFrame. The DataFrame
                      has the same index and columns as the input DataFrame.
    """
    distances = pairwise_distances(df.values, metric=metric)
    distances_df = pd.DataFrame(distances, index=df.index, columns=df.index)
    return distances_df
