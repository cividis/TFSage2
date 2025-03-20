import pandas as pd
from typing import Callable


def generate_ranked_list(
    distances_df: pd.DataFrame,
    query_id: str,
    metadata: pd.DataFrame | None = None,
    scoring_function: Callable | None = None,
) -> pd.DataFrame:
    """
    Generate a ranked list of experiments based on their similarity to a query.

    Args:
        distances_df (pd.DataFrame): A DataFrame containing pairwise distances
            between experiments. The index should include the query_id, and
            the columns should represent other experiments.
        query_id (str): The ID of the query experiment for which the ranking
            is to be generated.
        metadata (pd.DataFrame | None, optional): Additional metadata to be
            joined with the ranked list. Defaults to None.
        scoring_function (Callable | None, optional): A function to compute
            scores based on distances. If provided, the ranked list will be
            sorted by the computed scores in descending order. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the ranked list of experiments.
        It includes the distance to the query, and optionally metadata and
        scores if provided.
    """
    query_distances = distances_df.loc[query_id]
    ranked_list = query_distances.sort_values().to_frame(name="distance")
    if metadata is not None:
        ranked_list = ranked_list.join(metadata)
    if scoring_function is not None:
        ranked_list["score"] = ranked_list["distance"].apply(scoring_function)
        ranked_list = ranked_list.sort_values("score", ascending=False)

    return ranked_list
