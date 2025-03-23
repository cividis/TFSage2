import pandas as pd
import numpy as np
import pybedtools


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
