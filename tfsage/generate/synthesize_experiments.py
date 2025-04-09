import pandas as pd
import numpy as np
import pybedtools
from typing import List


def multiintersect(bed_files: List[str]) -> pybedtools.BedTool:
    bedtools = pybedtools.BedTool()
    result = bedtools.multi_intersect(i=bed_files)
    return result


def merge_nearby_peaks(
    result: pybedtools.BedTool, n: int, merge_distance: int = 0
) -> pybedtools.BedTool:
    if len(result):
        c = ",".join([str(i + 6) for i in range(n)])
        result = result.merge(c=c, o="max", d=merge_distance)
    return result


def bedtool_to_dataframe(result: pybedtools.BedTool, n: int) -> pd.DataFrame:
    base_cols = ["chrom", "start", "end"]
    column_names = base_cols + [f"file_{i}" for i in range(n)]

    df = result.to_dataframe(disable_auto_names=True)
    if df.empty:
        df = pd.DataFrame(columns=column_names)
    else:
        df.columns = column_names

    return df


def compute_weighted_sum(df: pd.DataFrame, weights: List[float]) -> pd.DataFrame:
    columns = [f"file_{i}" for i in range(len(weights))]
    df["sum"] = (2 * df[columns] - 1).mul(weights).sum(axis=1)
    # Add weights as columns for reference
    weights_df = pd.DataFrame(
        np.tile(weights, (df.shape[0], 1)),
        index=df.index,
        columns=[f"weight_{i}" for i in range(len(weights))],
    )
    df = pd.concat([df, weights_df], axis=1)
    return df


def synthesize_experiments(
    bed_files: List[str], weights: List[float] | None = None, merge_distance: int = 0
) -> pd.DataFrame:
    """
    Synthesizes experiments by processing and combining multiple BED files.

    This function performs the following steps:
    1. Computes the intersection of regions across the provided BED files.
    2. Merges nearby peaks based on the specified merge distance.
    3. Converts the resulting BEDTool object into a pandas DataFrame.
    4. Optionally computes a weighted sum of the regions if weights are provided.

    Args:
        bed_files (List[str]): A list of file paths to BED files to be processed.
        weights (List[float] | None, optional): A list of weights corresponding to each BED file.
            If provided, the weighted sum of the regions will be computed. Defaults to None.
        merge_distance (int, optional): The maximum distance between peaks to merge them.
            Defaults to 0 (no merging).

    Returns:
        pd.DataFrame: A DataFrame containing the processed and synthesized regions.
    """
    result = multiintersect(bed_files)
    result = merge_nearby_peaks(result, len(bed_files), merge_distance)
    result = bedtool_to_dataframe(result, len(bed_files))
    if weights is not None:
        result = compute_weighted_sum(result, weights)
    return result
