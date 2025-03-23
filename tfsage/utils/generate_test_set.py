import pandas as pd
import pybedtools
from typing import List


def generate_test_set(
    query_file: str, target_file: str, peak_width: int | None = 200
) -> pd.DataFrame:
    """
    Generate a test set containing both positive and negative regions based on
    genomic data.

    Args:
        query_file (str): Path to the query BED file containing genomic regions
            of interest.
        target_file (str): Path to the target BED file containing regions to
            compare against.
        peak_width (int | None, optional): The width of the peaks to be used
            for generating positive and negative sets. If specified, regions
            will be adjusted to this width. Defaults to 200.

    Returns:
        pd.DataFrame: A DataFrame containing the test set with columns:
            - "chrom": Chromosome name.
            - "start": Start position of the region.
            - "end": End position of the region.
            - "num": Number of overlapping regions.
            - "list": List of overlapping region indices.
            - "positive": Boolean indicating if the region is positive.
            - "negative": Boolean indicating if the region is negative.
    """
    bedtools = pybedtools.BedTool()

    positive_set = generate_positive_set(target_file, peak_width)
    negative_set = generate_negative_set(
        query_file, target_file, positive_set, peak_width
    )

    test_set = multiintersect([positive_set, negative_set])
    columns = [
        "chrom",
        "start",
        "end",
        "num",
        "list",
        "positive",
        "negative",
    ]
    df = test_set.to_dataframe(disable_auto_names=True, names=columns)
    df["positive"] = df["positive"].astype(bool)
    df["negative"] = df["negative"].astype(bool)
    return df


def generate_positive_set(
    target_file: str,
    peak_width: int | None = 200,
) -> pybedtools.BedTool:
    bed_file = pybedtools.BedTool(target_file)
    if peak_width is not None:
        bed_file = bed_file.each(set_midpoint).saveas()
        bed_file = bed_file.each(slop, b=peak_width // 2).saveas()

    return bed_file


def generate_negative_set(
    query_file: str,
    target_file: str,
    positive_set: pybedtools.BedTool,
    peak_width: int | None = 200,
) -> pybedtools.BedTool:
    bed_file = pybedtools.BedTool(query_file)
    bed_file = bed_file.subtract(target_file, A=True)
    bed_file = bed_file.subtract(positive_set, A=True)
    if peak_width is not None:
        bed_file = bed_file.each(set_midpoint).saveas()
        bed_file = bed_file.each(slop, b=peak_width // 2).saveas()
        bed_file = bed_file.subtract(target_file, A=True)
        bed_file = bed_file.subtract(positive_set, A=True)
    return bed_file


def multiintersect(bed_files: List[pybedtools.BedTool]) -> pybedtools.BedTool:
    bedtools = pybedtools.BedTool()
    bed_files = [f.sort() for f in bed_files]
    result = bedtools.multi_intersect(i=[f.fn for f in bed_files])
    return result


def set_midpoint(feature):
    feature.start = (feature.start + feature.end) // 2
    feature.end = feature.start + 0
    return feature


def slop(feature, b):
    feature.start = max(0, feature.start - b)
    feature.end += b
    return feature
