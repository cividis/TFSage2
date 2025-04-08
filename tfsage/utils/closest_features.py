import pandas as pd
import pybedtools
from importlib import resources
from typing import Literal
from lisa.core import genome_tools
import contextlib
import os
import sys
import warnings


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr


def load_features_bed(
    genome: Literal["hg38", "mm10"] = "hg38", bed_file: str | None = None
) -> genome_tools.RegionSet:
    """
    Load a BED file containing genomic features for a specified genome.

    Args:
        genome (Literal["hg38", "mm10"], optional): The genome version to load features for.
            Supported values are "hg38" (human) and "mm10" (mouse). Defaults to "hg38".
        bed_file (str | None, optional): Path to a custom BED file. If None, the default
            BED file for the specified genome will be used. Defaults to None.

    Returns:
        genome_tools.RegionSet: A sorted BedTool object representing the genomic features.

    Raises:
        ValueError: If the genome is not one of the supported values ("hg38" or "mm10").
    """
    if genome in ["hg38", "mm10"]:
        bed_file = resources.files("tfsage.assets").joinpath(
            f"{genome}/{genome}_refseq_TSS.bed"
        )
    elif bed_file is None:
        raise ValueError(
            f"Unsupported genome '{genome}'. Provide a valid genome or a custom BED file."
        )

    features_bed = pybedtools.BedTool(bed_file).sort()
    return features_bed


def field_count(bed: pybedtools.BedTool) -> int:
    try:
        field_count = bed.field_count(1)
    except OverflowError:
        field_count = bed.to_dataframe(disable_auto_names=True).shape[1]
    return field_count


def closest_features(
    peaks_bed: pybedtools.BedTool | str | pd.DataFrame,
    features_bed: pybedtools.BedTool | str | pd.DataFrame,
    k: int = 15,
) -> pd.DataFrame:
    """
    Find the closest genomic features to a set of peaks.

    Args:
        peaks_bed (pybedtools.BedTool | str | pd.DataFrame): Input representing the peaks.
            - If a `pybedtools.BedTool` object, it is used directly.
            - If a `str`, it is assumed to be a file path to a BED file.
            - If a `pd.DataFrame`, it is converted to a BedTool object.
        features_bed (pybedtools.BedTool | str | pd.DataFrame): Input representing the genomic features.
            - If a `pybedtools.BedTool` object, it is used directly.
            - If a `str`, it is assumed to be a file path to a BED file.
            - If a `pd.DataFrame`, it is converted to a BedTool object.
        k (int, optional): The number of closest features to find for each peak. Defaults to 15.

    Returns:
        pd.DataFrame: A DataFrame containing the closest features with the following columns:
            - chrom: Chromosome of the peak.
            - start: Start position of the peak.
            - end: End position of the peak.
            - name: Name of the closest feature.
            - distance: Distance from the peak to the feature.
    """
    # Convert peaks_bed to BedTool if necessary
    if isinstance(peaks_bed, str):
        peaks_bed = pybedtools.BedTool(peaks_bed).sort()
    elif isinstance(peaks_bed, pd.DataFrame):
        peaks_bed = pybedtools.BedTool.from_dataframe(peaks_bed).sort()

    # Convert features_bed to BedTool if necessary
    if isinstance(features_bed, str):
        features_bed = pybedtools.BedTool(features_bed).sort()
    elif isinstance(features_bed, pd.DataFrame):
        features_bed = pybedtools.BedTool.from_dataframe(features_bed).sort()

    # Define column names for the resulting DataFrame
    column_names = [
        *[f"a_{i}" for i in range(field_count(peaks_bed))],
        *[f"b_{i}" for i in range(field_count(features_bed))],
        "distance",
    ]

    # Perform the closest feature search and process the results
    with suppress_output():
        res = peaks_bed.closest(features_bed, D="ref", k=k)

    df = (
        res.to_dataframe(disable_auto_names=True, names=column_names)
        .filter(["a_0", "a_1", "a_2", "b_3", "distance"], axis=1)
        .rename(
            {
                "a_0": "chrom",
                "a_1": "start",
                "a_2": "end",
                "b_3": "name",
            },
            axis=1,
        )
    )
    return df


if __name__ == "__main__":
    """
    Example usage of the `closest_features` function.

    This script loads a set of peaks from a BED file, finds the closest genomic features
    for a specified genome, and filters the results based on a promoter region definition.
    """
    # Example usage with a BedTool object
    peaks_bed = pybedtools.BedTool("data/SRX502813.bed").sort()
    features_bed = load_features_bed("hg38")
    df = closest_features(peaks_bed, features_bed)
    df = df.query("-1000 < distance < 100")
    df["gene"] = df["name"].str.split(":").str[1]
    print(df, df["gene"].nunique())

    # Example usage with a BED file
    peaks_bed = "data/SRX502813.bed"
    df = closest_features(peaks_bed, features_bed)
    df = df.query("-1000 < distance < 100")
    df["gene"] = df["name"].str.split(":").str[1]
    print(df, df["gene"].nunique())

    # Example usage with a DataFrame
    peaks_bed = pd.read_csv("data/SRX502813.bed", sep="\t", header=None)
    df = closest_features(peaks_bed, features_bed)
    df = df.query("-1000 < distance < 100")
    df["gene"] = df["name"].str.split(":").str[1]
    print(df, df["gene"].nunique())
