import os
from typing import List
import numpy as np
import pandas as pd
from lisa.core import data_interface, genome_tools
from .prepare_assets import prepare_region_set

genome_tools.Region.slop = lambda self, d, genome: genome_tools.Region(
    self.chromosome, max(0, self.start - d), self.end + d
)


def is_empty_file(file: str) -> bool:
    return os.stat(file).st_size == 0


def extract_region_names(region_set: genome_tools.RegionSet) -> List[str]:
    region_names = [region.annotation[0] for region in region_set.regions]
    return region_names


def _extract_features(
    bed_file: str, gene_loc_set: genome_tools.RegionSet, decay: float = 10_000
) -> np.ndarray:
    if is_empty_file(bed_file):
        return np.zeros(len(gene_loc_set.regions))

    region_set = prepare_region_set(bed_file, gene_loc_set.genome)
    rp_map = data_interface.DataInterface._make_basic_rp_map(
        gene_loc_set, region_set, decay
    )
    features = rp_map.sum(axis=1).flatten().A1
    return features


def extract_features(
    bed_file: str, gene_loc_set: genome_tools.RegionSet, decay: float = 10_000
) -> pd.Series:
    """
    Extract features from a BED file and a gene location set.

    Parameters:
        bed_file (str): Path to the BED file.
        gene_loc_set (genome_tools.RegionSet): Gene location set.
        decay (float): Decay parameter for the RP map. Default is 10,000.

    Returns:
        pd.Series: A pandas Series containing the extracted features with region names as the index.
    """
    features = _extract_features(bed_file, gene_loc_set, decay)
    region_names = extract_region_names(gene_loc_set)
    features_series = pd.Series(features, index=region_names)
    return features_series
