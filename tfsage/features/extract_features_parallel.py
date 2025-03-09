from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from lisa.core import genome_tools
from .extract_features import _extract_features, extract_region_names

# Global variable for the worker processes
_gene_loc_set = None


def initializer(gene_loc_set: genome_tools.RegionSet) -> None:
    global _gene_loc_set
    _gene_loc_set = gene_loc_set  # Each worker gets its own copy


def process_bed_file(bed_file: str, decay_factor: float = 10_000) -> np.ndarray:
    features = _extract_features(bed_file, _gene_loc_set, decay_factor)
    return features


def extract_features_parallel(
    bed_files: List[str],
    gene_loc_set: genome_tools.RegionSet,
    decay_factor: float = 10_000,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Compute RP scores for a batch of BED files in parallel.

    Parameters:
        bed_files (List[str]): List of paths to the BED files.
        gene_loc_set (genome_tools.RegionSet): Gene location set.
        decay_factor (float): Decay parameter for the RP map. Default is 10,000.
        max_workers (int | None): Maximum number of workers for parallel processing. Default is None.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the computed RP scores with region names as the index.
    """
    results = [None] * len(bed_files)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initializer,
        initargs=(gene_loc_set,),
    ) as executor:
        futures = {
            executor.submit(process_bed_file, bed_file, decay_factor): i
            for i, bed_file in enumerate(bed_files)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]
            results[index] = future.result()

    features = np.stack(results).T
    features_df = pd.DataFrame(features, index=extract_region_names(gene_loc_set))
    return features_df
