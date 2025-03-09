import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, IO
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


def _extract_features_ripgrep(
    bed_file: str, name: str, stdout: IO, decay_factor: float = 10_000
) -> np.ndarray:
    subprocess.run(["rg", "-N", name, bed_file], stdout=stdout, check=True)
    features = _extract_features(stdout.name, _gene_loc_set, decay_factor)
    return features


def process_bed_file_for_name(
    bed_file: str,
    name: str,
    decay_factor: float = 10_000,
    data_dir: str | None = None,
) -> np.ndarray:
    def get_data_path() -> str | None:
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            return os.path.join(data_dir, f"{name}.bed")
        return None

    data_path = get_data_path()
    if data_path and os.path.exists(data_path):
        features = _extract_features(data_path, _gene_loc_set, decay_factor)
    else:
        with (
            open(data_path, "w") if data_path else tempfile.NamedTemporaryFile()
        ) as stdout:
            features = _extract_features_ripgrep(bed_file, name, stdout, decay_factor)
    return features


def extract_features_chip_atlas(
    bed_file: str,
    individual_names: List[str],
    gene_loc_set: genome_tools.RegionSet,
    decay_factor: float = 10_000,
    max_workers: int | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """
    Compute RP scores for a batch of individual names from a merged BED file in parallel.

    Parameters:
        bed_file (str): Path to the merged BED file.
        individual_names (List[str]): List of individual names to process from the merged BED file.
        gene_loc_set (genome_tools.RegionSet): Gene location set.
        decay_factor (float): Decay parameter for the RP map. Default is 10,000.
        max_workers (int | None): Maximum number of workers for parallel processing. Default is None.
        data_dir (str | None): Directory to save or retrieve the filtered BED files. If None, temporary files are used.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the computed RP scores with region names as the index.
    """
    results = [None] * len(individual_names)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initializer,
        initargs=(gene_loc_set,),
    ) as executor:
        futures = {
            executor.submit(
                process_bed_file_for_name, bed_file, name, decay_factor, data_dir
            ): i
            for i, name in enumerate(individual_names)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]
            results[index] = future.result()

    features = np.stack(results).T
    features_df = pd.DataFrame(features, index=extract_region_names(gene_loc_set))
    return features_df
