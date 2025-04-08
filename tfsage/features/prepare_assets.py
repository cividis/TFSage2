import pandas as pd
from importlib import resources
from typing import Literal
from lisa.core import genome_tools

genome_tools.Genome.check_region = lambda self, region: None


def prepare_genome(genome_file: str) -> genome_tools.Genome:
    """
    Prepare a Genome object from a genome file.

    Parameters:
        genome_file (str): Path to the genome file containing chromosome names and lengths.

    Returns:
        genome_tools.Genome: A Genome object initialized with the chromosome names and lengths.
    """
    df = pd.read_csv(
        genome_file, sep="\t", header=None, usecols=[0, 1], dtype={0: str, 1: int}
    )
    chromosomes = df[0].values
    lengths = df[1].values
    genome = genome_tools.Genome(chromosomes, lengths)
    return genome


def prepare_region_set(
    region_file: str, genome: genome_tools.Genome
) -> genome_tools.RegionSet:
    """
    Prepare a RegionSet object from a region file and a Genome object.

    Parameters:
        region_file (str): Path to the region file in BED format.
        genome (genome_tools.Genome): A Genome object.

    Returns:
        genome_tools.RegionSet: A RegionSet object initialized with the regions from the region file and the Genome object.
    """
    regions = genome_tools.Region.read_bedfile(region_file)
    region_set = genome_tools.RegionSet(regions, genome)
    return region_set


def load_region_set(genome: Literal["hg38", "mm10"] = "hg38") -> genome_tools.RegionSet:
    """
    Load a RegionSet object for a specified genome.

    Parameters:
        genome (Literal["hg38", "mm10"], optional): The genome version to load (e.g., "hg38" or "mm10"). Default is "hg38".

    Returns:
        genome_tools.RegionSet: A RegionSet object for the specified genome.
    """
    genome_file = resources.files("tfsage.assets").joinpath(f"{genome}/{genome}.len")
    region_file = resources.files("tfsage.assets").joinpath(
        f"{genome}/{genome}_refseq_TSS.bed"
    )
    genome = prepare_genome(genome_file)
    region_set = prepare_region_set(region_file, genome)
    return region_set


if __name__ == "__main__":
    region_set = load_region_set("hg38")
    print(region_set)
