from typing import Literal
from .curl_and_sort import curl_and_sort


def download_chip_atlas(
    experiment_id: str,
    output_file: str,
    threshold: Literal["05", "10", "20", "50"] = "05",
    genome: str = "hg38",
) -> None:
    """
    Download a ChIP-Atlas BED file for a given experiment ID and save it to the specified output file.

    Parameters:
        experiment_id (str): The ID of the experiment to download.
        output_file (str): The path to the output file where the downloaded data will be saved.
        threshold (Literal["05", "10", "20", "50"], optional): The threshold for the ChIP-Atlas data. Default is "05".
        genome (str, optional): The genome version (e.g., "hg38"). Default is "hg38".

    Returns:
        None
    """
    url = "https://chip-atlas.dbcls.jp/data/{genome}/eachData/bed{threshold}/{experiment_id}.{threshold}.bed".format(
        experiment_id=experiment_id,
        threshold=threshold,
        genome=genome,
    )
    curl_and_sort(url, output_file)
