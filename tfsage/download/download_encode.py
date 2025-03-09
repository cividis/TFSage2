from .curl_and_sort import curl_and_sort


def download_encode(experiment_id: str, output_file: str) -> None:
    """
    Download an ENCODE BED file for a given experiment ID and save it to the specified output file.

    Parameters:
        experiment_id (str): The ID of the experiment to download.
        output_file (str): The path to the output file where the downloaded data will be saved.

    Returns:
        None
    """
    url = "https://www.encodeproject.org/files/{experiment_id}/@@download/{experiment_id}.bed.gz".format(
        experiment_id=experiment_id
    )
    curl_and_sort(url, output_file)
