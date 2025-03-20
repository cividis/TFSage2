import subprocess
from importlib import resources
from typing import List
import pandas as pd
import tempfile
import os


def run_seurat_integration(
    rp_matrix: str,
    metadata: str,
    output_dir: str,
    align_key: str = "Assay",
    methods: List[str] = ["FastMNNIntegration"],
) -> None:
    """
    Generate embeddings using Seurat by calling the seurat_integration.R script.

    Parameters:
        rp_matrix (str): Path to the RP matrix file (input, Parquet format).
        metadata (str): Path to the metadata file (input, Parquet format).
        output_dir (str): Directory to save embeddings files (output).
        align_key (str, optional): Alignment key used to split metadata. Default is 'Assay'.
        methods (List[str], optional): List of embedding generation methods. Default is ['FastMNNIntegration'].

    Returns:
        None
    """
    methods_str = ",".join(methods)
    cmd = [
        "Rscript",
        resources.files("tfsage.assets").joinpath("seurat_integration.R"),
        "--rp-matrix",
        rp_matrix,
        "--metadata",
        metadata,
        "--output-dir",
        output_dir,
        "--align-key",
        align_key,
        "--method",
        methods_str,
    ]
    subprocess.run(cmd, check=True)


def generate_embeddings(
    rp_matrix_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    align_key: str = "Assay",
    method: str = "FastMNNIntegration",
) -> pd.DataFrame:
    """
    Wrapper function to generate embeddings using Seurat.

    Parameters:
        rp_matrix_df (pd.DataFrame): RP matrix as a pandas DataFrame.
        metadata_df (pd.DataFrame): Metadata as a pandas DataFrame.
        align_key (str, optional): Alignment key used to split metadata. Default is 'Assay'.
        method (str, optional): Embedding generation method. Default is 'FastMNNIntegration'.

    Returns:
        pd.DataFrame: The resulting embedding matrix as a pandas DataFrame.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        rp_matrix_path = os.path.join(temp_dir, "rp_matrix.parquet")
        metadata_path = os.path.join(temp_dir, "metadata.parquet")

        # Remove index name
        rp_matrix_df.index.name = None
        metadata_df.index.name = None

        # Save the RP matrix and metadata as Parquet files
        rp_matrix_df.to_parquet(rp_matrix_path)
        metadata_df.to_parquet(metadata_path)

        # Generate embeddings
        run_seurat_integration(
            rp_matrix=rp_matrix_path,
            metadata=metadata_path,
            output_dir=temp_dir,
            align_key=align_key,
            methods=[method],
        )

        # Read and return the resulting embedding matrix
        embeddings_path = os.path.join(temp_dir, f"{method}.parquet")
        embedding_df = pd.read_parquet(embeddings_path)
        embedding_df.set_index("__index_level_0__", inplace=True)
        embedding_df.index.name = None
        return embedding_df
