# ------------------------------------------------------------------------------
# Script: seurat_integration.R
# Description:
#   This script generates embeddings using Seurat. It takes an RP matrix and
#   metadata in Parquet format as input, preprocesses the data, and optionally
#   performs integration. The embeddings are saved in Parquet format for each
#   specified method.
#
# Usage:
#   Rscript seurat_integration.R \
#     --rp-matrix /path/to/rp_matrix.parquet \
#     --metadata /path/to/metadata.parquet \
#     --output-dir /path/to/output \
#     --align-key Assay \
#     --method CCAIntegration,HarmonyIntegration
#
# Required Arguments:
#   --rp-matrix    Path to RP matrix file (input, Parquet format).
#   --metadata     Path to metadata file (input, Parquet format).
#   --output-dir   Directory to save embeddings files (output).
#
# Optional Arguments:
#   --align-key    Alignment key used to split metadata (default: 'Assay').
#   --method       Comma-separated list of embedding generation methods.
#                  Options: CCAIntegration, HarmonyIntegration,
#                  JointPCAIntegration, RPCAIntegration, FastMNNIntegration,
#                  none (skip integration and use PCA embeddings).
#                  Default: 'FastMNNIntegration'.
#
# Output:
#   - Parquet files containing the generated embeddings for each specified
#     method.
#
# Requirements:
#   - R packages: optparse, arrow, Seurat, dplyr, tibble, SeuratWrappers
#   - Input files in Parquet format.
# ------------------------------------------------------------------------------

library(optparse)
library(arrow)
library(Seurat)
library(dplyr)
library(tibble)
library(SeuratWrappers)

# Define command-line options
option_list <- list(
  make_option(c("--rp-matrix"),
    type = "character",
    help = "Path to RP matrix file (input, Parquet format).",
    metavar = "file"
  ),
  make_option(c("--metadata"),
    type = "character",
    help = "Path to metadata file (input, Parquet format).",
    metavar = "file"
  ),
  make_option(c("--output-dir"),
    type = "character",
    help = "Directory to save embeddings files (output). Each file is named after its method.", # nolint: line_length_linter.
    metavar = "dir"
  ),
  make_option(c("--align-key"),
    type = "character",
    help = "Alignment key used to split metadata (default: 'Assay').",
    metavar = "key",
    default = "Assay"
  ),
  make_option(c("--method"),
    type = "character",
    help = "Comma-separated list of embedding generation methods (default: 'FastMNNIntegration'). Options: CCAIntegration, HarmonyIntegration, JointPCAIntegration, RPCAIntegration, FastMNNIntegration, none.", # nolint: line_length_linter.
    metavar = "methods",
    default = "FastMNNIntegration"
  )
)

# Parse arguments
option_parser <- OptionParser(option_list = option_list)
args <- parse_args(option_parser)

# Parse multiple methods
methods <- strsplit(args$method, ",")[[1]]

# Define valid methods
valid_methods <- c(
  "CCAIntegration",
  "HarmonyIntegration",
  "JointPCAIntegration",
  "RPCAIntegration",
  "FastMNNIntegration",
  "none"
)

# Check for invalid methods
invalid_methods <- setdiff(methods, valid_methods)
if (length(invalid_methods) > 0) {
  stop(paste(
    "Invalid method(s):",
    paste(invalid_methods, collapse = ", "),
    "\nValid options are:",
    paste(valid_methods, collapse = ", ")
  ))
}

# Create output directory if it doesn't exist
output_dir <- args$`output-dir`
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Generate file paths for each method
embedding_paths <- file.path(output_dir, paste0(methods, ".parquet"))

# Load data
seurat_object <- CreateSeuratObject(
  counts = read_parquet(file = args$`rp-matrix`) %>%
    select(-`__index_level_0__`) %>%
    as.matrix() %>%
    as("dgCMatrix"),
  meta.data = read_parquet(file = args$metadata) %>%
    rename(split_key = !!rlang::sym(args$`align-key`)) %>%
    select(`__index_level_0__`, split_key) %>%
    column_to_rownames(var = "__index_level_0__")
)

# Split assay
seurat_object[["RNA"]] <-
  split(seurat_object[["RNA"]], f = seurat_object$split_key)

# Preprocess data once
seurat_object <- seurat_object %>%
  NormalizeData() %>%
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA()

# Generate embeddings for each method
for (i in seq_along(methods)) {
  method <- methods[i]
  embeddings_path <- embedding_paths[i]

  # Perform integration if needed
  if (method == "CCAIntegration") {
    seurat_object <- IntegrateLayers(
      object = seurat_object,
      method = CCAIntegration,
      new.reduction = "integrated.cca"
    )
    embeddings <- seurat_object@reductions$integrated.cca@cell.embeddings
  } else if (method == "HarmonyIntegration") {
    seurat_object <- IntegrateLayers(
      object = seurat_object,
      method = HarmonyIntegration,
      new.reduction = "harmony"
    )
    embeddings <- seurat_object@reductions$harmony@cell.embeddings
  } else if (method == "JointPCAIntegration") {
    seurat_object <- IntegrateLayers(
      object = seurat_object,
      method = JointPCAIntegration,
      new.reduction = "integrated.jpca"
    )
    embeddings <- seurat_object@reductions$integrated.jpca@cell.embeddings
  } else if (method == "RPCAIntegration") {
    seurat_object <- IntegrateLayers(
      object = seurat_object,
      method = RPCAIntegration,
      new.reduction = "integrated.rpca"
    )
    embeddings <- seurat_object@reductions$integrated.rpca@cell.embeddings
  } else if (method == "FastMNNIntegration") {
    seurat_object <- IntegrateLayers(
      object = seurat_object,
      method = FastMNNIntegration,
      new.reduction = "integrated.mnn"
    )
    embeddings <- seurat_object@reductions$integrated.mnn@cell.embeddings
  } else if (method == "none") {
    embeddings <- seurat_object@reductions$pca@cell.embeddings
  }

  # Save embeddings
  embeddings %>%
    as.data.frame() %>%
    rownames_to_column("__index_level_0__") %>%
    write_parquet(embeddings_path)
}
