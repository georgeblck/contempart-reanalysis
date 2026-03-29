# Visualization of contempArt CLIP embeddings
# Mirrors the 2020 ECCV paper figures but with CLIP C-vectors and A-vectors
#
# Usage:
#   Rscript src/plot_embeddings.R --results-dir results --metadata data/finalData.csv
#
# Produces t-SNE/UMAP scatter plots colored by:
#   - Art school (Hochschule)
#   - Gender
#   - Nationality / region
#   - Instagram engagement (likes per image)
#   - Embedding type comparison (C-vector vs A-vector)

library(tidyverse)
library(umap)
library(Rtsne)

# Okabe-Ito palette (colorblind-safe, max 8 groups)
okabe_ito <- c(
  "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7", "#999999"
)

# Paul Tol muted (colorblind-safe, up to 10 groups)
tol_muted <- c(
  "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
  "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD"
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

load_artist_data <- function(results_dir, metadata_path) {
  # Load PCA-reduced artist embeddings
  # Python saves as .npy, read via RcppCNPy or pre-convert to CSV
  # For simplicity, expect CSV versions saved by analyze.py

  files <- list(
    c_pca = file.path(results_dir, "c_vectors_artist_pca.csv"),
    a_pca = file.path(results_dir, "a_vectors_artist_pca.csv"),
    artists = file.path(results_dir, "artists.csv")
  )

  data <- list()

  if (file.exists(files$c_pca)) {
    data$c_pca <- read_csv(files$c_pca, show_col_types = FALSE)
  }
  if (file.exists(files$a_pca)) {
    data$a_pca <- read_csv(files$a_pca, show_col_types = FALSE)
  }
  if (file.exists(files$artists)) {
    data$artists <- read_csv(files$artists, show_col_types = FALSE)
  }

  if (file.exists(metadata_path)) {
    data$metadata <- read_csv(metadata_path, show_col_types = FALSE)
  }

  data
}

# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

compute_tsne <- function(emb_matrix, perplexity = 30, seed = 42) {
  set.seed(seed)
  tsne <- Rtsne(
    emb_matrix,
    dims = 2,
    perplexity = perplexity,
    max_iter = 5000,
    check_duplicates = FALSE
  )
  data.frame(tsne1 = tsne$Y[, 1], tsne2 = tsne$Y[, 2])
}

compute_umap <- function(emb_matrix, n_neighbors = 15, seed = 42) {
  set.seed(seed)
  um <- umap(emb_matrix, n_neighbors = n_neighbors)
  data.frame(umap1 = um$layout[, 1], umap2 = um$layout[, 2])
}

# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

plot_scatter <- function(df, x, y, color_var, title,
                         palette = NULL, shape_var = NULL) {
  p <- ggplot(df, aes(x = .data[[x]], y = .data[[y]],
                       colour = .data[[color_var]])) +
    geom_point(size = 1.5, alpha = 0.7) +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 11),
      panel.grid.minor = element_blank()
    )

  if (!is.null(shape_var) && shape_var %in% names(df)) {
    p <- p + aes(shape = .data[[shape_var]])
  }

  n_groups <- length(unique(df[[color_var]]))
  if (!is.null(palette)) {
    p <- p + scale_colour_manual(values = palette[seq_len(n_groups)])
  } else if (n_groups <= 8) {
    p <- p + scale_colour_manual(values = okabe_ito[seq_len(n_groups)])
  } else {
    p <- p + scale_colour_manual(values = tol_muted[seq_len(min(n_groups, 10))])
  }

  p
}

plot_spread_comparison <- function(spread_df) {
  ggplot(spread_df, aes(x = reorder(group, spread), y = spread)) +
    geom_col(fill = "#56B4E9", alpha = 0.8) +
    geom_text(aes(label = n), hjust = -0.2, size = 3) +
    coord_flip() +
    labs(
      title = "Embedding spread by art school",
      x = NULL,
      y = "Mean cosine distance (higher = more diverse)"
    ) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank())
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  results_dir <- if (length(args) >= 2 && args[1] == "--results-dir") args[2] else "results"
  metadata_path <- if (length(args) >= 4 && args[3] == "--metadata") args[4] else "data/finalData.csv"
  output_dir <- "plots"

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  cat("Loading data...\n")
  data <- load_artist_data(results_dir, metadata_path)

  cat("Ready for plotting.\n")
  cat("Available data:\n")
  for (name in names(data)) {
    if (is.data.frame(data[[name]])) {
      cat(sprintf("  %s: %d rows x %d cols\n", name, nrow(data[[name]]), ncol(data[[name]])))
    }
  }
}

main()
