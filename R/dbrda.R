# Distance-based Redundancy Analysis (db-RDA)
#
# Tests the unique contribution of each demographic variable to embedding
# distances, controlling for confounding between variables.
#
# Addresses the key limitation of Mantel tests: school and professor are
# confounded (professors work at specific schools). db-RDA partitions
# variance so we can see each variable's independent contribution.
#
# Run: Rscript R/dbrda.R

library(vegan)
library(readr)
library(dplyr)

results_dir <- "results"

for (vec_name in c("c_vectors", "a_vectors")) {
  emb_path <- file.path(results_dir, paste0(vec_name, "_artist_emb.npy"))
  if (!file.exists(emb_path)) {
    message("Skipping ", vec_name, " (no embeddings)")
    next
  }

  label <- if (vec_name == "c_vectors") "C-vectors (content)" else "A-vectors (appearance)"
  message("\n", paste(rep("=", 60), collapse = ""))
  message("db-RDA: ", label)
  message(paste(rep("=", 60), collapse = ""))

  # Load UMAP data (has metadata merged in)
  df <- read_csv(
    file.path(results_dir, paste0(vec_name, "_umap_data.csv")),
    show_col_types = FALSE
  )

  # Load the artist distance matrix (cosine, from PCA embeddings)
  # We need the raw artist embeddings to compute a distance matrix
  # Use reticulate to load .npy, or compute from UMAP data
  # Actually, we should use the cosine distance on the full embeddings
  # Let's load via reticulate
  library(reticulate)
  np <- import("numpy", convert = FALSE)
  emb_raw <- py_to_r(np$load(emb_path))

  # Match ordering: embeddings are sorted by artist_id
  artist_order <- sort(unique(df$artist_id))
  # df is already sorted by artist_id from the merge
  stopifnot(nrow(emb_raw) == length(artist_order))

  # Compute cosine distance matrix (vegan doesn't support cosine natively)
  # cosine distance = 1 - cosine similarity
  norm_emb <- emb_raw / sqrt(rowSums(emb_raw^2))
  cos_sim <- tcrossprod(norm_emb)
  cos_dist <- as.dist(1 - cos_sim)

  # Prepare predictors (factors)
  env <- df |>
    mutate(
      school = factor(school),
      gender = factor(tidyr::replace_na(gender, "Unknown")),
      professor_class = factor(tidyr::replace_na(professor_class, "Unknown")),
      continent = factor(tidyr::replace_na(continent, "Unknown"))
    ) |>
    select(school, gender, professor_class, continent)

  # ---- Full model: all variables ----
  message("\n--- Full model (all variables) ---")
  full_model <- dbrda(cos_dist ~ school + gender + professor_class + continent, data = env)
  full_anova <- anova(full_model, permutations = 9999)
  message("Total constrained variance: ",
          round(full_model$CCA$tot.chi / full_model$tot.chi * 100, 1), "%")
  message("Global test: F=", round(full_anova$F[1], 2),
          ", p=", format(full_anova$`Pr(>F)`[1], digits = 4))

  # ---- Marginal tests: each variable controlling for the others ----
  message("\n--- Marginal tests (Type II, each controlling for others) ---")
  margin_anova <- anova(full_model, by = "margin", permutations = 9999)
  print(margin_anova)

  # ---- Variance partitioning ----
  message("\n--- Variance partitioning: school vs professor ---")
  # How much do school and professor overlap?
  vp <- varpart(cos_dist, ~ school, ~ professor_class, data = env)
  message("School alone: ", round(vp$part$indfract$Adj.R.squared[1] * 100, 2), "%")
  message("Professor alone: ", round(vp$part$indfract$Adj.R.squared[2] * 100, 2), "%")
  message("Shared: ", round(vp$part$indfract$Adj.R.squared[3] * 100, 2), "%")

  # ---- Save results ----
  # Marginal tests as CSV
  ma <- as.data.frame(margin_anova)
  margin_df <- data.frame(
    variable = rownames(ma),
    Df = ma$Df,
    variance = ma$SumOfSqs,
    F_stat = ma$F,
    p_value = ma$`Pr(>F)`,
    row.names = NULL
  )
  # Remove the Residual row for cleaner output
  margin_df <- margin_df[!is.na(margin_df$F_stat), ]

  out_path <- file.path(results_dir, paste0(vec_name, "_dbrda.csv"))
  write_csv(margin_df, out_path)
  message("\nSaved to ", out_path)

  # Print a clean summary
  message("\n--- Summary ---")
  message(sprintf("%-20s %8s %8s %8s", "Variable", "F", "p-value", "Sig?"))
  message(paste(rep("-", 50), collapse = ""))
  for (i in seq_len(nrow(margin_df))) {
    sig <- if (!is.na(margin_df$p_value[i]) && margin_df$p_value[i] < 0.05) "yes" else "no"
    message(sprintf("%-20s %8.2f %8.4f %8s",
                    margin_df$variable[i],
                    margin_df$F_stat[i],
                    margin_df$p_value[i],
                    sig))
  }
}

message("\nDone.")
