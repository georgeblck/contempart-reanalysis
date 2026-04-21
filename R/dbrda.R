# Distance-based Redundancy Analysis (db-RDA) for every registered head.
#
# Reads `results/heads.csv` (written by `src.step1_link`) and loops over
# every head producing:
#   results/<head>_dbrda.csv         marginal tests (each var controlling
#                                     for the others)
#   results/<head>_varpart.csv       school-vs-professor variance partition
#
# Finally writes a combined wide CSV:
#   results/all_dbrda.csv            one row per (head, variable)
#
# Run: Rscript R/dbrda.R [head1 head2 ...]

suppressPackageStartupMessages({
  library(vegan)
  library(readr)
  library(dplyr)
  library(reticulate)
  library(tidyr)
  library(parallel)
})

results_dir <- "results"
heads <- read_csv(file.path(results_dir, "heads.csv"), show_col_types = FALSE)

n_perm <- 999
n_cores <- max(1, detectCores() - 1)
message("Using ", n_cores, " cores, ", n_perm, " permutations")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  heads <- heads |> filter(name %in% args)
}

np <- import("numpy", convert = FALSE)

all_marginal <- list()
all_varpart <- list()

for (i in seq_len(nrow(heads))) {
  head_name <- heads$name[i]
  head_display <- heads$display[i]

  emb_path <- file.path(results_dir, paste0(head_name, "_artist_emb.npy"))
  meta_path <- file.path(results_dir, paste0(head_name, "_metadata.csv"))
  if (!file.exists(emb_path) || !file.exists(meta_path)) {
    message("Skipping ", head_name, " (missing artifacts)")
    next
  }

  message("\n", paste(rep("=", 60), collapse = ""))
  message("db-RDA: ", head_display, " [", head_name, "]")
  message(paste(rep("=", 60), collapse = ""))

  emb_raw <- py_to_r(np$load(emb_path))
  df <- read_csv(meta_path, show_col_types = FALSE)
  stopifnot(nrow(emb_raw) == nrow(df))

  # Cosine distance (row-normalize, then 1 - dot product)
  norms <- sqrt(rowSums(emb_raw^2))
  norm_emb <- emb_raw / norms
  cos_sim <- tcrossprod(norm_emb)
  cos_dist <- as.dist(1 - cos_sim)

  env <- df |>
    mutate(
      school = factor(school),
      gender = factor(replace_na(gender, "Unknown")),
      professor_class = factor(replace_na(professor_class, "Unknown")),
      continent = factor(replace_na(continent, "Unknown"))
    ) |>
    select(school, gender, professor_class, continent)

  full_model <- dbrda(
    cos_dist ~ school + gender + professor_class + continent,
    data = env
  )
  full_anova <- anova(full_model, permutations = n_perm, parallel = n_cores)
  total_pct <- full_model$CCA$tot.chi / full_model$tot.chi * 100
  message(sprintf(
    "Total constrained: %.1f%%  Global F=%.2f  p=%.4f",
    total_pct, full_anova$F[1], full_anova$`Pr(>F)`[1]
  ))

  margin_anova <- anova(full_model, by = "margin", permutations = n_perm, parallel = n_cores)
  ma <- as.data.frame(margin_anova)
  ma$variable <- rownames(ma)
  margin_df <- ma |>
    filter(!is.na(F)) |>
    transmute(
      head = head_name,
      display = head_display,
      variable = variable,
      Df = Df,
      variance = SumOfSqs,
      variance_pct = SumOfSqs / full_model$tot.chi * 100,
      F_stat = F,
      p_value = `Pr(>F)`
    )
  write_csv(margin_df, file.path(results_dir, paste0(head_name, "_dbrda.csv")))
  all_marginal[[head_name]] <- margin_df

  # school vs professor variance partition
  vp <- varpart(cos_dist, ~ school, ~ professor_class, data = env)
  frac <- vp$part$indfract$Adj.R.squared
  vp_df <- tibble(
    head = head_name,
    display = head_display,
    school_only = frac[1],
    professor_only = frac[3],
    shared = frac[2],
    residual = frac[4]
  )
  write_csv(vp_df, file.path(results_dir, paste0(head_name, "_varpart.csv")))
  all_varpart[[head_name]] <- vp_df

  for (j in seq_len(nrow(margin_df))) {
    sig <- if (!is.na(margin_df$p_value[j]) && margin_df$p_value[j] < 0.05) "*" else " "
    message(sprintf(
      "  %-18s var=%5.2f%%  F=%5.2f  p=%.4f %s",
      margin_df$variable[j],
      margin_df$variance_pct[j],
      margin_df$F_stat[j],
      margin_df$p_value[j],
      sig
    ))
  }
  message(sprintf(
    "  varpart: school-only=%.1f%%  prof-only=%.1f%%  shared=%.1f%%",
    vp_df$school_only * 100, vp_df$professor_only * 100, vp_df$shared * 100
  ))
}

combined <- bind_rows(all_marginal)
write_csv(combined, file.path(results_dir, "all_dbrda.csv"))

vp_combined <- bind_rows(all_varpart)
write_csv(vp_combined, file.path(results_dir, "all_varpart.csv"))

message("\nDone. ", nrow(combined), " rows -> results/all_dbrda.csv")
