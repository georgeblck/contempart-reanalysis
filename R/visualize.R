# Step 3: UMAP visualizations for contempArt CLIP analysis
#
# Reads UMAP coordinates + metadata, produces publication-quality plots.
# Run: Rscript R/visualize.R
# Expects: results/c_vectors_umap_data.csv, results/c_vectors_spread_by_school.csv

library(ggplot2)
library(dplyr)
library(readr)
library(khroma)

plots_dir <- "plots"
dir.create(plots_dir, showWarnings = FALSE)

# Tufte-inspired minimal theme
theme_tufte_minimal <- function(base_size = 11) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks = element_line(colour = "grey50", linewidth = 0.3),
      axis.text = element_blank(),
      axis.title = element_text(size = rel(0.9), colour = "grey30"),
      legend.key.size = unit(0.8, "lines"),
      legend.text = element_text(size = rel(0.75)),
      legend.title = element_text(size = rel(0.85)),
      plot.title = element_text(size = rel(1.1), face = "plain", hjust = 0),
      plot.subtitle = element_text(size = rel(0.85), colour = "grey40", hjust = 0),
      strip.text = element_text(size = rel(0.85))
    )
}

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

vec_types <- c("c_vectors", "a_vectors")

for (vec_name in vec_types) {
  umap_file <- file.path("results", paste0(vec_name, "_umap_data.csv"))
  spread_file <- file.path("results", paste0(vec_name, "_spread_by_school.csv"))

  if (!file.exists(umap_file)) {
    message("Skipping ", vec_name, " (no UMAP data)")
    next
  }

  df <- read_csv(umap_file, show_col_types = FALSE)
  label <- if (vec_name == "c_vectors") "C-vectors (content)" else "A-vectors (appearance)"
  prefix <- vec_name

  message("Plotting ", label, ": ", nrow(df), " artists")

  # ------ UMAP by school (15 schools, Brewer Paired for >8 groups) ------
  n_schools <- n_distinct(df$school)
  # Paired palette handles 12; extend with greys for 13+
  paired_12 <- c("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
                 "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928")
  school_pal <- c(paired_12, "#666666", "#999999", "#CCCCCC")[seq_len(n_schools)]
  school_shapes <- rep(c(16, 17, 15, 18, 3, 4, 8, 1, 2, 0, 5, 6, 7, 9, 10), length.out = n_schools)

  p_school <- ggplot(df, aes(umap_1, umap_2, colour = school, shape = school)) +
    geom_point(size = 1.5, alpha = 0.7) +
    scale_colour_manual(values = school_pal) +
    scale_shape_manual(values = school_shapes) +
    labs(
      title = paste0("UMAP by art school (", label, ")"),
      subtitle = paste0(nrow(df), " artists, ", n_schools, " schools"),
      x = "UMAP 1", y = "UMAP 2",
      colour = "School", shape = "School"
    ) +
    theme_tufte_minimal() +
    guides(
      colour = guide_legend(ncol = 2, override.aes = list(size = 2.5)),
      shape = guide_legend(ncol = 2, override.aes = list(size = 2.5))
    )

  ggsave(file.path(plots_dir, paste0(prefix, "_umap_school.png")),
         p_school, width = 6, height = 4, dpi = 300)

  # ------ UMAP by gender (Okabe-Ito) ------
  df_gender <- df |> mutate(gender = tidyr::replace_na(gender, "Unknown"))
  gender_pal <- c("F" = "#E69F00", "M" = "#56B4E9", "Unknown" = "#999999")
  gender_shapes <- c("F" = 16, "M" = 17, "Unknown" = 4)

  p_gender <- ggplot(df_gender, aes(umap_1, umap_2, colour = gender, shape = gender)) +
    geom_point(size = 1.8, alpha = 0.7) +
    scale_colour_manual(values = gender_pal) +
    scale_shape_manual(values = gender_shapes) +
    labs(
      title = paste0("UMAP by gender (", label, ")"),
      subtitle = paste0(nrow(df), " artists"),
      x = "UMAP 1", y = "UMAP 2",
      colour = "Gender", shape = "Gender"
    ) +
    theme_tufte_minimal()

  ggsave(file.path(plots_dir, paste0(prefix, "_umap_gender.png")),
         p_gender, width = 6, height = 4, dpi = 300)

  # ------ UMAP by continent (Okabe-Ito) ------
  df_cont <- df |> mutate(continent = tidyr::replace_na(continent, "Unknown"))
  n_cont <- n_distinct(df_cont$continent)
  cont_pal <- colour("bright")(min(n_cont, 7))
  cont_shapes <- c(16, 17, 15, 18, 3, 4, 8)[seq_len(n_cont)]

  p_cont <- ggplot(df_cont, aes(umap_1, umap_2, colour = continent, shape = continent)) +
    geom_point(size = 1.8, alpha = 0.7) +
    scale_colour_manual(values = rep(cont_pal, length.out = n_cont)) +
    scale_shape_manual(values = cont_shapes) +
    labs(
      title = paste0("UMAP by continent (", label, ")"),
      subtitle = paste0(sum(df_cont$continent != "Unknown"), " artists with nationality data"),
      x = "UMAP 1", y = "UMAP 2",
      colour = "Continent", shape = "Continent"
    ) +
    theme_tufte_minimal()

  ggsave(file.path(plots_dir, paste0(prefix, "_umap_continent.png")),
         p_cont, width = 6, height = 4, dpi = 300)

  # ------ UMAP by professor class (top 10 + Other) ------
  top_profs <- df |>
    filter(!is.na(professor_class)) |>
    count(professor_class, sort = TRUE) |>
    slice_head(n = 10) |>
    pull(professor_class)

  df_prof <- df |>
    mutate(professor_group = case_when(
      professor_class %in% top_profs ~ professor_class,
      is.na(professor_class) ~ "Unknown",
      TRUE ~ "Other"
    ))

  n_groups <- n_distinct(df_prof$professor_group)
  prof_pal <- c(paired_12, "#666666")[seq_len(min(n_groups, 13))]
  prof_shapes <- rep(c(16, 17, 15, 18, 3, 4, 8, 1, 2, 0, 5, 6, 7), length.out = n_groups)

  p_prof <- ggplot(df_prof, aes(umap_1, umap_2, colour = professor_group, shape = professor_group)) +
    geom_point(size = 1.8, alpha = 0.7) +
    scale_colour_manual(values = rep(prof_pal, length.out = n_groups)) +
    scale_shape_manual(values = prof_shapes) +
    labs(
      title = paste0("UMAP by professor class (", label, ")"),
      subtitle = "Top 10 classes shown, rest grouped as Other",
      x = "UMAP 1", y = "UMAP 2",
      colour = "Professor", shape = "Professor"
    ) +
    theme_tufte_minimal() +
    guides(
      colour = guide_legend(ncol = 2, override.aes = list(size = 2.5)),
      shape = guide_legend(ncol = 2, override.aes = list(size = 2.5))
    )

  ggsave(file.path(plots_dir, paste0(prefix, "_umap_professor.png")),
         p_prof, width = 6, height = 4, dpi = 300)

  # ------ Spread by school (horizontal bar) ------
  if (file.exists(spread_file)) {
    spread <- read_csv(spread_file, show_col_types = FALSE) |>
      arrange(spread)
    spread$group <- factor(spread$group, levels = spread$group)

    p_spread <- ggplot(spread, aes(spread, group)) +
      geom_point(colour = "#56B4E9", size = 3) +
      geom_segment(aes(x = 0, xend = spread, y = group, yend = group),
                   colour = "#56B4E9", linewidth = 0.5) +
      geom_text(aes(label = paste0("n=", n)), hjust = -0.3, size = 3, colour = "grey40") +
      scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
      labs(
        title = paste0("Intra-school spread (", label, ")"),
        subtitle = "Mean pairwise cosine distance per school (higher = more diverse)",
        x = "Mean cosine distance", y = NULL
      ) +
      theme_minimal(base_size = 11) +
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks.y = element_blank()
      )

    ggsave(file.path(plots_dir, paste0(prefix, "_spread_school.png")),
           p_spread, width = 6, height = 4, dpi = 300)
  }

  message("  Done: ", prefix)
}

message("All plots saved to ", plots_dir, "/")
