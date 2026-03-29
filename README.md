# contempart-clip

Re-analysis of the contempArt dataset (Huckle, Garcia & Nakashima, ECCV Workshop 2020) with modern embeddings.

- [Background](#background)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Output](#output)
- [Data](#data)
- [Key question](#key-question)

## Background

The original paper used VGG-19 Gram matrices (style/texture) and found no correlation between visual style and artist demographics (gender, nationality, art school) or social proximity (Instagram follower graph). This project re-runs the analysis with CLIP C-vectors (semantic content) and Stable Diffusion A-vectors (visual appearance) to test whether the null result holds for content, not just style.

See [results/comparison.md](results/comparison.md) for a detailed side-by-side comparison of methods, results, and what changed.

## Setup

```bash
uv sync                        # Python dependencies
Rscript -e 'renv::restore()'   # R dependencies (for ggplot2 visualizations)
```

## Pipeline

Steps 0-2 and 4 are Python. Step 3 is R/ggplot2. Run in order.

```bash
# 0. Initialize fresh report
uv run python -m src.step0_init_report

# 1. Extract embeddings (~30 min MPS, ~2 hrs CPU)
uv run python -m src.step1_embed
uv run python -m src.step1_embed --vectors c    # C-vectors only
uv run python -m src.step1_embed --vectors a    # A-vectors only

# 2. Statistical tests: Mantel, PERMANOVA, spread (~5 min)
uv run python -m src.step2_statistics

# 3. Visualizations: UMAP plots (R/ggplot2)
Rscript R/visualize.R

# 4. Social network: node2vec + correlation with embeddings (~5 min)
uv run python -m src.step4_graph
```

Step 1 supports checkpointing: if interrupted, it resumes from the last checkpoint on re-run. Checkpoints are saved every 100 batches to `embeddings/*_checkpoint.npz`.

## Output

```
results/
  report.md                    <- main output, all findings
  comparison.md                <- side-by-side with original paper
  c_vectors_test_results.csv   <- Mantel/PERMANOVA p-values
  c_vectors_umap_data.csv      <- UMAP coordinates + metadata (for R)
  *_artist_emb.npy             <- mean embedding per artist
  *_artist_pca.npy             <- PCA-reduced
  *_spread_by_school.csv

plots/
  *_umap_school.png            <- UMAP colored by art school
  *_umap_gender.png
  *_umap_continent.png
  *_umap_professor.png
  *_spread_school.png          <- lollipop chart of spread per school

R/
  visualize.R                  <- ggplot2 script (step 3)

graphs/
  artist_graph.graphml         <- artist social graph
  node2vec_embeddings.npy      <- node2vec embeddings
```

## Data

See [data/README.md](data/README.md). 442 artists, 15 German art schools, 14,559 artworks. Data is not included in this repo (see .gitignore). Place the contempArt dataset at `data/` following the structure in data/README.md.

## Key question

Does CLIP (content) reveal demographic correlations that VGG (style) missed? Or did the original paper miss significant effects by not running formal hypothesis tests?
