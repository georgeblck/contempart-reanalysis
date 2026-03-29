# contempart-clip

Re-analysis of the [contempArt](https://arxiv.org/abs/2008.09558) dataset (Huckle, Garcia & Nakashima, ECCV Workshop 2020) using CLIP and Stable Diffusion embeddings.

## Why

The original paper used VGG-19 Gram matrices to measure artistic style and found no correlation between style and artist demographics (school, gender, nationality) or social proximity (Instagram follower graph). That analysis captured texture and brushwork, but not what the paintings actually depict.

This project asks: does the null result hold when we measure content (what is painted) and appearance (how it looks) separately?

## Embeddings

Two modern embeddings replace VGG (following Kim et al. 2025):

- C-vectors (CLIP ViT-L/14, 768-dim): semantic content, what the painting depicts
- A-vectors (SD 2.0 VAE, 16,384-dim): visual appearance, colors, composition, texture

442 artists, 14,393 artworks, 15 German art schools.

## Results

All tests use the Mantel test (permutation-based correlation between distance matrices, 9,999 permutations), the same family of test used in the original paper but with formal p-values.

### Content (C-vectors) reveals institutional effects that style did not

| Variable | C-vectors (content) | A-vectors (appearance) | VGG (style, 2020) |
|----------|--------------------|-----------------------|-------------------|
| School | r=0.030, p=0.0001 | r=0.000, p=0.99 | not formally tested |
| Professor class | r=0.028, p=0.0001 | r=0.003, p=0.43 | not tested |
| Gender | r=0.010, p=0.18 | r=0.020, p=0.007 | not formally tested |
| Nationality | r=-0.093, p=0.08 | r=-0.010, p=0.77 | not formally tested |

School and professor class predict content similarity (C-vectors) but not appearance (A-vectors). Gender is the reverse: a small appearance effect but no content effect.

### Social network correlation matches content, not style

| Embedding | vs G^U (artist follows artist) | vs G^Y (full network) |
|-----------|-------------------------------|----------------------|
| C-vectors | r=0.111, p=0.009 | r=0.002, p=0.96 |
| A-vectors | r=0.013, p=0.66 | r=0.038, p=0.13 |
| VGG (2020) | rho=0.007 | rho=-0.032 |

Tested on the original paper's pre-computed node2vec distance matrices (same social network data, same 364 artists). Only C-vectors show a significant correlation with who follows whom. A-vectors and VGG style do not.

### What this means

The original paper's conclusion that "artistic style [is] entirely independent of any non-visual data" holds for appearance and style features (A-vectors and VGG confirm the null). But content is different. Artists at the same school or under the same professor produce more semantically similar work. Artists who follow each other on Instagram paint more similar subjects. These effects are small (r = 0.03 to 0.11) but significant, and they were invisible to texture-based features.

The gender result is interesting in the opposite direction: men and women produce art that looks slightly different (A-vectors, r=0.020) but not art that depicts different things (C-vectors, r=0.010).

For the full analysis, methodology comparison, and paper quotes, see [results/comparison.md](results/comparison.md) and [results/report.md](results/report.md).

## Setup

```bash
uv sync                        # Python dependencies
Rscript -e 'renv::restore()'   # R dependencies (for ggplot2 visualizations)
```

## Pipeline

```bash
uv run python -m src.step0_init_report          # 0. Initialize report
uv run python -m src.step1_embed                # 1. Extract embeddings (~2 hrs)
uv run python -m src.step2_statistics           # 2. Mantel + PERMANOVA tests
Rscript R/visualize.R                           # 3. UMAP plots (ggplot2)
uv run python -m src.step4_graph                # 4. Social network correlation
```

Step 1 supports checkpointing (resumes from last saved batch if interrupted). Use `--vectors c` or `--vectors a` to extract one type only.

## Data

442 artists, 15 German art schools, 14,559 artworks. Data is not included in this repo. See [data/README.md](data/README.md) for the expected structure.

Social network distance matrices from the original 2020 analysis are included in [data/original_2020/](data/original_2020/) for direct comparison.
