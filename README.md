# contempart-reanalysis

A re-analysis of the [contempArt study](https://arxiv.org/abs/2008.09558)
(Huckle, Garcia & Nakashima, ECCV Workshop 2020) across 16 modern image
embeddings (content, style, appearance, general, duplicate).

See also:
[contempart-eccv2020](https://github.com/georgeblck/contempart-eccv2020)
(2020 reproduction) and
[contempart](https://github.com/georgeblck/contempart) (dataset).

## TL;DR

The original paper found no link between artistic style and artist
demographics. Using 16 embedding heads across 9 backbones (CLIP,
OpenCLIP, DINOv2, CSD, SCFlow, GOYA, SD 2.1 VAE, SSCD) on the same
441-artist corpus, the same pattern holds for pure-appearance features
(SD VAE / KIM-A), while every content-oriented head picks up a strong
professor-class signal (~10% of variance explained). Most of the school
effect in the old paper is absorbed by professor class.

## Contents

- [Embeddings](#embeddings)
- [db-RDA: which demographics predict embedding distances?](#db-rda-which-demographics-predict-embedding-distances)
- [Social network: do embedding distances track follower-graph distances?](#social-network-do-embedding-distances-track-follower-graph-distances)
- [Reproduce](#reproduce)
- [Data provenance](#data-provenance)

## Embeddings

16 embedding heads extracted from 9 backbones. Files live in
`embeddings/<backbone>/<backbone>_contempart.npz`. 14,390 images,
441 artists.

| Head | Role | Backbone | Dim | Paper | Code |
|---|---|---|---:|---|---|
| **CLIP-L** | content | CLIP ViT-L/14 (OpenAI) | 768 | [Radford 2021](https://arxiv.org/abs/2103.00020) | [openai/CLIP](https://github.com/openai/CLIP) |
| **CLIP-B32** | content | CLIP ViT-B/32 (OpenAI) | 512 | [Radford 2021](https://arxiv.org/abs/2103.00020) | [openai/CLIP](https://github.com/openai/CLIP) |
| **OpenCLIP-L** | content | OpenCLIP ViT-L/14 (LAION-2B) | 768 | [Cherti 2023](https://arxiv.org/abs/2212.07143) | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| **CSD-Content** | content | CSD ViT-L/14 | 768 | [Somepalli 2024](https://arxiv.org/abs/2404.01292) | [learn2phoenix/CSD](https://github.com/learn2phoenix/CSD) |
| **SCFlow-Content** | content | SCFlow (ICCV 2025) | 768 | [Ma 2025](https://arxiv.org/abs/2503.11478) | [compvis/scflow](https://github.com/CompVis/scflow) |
| **GOYA-Content** | content | GOYA MLP (on CLIP-B32) | 2048 | [Gou 2023](https://arxiv.org/abs/2305.13770) | [yankungou/GOYA](https://github.com/yankungou/GOYA) |
| **KIM-C** | content | OpenCLIP ViT-H/14 (LAION-2B) | 1024 | [Kim 2025](https://arxiv.org/abs/2503.13531) | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| **DINOv2-Style** | style | DINOv2 ViT-L/14 | 1024 | [Oquab 2024](https://arxiv.org/abs/2304.07193) + [fruit-SALAD (Schaldenbrand 2024)](https://arxiv.org/abs/2406.01278) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **DINOv2-Gram** | style | DINOv2 ViT-L/14 | 1024 | [Oquab 2024](https://arxiv.org/abs/2304.07193) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **CSD-Style** | style | CSD ViT-L/14 | 768 | [Somepalli 2024](https://arxiv.org/abs/2404.01292) | [learn2phoenix/CSD](https://github.com/learn2phoenix/CSD) |
| **SCFlow-Style** | style | SCFlow (ICCV 2025) | 768 | [Ma 2025](https://arxiv.org/abs/2503.11478) | [compvis/scflow](https://github.com/CompVis/scflow) |
| **GOYA-Style** | style | GOYA MLP (on CLIP-B32) | 2048 | [Gou 2023](https://arxiv.org/abs/2305.13770) | [yankungou/GOYA](https://github.com/yankungou/GOYA) |
| **DINOv2-CLS** | general | DINOv2 ViT-L/14 | 1024 | [Oquab 2024](https://arxiv.org/abs/2304.07193) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **DINOv2-L12** | general | DINOv2 ViT-L/14 | 1024 | [Oquab 2024](https://arxiv.org/abs/2304.07193) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **KIM-A** | appearance | SD 2.1 VAE | 16384 | [Rombach 2022](https://arxiv.org/abs/2112.10752) | [huggingface.co/sd2-community](https://huggingface.co/sd2-community/stable-diffusion-2-1) |
| **SSCD** | duplicate | SSCD ResNet50 | 512 | [Pizzi 2022](https://arxiv.org/abs/2202.10261) | [facebookresearch/sscd](https://github.com/facebookresearch/sscd-copy-detection) |

Roles: *content* (what is depicted), *style* (how it looks at texture
level), *appearance* (colour / composition), *general* (generic visual
features), *duplicate* (near-duplicate detection).

## db-RDA: which demographics predict embedding distances?

Distance-based Redundancy Analysis ([Legendre & Anderson
1999](https://www.jstor.org/stable/2641010)) on cosine distances over
artist-aggregated embeddings. **Effect size = partial variance
explained**: the fraction of total inertia in embedding distances that
each variable accounts for *after controlling for the other three*.
Columns are sorted by mean effect size across heads; rows are grouped by
role. Checkmarks mark p < 0.05 (999 permutations, marginal test). The
column-max (per demographic) is bolded to highlight which head picks up
the strongest signal, regardless of significance.

| Head | Role | professor | continent | gender | school |
|---|---|---:|---:|---:|---:|
| **CLIP-L** | content | 9.8% ✓ | 2.4% ✓ | 1.5% ✓ | 0.6% ✓ |
| **CLIP-B32** | content | 10.3% ✓ | 1.2% ✓ | 1.8% ✓ | 0.8% ✓ |
| **OpenCLIP-L** | content | 10.5% ✓ | 1.7% ✓ | 1.5% ✓ | 0.6% ✓ |
| **CSD-Content** | content | 10.3% ✓ | 2.3% ✓ | 1.1% ✓ | 0.4%   |
| **SCFlow-Content** | content | 8.3% ✓ | 1.6% ✓ | 1.5% ✓ | 0.6%   |
| **GOYA-Content** | content | 6.5%   | 0.4%   | 0.6%   | 0.3%   |
| **KIM-C** | content | 10.5% ✓ | 1.9% ✓ | 1.3% ✓ | 0.4%   |
| **DINOv2-Style** | style | 9.5% ✓ | 2.1% ✓ | **2.1% ✓** | **1.0% ✓** |
| **DINOv2-Gram** | style | 9.5% ✓ | **2.8% ✓** | 1.8% ✓ | 0.6%   |
| **CSD-Style** | style | 10.2% ✓ | 2.5% ✓ | 1.2% ✓ | 0.3%   |
| **SCFlow-Style** | style | 9.5% ✓ | 1.6% ✓ | 1.3% ✓ | 0.3%   |
| **GOYA-Style** | style | **12.0% ✓** | 1.3% ✓ | 0.9% ✓ | 0.8% ✓ |
| **DINOv2-CLS** | general | 10.5% ✓ | 0.6%   | 1.4% ✓ | 0.6% ✓ |
| **DINOv2-L12** | general | 9.0% ✓ | 2.1% ✓ | 1.9% ✓ | **1.0% ✓** |
| **KIM-A** | appearance | 7.1%   | 1.6% ✓ | 1.2% ✓ | 0.3%   |
| **SSCD** | duplicate | 7.5% ✓ | 0.9% ✓ | 0.9% ✓ | 0.2%   |

**Take-aways.** Professor class is the dominant signal across nearly
every head (~8-12% of variance). Pure-appearance **KIM-A** (SD VAE) and
specialised **GOYA-Content** lose the professor effect, matching the
2020 null result for visual style. Continent and gender are consistently
smaller but significant; school barely registers once professor is
controlled for (school-unique variance is < 1% everywhere; the bulk is
shared with professor).

## Social network: do embedding distances track follower-graph distances?

For each head, Mantel correlation between the artist-aggregated cosine
distance matrix and a pre-computed 2020 node2vec distance matrix from
the Instagram follower graph (364-artist subset). Positive *r* means
artists who are closer in embedding space are also closer in the
follower graph. Checkmarks mark p < 0.05 (9,999 permutations). The
column-max is bolded to highlight the top head, regardless of
significance.

- **G^U**: artist-to-artist sub-graph (only edges between artists)
- **G^Y**: full network (includes galleries, friends, non-artist accounts)

| Head | Role | G^U (artist-to-artist) | G^Y (full network) |
|---|---|---|---|
| **CLIP-L** | content | **r = +0.105 ✓** | r = +0.020 |
| **CLIP-B32** | content | r = +0.089 ✓ | r = -0.003 |
| **OpenCLIP-L** | content | r = +0.096 ✓ | r = -0.006 |
| **CSD-Content** | content | r = +0.080 ✓ | r = +0.025 |
| **SCFlow-Content** | content | **r = +0.105 ✓** | r = +0.003 |
| **GOYA-Content** | content | r = -0.066 | r = -0.022 |
| **KIM-C** | content | r = +0.089 ✓ | r = -0.008 |
| **DINOv2-Style** | style | r = +0.031 | r = -0.030 |
| **DINOv2-Gram** | style | r = -0.005 | r = +0.001 |
| **CSD-Style** | style | r = +0.042 | **r = +0.049** |
| **SCFlow-Style** | style | **r = +0.105 ✓** | r = -0.024 |
| **GOYA-Style** | style | r = +0.020 | r = -0.052 |
| **DINOv2-CLS** | general | r = +0.061 | r = -0.028 |
| **DINOv2-L12** | general | r = +0.037 | r = -0.019 |
| **KIM-A** | appearance | r = +0.011 | r = +0.039 |
| **SSCD** | duplicate | r = +0.037 | r = +0.003 |
| **VGG-2020** *(2020 paper baseline)* | — | r = +0.042 | r = -0.036 |

**Take-aways.** Content-oriented CLIP variants and SCFlow correlate
significantly with G^U at r = 0.08-0.11, higher than the VGG-2020
baseline (r = 0.04, n.s. in the 2020 paper). Pure style heads
(DINOv2-Style/Gram, CSD-Style, GOYA-Style) show no social signal.
G^Y is noise for every head, matching the original paper: adding
non-artist accounts drowns the signal.

The supplementary Mantel-plus-PERMANOVA table (same variables, no
partial controls) and cross-embedding VGG-2020 correlations live at
[`results/all_mantel_permanova.csv`](results/all_mantel_permanova.csv)
and [`results/all_social.csv`](results/all_social.csv).

## Reproduce

```bash
# 1. Environment
uv sync                        # Python deps
Rscript -e 'renv::restore()'   # R deps (vegan, parallel, etc.)

# 2. Data (from Zenodo: images + metadata + 16 embedding npz files)
uv run python scripts/setup_data.py /path/to/zenodo/download

# 3. Pipeline (or just: uv run python scripts/run_all.py)
uv run python -m src.step1_link           # validate 16 npz files, build manifest
uv run python -m src.step2_statistics     # Mantel + PERMANOVA, all heads
Rscript R/dbrda.R                          # db-RDA + varpart, all heads (parallel)
uv run python -m src.step4_graph          # social network, all heads
uv run python scripts/make_readme_tables.py
```

Full pipeline runs in ~15 min on an M1 with 8 cores (db-RDA dominates
the cost). Permutation counts: 9,999 for Mantel / PERMANOVA / social
Mantel; 999 for db-RDA (multi-core).

## Data provenance

| Source | License | Refresh | Location |
|---|---|---|---|
| contempArt images + metadata | CC-BY 4.0 | static | Zenodo DOI [10.5281/zenodo.19365430](https://doi.org/10.5281/zenodo.19365430) |
| 16 embedding npz files | CC-BY-NC 4.0 | static | Zenodo DOI [10.5281/zenodo.19685514](https://doi.org/10.5281/zenodo.19685514) (concept: [19685513](https://doi.org/10.5281/zenodo.19685513)) |
| 2020 node2vec + VGG distances | as original paper | static | included in `data/original_2020/` |
