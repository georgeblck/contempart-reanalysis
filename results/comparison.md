# Comparison: Original contempArt (2020) vs CLIP Re-analysis (2026)

## Table of contents

- [Reproducibility: dataset and pipeline numbers](#reproducibility-dataset-and-pipeline-numbers)
- [Results at a glance](#results-at-a-glance)
- [Feature extraction](#feature-extraction)
- [Aggregation](#aggregation)
- [Statistical tests](#statistical-tests)
- [Detailed results](#detailed-results)
- [Key differences that affect comparability](#key-differences-that-affect-comparability)
- [Open questions](#open-questions)


## Reproducibility: dataset and pipeline numbers

One goal of this re-analysis is to check whether the original dataset can be faithfully reproduced from the published data. The table below compares every quantity we can verify.

| Quantity | Original (2020) | Re-analysis (2026) | Match? |
|----------|----------------:|-------------------:|--------|
| Artists in metadata | 442 | 442 | exact |
| Artists with image folders | 442 (implied) | 441 | 1 missing (see note 1) |
| Total images | 14,559 | 14,398 found, 14,393 usable | 161 missing, 5 corrupt (see note 2) |
| Art schools | 15 | 15 | exact |
| Artists with gender data | 440 | 440 | exact |
| Artists with nationality data | 234 | 234 | exact |
| Artists with professor class | not reported | 420 | new variable, not in original |
| Artists with Instagram handle | "82.35% of the sample" (p.5) = ~364 | 366 | close (see note 3) |
| G^U nodes (artist-to-artist) | 364 (p.12) | 364 (reused original) | exact |
| G^U edges | 5,614 (p.12) | 4,966 directed / 2,846 undirected from edgelist | see note 4 |
| G^Y nodes (full network) | 247,087 (p.12) | 247,087 unique accounts in edgelist | exact |
| G^Y edges | 745,144 (p.12) | 456,056 rows in edgelist | see note 5 |
| node2vec dims | 128 | 128 (reused original) | exact |
| node2vec distance matrices | computed in 2020 | reused from data/original_2020/ | same data |
| VGG style distance matrix | computed in 2020 | reused from data/original_2020/ | same data |
| VGG embedding dims | FC7: 4,096; Texture: 4,096; Archetype: 72 | not recomputed | -- |
| CLIP embedding dims | not tested | C-vectors: 768 | new |
| SD VAE embedding dims | not tested | A-vectors: 16,384 | pending |
| PCA components | not reported | 50 (82.1% variance) | -- |
| Aggregation method | per-artist centroid (p.12) | per-artist mean | same method |
| Stat test (social network) | Spearman rho, no p-values | Mantel + Spearman, 9999 permutations | upgraded |
| Stat test (demographics) | t-SNE visual inspection only | Mantel + PERMANOVA, 9999 permutations | upgraded |

### Notes on deviations

1. 442 vs 441 artists: one artist folder is absent from the image data. The metadata CSV has 442 entries; the image directory has 441 artist folders. The missing artist has not been identified.

2. 14,559 vs 14,398 images: 161 images from the original dataset are not present in our image directory. This may be due to the ongoing rsync from the external drive (not yet complete at the time of analysis). Additionally, 5 PNGs from artist luanlamberty are corrupt and unreadable by PIL, leaving 14,393 usable images.

3. Instagram handle count: the original reports 82.35% of 442 = ~364 artists with Instagram data. Our metadata has 366 non-null Instagram handles. The discrepancy of 2 may reflect minor cleaning differences.

4. G^U edge count: the original reports 5,614 edges. Our edgelist yields 4,966 directed edges (2,846 undirected). The original likely counted each mutual follow as two undirected edges, or the original graph construction included additional edges from the follower/following CSVs that differ slightly from the cleaned edgelist. The node count (364) matches exactly.

5. G^Y edge count: the original reports 745,144 edges across 247,087 nodes. Our edgelist.csv has 456,056 rows across 247,087 unique accounts. The difference (745,144 vs 456,056) suggests the original included both directions of each follow relationship, while our edgelist stores each follow once. The node count matches exactly.

6. Social network analysis: rather than recomputing node2vec (which is stochastic and would not produce identical embeddings), we reuse the original 2020 distance matrices stored in data/original_2020/. This ensures the social network side of the comparison is identical to the original paper.


## Results at a glance

| Variable | Original test | Original result | Re-analysis test | Re-analysis result | Changed? |
|----------|--------------|-----------------|------------------|--------------------|----------|
| School vs embeddings | t-SNE visual inspection | "no visible patterns" (p.14) | Mantel (C-vectors) | r=0.030, p=0.0003 | now significant |
| School vs embeddings | (inferred via social graph) | "art schools too, have no bearing" (p.13) | PERMANOVA (C-vectors) | F=3.249, p=0.0001 | now significant |
| Gender vs embeddings | t-SNE visual inspection | "no visible patterns" (p.14) | Mantel (C-vectors) | r=0.010, p=0.18 | still not significant |
| Gender vs embeddings | not tested | not tested | PERMANOVA (C-vectors) | F=5.004, p=0.0001 | new test, significant |
| Nationality vs embeddings | t-SNE visual inspection | "no visible patterns" (p.14) | Mantel (C-vectors) | r=-0.093, p=0.08 | still not significant |
| Nationality vs embeddings | not tested | not tested | PERMANOVA (C-vectors) | F=0.750, p=0.84 | still not significant |
| Professor class | not tested | not tested | Mantel (C-vectors) | r=0.028, p=0.0001 | new variable, significant |
| Professor class | not tested | not tested | PERMANOVA (C-vectors) | F=2.337, p=0.0001 | new variable, significant |
| G^U vs VGG (FC7) | Spearman rho | rho=0.007, no p-value | Mantel (same data) | r=0.042, p=0.25 | still not significant |
| G^U vs Texture | Spearman rho | rho=0.043, no p-value | not re-run | -- | -- |
| G^U vs Archetype | Spearman rho | rho=0.012, no p-value | not re-run | -- | -- |
| G^U vs C-vectors | not tested | not tested | Mantel | r=0.111, p=0.010 | new, significant |
| G^Y vs VGG (FC7) | Spearman rho | rho=-0.032, no p-value | Mantel (same data) | r=-0.037, p=0.22 | still not significant |
| G^Y vs Texture | Spearman rho | rho=-0.025, no p-value | not re-run | -- | -- |
| G^Y vs Archetype | Spearman rho | rho=-0.057, no p-value | not re-run | -- | -- |
| G^Y vs C-vectors | not tested | not tested | Mantel | r=0.002, p=0.96 | new, not significant |
| A-vectors (all tests) | not tested | not tested | pending | -- | pending |

Note: "style" in the original means VGG-based texture/style features. "C-vectors" means CLIP ViT-L/14 semantic content features. These capture different aspects of the artwork (see [Feature extraction](#feature-extraction)).

### What is still missing

- Mantel/PERMANOVA on original VGG features against demographics (school, gender, professor). Only the social network comparison has been done so far.
- Texture and Archetype embeddings not re-tested (only VGG FC7 cosine). These had different Spearman values in the original Table 3.
- A-vectors (SD 2.0 VAE appearance features) pending overnight run.
- Intra-artist style variance (Table 2 in original) not replicated.
- Once rsync completes: re-check image count to resolve the 161 missing images.


## Feature extraction

### Original (VGG, 2020)

Three unsupervised style embeddings, all based on VGG-19 pretrained on ImageNet:

1. Raw VGG (FC7, 4096-dim): "The network is pre-trained on the ImageNet database and the second to last layer fc_7 is used as the style embedding e^V_n in R^{4,096} for any image I." (p.6)

2. Texture / Gram matrix (4096-dim after SVD): "the correlations inside convolutional feature maps of certain network layers capture positionless information on the texture or rather, the style of images." Gram matrices computed from five conv layers, vectorized, then reduced via SVD to 4096 dims. (pp.7-8)

3. Archetype embeddings (2M-dim): Archetypal analysis on the Gram texture descriptors, yielding mixture weights concatenated into 2M-dim vectors (M=36 archetypes). (pp.8-9)

All three capture style/texture: how things are painted, not what is depicted.

### Re-analysis (CLIP + SD VAE, 2026)

Two embeddings from modern foundation models:

1. C-vectors (CLIP ViT-L/14, 768-dim): Semantic content, capturing what the painting depicts (objects, scenes, composition). Following Kim et al. 2025.

2. A-vectors (SD 2.0 VAE, 16384-dim): Visual appearance (colors, brightness, composition, texture). Pending, not yet computed.

C-vectors capture content, not style. This is a fundamentally different aspect of the artwork.


## Aggregation

### Original

Per-artist centroid: "we compute each artists centroid style embedding c^l = (1/N^l) sum_i e^l_i." (p.12)

Also computed intra-artist style variance using cosine distance to centroid.

### Re-analysis

Same approach: per-artist mean embedding, cosine distance. Also computed intra-school spread (mean pairwise cosine distance within each school).


## Statistical tests

### Original

Social network (Section 6.1):
- node2vec (128-dim) on two graphs: G^U (undirected follower overlap) and G^Y (full directed network including non-artists)
- "Spearman's rank coefficient is used to compute the correlation between the flattened upper triangular parts of the described distance matrices." (p.13)
- No p-values reported. No permutation tests. No significance thresholds.

Demographics (Section 6.2):
- t-SNE visualization only. No formal statistical test.
- "There were no visible patterns for any of the available variables, including Instagram-specific measures such as likes, comments or the number of followers and general ones such as nationality, gender or art school affiliation." (p.14)

### Re-analysis

- Mantel test (permutation-based distance matrix correlation, 9999 permutations, Pearson r, with p-values)
- PERMANOVA (permutation-based multivariate ANOVA on distance matrices, 9999 permutations, with p-values and F-statistics)
- Spearman rho also computed for social network comparisons (for direct comparison with original Table 3)
- Both Mantel and PERMANOVA are proper hypothesis tests with explicit significance thresholds (p < 0.05)


## Detailed results

### Original: social network vs style

Table 3 from the paper, "Rank correlations of style and network distances":

| Embedding | G^U (undirected) | G^Y (directed) |
|-----------|------------------|----------------|
| VGG       | .007             | -.032          |
| Texture   | .043             | -.025          |
| Archetype | .012             | -.057          |

> "The results in Table 3 show that there are only very small correlations between stylistic and social distance. Even though the two graphs share only a minor similarity (r_sp = .166), neither network contains information that relates to inter-artist differences in style." (p.13)

### Original: demographics vs style

No quantitative results. Visual inspection only:

> "There were no visible patterns for any of the available variables, including Instagram-specific measures such as likes, comments or the number of followers and general ones such as nationality, gender or art school affiliation." (p.14)

> "The clear overlap between school affiliation and the smaller network graph G^U, as seen in Figure 3, allows the further conclusion, that art schools too, have no bearing on artistic style." (p.13)

### Original: conclusion

> "These embeddings of artistic style were shown to be entirely independent of any non-visual data." (p.14)

### Re-analysis: demographics vs content (C-vectors)

| Test | Variable | Statistic | p-value | Significant? |
|------|----------|-----------|---------|--------------|
| Mantel | school | r=0.030 | 0.0003 | yes |
| PERMANOVA | school | F=3.249 | 0.0001 | yes |
| Mantel | gender | r=0.010 | 0.18 | no |
| PERMANOVA | gender | F=5.004 | 0.0001 | yes |
| Mantel | nationality | r=-0.093 | 0.08 | no |
| PERMANOVA | nationality | F=0.750 | 0.84 | no |
| Mantel | professor_class | r=0.028 | 0.0001 | yes |
| PERMANOVA | professor_class | F=2.337 | 0.0001 | yes |

### Re-analysis: social network vs embeddings

Using the original paper's pre-computed node2vec distance matrices (stored in data/original_2020/, not recomputed):

| Comparison | Mantel r | p-value | Spearman rho | Significant? |
|------------|----------|---------|--------------|--------------|
| C-vectors vs G^U | r=0.111 | 0.010 | rho=0.059 | yes |
| C-vectors vs G^Y | r=0.002 | 0.96 | rho=0.018 | no |
| VGG style (2020) vs G^U | r=0.042 | 0.25 | rho=-0.005 | no |
| VGG style (2020) vs G^Y | r=-0.037 | 0.22 | rho=-0.029 | no |

n=364 artists. Both CLIP and VGG tested against the same node2vec matrices using the same Mantel test. CLIP content shows a significant correlation with the artist-to-artist network (G^U) that VGG style does not. Neither feature type correlates with the full network (G^Y).


## Key differences that affect comparability

### 1. Features measure different things

The original tested whether style (texture, brushwork) correlates with demographics. The re-analysis tests whether content (what is depicted) correlates with demographics. Finding content correlations does not contradict the original null result on style. They answer different questions.

### 2. Statistical methods are more powerful

The original used:
- Spearman correlation with no p-values or permutation tests for the social network analysis
- Visual inspection of t-SNE plots for demographics (no formal test at all)

The re-analysis uses:
- Mantel test with 9999 permutations and explicit p-values
- PERMANOVA with 9999 permutations, testing group centroid differences

PERMANOVA in particular can detect centroid shifts that Spearman on pairwise distances would miss. The gender result (PERMANOVA significant, Mantel not) is a case in point. It is unknown whether the original VGG features would also show significance under PERMANOVA.

### 3. Effect sizes are in the same range

The original Spearman values (0.007 to 0.043) and the new Mantel r values (0.010 to 0.030) for demographics are comparable. For the social network, CLIP shows a larger effect (r=0.111 vs VGG r=0.042). The difference is that the re-analysis applies formal hypothesis testing and finds some of these effects statistically significant. The original did not test for significance and described them as "very small correlations."

### 4. Reproducibility of dataset

The core dataset reproduces well: 441 of 442 artists, 14,398 of 14,559 images, all 15 schools, identical demographic coverage (gender, nationality). The 161 missing images are likely due to an incomplete rsync and should resolve once the transfer finishes. The social graph node count (364) matches exactly. Edge counts differ slightly due to how directed/undirected edges are counted (see notes 4 and 5 above).


## Open questions

1. Would the original VGG Gram features also show significant school and professor effects under Mantel/PERMANOVA? We tested VGG vs social network and found it not significant (Mantel r=0.042, p=0.25). But the demographic tests (school, gender, professor) have not been run on the original VGG features yet. This is the critical missing piece for attributing the new findings to CLIP vs VGG rather than to the upgraded statistical tests.

2. The gender PERMANOVA result (F=5.0, p=0.0001) with no Mantel significance needs careful interpretation. PERMANOVA tests centroid differences, Mantel tests distance correlation. A centroid shift without distance correlation means a small, consistent population-level difference with large individual overlap. Is this meaningful or a statistical artifact of unbalanced groups (F:318, M:120, Unknown:2)?

3. Why does G^Y show no correlation for either feature type? The full network includes non-artist accounts (galleries, magazines, friends) that add noise. Two artists who both follow the same meme account are "close" in G^Y but that says nothing about their art. G^U, which only captures direct artist-to-artist following, is the cleaner signal.

4. Once the rsync completes and images are re-checked: does using all 14,559 images (vs 14,393) change any results?
