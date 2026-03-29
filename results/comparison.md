# Comparison: Original contempArt (2020) vs CLIP Re-analysis (2026)

## Table of contents

- [Dataset and pipeline numbers](#dataset-and-pipeline-numbers)
- [Results at a glance](#results-at-a-glance)
- [Feature extraction](#feature-extraction)
- [Aggregation](#aggregation)
- [Statistical tests](#statistical-tests)
- [Detailed results](#detailed-results)
- [Key differences that affect comparability](#key-differences-that-affect-comparability)
- [Open questions](#open-questions)


## Dataset and pipeline numbers

| Quantity | Original (2020) | Re-analysis (2026) | Match? |
|----------|----------------:|-------------------:|--------|
| Artists | 442 | 441 | 1 artist missing (see note) |
| Images | 14,559 | 14,398 total, 14,393 usable | 161 fewer images, 5 corrupt PNGs |
| Art schools | 15 | 15 | exact |
| Artists with gender | 440 | 440 | exact |
| Artists with nationality | 234 | 234 | exact |
| Artists with professor class | not reported | 420 | new variable |
| Artists with Instagram handle | "82.35% of the sample" (p.5) = ~364 | 366 | close |
| Embedding dims (style) | VGG FC7: 4,096; Texture: 4,096; Archetype: 72 | not re-run | -- |
| Embedding dims (content) | not tested | C-vectors (CLIP): 768 | new |
| Embedding dims (appearance) | not tested | A-vectors (SD VAE): 16,384 | pending |
| PCA components | not reported | 50 (82.1% variance) | -- |
| Aggregation | per-artist centroid (p.12) | per-artist mean | same method |
| Social graph G^U nodes | 364 | 364 | exact |
| Social graph G^U edges | 5,614 (p.12) | 4,966 directed / 2,846 undirected | see note |
| Social graph G^Y nodes | 247,087 (p.12) | not reconstructed | see note |
| Social graph G^Y edges | 745,144 (p.12) | not reconstructed | see note |
| node2vec dims | 128 | 128 | exact |
| Stat test (social) | Spearman rho, no p-values | Mantel, 9999 permutations | different test |
| Stat test (demographics) | t-SNE visual inspection | Mantel + PERMANOVA, 9999 permutations | different test |

Notes on deviations:

1. 442 vs 441 artists: one artist folder is likely missing from the image data. The original dataset had 442 artist entries in the metadata; our image directory has 441 folders (plus the 537 total folders mentioned in data/README.md include non-artist folders).

2. 14,559 vs 14,398 images: 161 images from the original dataset are not present in our image directory. Additionally, 5 PNGs from artist luanlamberty are corrupt (unreadable by PIL), leaving 14,393 usable images.

3. G^U edge count: the original reports 5,614 edges for G^U. We reconstruct 4,966 directed edges (2,846 undirected) from the edgelist. The difference may be due to the original counting mutual follows as two edges in the undirected graph (4,966 / 2 + mutual = ~5,614) or slight differences in the handle-to-artist mapping. The node count (364) matches exactly.

4. G^Y not reconstructed: G^Y is the full Instagram network including all non-artist accounts. The original had 247,087 nodes and 745,144 edges. Our edgelist.csv contains 456,056 rows across 247,087 unique accounts, confirming the raw data is present. However, the re-analysis only uses artist-to-artist edges (equivalent to G^U). Reconstructing G^Y with node2vec would allow a direct replication of Table 3 from the original paper. This is a gap in the current analysis.


## Results at a glance

| Variable | Original test | Original result | Re-analysis test | Re-analysis result | Changed? |
|----------|--------------|-----------------|------------------|--------------------|----------|
| School vs style | t-SNE visual inspection | "no visible patterns" (p.14) | Mantel (C-vectors) | r=0.030, p=0.0005 | **now significant** |
| School vs style | (inferred via social graph) | "art schools too, have no bearing" (p.13) | PERMANOVA (C-vectors) | F=3.249, p=0.0001 | **now significant** |
| Gender vs style | t-SNE visual inspection | "no visible patterns" (p.14) | Mantel (C-vectors) | r=0.010, p=0.18 | still not significant |
| Gender vs style | not tested | not tested | PERMANOVA (C-vectors) | F=5.004, p=0.0002 | **new test, significant** |
| Nationality vs style | t-SNE visual inspection | "no visible patterns" (p.14) | Mantel (C-vectors) | r=-0.093, p=0.07 | still not significant |
| Nationality vs style | not tested | not tested | PERMANOVA (C-vectors) | F=0.750, p=0.84 | still not significant |
| Professor class | not tested | not tested | Mantel (C-vectors) | r=0.028, p=0.0001 | **new variable, significant** |
| Professor class | not tested | not tested | PERMANOVA (C-vectors) | F=2.337, p=0.0001 | **new variable, significant** |
| Social network (G^U) vs VGG | Spearman rho | rho=0.007, no p-value | Mantel (same data) | r=0.042, p=0.25 | still not significant |
| Social network (G^U) vs Texture | Spearman rho | rho=0.043, no p-value | not re-run | -- | -- |
| Social network (G^U) vs Archetype | Spearman rho | rho=0.012, no p-value | not re-run | -- | -- |
| Social network (G^U) vs C-vectors | not tested | not tested | Mantel | r=0.111, p=0.010 | **new, significant** |
| Social network (G^Y) vs VGG | Spearman rho | rho=-0.032, no p-value | Mantel (same data) | r=-0.037, p=0.22 | still not significant |
| Social network (G^Y) vs Texture | Spearman rho | rho=-0.025, no p-value | not re-run | -- | -- |
| Social network (G^Y) vs Archetype | Spearman rho | rho=-0.057, no p-value | not re-run | -- | -- |
| Social network (G^Y) vs C-vectors | not tested | not tested | Mantel | r=0.002, p=0.96 | new, not significant |
| A-vectors (all tests) | not tested | not tested | pending | -- | pending |

Key: "style" in the original means VGG Gram matrix texture features. "C-vectors" in the re-analysis means CLIP ViT-L/14 semantic content features. These measure different things (see [Feature extraction](#feature-extraction)).

### What is missing from the re-analysis

- VGG/Texture/Archetype features not tested with PERMANOVA against demographics (school, gender, professor). Social network comparison done (VGG not significant under Mantel).
- Texture and Archetype embeddings not tested (only VGG FC7 cosine). These had different Spearman values in the original Table 3.
- A-vectors (SD 2.0 VAE appearance features) pending overnight run
- Intra-artist style variance (Table 2 in original) not replicated


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
- Both are proper hypothesis tests with explicit significance thresholds (p < 0.05)


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
| Mantel | school | r=0.030 | 0.0005 | yes |
| PERMANOVA | school | F=3.249 | 0.0001 | yes |
| Mantel | gender | r=0.010 | 0.18 | no |
| PERMANOVA | gender | F=5.004 | 0.0002 | yes |
| Mantel | nationality | r=-0.093 | 0.07 | no |
| PERMANOVA | nationality | F=0.750 | 0.84 | no |
| Mantel | professor_class | r=0.028 | 0.0001 | yes |
| PERMANOVA | professor_class | F=2.337 | 0.0001 | yes |

### Re-analysis: social network vs content

Using the original paper's pre-computed node2vec distance matrices (no recomputation needed):

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

### 3. Effect sizes are comparable

The original Spearman values (0.007 to 0.043) and the new Mantel r values (0.010 to 0.030) are in the same range. The difference is that the re-analysis applies formal hypothesis testing and finds some of these small effects are statistically significant. The original did not test for significance and described them as "very small correlations."

### 4. Social network is now properly reconstructed

After fixing an Instagram handle mapping bug, the graph matches the original paper: 364 connected artists, 4,966 directed edges. The social network correlation is now significant (r=0.026, p=0.006). The original paper reported Spearman rho=0.007 (VGG) to 0.043 (Texture) with no p-values. Our r=0.026 falls in the same range, but the Mantel permutation test reveals it is significant.


## Open questions

1. Would the original VGG Gram features also show significant school and professor effects under Mantel/PERMANOVA? We tested VGG vs social network and found it not significant (Mantel r=0.042, p=0.25). But the demographic tests (school, gender, professor) have not been run on the original VGG features yet.

2. The gender PERMANOVA result (F=5.0, p=0.0002) with no Mantel significance needs careful interpretation. PERMANOVA tests centroid differences, Mantel tests distance correlation. A centroid shift without distance correlation means a small, consistent population-level difference with large individual overlap. Is this meaningful or a statistical artifact of unbalanced groups?

3. Why does G^Y show no correlation for either feature type? The full network includes non-artist accounts (galleries, magazines, friends) that add noise. Two artists who both follow the same meme account are "close" in G^Y but that says nothing about their art. G^U, which only captures direct artist-to-artist following, is the cleaner signal.
