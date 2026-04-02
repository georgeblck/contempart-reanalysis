# contempart-clip

A re-analysis of the [contempArt study](https://arxiv.org/abs/2008.09558) (Huckle, Garcia & Nakashima, ECCV Workshop 2020) using CLIP and Stable Diffusion embeddings.

See also: [contempart-eccv2020](https://github.com/georgeblck/contempart-eccv2020) (reproduction of the original analysis) and [contempart](https://github.com/georgeblck/contempart) (dataset).

## TL;DR

The original study found no link between artistic style and artist demographics. We re-test with modern embeddings that separate content (what is painted) from appearance (how it looks). Institutional factors (school, professor, social network) shape content but not appearance. The original null result holds for visual style, but content tells a different story.

## Why

The original paper used VGG-19 Gram matrices to measure artistic style and found no correlation between style and artist demographics (school, gender, nationality) or social proximity (Instagram follower graph). That analysis captured texture and brushwork, but not what the paintings actually depict.

This project asks: does the null result hold when we measure content (what is painted) and appearance (how it looks) separately?

## Embeddings

Two modern embeddings replace VGG (following Kim et al. 2025):

- C-vectors (CLIP ViT-L/14, 768-dim): semantic content, what the painting depicts
- A-vectors (SD 2.0 VAE, 16,384-dim): visual appearance, colors, composition, texture

442 artists, 14,393 artworks, 15 German art schools.

## Results

### db-RDA: which demographics predict embedding distances?

[db-RDA](https://en.wikipedia.org/wiki/Redundancy_analysis) (distance-based Redundancy Analysis) is multivariate regression on distance matrices. It tests each variable's unique contribution while controlling for the others, resolving confounding (e.g. professors are nested within schools).

C-vectors (content), 24.5% total variance explained (F=2.71, p=0.0001):

| Variable | Var. explained | F | p-value | |
|----------|---------------:|----:|--------:|---|
| Professor class | 10.5% | 1.76 | 0.0001 | ✅ |
| Continent | 3.5% | 5.49 | 0.0001 | ✅ |
| Gender | 1.4% | 3.17 | 0.013 | ✅ |
| School | 0.6% | 2.95 | 0.044 | ✅ (weak after controlling for professor) |

Variance partition (school vs professor): school unique 0.4%, professor unique 5.0%, shared 6.3%. The school effect is almost entirely explained by who teaches there.

A-vectors (appearance), 16.4% total variance explained (F=1.64, p=0.0007):

| Variable | Var. explained | F | p-value | |
|----------|---------------:|----:|--------:|---|
| Gender | 2.0% | 2.98 | 0.023 | ✅ |
| Continent | 2.5% | 2.48 | 0.026 | ✅ |
| Professor class | 11.6% | 1.23 | 0.12 | ❌ |
| School | 0.4% | 1.09 | 0.33 | ❌ |

Different pattern: gender and continent predict appearance, school and professor do not. The institutional effects that shape content have no impact on how the art looks.

### Social network: who follows whom predicts content, not style

| Embedding | vs G^U (artist follows artist) | | vs G^Y (full network) | |
|-----------|-------------------------------|---|----------------------|---|
| C-vectors (content) | r=0.111, p=0.009 | ✅ | r=0.002, p=0.96 | ❌ |
| A-vectors (appearance) | r=0.013, p=0.66 | ❌ | r=0.038, p=0.13 | ❌ |
| VGG style (2020) | rho=0.007 | ❌ | rho=-0.032 | ❌ |

Tested on the original paper's pre-computed node2vec distance matrices (same 364 artists). G^U = direct artist-to-artist follows. G^Y = full network including non-artist accounts (galleries, friends, etc.). Only C-vectors vs G^U show a significant correlation.

### What this means

The original paper's conclusion that "artistic style [is] entirely independent of any non-visual data" holds for appearance and style features (A-vectors and VGG confirm the null). But content is different. Artists under the same professor produce more semantically similar work, and this explains most of the school-level effect. Artists who follow each other on Instagram paint more similar subjects. Continent of origin also predicts content once other factors are controlled for.

The gender result is interesting in the opposite direction: men and women produce art that looks slightly different (A-vectors) but not art that depicts different things (C-vectors).

For the full analysis, methodology comparison, and paper quotes, see [results/comparison.md](results/comparison.md) and [results/report.md](results/report.md).

## Setup

Download the contempArt dataset from [Zenodo](https://doi.org/10.5281/zenodo.19365430), then run:

```bash
uv sync
uv run python scripts/setup_data.py /path/to/zenodo/download
```

This links the images and metadata into the expected layout. The original 2020 social network distance matrices are already included in `data/original_2020/`.

Then run the pipeline:

```bash
uv run python -m src.step1_embed       # extract CLIP + SD embeddings
uv run python -m src.step2_statistics   # db-RDA, Mantel, PERMANOVA
uv run python -m src.step3_visualize    # t-SNE plots
uv run python -m src.step4_graph        # social network comparison
```

## References

- Huckle, N., Garcia, N., & Nakashima, Y. (2020). contempArt: A dataset of contemporary artworks and socio-demographic data. *ECCV Workshop on Computer Vision for Fashion, Art and Design*. [arXiv:2008.09558](https://arxiv.org/abs/2008.09558)
- Kim, J., Lee, B., You, T., & Yun, J. (2025). Context-aware Multimodal AI Reveals Hidden Pathways in Five Centuries of Art Evolution. *arXiv preprint*. [arXiv:2503.13531](https://arxiv.org/abs/2503.13531)
