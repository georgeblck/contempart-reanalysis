# contempArt embeddings

**16 image-embedding heads across 9 backbones** for the contempArt dataset
(441 contemporary German artists, 14,390 artworks). Precomputed
embeddings ready for downstream analysis — no GPU required to reuse.

Derived from [contempArt (Huckle 2020, Zenodo 19365430)](https://doi.org/10.5281/zenodo.19365430).
Analysis code: <https://github.com/georgeblck/contempart-reanalysis>.

## Files

Each `*.npz` is a numpy-compressed archive keyed by
`filenames` (shape `(14390,)`, strings of the form `<artist>__<file>`)
plus one or more embedding arrays. See the Heads table below for which
keys live in which file.

| File | Size | Heads inside |
|---|---:|---|
| `clip_openai/clip_openai_contempart.npz` | 49 MB | CLIP-L |
| `clip_openai_b32/clip_openai_b32_contempart.npz` | 63 MB | CLIP-B32 |
| `clip_openclip/clip_openclip_contempart.npz` | 49 MB | OpenCLIP-L |
| `dinov2/dinov2_contempart.npz` | 232 MB | DINOv2-Style, -Gram, -CLS, -L12 |
| `csd/csd_contempart.npz` | 91 MB | CSD-Style, -Content |
| `scflow/scflow_contempart.npz` | 91 MB | SCFlow-Style, -Content |
| `goya/goya_contempart.npz` | 209 MB | GOYA-Style, -Content |
| `sd2_kim/sd2_kim_contempart.npz` | 963 MB | KIM-C, KIM-A |
| `sscd/sscd_contempart.npz` | 35 MB | SSCD |
| `image_manifest.csv` | 2 MB | filename → artist mapping (14,390 rows) |
| `heads.csv` | 2 KB | full head registry (name, role, dim, npz key) |

Total: ~1.85 GB across 9 embedding files.

## Quickstart

```python
import numpy as np
import pandas as pd

manifest = pd.read_csv("image_manifest.csv")             # filename, artist
heads = pd.read_csv("heads.csv")                         # registry

d = np.load("clip_openai/clip_openai_contempart.npz", allow_pickle=True)
clip_l = d["embeddings"]            # (14390, 768) float32, L2-normalized
filenames = d["filenames"]          # aligns row-for-row with clip_l

# DINOv2 carries four heads in one file:
d = np.load("dinov2/dinov2_contempart.npz", allow_pickle=True)
d["embeddings"]          # (14390, 1024) style head (fruit-SALAD recipe)
d["gram_embeddings"]     # (14390, 1024) gram-matrix @ layer 12
d["cls_embeddings"]      # (14390, 1024) DINOv2 CLS token
d["layer12_embeddings"]  # (14390, 1024) mean-pooled patches @ layer 12
```

All embedding rows are aligned by position with the `filenames` array in
the same `.npz`, and `filenames` is identical across every file. Join
back to artist metadata (demographics) using `image_manifest.csv` and
the artist CSV in the [original contempArt Zenodo record](https://doi.org/10.5281/zenodo.19365430).

## Heads

| Head | Role | Backbone | Dim | npz key | Normalized |
|---|---|---|---:|---|---|
| CLIP-L | content | CLIP ViT-L/14 (OpenAI) | 768 | `embeddings` | L2 |
| CLIP-B32 | content | CLIP ViT-B/32 (OpenAI) | 512 | `embeddings` | L2 |
| OpenCLIP-L | content | OpenCLIP ViT-L/14 (LAION-2B) | 768 | `embeddings` | L2 |
| CSD-Content | content | CSD ViT-L/14 | 768 | `content_embeddings` | L2 |
| SCFlow-Content | content | SCFlow (ICCV 2025) | 768 | `content_embeddings` | no |
| GOYA-Content | content | GOYA MLP (on CLIP-B32) | 2048 | `content_embeddings` | no |
| KIM-C | content | OpenCLIP ViT-H/14 (LAION-2B) | 1024 | `embeddings` | L2 |
| DINOv2-Style | style | DINOv2 ViT-L/14, fruit-SALAD recipe | 1024 | `embeddings` | L2 |
| DINOv2-Gram | style | DINOv2 ViT-L/14 | 1024 | `gram_embeddings` | L2 |
| CSD-Style | style | CSD ViT-L/14 | 768 | `embeddings` | L2 |
| SCFlow-Style | style | SCFlow (ICCV 2025) | 768 | `embeddings` | no |
| GOYA-Style | style | GOYA MLP (on CLIP-B32) | 2048 | `style_embeddings` | no |
| DINOv2-CLS | general | DINOv2 ViT-L/14 | 1024 | `cls_embeddings` | L2 |
| DINOv2-L12 | general | DINOv2 ViT-L/14 | 1024 | `layer12_embeddings` | L2 |
| KIM-A | appearance | SD 2.1 VAE | 16384 | `a_vectors` | no |
| SSCD | duplicate | SSCD ResNet50 | 512 | `embeddings` | L2 |

All embeddings were computed in fp16 on a single NVIDIA A6000 pod and
saved as fp32 `.npz`.

## License

Released under **CC-BY-NC 4.0** (non-commercial). Chosen conservatively
because SSCD carries a non-commercial clause that propagates to derived
works. Other backbones are more permissive individually; the bundle
license is bound by the most restrictive component.

Per-model licenses for the weights used to compute each head:

| Backbone | License |
|---|---|
| CLIP ViT-L/14, ViT-B/32 (OpenAI) | MIT |
| OpenCLIP ViT-L/14, ViT-H/14 (LAION-2B) | MIT |
| DINOv2 ViT-L/14 | Apache-2.0 |
| CSD ViT-L/14 | MIT |
| SCFlow | check upstream (ICCV 2025) |
| GOYA MLP | MIT |
| SD 2.1 VAE | CreativeML Open RAIL++-M |
| SSCD ResNet50 | **CC-BY-NC 4.0** |

The underlying contempArt images are CC-BY 4.0.

## Citation

```bibtex
@dataset{huckle_2026_contempart_embeddings,
  author       = {Huckle, Nikolai},
  title        = {{contempArt embeddings: 16 image-embedding heads across 9 backbones}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.TBD},
  url          = {https://doi.org/10.5281/zenodo.TBD}
}
```

Please also cite the original contempArt dataset:

```bibtex
@inproceedings{huckle_2020_contempart,
  author       = {Huckle, Nikolai and Garcia, Noa and Nakashima, Yuta},
  title        = {{contempArt: A Dataset of Contemporary Artworks and Socio-demographic Data}},
  booktitle    = {ECCV Workshop on Computer Vision for Fashion, Art and Design},
  year         = 2020,
  eprint       = {2008.09558},
  archivePrefix= {arXiv}
}
```

and the backbones your analysis actually uses — see the Heads table in
the [GitHub README](https://github.com/georgeblck/contempart-reanalysis) for
per-model paper references.
