# archive/

Pre-2026-04 artifacts preserved for reproducibility.

## `old_embeddings/`

The original in-project CLIP ViT-L/14 + SD 2.0 VAE embeddings, computed
locally via `src/embed_clip.py` (see `src/`). Replaced by the full 16-head
battery shipped through the project's Zenodo record.

The A-vectors here (SD 2.0 VAE, fp32) use the same weights as the new KIM-A
head (SD 2.1 VAE, fp16; SHA256 `a1d99348...` verified identical across 2.0
and 2.1). Only precision differs.

- `c_vectors.npy`: (14393, 768) fp32
- `a_vectors.npy`: (14393, 16384) fp32
- `image_manifest.csv`, `c_vectors_manifest.csv`, `a_vectors_manifest.csv`

## `src/`

- `step1_embed.py`: driver for the old two-head pipeline
- `embed_clip.py`: CLIP + VAE inference code
