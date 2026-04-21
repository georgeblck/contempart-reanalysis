# contempArt CLIP Analysis Report

Generated: 2026-04-21 19:55

Each section below was produced by a separate pipeline step.
Re-run any step to update its section (append mode).

## Step 1: Embedding validation

- Heads registered: 16
- Images: 14,390
- Artists: 441
- Manifest: `embeddings/image_manifest.csv`

| Head | Display | Role | N | Dim | Dtype |
|---|---|---|---|---|---|
| clip_l | CLIP-L | content | 14390 | 768 | float32 |
| clip_b32 | CLIP-B32 | content | 14390 | 512 | float32 |
| openclip_l | OpenCLIP-L | content | 14390 | 768 | float32 |
| dinov2_cls | DINOv2-CLS | general | 14390 | 1024 | float32 |
| dinov2_l12 | DINOv2-L12 | general | 14390 | 1024 | float32 |
| dinov2_style | DINOv2-Style | style | 14390 | 1024 | float32 |
| dinov2_gram | DINOv2-Gram | style | 14390 | 1024 | float32 |
| csd_content | CSD-Content | content | 14390 | 768 | float32 |
| csd_style | CSD-Style | style | 14390 | 768 | float32 |
| scflow_content | SCFlow-Content | content | 14390 | 768 | float32 |
| scflow_style | SCFlow-Style | style | 14390 | 768 | float32 |
| goya_content | GOYA-Content | content | 14390 | 2048 | float32 |
| goya_style | GOYA-Style | style | 14390 | 2048 | float32 |
| kim_c | KIM-C | content | 14390 | 1024 | float32 |
| kim_a | KIM-A | appearance | 14390 | 16384 | float32 |
| sscd | SSCD | duplicate | 14390 | 512 | float32 |


## Step 2: Mantel + PERMANOVA (all heads)

- Heads: 1
- Artists: 441
- Permutations: 9,999

Per-head CSVs in `results/<head>_test_results.csv`.
Combined long table: `results/all_mantel_permanova.csv`.

## Step 1: Embedding validation

- Heads registered: 16
- Images: 14,390
- Artists: 441
- Manifest: `embeddings/image_manifest.csv`

| Head | Display | Role | N | Dim | Dtype |
|---|---|---|---|---|---|
| clip_l | CLIP-L | content | 14390 | 768 | float32 |
| clip_b32 | CLIP-B32 | content | 14390 | 512 | float32 |
| openclip_l | OpenCLIP-L | content | 14390 | 768 | float32 |
| dinov2_cls | DINOv2-CLS | general | 14390 | 1024 | float32 |
| dinov2_l12 | DINOv2-L12 | general | 14390 | 1024 | float32 |
| dinov2_style | DINOv2-Style | style | 14390 | 1024 | float32 |
| dinov2_gram | DINOv2-Gram | style | 14390 | 1024 | float32 |
| csd_content | CSD-Content | content | 14390 | 768 | float32 |
| csd_style | CSD-Style | style | 14390 | 768 | float32 |
| scflow_content | SCFlow-Content | content | 14390 | 768 | float32 |
| scflow_style | SCFlow-Style | style | 14390 | 768 | float32 |
| goya_content | GOYA-Content | content | 14390 | 2048 | float32 |
| goya_style | GOYA-Style | style | 14390 | 2048 | float32 |
| kim_c | KIM-C | content | 14390 | 1024 | float32 |
| kim_a | KIM-A | appearance | 14390 | 16384 | float32 |
| sscd | SSCD | duplicate | 14390 | 512 | float32 |


## Step 2: Mantel + PERMANOVA (all heads)

- Heads: 16
- Artists: 441
- Permutations: 9,999

Per-head CSVs in `results/<head>_test_results.csv`.
Combined long table: `results/all_mantel_permanova.csv`.
