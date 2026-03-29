# Original 2020 Results (DO NOT MODIFY)

Pre-computed outputs from the original contempArt analysis (Huckle, Garcia & Nakashima, ECCV Workshop 2020). These files are frozen artifacts used for direct comparison with the 2026 re-analysis. Do not regenerate, edit, or overwrite them.

## Files

| File | Shape | Description |
|------|-------|-------------|
| gu_n2v_cosine.npy | 364 x 364 | Pairwise cosine distance from node2vec on G^U (artist-to-artist network) |
| gy_n2v_cosine.npy | 364 x 364 | Pairwise cosine distance from node2vec on G^Y (full network, 247k nodes, artist subset) |
| vgg_style_cosine.npy | 364 x 364 | Pairwise cosine distance from VGG FC7 centroids (50th percentile aggregation, symmetric) |
| finalData.csv | 442 rows | Original artist metadata (semicolon-separated). Provides the 442-artist ordering for all_* matrices. |
| smallGraph.csv | 364 rows | Node2vec embeddings for G^U with Instagram handles. Provides the 364-artist ordering for all three .npy files above. |

## Artist ordering

The 364 x 364 matrices are indexed by the artist order in smallGraph.csv. To map rows to artist IDs:

```python
allDat = pd.read_csv("finalData.csv", sep=";")
allDat.rename(columns={"instagramHandle.y": "instagramHandleY"}, inplace=True)
insta2ID = allDat.dropna(subset=["instagramHandleY"]).set_index("instagramHandleY")["ID"].to_dict()
n2vDat = pd.read_csv("smallGraph.csv")
n2vDat["artist_id"] = n2vDat["instagramHandleCheck2"].map(insta2ID)
artist_order = n2vDat["artist_id"].tolist()  # length 364
```

## Source

Copied from `/Users/nikolaihuckle/Documents/projects/artAnalysis/visart2020/` on 2026-03-29.

Original filenames:
- distances/small_n2v_cos.npy -> gu_n2v_cosine.npy
- distances/big_n2v_cos.npy -> gy_n2v_cosine.npy
- distances/insta_cos_50_True.npy -> vgg_style_cosine.npy
- nodeResults/finalData.csv -> finalData.csv
- nodeResults/smallGraphs/smallGraph.csv -> smallGraph.csv
