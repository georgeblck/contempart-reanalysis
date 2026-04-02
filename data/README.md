# Data

Run `uv run python scripts/setup_data.py /path/to/zenodo/download` to set up.

The setup script creates symlinks from the [Zenodo contempArt dataset](https://doi.org/10.5281/zenodo.19365430) into the expected layout:

```
data/
  images/visart2020/         <- symlinked from Zenodo (14,398 images, 441 artists)
  metadata/artists.csv       <- symlinked from Zenodo
  original_2020/             <- committed in repo (node2vec distances, VGG style distances)
```

The `original_2020/` directory contains pre-computed distance matrices from the 2020 paper, used for the social network comparison in step 4. These are small (364x364 arrays) and already tracked in git.
