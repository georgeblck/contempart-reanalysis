"""Step 1: validate embedding files and build unified image manifest.

Replaces the old local-inference pipeline (`archive/src/embed_clip.py`).
All embeddings come precomputed (shipped as
`embeddings/<model>/<model>_contempart.npz` via the project's Zenodo
record). This step only:

1. Checks every registered head exists on disk with the expected array key.
2. Confirms filename orderings are identical across heads.
3. Writes `embeddings/image_manifest.csv` (path, artist, filename) from the
   shared filename list, in the same order as the embedding rows.

Usage:
    uv run python -m src.step1_link
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .registry import EMBEDDINGS_DIR, HEADS

IMAGE_ROOT = Path("data/images/visart2020")
MANIFEST_PATH = EMBEDDINGS_DIR / "image_manifest.csv"


def parse_filename(fname: str) -> tuple[str, str]:
    """Split `{artist}__{filename}` into (artist, filename)."""
    artist, _, file = fname.partition("__")
    if not file:
        raise ValueError(f"filename '{fname}' does not match artist__file pattern")
    return artist, file


def main() -> None:
    reference_filenames: np.ndarray | None = None
    reference_head: str | None = None
    rows: list[dict[str, object]] = []

    for head in HEADS:
        if not head.path.exists():
            raise FileNotFoundError(f"missing embedding file: {head.path}")
        data = np.load(head.path, allow_pickle=True)
        if head.key not in data.files:
            raise KeyError(f"{head.path}: key '{head.key}' not in {data.files}")
        emb = data[head.key]
        fnames = data["filenames"]
        if emb.shape[1] != head.dim:
            raise ValueError(
                f"{head.name}: expected dim {head.dim}, got {emb.shape[1]}"
            )
        if reference_filenames is None:
            reference_filenames = fnames
            reference_head = head.name
        elif not np.array_equal(fnames, reference_filenames):
            raise ValueError(
                f"{head.name} filenames diverge from reference head {reference_head}"
            )
        rows.append(
            {
                "name": head.name,
                "display": head.display,
                "role": head.role,
                "n": emb.shape[0],
                "dim": emb.shape[1],
                "dtype": str(emb.dtype),
            }
        )

    assert reference_filenames is not None
    artists, files = zip(*(parse_filename(f) for f in reference_filenames))
    manifest = pd.DataFrame(
        {
            "path": [str(IMAGE_ROOT / a / f) for a, f in zip(artists, files)],
            "artist": artists,
            "filename": files,
        }
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    Path("results").mkdir(exist_ok=True)
    heads_df = pd.DataFrame(
        [
            {
                "name": h.name,
                "display": h.display,
                "file": h.file,
                "key": h.key,
                "dim": h.dim,
                "role": h.role,
                "backbone": h.backbone,
                "normalized": h.normalized,
            }
            for h in HEADS
        ]
    )
    heads_df.to_csv("results/heads.csv", index=False)

    print(f"Step 1 complete. {len(HEADS)} heads registered.")
    print(f"  Images:  {len(manifest):,}")
    print(f"  Artists: {manifest['artist'].nunique()}")
    print(f"  Manifest: {MANIFEST_PATH}")
    for r in rows:
        print(f"  {r['name']:<16} {r['display']:<18} {r['role']:<10} n={r['n']:<5} dim={r['dim']:<6} {r['dtype']}")


if __name__ == "__main__":
    main()
