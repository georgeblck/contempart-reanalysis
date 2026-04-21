"""Step 2: aggregate-by-artist, Mantel + PERMANOVA for every registered head.

For each head in `src.registry`:
  1. Load embeddings, aggregate by artist (mean).
  2. Save `results/<head>_artist_emb.npy` (for R/dbrda.R) and
     `results/<head>_metadata.csv` (for R/dbrda.R).
  3. Compute cosine distance matrix.
  4. Run Mantel + PERMANOVA against school, gender, nationality, professor.
  5. Save per-head results CSV.

Finally write `results/all_mantel_permanova.csv` with one row per
(head, variable, test).

Usage:
    uv run python -m src.step2_statistics
    uv run python -m src.step2_statistics --heads clip_l kim_a    # subset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.distance import mantel as skbio_mantel
from skbio.stats.distance import permanova as skbio_permanova

from .registry import HEADS, HEADS_BY_NAME, Head

EMBEDDING_DIR = Path("embeddings")
METADATA_PATH = Path("data/metadata/artists.csv")
RESULTS_DIR = Path("results")

PERMUTATIONS = 9999
VARIABLES = ["school", "gender", "country_iso3", "professor_class"]
VARIABLE_DISPLAY = {
    "school": "school",
    "gender": "gender",
    "country_iso3": "nationality",
    "professor_class": "professor",
}


def aggregate_by_artist(emb: np.ndarray, artists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean embedding per artist, sorted by artist id."""
    unique = np.array(sorted(set(artists)))
    out = np.zeros((len(unique), emb.shape[1]), dtype=np.float32)
    for i, a in enumerate(unique):
        out[i] = emb[artists == a].mean(axis=0)
    return out, unique


def categorical_distance(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=object)
    eq = arr[:, None] == arr[None, :]
    return (~eq).astype(float)


def run_head_tests(
    cos_dist: np.ndarray,
    merged: pd.DataFrame,
    head_name: str,
) -> list[dict[str, object]]:
    """Run Mantel + PERMANOVA for each demographic variable."""
    rows: list[dict[str, object]] = []
    ids = [str(i) for i in range(cos_dist.shape[0])]

    for var in VARIABLES:
        mask = merged[var].notna().values
        if mask.sum() < 10:
            continue
        sub_dist = cos_dist[np.ix_(mask, mask)]
        labels = merged.loc[mask, var].values
        cat_dist = categorical_distance(labels)

        dm_x = DistanceMatrix(sub_dist)
        dm_y = DistanceMatrix(cat_dist)
        r, p_m, _ = skbio_mantel(dm_x, dm_y, method="pearson", permutations=PERMUTATIONS)

        sub_ids = [str(i) for i in range(mask.sum())]
        dm = DistanceMatrix(sub_dist, ids=sub_ids)
        grouping = pd.Series(labels, index=sub_ids, name="group")
        perm = skbio_permanova(dm, grouping, permutations=PERMUTATIONS)
        f_stat = float(perm["test statistic"])
        p_p = float(perm["p-value"])

        rows.append({
            "head": head_name,
            "variable": VARIABLE_DISPLAY[var],
            "n": int(mask.sum()),
            "mantel_r": float(r),
            "mantel_p": float(p_m),
            "permanova_f": f_stat,
            "permanova_p": p_p,
        })
    return rows


def process_head(head: Head, manifest: pd.DataFrame, metadata: pd.DataFrame) -> list[dict[str, object]]:
    print(f"\n[{head.name}] {head.display} ({head.dim}d, {head.role})")

    data = np.load(head.path, allow_pickle=True)
    emb = data[head.key].astype(np.float32)
    fnames = data["filenames"]
    if len(fnames) != len(manifest):
        raise ValueError(f"{head.name}: manifest mismatch ({len(fnames)} vs {len(manifest)})")

    artists_arr = manifest["artist"].to_numpy()
    artist_emb, artist_names = aggregate_by_artist(emb, artists_arr)
    del emb  # kim_a is ~900 MB, free it before the next head loads

    np.save(RESULTS_DIR / f"{head.name}_artist_emb.npy", artist_emb)

    artist_df = pd.DataFrame({"artist_id": artist_names})
    merged = artist_df.merge(metadata, on="artist_id", how="left")
    merged.to_csv(RESULTS_DIR / f"{head.name}_metadata.csv", index=False)

    cos_dist = squareform(pdist(artist_emb, metric="cosine"))
    rows = run_head_tests(cos_dist, merged, head.name)

    per_head = pd.DataFrame(rows)
    per_head.to_csv(RESULTS_DIR / f"{head.name}_test_results.csv", index=False)

    for r in rows:
        sig_m = "*" if r["mantel_p"] < 0.05 else " "
        sig_p = "*" if r["permanova_p"] < 0.05 else " "
        print(
            f"  {r['variable']:<12} n={r['n']:>3}  "
            f"Mantel r={r['mantel_r']:+.4f} p={r['mantel_p']:.4f}{sig_m}  "
            f"PERMANOVA F={r['permanova_f']:6.2f} p={r['permanova_p']:.4f}{sig_p}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="*", help="Subset of head names; default = all")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    manifest = pd.read_csv(EMBEDDING_DIR / "image_manifest.csv")

    heads = (
        [HEADS_BY_NAME[n] for n in args.heads] if args.heads else HEADS
    )

    all_rows: list[dict[str, object]] = []
    for head in heads:
        all_rows.extend(process_head(head, manifest, metadata))

    combined = pd.DataFrame(all_rows)
    combined.to_csv(RESULTS_DIR / "all_mantel_permanova.csv", index=False)

    print(f"\nStep 2 complete. {len(combined)} rows -> {RESULTS_DIR/'all_mantel_permanova.csv'}")


if __name__ == "__main__":
    main()
