"""
Build demographic and social distance matrices for Mantel/PERMANOVA tests.

Takes artist metadata (school, gender, nationality) and produces
pairwise distance matrices (0/1 for categorical: 0 = same group, 1 = different).

Usage:
    uv run python src/build_distances.py \
        --metadata data/metadata/artists.csv \
        --embedding-dir results \
        --output-dir distances
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def categorical_distance(labels: np.ndarray) -> np.ndarray:
    """Binary distance matrix: 0 if same category, 1 if different."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = 0.0 if labels[i] == labels[j] else 1.0
    return dist


def build_all_distances(
    artists: np.ndarray,
    metadata: pd.DataFrame,
    embedding_dir: Path,
) -> dict:
    """Build distance matrices for all available variables.

    Returns dict of {name: (distance_matrix, artist_subset)} where
    artist_subset is the mask of artists that have data for that variable.
    """
    # Merge artists with metadata
    artist_df = pd.DataFrame({"artist": artists})

    # Try common column names for the join
    meta = metadata.copy()
    join_col = None
    for candidate in ["artist", "artist_id", "ID", "labels", "labelsCat", "name"]:
        if candidate in meta.columns:
            join_col = candidate
            break

    if join_col is None:
        print("Could not find artist join column in metadata.")
        print(f"Available columns: {list(meta.columns)}")
        return {}

    if join_col != "artist":
        meta = meta.rename(columns={join_col: "artist"})

    merged = artist_df.merge(meta, on="artist", how="left")

    results = {}

    # School / Hochschule
    for col in ["school", "university", "hochschule", "hs"]:
        if col in merged.columns:
            mask = merged[col].notna()
            if mask.sum() >= 10:
                labels = merged.loc[mask, col].values
                results["school"] = {
                    "dist": categorical_distance(labels),
                    "mask": mask.values,
                    "n_groups": len(set(labels)),
                    "labels": labels,
                }
                print(f"School: {mask.sum()} artists, {results['school']['n_groups']} groups")
            break

    # Gender
    for col in ["gender", "sex"]:
        if col in merged.columns:
            mask = merged[col].notna()
            if mask.sum() >= 10:
                labels = merged.loc[mask, col].values
                results["gender"] = {
                    "dist": categorical_distance(labels),
                    "mask": mask.values,
                    "n_groups": len(set(labels)),
                    "labels": labels,
                }
                print(f"Gender: {mask.sum()} artists, {results['gender']['n_groups']} groups")
            break

    # Nationality / region
    for col in ["country_iso3", "nationality", "region", "country", "nat"]:
        if col in merged.columns:
            mask = merged[col].notna()
            if mask.sum() >= 10:
                labels = merged.loc[mask, col].values
                results["nationality"] = {
                    "dist": categorical_distance(labels),
                    "mask": mask.values,
                    "n_groups": len(set(labels)),
                    "labels": labels,
                }
                print(f"Nationality: {mask.sum()} artists, {results['nationality']['n_groups']} groups")
            break

    # Embedding distances (cosine) for each vector type
    for vec_name in ["c_vectors", "a_vectors"]:
        artist_emb_path = embedding_dir / f"{vec_name}_artist_emb.npy" if embedding_dir else None
        if artist_emb_path and artist_emb_path.exists():
            emb = np.load(artist_emb_path)
            dist = squareform(pdist(emb, metric="cosine"))
            results[f"{vec_name}_cosine"] = {
                "dist": dist,
                "mask": np.ones(len(artists), dtype=bool),
                "labels": None,
            }
            print(f"{vec_name} cosine distances: {dist.shape}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Build distance matrices")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, default=Path("results"))
    parser.add_argument("--manifest", type=Path, default=Path("embeddings/image_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("distances"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    artists = np.array(sorted(manifest["artist"].unique()))
    metadata = pd.read_csv(args.metadata)

    print(f"Artists: {len(artists)}")
    print(f"Metadata columns: {list(metadata.columns)}")

    distances = build_all_distances(artists, metadata, args.embedding_dir)

    for name, data in distances.items():
        np.save(args.output_dir / f"dist_{name}.npy", data["dist"])
        np.save(args.output_dir / f"mask_{name}.npy", data["mask"])
        if data["labels"] is not None:
            pd.Series(data["labels"]).to_csv(
                args.output_dir / f"labels_{name}.csv", index=False
            )

    print(f"\nSaved {len(distances)} distance matrices to {args.output_dir}/")


if __name__ == "__main__":
    main()
