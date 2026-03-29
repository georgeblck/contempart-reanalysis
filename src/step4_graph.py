"""
Step 4: Compare CLIP embeddings to pre-computed social network distances.

Uses the original node2vec distance matrices from the 2020 paper:
  G^U: artist-to-artist network (364 artists, small_n2v_cos.npy)
  G^Y: full network including non-artists (364 artists, big_n2v_cos.npy)

Also tests the original VGG style distances for direct comparison.

Usage:
    uv run python -m src.step4_graph
"""

from pathlib import Path

import numpy as np
import pandas as pd
from .report import Report
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from skbio import DistanceMatrix
from skbio.stats.distance import mantel as skbio_mantel


ORIGINAL_DIR = Path("data/original_2020")
EMBEDDING_DIR = Path("embeddings")
RESULTS_DIR = Path("results")


def load_original_artist_order() -> list[str]:
    """Reconstruct the 364-artist ordering used in the original distance matrices."""
    allDat = pd.read_csv(ORIGINAL_DIR / "finalData.csv", sep=";")
    allDat.rename(
        columns={"instagramHandle.x": "instagramHandleX", "instagramHandle.y": "instagramHandleY"},
        inplace=True,
    )
    allDat_insta = allDat.dropna(subset=["instagramHandleY"])
    insta2ID = allDat_insta.set_index("instagramHandleY")["ID"].to_dict()

    n2vDat = pd.read_csv(ORIGINAL_DIR / "smallGraph.csv", sep=",")
    n2vDat["ID"] = n2vDat["instagramHandleCheck2"].map(insta2ID)
    return n2vDat["ID"].tolist()


def run_mantel(dist_a: np.ndarray, dist_b: np.ndarray) -> tuple[float, float]:
    """Run Mantel test, return (r, p)."""
    dm_a = DistanceMatrix(dist_a)
    dm_b = DistanceMatrix(dist_b)
    r, p, _ = skbio_mantel(dm_a, dm_b, method="pearson", permutations=9999)
    return r, p


def run_spearman(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Compute Spearman rho on flattened upper triangle (matching original paper)."""
    flat_a = dist_a[np.triu_indices_from(dist_a, k=1)]
    flat_b = dist_b[np.triu_indices_from(dist_b, k=1)]
    rho, _ = spearmanr(flat_a, flat_b)
    return rho


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load artist orderings
    orig_artists = load_original_artist_order()
    manifest = pd.read_csv(EMBEDDING_DIR / "image_manifest.csv")
    our_artists = np.array(sorted(manifest["artist"].unique()))

    shared = [a for a in orig_artists if a in set(our_artists)]
    idx_orig = [orig_artists.index(a) for a in shared]
    idx_ours = [list(our_artists).index(a) for a in shared]

    print(f"Original 364 artists, shared with our data: {len(shared)}")

    # Load pre-computed social network distance matrices
    gu_dist = np.load(ORIGINAL_DIR / "gu_n2v_cosine.npy")
    gy_dist = np.load(ORIGINAL_DIR / "gy_n2v_cosine.npy")
    gu_sub = gu_dist[np.ix_(idx_orig, idx_orig)]
    gy_sub = gy_dist[np.ix_(idx_orig, idx_orig)]

    # Load original VGG style distance matrix (cosine, 50th percentile, symmetric)
    vgg_dist = np.load(ORIGINAL_DIR / "vgg_style_cosine.npy")
    vgg_sub = vgg_dist[np.ix_(idx_orig, idx_orig)]

    report = Report()
    report.header("## Step 4: Social Network Analysis")

    report.line(f"- Using pre-computed node2vec from original paper (2020)")
    report.line(f"- G^U: artist-to-artist network, 364 nodes, cosine distance")
    report.line(f"- G^Y: full network (247k nodes), artist subset, cosine distance")
    report.line(f"- Artists in common with CLIP embeddings: {len(shared)}")

    report.blank()
    report.line("### Embedding vs social network (Mantel test, 9999 permutations)")
    report.blank()

    test_rows = []

    # Test each embedding type against both graphs
    for vec_name in ["c_vectors", "a_vectors"]:
        emb_path = RESULTS_DIR / f"{vec_name}_artist_emb.npy"
        if not emb_path.exists():
            continue

        artist_emb = np.load(emb_path)
        emb_sub = artist_emb[idx_ours]
        emb_dist = squareform(pdist(emb_sub, metric="cosine"))

        label = "C-vectors" if vec_name == "c_vectors" else "A-vectors"

        r_gu, p_gu = run_mantel(emb_dist, gu_sub)
        rho_gu = run_spearman(emb_dist, gu_sub)
        sig_gu = "yes" if p_gu < 0.05 else "no"
        test_rows.append([f"{label} vs G^U", f"r={r_gu:.4f}", f"{p_gu:.4f}", f"rho={rho_gu:.4f}", sig_gu])

        r_gy, p_gy = run_mantel(emb_dist, gy_sub)
        rho_gy = run_spearman(emb_dist, gy_sub)
        sig_gy = "yes" if p_gy < 0.05 else "no"
        test_rows.append([f"{label} vs G^Y", f"r={r_gy:.4f}", f"{p_gy:.4f}", f"rho={rho_gy:.4f}", sig_gy])

    # Also test original VGG style for direct comparison
    r_vgg_gu, p_vgg_gu = run_mantel(vgg_sub, gu_sub)
    rho_vgg_gu = run_spearman(vgg_sub, gu_sub)
    sig_vgg_gu = "yes" if p_vgg_gu < 0.05 else "no"
    test_rows.append([f"VGG style (2020) vs G^U", f"r={r_vgg_gu:.4f}", f"{p_vgg_gu:.4f}", f"rho={rho_vgg_gu:.4f}", sig_vgg_gu])

    r_vgg_gy, p_vgg_gy = run_mantel(vgg_sub, gy_sub)
    rho_vgg_gy = run_spearman(vgg_sub, gy_sub)
    sig_vgg_gy = "yes" if p_vgg_gy < 0.05 else "no"
    test_rows.append([f"VGG style (2020) vs G^Y", f"r={r_vgg_gy:.4f}", f"{p_vgg_gy:.4f}", f"rho={rho_vgg_gy:.4f}", sig_vgg_gy])

    report.table(
        ["Comparison", "Mantel r", "p-value", "Spearman rho", "Significant?"],
        test_rows,
    )

    report.line(f"n={len(shared)} artists. Mantel: Pearson r on distance matrices with permutation p-value.")
    report.line("Spearman rho: rank correlation on flattened upper triangle (same method as original paper Table 3).")
    report.line("VGG style = original cosine distance from VGG FC7 centroids (50th percentile, symmetric).")

    report.save()
    print("Step 4 complete.")


if __name__ == "__main__":
    main()
