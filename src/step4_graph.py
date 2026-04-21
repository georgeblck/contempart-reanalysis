"""Step 4: every head vs the 2020 social-network distance matrices.

For each head in the registry, Mantel (+ Spearman) against:
  G^U   artist-to-artist follows (364 artists)
  G^Y   full network, artist subset (364 artists)
  VGG   original 2020 VGG-style cosine distances (for direct comparison)

Outputs a combined long table: `results/all_social.csv`.

Usage:
    uv run python -m src.step4_graph
    uv run python -m src.step4_graph --heads clip_l kim_a
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from skbio import DistanceMatrix
from skbio.stats.distance import mantel as skbio_mantel

from .registry import HEADS, HEADS_BY_NAME, Head

ORIGINAL_DIR = Path("data/original_2020")
EMBEDDING_DIR = Path("embeddings")
RESULTS_DIR = Path("results")

PERMUTATIONS = 9999


def load_original_artist_order() -> list[str]:
    """Reconstruct the 364-artist ordering used in the 2020 distance matrices."""
    all_dat = pd.read_csv(ORIGINAL_DIR / "finalData.csv", sep=";")
    all_dat = all_dat.rename(
        columns={
            "instagramHandle.x": "instagramHandleX",
            "instagramHandle.y": "instagramHandleY",
        }
    )
    insta = all_dat.dropna(subset=["instagramHandleY"])
    insta2id = insta.set_index("instagramHandleY")["ID"].to_dict()
    n2v = pd.read_csv(ORIGINAL_DIR / "smallGraph.csv", sep=",")
    n2v["ID"] = n2v["instagramHandleCheck2"].map(insta2id)
    return n2v["ID"].tolist()


def mantel_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    r, p, _ = skbio_mantel(
        DistanceMatrix(a), DistanceMatrix(b),
        method="pearson", permutations=PERMUTATIONS,
    )
    return float(r), float(p)


def spearman_upper(a: np.ndarray, b: np.ndarray) -> float:
    ia = np.triu_indices_from(a, k=1)
    rho, _ = spearmanr(a[ia], b[ia])
    return float(rho)


def process_head(head: Head, idx_ours: list[int], graphs: dict[str, np.ndarray]) -> list[dict[str, object]]:
    emb_path = RESULTS_DIR / f"{head.name}_artist_emb.npy"
    if not emb_path.exists():
        print(f"  skip {head.name} (no artist_emb.npy, run step2 first)")
        return []

    artist_emb = np.load(emb_path)
    sub = artist_emb[idx_ours]
    emb_dist = squareform(pdist(sub, metric="cosine"))

    rows: list[dict[str, object]] = []
    for graph_name, graph_dist in graphs.items():
        r, p = mantel_pair(emb_dist, graph_dist)
        rho = spearman_upper(emb_dist, graph_dist)
        rows.append({
            "head": head.name,
            "display": head.display,
            "graph": graph_name,
            "mantel_r": r,
            "mantel_p": p,
            "spearman_rho": rho,
        })
        sig = "*" if p < 0.05 else " "
        print(
            f"  {head.name:<16} vs {graph_name:<4}  "
            f"r={r:+.4f} p={p:.4f}{sig}  rho={rho:+.4f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="*", help="Subset of head names; default = all")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    orig_artists = load_original_artist_order()
    manifest = pd.read_csv(EMBEDDING_DIR / "image_manifest.csv")
    our_artists = np.array(sorted(manifest["artist"].unique()))
    shared = [a for a in orig_artists if a in set(our_artists)]
    idx_orig = [orig_artists.index(a) for a in shared]
    idx_ours = [list(our_artists).index(a) for a in shared]
    print(f"Shared artists: {len(shared)} / 364 original")

    gu = np.load(ORIGINAL_DIR / "gu_n2v_cosine.npy")[np.ix_(idx_orig, idx_orig)]
    gy = np.load(ORIGINAL_DIR / "gy_n2v_cosine.npy")[np.ix_(idx_orig, idx_orig)]
    vgg = np.load(ORIGINAL_DIR / "vgg_style_cosine.npy")[np.ix_(idx_orig, idx_orig)]
    graphs = {"GU": gu, "GY": gy, "VGG": vgg}

    heads = (
        [HEADS_BY_NAME[n] for n in args.heads] if args.heads else HEADS
    )

    all_rows: list[dict[str, object]] = []
    for head in heads:
        print(f"\n[{head.name}] {head.display}")
        all_rows.extend(process_head(head, idx_ours, graphs))

    # Also run the 2020 VGG baseline vs graphs (for reference in README).
    for graph_name, graph_dist in {"GU": gu, "GY": gy}.items():
        r, p = mantel_pair(vgg, graph_dist)
        rho = spearman_upper(vgg, graph_dist)
        all_rows.append({
            "head": "_vgg_2020",
            "display": "VGG-2020",
            "graph": graph_name,
            "mantel_r": r,
            "mantel_p": p,
            "spearman_rho": rho,
        })

    combined = pd.DataFrame(all_rows)
    combined.to_csv(RESULTS_DIR / "all_social.csv", index=False)

    print(f"\nStep 4 complete. {len(combined)} rows -> {RESULTS_DIR/'all_social.csv'}")


if __name__ == "__main__":
    main()
