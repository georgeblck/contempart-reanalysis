"""
Step 3: UMAP plots colored by demographics.

Appends visualization figures to report.

Usage:
    uv run python -m src.step3_visualize
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from .report import Report


RESULTS_DIR = Path("results")
EMBEDDING_DIR = Path("embeddings")
METADATA_PATH = Path("data/metadata/artists.csv")
PLOTS_DIR = Path("plots")

TOL_MUTED = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD",
    "#661100", "#6699CC", "#AA4466", "#BBCC33", "#AAAA00",
]
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
]

MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*", "p", "h"]


def plot_scatter(emb_2d, labels, title, palette, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=palette[i % len(palette)],
            marker=MARKERS[i % len(MARKERS)],
            label=lab, s=20, alpha=0.7, edgecolors="none",
        )
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=8, markerscale=1.5, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    report = Report()

    for vec_name in ["c_vectors", "a_vectors"]:
        pca_path = RESULTS_DIR / f"{vec_name}_artist_pca.npy"
        if not pca_path.exists():
            continue

        artist_pca = np.load(pca_path)
        label = "C-vectors (content)" if vec_name == "c_vectors" else "A-vectors (appearance)"

        report.header(f"## Step 3: Visualizations, {label}")

        # Get artist names from manifest (prefer per-vector manifest)
        vec_manifest = EMBEDDING_DIR / f"{vec_name}_manifest.csv"
        if vec_manifest.exists():
            manifest = pd.read_csv(vec_manifest)
        else:
            manifest = pd.read_csv(EMBEDDING_DIR / "image_manifest.csv")
        artist_names = np.array(sorted(manifest["artist"].unique()))

        # Guard against mismatch between PCA and manifest
        if len(artist_names) != artist_pca.shape[0]:
            print(f"Warning: {len(artist_names)} artists in manifest, {artist_pca.shape[0]} in PCA, truncating")
            n = min(len(artist_names), artist_pca.shape[0])
            artist_names = artist_names[:n]
            artist_pca = artist_pca[:n]

        artist_df = pd.DataFrame({"artist_id": artist_names})
        merged = artist_df.merge(metadata, on="artist_id", how="left")

        # UMAP
        print(f"  Computing UMAP for {vec_name}...")
        reducer = umap.UMAP(n_neighbors=15, random_state=42)
        emb_umap = reducer.fit_transform(artist_pca)
        np.save(RESULTS_DIR / f"{vec_name}_umap.npy", emb_umap)

        # Plot by school
        schools = merged["school"].fillna("Unknown").values
        plot_scatter(emb_umap, schools, f"UMAP by school ({label})", TOL_MUTED,
                     PLOTS_DIR / f"{vec_name}_umap_school.png")
        report.image(f"../plots/{vec_name}_umap_school.png", f"UMAP by school")

        # Plot by gender
        genders = merged["gender"].fillna("Unknown").values
        plot_scatter(emb_umap, genders, f"UMAP by gender ({label})", OKABE_ITO,
                     PLOTS_DIR / f"{vec_name}_umap_gender.png")
        report.image(f"../plots/{vec_name}_umap_gender.png", f"UMAP by gender")

        # Plot by continent
        continents = merged["continent"].fillna("Unknown").values
        plot_scatter(emb_umap, continents, f"UMAP by continent ({label})", OKABE_ITO,
                     PLOTS_DIR / f"{vec_name}_umap_continent.png")
        report.image(f"../plots/{vec_name}_umap_continent.png", f"UMAP by continent")

        # Plot by professor class (top 10 for readability)
        classes = merged["professor_class"].fillna("Unknown").values
        top_classes = pd.Series(classes).value_counts().head(10).index.tolist()
        class_labels = np.array([c if c in top_classes else "Other" for c in classes])
        plot_scatter(emb_umap, class_labels, f"UMAP by professor (top 10) ({label})", TOL_MUTED,
                     PLOTS_DIR / f"{vec_name}_umap_professor.png")
        report.image(f"../plots/{vec_name}_umap_professor.png", f"UMAP by professor class (top 10)")

    report.save()
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
