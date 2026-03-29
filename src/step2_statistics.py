"""
Step 2: Aggregate by artist, compute spreads, run Mantel + PERMANOVA tests.

Appends statistical results and spread plots to report.

Usage:
    uv run python src/step2_statistics.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .report import Report
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.distance import mantel as skbio_mantel
from skbio.stats.distance import permanova as skbio_permanova
from sklearn.decomposition import PCA


EMBEDDING_DIR = Path("embeddings")
METADATA_PATH = Path("data/metadata/artists.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def aggregate_by_artist(emb, artists):
    unique = sorted(set(artists))
    out = np.zeros((len(unique), emb.shape[1]))
    for i, a in enumerate(unique):
        out[i] = emb[artists == a].mean(axis=0)
    return out, np.array(unique)


def categorical_distance(labels):
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = 0.0 if labels[i] == labels[j] else 1.0
    return dist


def run_test(cos_dist, labels, name, report, test_type="both"):
    """Run Mantel and/or PERMANOVA, append results to report rows."""
    cat_dist = categorical_distance(labels)
    rows = []

    if test_type in ("both", "mantel"):
        dm_x = DistanceMatrix(cos_dist)
        dm_y = DistanceMatrix(cat_dist)
        r, p, _ = skbio_mantel(dm_x, dm_y, method="pearson", permutations=9999)
        sig = "yes" if p < 0.05 else "no"
        rows.append(["Mantel", name, f"r={r:.4f}", f"{p:.4f}", sig])

    if test_type in ("both", "permanova"):
        ids = [str(i) for i in range(len(labels))]
        dm = DistanceMatrix(cos_dist, ids=ids)
        grouping = pd.Series(labels, index=ids, name="group")
        result = skbio_permanova(dm, grouping, permutations=9999)
        f_stat = float(result["test statistic"])
        p = float(result["p-value"])
        sig = "yes" if p < 0.05 else "no"
        rows.append(["PERMANOVA", name, f"F={f_stat:.4f}", f"{p:.4f}", sig])

    return rows


def plot_spread_bars(spread_df, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    spread_df = spread_df.sort_values("spread")
    bars = ax.barh(spread_df["group"], spread_df["spread"], color="#56B4E9", alpha=0.8)
    for bar, n in zip(bars, spread_df["n"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=9)
    ax.set_xlabel("Mean cosine distance (higher = more diverse)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)

    report = Report()

    for vec_name in ["c_vectors", "a_vectors"]:
        vec_path = EMBEDDING_DIR / f"{vec_name}.npy"
        if not vec_path.exists():
            continue

        # Use per-vector manifest if available (accounts for failed images)
        vec_manifest = EMBEDDING_DIR / f"{vec_name}_manifest.csv"
        if vec_manifest.exists():
            manifest = pd.read_csv(vec_manifest)
        else:
            manifest = pd.read_csv(EMBEDDING_DIR / "image_manifest.csv")

        emb = np.load(vec_path)
        artists_arr = manifest["artist"].values

        if len(artists_arr) != emb.shape[0]:
            print(f"Warning: manifest ({len(artists_arr)}) != embeddings ({emb.shape[0]}), truncating")
            n = min(len(artists_arr), emb.shape[0])
            artists_arr = artists_arr[:n]
            emb = emb[:n]

        label = "C-vectors (content)" if vec_name == "c_vectors" else "A-vectors (appearance)"

        report.header(f"## Step 2: Statistical Tests, {label}")

        # Aggregate
        artist_emb, artist_names = aggregate_by_artist(emb, artists_arr)

        # PCA
        n_comp = min(50, len(artist_names) - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        artist_pca = pca.fit_transform(artist_emb)
        var10 = pca.explained_variance_ratio_[:10].sum()
        var_all = pca.explained_variance_ratio_.sum()
        report.line(f"- Artists: {len(artist_names)}")
        report.line(f"- PCA: {var10:.1%} variance in 10 PCs, {var_all:.1%} in {n_comp} PCs")

        # Overall spread
        spread = float(np.mean(pdist(artist_emb, metric="cosine")))
        report.line(f"- Overall spread: {spread:.4f}")

        # Save
        np.save(RESULTS_DIR / f"{vec_name}_artist_emb.npy", artist_emb)
        np.save(RESULTS_DIR / f"{vec_name}_artist_pca.npy", artist_pca)

        # Merge metadata
        artist_df = pd.DataFrame({"artist_id": artist_names})
        merged = artist_df.merge(metadata, on="artist_id", how="left")

        # Cosine distance matrix
        cos_dist = squareform(pdist(artist_emb, metric="cosine"))

        # Run tests
        test_rows = []

        # School (all artists have this)
        schools = merged["school"].fillna("Unknown").values
        test_rows.extend(run_test(cos_dist, schools, "school", report))

        # Gender
        g_mask = merged["gender"].notna().values
        if g_mask.sum() >= 10:
            g_dist = cos_dist[np.ix_(g_mask, g_mask)]
            test_rows.extend(run_test(g_dist, merged.loc[g_mask, "gender"].values, "gender", report))

        # Nationality
        n_mask = merged["country_iso3"].notna().values
        if n_mask.sum() >= 10:
            n_dist = cos_dist[np.ix_(n_mask, n_mask)]
            test_rows.extend(run_test(n_dist, merged.loc[n_mask, "country_iso3"].values, "nationality", report))

        # Professor class
        c_mask = merged["professor_class"].notna().values
        if c_mask.sum() >= 10:
            c_dist = cos_dist[np.ix_(c_mask, c_mask)]
            test_rows.extend(run_test(c_dist, merged.loc[c_mask, "professor_class"].values, "professor_class", report))

        report.blank()
        report.table(
            ["Test", "Variable", "Statistic", "p-value", "Significant (p<0.05)?"],
            test_rows,
        )

        # Save test results as CSV too
        test_df = pd.DataFrame(test_rows, columns=["test", "variable", "statistic", "p_value", "significant"])
        test_df.to_csv(RESULTS_DIR / f"{vec_name}_test_results.csv", index=False)

        # Spread by school
        spread_records = []
        for school in sorted(set(schools)):
            mask = schools == school
            if mask.sum() >= 2:
                s = float(np.mean(pdist(artist_emb[mask], metric="cosine")))
                spread_records.append({"group": school, "spread": s, "n": int(mask.sum())})
        spread_df = pd.DataFrame(spread_records)
        spread_df.to_csv(RESULTS_DIR / f"{vec_name}_spread_by_school.csv", index=False)
        plot_spread_bars(spread_df, f"Spread by school ({label})",
                         PLOTS_DIR / f"{vec_name}_spread_school.png")
        report.image(f"../plots/{vec_name}_spread_school.png", f"Spread by school ({label})")

    report.save()
    print("Step 2 complete.")


if __name__ == "__main__":
    main()
