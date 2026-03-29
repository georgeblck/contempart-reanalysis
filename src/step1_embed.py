"""
Step 1: Extract C-vectors (CLIP) and A-vectors (SD VAE) from images.

Appends embedding stats to report.

Usage:
    uv run python src/step1_embed.py
    uv run python src/step1_embed.py --vectors c       # C-vectors only (faster)
    uv run python src/step1_embed.py --vectors a       # A-vectors only
    uv run python src/step1_embed.py --vectors both    # both (default)
"""

import argparse
from pathlib import Path

from .embed_clip import embed_a_vectors, embed_c_vectors, load_image_paths
from .report import Report

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/images/visart2020"))
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vectors", type=str, default="both", choices=["c", "a", "both"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_image_paths(args.data_dir)
    df.to_csv(args.output_dir / "image_manifest.csv", index=False)

    report = Report()
    report.header("## Step 1: Embeddings")

    if args.vectors in ("c", "both"):
        c_emb, c_failed = embed_c_vectors(df, batch_size=args.batch_size)
        np.save(args.output_dir / "c_vectors.npy", c_emb)
        failed_paths = {p for p, _ in c_failed}
        df_clean = df[~df["path"].isin(failed_paths)].reset_index(drop=True)
        df_clean.to_csv(args.output_dir / "c_vectors_manifest.csv", index=False)
        report.line(f"- C-vectors (CLIP ViT-L/14): {c_emb.shape[0]} images, {c_emb.shape[1]} dims")
        if c_failed:
            report.line(f"- C-vector failures: {len(c_failed)}")

    if args.vectors in ("a", "both"):
        a_batch = min(args.batch_size, 8)
        a_emb, a_failed = embed_a_vectors(df, batch_size=a_batch)
        np.save(args.output_dir / "a_vectors.npy", a_emb)
        failed_paths = {p for p, _ in a_failed}
        df_clean = df[~df["path"].isin(failed_paths)].reset_index(drop=True)
        df_clean.to_csv(args.output_dir / "a_vectors_manifest.csv", index=False)
        report.line(f"- A-vectors (SD 2.0 VAE): {a_emb.shape[0]} images, {a_emb.shape[1]} dims")
        if a_failed:
            report.line(f"- A-vector failures: {len(a_failed)}")

    report.line(f"- Manifest: {len(df)} images, {df['artist'].nunique()} artists")
    report.save()


if __name__ == "__main__":
    main()
