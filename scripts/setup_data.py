"""Set up data directory from a Zenodo contempArt download.

Downloads are available at: https://doi.org/10.5281/zenodo.19365430

Usage:
    uv run python scripts/setup_data.py /path/to/zenodo/download

    # If images are in a separate location:
    uv run python scripts/setup_data.py /path/to/zenodo/download --images /path/to/images

The script creates symlinks where possible (no disk duplication).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images" / "visart2020"
METADATA_DIR = DATA_DIR / "metadata"


def find_artists_csv(zenodo_dir: Path) -> Path | None:
    """Find artists.csv in the Zenodo download."""
    for candidate in [
        zenodo_dir / "artists.csv",
        zenodo_dir / "data" / "artists.csv",
    ]:
        if candidate.exists():
            return candidate
    return None


def find_images_dir(zenodo_dir: Path) -> Path | None:
    """Find the image directory (folder containing artist subfolders)."""
    for candidate in [
        zenodo_dir / "images",
        zenodo_dir / "contempart_images",
        zenodo_dir / "visart2020",
        zenodo_dir / "data" / "images",
        zenodo_dir / "data" / "contempart_images",
    ]:
        if candidate.is_dir():
            # Verify it contains artist subfolders (not just a single nested dir)
            subdirs = [p for p in candidate.iterdir() if p.is_dir()]
            if len(subdirs) > 10:
                return candidate
    # Maybe the zenodo dir itself contains artist folders
    subdirs = [p for p in zenodo_dir.iterdir() if p.is_dir()]
    if len(subdirs) > 100:
        return zenodo_dir
    return None


def symlink_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink, falling back to copy if symlinks aren't supported."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        print(f"  Already exists: {dst}")
        return
    try:
        dst.symlink_to(src.resolve())
        print(f"  Linked: {dst} -> {src}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"  Copied: {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up contempart-clip data from a Zenodo download."
    )
    parser.add_argument("zenodo_dir", type=Path, help="Path to the Zenodo download")
    parser.add_argument(
        "--images", type=Path, default=None,
        help="Path to images if stored separately from the Zenodo metadata",
    )
    args = parser.parse_args()

    zenodo = args.zenodo_dir.resolve()
    if not zenodo.exists():
        print(f"Error: {zenodo} does not exist")
        sys.exit(1)

    # Find artists.csv
    artists_csv = find_artists_csv(zenodo)
    if artists_csv is None:
        print(f"Error: could not find artists.csv in {zenodo}")
        print("  Expected at: {zenodo}/artists.csv or {zenodo}/data/artists.csv")
        sys.exit(1)

    # Find images
    images_src = args.images or find_images_dir(zenodo)
    if images_src is None:
        print(f"Error: could not find image directory in {zenodo}")
        print("  Use --images /path/to/images to specify manually")
        sys.exit(1)

    # Set up metadata
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    symlink_or_copy(artists_csv, METADATA_DIR / "artists.csv")

    # Set up images
    IMAGES_DIR.parent.mkdir(parents=True, exist_ok=True)
    if IMAGES_DIR.exists() or IMAGES_DIR.is_symlink():
        print(f"  Already exists: {IMAGES_DIR}")
    else:
        try:
            IMAGES_DIR.symlink_to(images_src.resolve())
            print(f"  Linked: {IMAGES_DIR} -> {images_src}")
        except OSError:
            print(f"  Cannot symlink, copying images (this may take a while)...")
            shutil.copytree(images_src, IMAGES_DIR)
            print(f"  Copied: {images_src} -> {IMAGES_DIR}")

    # Verify
    print()
    n_artists = sum(1 for p in IMAGES_DIR.iterdir() if p.is_dir())
    n_images = sum(1 for p in IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    has_meta = (METADATA_DIR / "artists.csv").exists()
    has_original = (DATA_DIR / "original_2020").is_dir()

    print("Setup complete:")
    print(f"  Images:    {n_images} files across {n_artists} artist folders")
    print(f"  Metadata:  {'OK' if has_meta else 'MISSING'}")
    print(f"  2020 data: {'OK (committed in repo)' if has_original else 'MISSING'}")
    print()
    print("Ready to run the pipeline:")
    print("  uv run python -m src.step1_link       # validate embeddings, build manifest")
    print("  uv run python -m src.step2_statistics # Mantel + PERMANOVA (16 heads)")
    print("  Rscript R/dbrda.R                      # db-RDA (16 heads)")
    print("  uv run python -m src.step4_graph       # social-network analysis")


if __name__ == "__main__":
    main()
