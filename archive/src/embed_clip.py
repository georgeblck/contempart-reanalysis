"""
Dual embedding pipeline for contempArt dataset.

Extracts two vector types per image (following Kim et al. 2025):
  C-vector (CLIP ViT-L/14, 768-dim): semantic content, what the painting depicts
  A-vector (SD 2.0 autoencoder, 16384-dim): visual appearance, colors, composition

Original 2020 pipeline used VGG-19 FC7 (4096-dim) + Gram matrices.
Kim et al. showed C-vectors predict time (R²=0.87), A-vectors don't (R²=0.20).

Usage:
    uv run python src/embed_clip.py --data-dir data/images/visart2020
    uv run python src/embed_clip.py --data-dir data/images/visart2020 --vectors c      # C-vectors only
    uv run python src/embed_clip.py --data-dir data/images/visart2020 --vectors a      # A-vectors only
    uv run python src/embed_clip.py --data-dir data/images/visart2020 --vectors both   # both (default)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def load_image_paths(data_dir: Path) -> pd.DataFrame:
    """Load all images from artist subdirectories.

    Matches the original contempArt structure:
    visart2020/
        agneswrobel/
            image1.jpg
            ...
    """
    records = []
    for artist_dir in sorted(data_dir.iterdir()):
        if not artist_dir.is_dir() or artist_dir.name.startswith("."):
            continue
        for img_path in sorted(artist_dir.iterdir()):
            suffix = img_path.suffix.lower()
            if suffix in (".jpg", ".jpeg", ".png", ".webp"):
                records.append(
                    {
                        "path": str(img_path),
                        "artist": artist_dir.name,
                        "filename": img_path.name,
                    }
                )
    df = pd.DataFrame(records)
    print(f"Found {len(df)} images from {df['artist'].nunique()} artists")
    return df


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_c_vectors(
    df: pd.DataFrame,
    batch_size: int = 32,
    device: str = "auto",
    checkpoint_dir: Path = Path("embeddings"),
    checkpoint_every: int = 100,
) -> tuple[np.ndarray, list]:
    """Extract C-vectors via CLIP ViT-L/14 (768-dim).

    Captures semantic content: what the painting depicts.
    Saves checkpoints every checkpoint_every batches so progress survives interruptions.
    """
    import open_clip

    if device == "auto":
        device = get_device()
    print(f"C-vectors: CLIP ViT-L/14 on {device}")

    ckpt_path = checkpoint_dir / "c_vectors_checkpoint.npz"
    start_batch = 0
    all_embeddings = []
    failed = []

    if ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        all_embeddings = [ckpt["embeddings"]]
        start_batch = int(ckpt["next_batch"])
        print(f"  Resuming from batch {start_batch} ({all_embeddings[0].shape[0]} images done)")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device)
    model.eval()

    batch_starts = list(range(0, len(df), batch_size))
    for batch_idx, i in enumerate(
        tqdm(batch_starts[start_batch:], desc="C-vectors", initial=start_batch, total=len(batch_starts)),
    ):
        batch_paths = df["path"].iloc[i : i + batch_size].tolist()
        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                failed.append((path, str(e)))

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

        if (batch_idx + 1) % checkpoint_every == 0:
            merged = np.concatenate(all_embeddings, axis=0)
            np.savez(ckpt_path, embeddings=merged, next_batch=start_batch + batch_idx + 1)
            all_embeddings = [merged]
            print(f"  Checkpoint: {merged.shape[0]} images at batch {start_batch + batch_idx + 1}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"C-vectors: {embeddings.shape[0]} images, {embeddings.shape[1]} dims")
    return embeddings, failed


def embed_a_vectors(
    df: pd.DataFrame,
    batch_size: int = 8,
    device: str = "auto",
    checkpoint_dir: Path = Path("embeddings"),
    checkpoint_every: int = 100,
) -> tuple[np.ndarray, list]:
    """Extract A-vectors via Stable Diffusion 2.0 autoencoder (16384-dim).

    Captures visual appearance: colors, brightness, composition, texture.
    The autoencoder compresses 512x512x3 images to 64x64x4 latent space,
    which we flatten to 16384-dim vectors.
    Saves checkpoints every checkpoint_every batches so progress survives interruptions.
    """
    from diffusers import AutoencoderKL
    from torchvision import transforms

    if device == "auto":
        device = get_device()
    print(f"A-vectors: SD 2.0 VAE on {device}")

    ckpt_path = checkpoint_dir / "a_vectors_checkpoint.npz"
    start_batch = 0
    all_embeddings = []
    failed = []

    if ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        all_embeddings = [ckpt["embeddings"]]
        start_batch = int(ckpt["next_batch"])
        print(f"  Resuming from batch {start_batch} ({all_embeddings[0].shape[0]} images done)")

    vae = AutoencoderKL.from_pretrained(
        "sd2-community/stable-diffusion-2", subfolder="vae"
    )
    vae = vae.to(device)
    vae.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    batch_starts = list(range(0, len(df), batch_size))
    for batch_idx, i in enumerate(
        tqdm(batch_starts[start_batch:], desc="A-vectors", initial=start_batch, total=len(batch_starts)),
    ):
        batch_paths = df["path"].iloc[i : i + batch_size].tolist()
        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                failed.append((path, str(e)))

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            latent = vae.encode(batch_tensor).latent_dist.mean
            flat = latent.reshape(latent.shape[0], -1)
            all_embeddings.append(flat.cpu().numpy())

        if (batch_idx + 1) % checkpoint_every == 0:
            merged = np.concatenate(all_embeddings, axis=0)
            np.savez(ckpt_path, embeddings=merged, next_batch=start_batch + batch_idx + 1)
            all_embeddings = [merged]
            print(f"  Checkpoint: {merged.shape[0]} images at batch {start_batch + batch_idx + 1}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"A-vectors: {embeddings.shape[0]} images, {embeddings.shape[1]} dims")
    return embeddings, failed


def main():
    parser = argparse.ArgumentParser(
        description="Dual CLIP + SD embeddings for contempArt"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to visart2020/ directory with artist subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("embeddings"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--vectors",
        type=str,
        default="both",
        choices=["c", "a", "both"],
        help="Which vectors to extract: c (CLIP), a (SD autoencoder), or both",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_image_paths(args.data_dir)

    all_failed = []

    if args.vectors in ("c", "both"):
        c_embeddings, c_failed = embed_c_vectors(df, batch_size=args.batch_size)
        np.save(args.output_dir / "c_vectors.npy", c_embeddings)
        all_failed.extend(c_failed)

    if args.vectors in ("a", "both"):
        a_batch = min(args.batch_size, 8)  # A-vectors need more memory
        a_embeddings, a_failed = embed_a_vectors(df, batch_size=a_batch)
        np.save(args.output_dir / "a_vectors.npy", a_embeddings)
        all_failed.extend(a_failed)

    df.to_csv(args.output_dir / "image_manifest.csv", index=False)

    if all_failed:
        print(f"\nFailed to load {len(all_failed)} images total:")
        for path, err in all_failed[:10]:
            print(f"  {path}: {err}")

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
