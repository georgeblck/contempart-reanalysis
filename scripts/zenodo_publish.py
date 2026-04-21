"""Upload the contempArt embeddings bundle to Zenodo via the REST API.

One record, per-backbone files (9 npz + image_manifest.csv + heads.csv +
README.md). Creates a draft — does NOT publish by default. Review the
draft in the Zenodo UI, then publish manually (or re-run with
`--publish`).

Usage:
    export ZENODO_TOKEN=<your-token>          # production
    export ZENODO_SANDBOX_TOKEN=<your-token>   # sandbox
    uv run python scripts/zenodo_publish.py            # dry-run, lists files
    uv run python scripts/zenodo_publish.py --sandbox  # upload to sandbox
    uv run python scripts/zenodo_publish.py            # upload to production
    uv run python scripts/zenodo_publish.py --publish  # also publish (irreversible)

After creation the script prints the deposit ID + edit URL so you can
review and publish from the web UI.

Zenodo docs: https://developers.zenodo.org/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"
ZENODO_DIR = ROOT / "zenodo"

# One file per backbone. Keeps downloads selective.
NPZ_FILES = [
    "clip_openai/clip_openai_contempart.npz",
    "clip_openai_b32/clip_openai_b32_contempart.npz",
    "clip_openclip/clip_openclip_contempart.npz",
    "dinov2/dinov2_contempart.npz",
    "csd/csd_contempart.npz",
    "scflow/scflow_contempart.npz",
    "goya/goya_contempart.npz",
    "sd2_kim/sd2_kim_contempart.npz",
    "sscd/sscd_contempart.npz",
]


def collect_files() -> list[tuple[Path, str]]:
    """Return (local_path, upload_name) pairs for every file in the bundle."""
    items: list[tuple[Path, str]] = []
    for rel in NPZ_FILES:
        p = EMBEDDINGS_DIR / rel
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        items.append((p, Path(rel).name))
    items.append((EMBEDDINGS_DIR / "image_manifest.csv", "image_manifest.csv"))
    items.append((RESULTS_DIR / "heads.csv", "heads.csv"))
    items.append((ZENODO_DIR / "README.md", "README.md"))
    for src, _ in items:
        if not src.exists():
            raise FileNotFoundError(f"missing: {src}")
    return items


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_metadata() -> dict:
    with open(ZENODO_DIR / "metadata.json") as f:
        return json.load(f)


def zenodo_host(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"


def token_for(sandbox: bool) -> str:
    env = "ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_TOKEN"
    tok = os.environ.get(env)
    if not tok:
        sys.exit(f"set {env}")
    return tok


def create_deposit(client: httpx.Client, host: str, metadata: dict) -> dict:
    r = client.post(f"{host}/api/deposit/depositions", json={})
    r.raise_for_status()
    dep = r.json()
    r = client.put(f"{host}/api/deposit/depositions/{dep['id']}", json=metadata)
    r.raise_for_status()
    return r.json()


def upload_file(client: httpx.Client, bucket_url: str, src: Path, name: str) -> None:
    size = src.stat().st_size
    print(f"  -> {name}  ({human_size(size)})", flush=True)
    with src.open("rb") as f:
        r = client.put(f"{bucket_url}/{name}", content=f, timeout=None)
    r.raise_for_status()


def publish(client: httpx.Client, host: str, deposit_id: int) -> dict:
    r = client.post(f"{host}/api/deposit/depositions/{deposit_id}/actions/publish")
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sandbox", action="store_true",
                    help="target sandbox.zenodo.org (test). Requires ZENODO_SANDBOX_TOKEN.")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files + metadata, do not create a deposit")
    ap.add_argument("--publish", action="store_true",
                    help="publish the draft immediately (irreversible once DOI is minted)")
    args = ap.parse_args()

    items = collect_files()
    metadata = load_metadata()
    total = sum(p.stat().st_size for p, _ in items)

    print(f"Target:   {'sandbox' if args.sandbox else 'PRODUCTION'}")
    print(f"Title:    {metadata['metadata']['title']}")
    print(f"Version:  {metadata['metadata']['version']}")
    print(f"License:  {metadata['metadata']['license']}")
    print(f"Creator:  {metadata['metadata']['creators'][0]['name']}")
    print(f"Files:    {len(items)}  (~{human_size(total)})")
    for src, name in items:
        print(f"  {name:<45} {human_size(src.stat().st_size)}")

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return

    host = zenodo_host(args.sandbox)
    tok = token_for(args.sandbox)

    with httpx.Client(
        headers={"Authorization": f"Bearer {tok}"},
        timeout=httpx.Timeout(60.0, read=300.0),
    ) as client:
        print("\nCreating deposit ...")
        dep = create_deposit(client, host, metadata)
        deposit_id = dep["id"]
        bucket_url = dep["links"]["bucket"]
        edit_url = dep["links"].get("html", f"{host}/deposit/{deposit_id}")
        print(f"  deposit_id = {deposit_id}")
        print(f"  edit URL   = {edit_url}")

        print("\nUploading files ...")
        for src, name in items:
            upload_file(client, bucket_url, src, name)

        if args.publish:
            print("\nPublishing ...")
            pub = publish(client, host, deposit_id)
            doi = pub.get("doi") or pub.get("metadata", {}).get("doi")
            print(f"  DOI: {doi}")
            print(f"  URL: {pub['links'].get('record_html', pub['links'].get('html'))}")
        else:
            print("\nDraft created. Review in the Zenodo UI, then publish from there")
            print("(or re-run this script with --publish).")
            print(f"  {edit_url}")


if __name__ == "__main__":
    main()
