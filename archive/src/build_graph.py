"""
Load Instagram follower data and build the social graph for node2vec.

Reads the follower/following CSVs from the original contempArt dataset
and constructs a directed NetworkX graph of artist-to-artist connections.

Usage:
    uv run python src/build_graph.py \
        --follow-dir data/social/followers \
        --following-dir data/social/following \
        --manifest embeddings/image_manifest.csv \
        --output-dir graphs
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec


def load_follow_data(follow_dir: Path, following_dir: Path) -> list:
    """Load follower/following CSVs and extract edges.

    Each CSV in followDat/ contains followers of one artist.
    Each CSV in followingDat/ contains who one artist follows.
    """
    edges = []

    # followDat: each file lists followers of the filename's artist
    if follow_dir.exists():
        for csv_path in sorted(follow_dir.glob("*.csv")):
            artist = csv_path.stem
            try:
                df = pd.read_csv(csv_path, header=None)
                for follower in df.iloc[:, 0]:
                    edges.append((str(follower), artist))  # follower -> artist
            except Exception:
                continue

    # followingDat: each file lists who the filename's artist follows
    if following_dir.exists():
        for csv_path in sorted(following_dir.glob("*.csv")):
            artist = csv_path.stem
            try:
                df = pd.read_csv(csv_path, header=None)
                for followed in df.iloc[:, 0]:
                    edges.append((artist, str(followed)))  # artist -> followed
            except Exception:
                continue

    print(f"Loaded {len(edges)} raw edges")
    return edges


def build_artist_graph(
    edges: list,
    artists: set,
    artist_only: bool = True,
    handle_to_artist: dict | None = None,
) -> nx.DiGraph:
    """Build directed graph, optionally filtering to artist-to-artist edges only.

    If handle_to_artist is provided, edges (which use Instagram handles) are
    translated to artist_ids before filtering.
    """
    G = nx.DiGraph()
    G.add_nodes_from(artists)

    kept = 0
    for src, dst in edges:
        if handle_to_artist:
            src = handle_to_artist.get(src, src)
            dst = handle_to_artist.get(dst, dst)
        if artist_only:
            if src in artists and dst in artists:
                G.add_edge(src, dst)
                kept += 1
        else:
            G.add_edge(src, dst)
            kept += 1

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if artist_only:
        print(f"Filtered to artist-only: {kept} edges from {len(edges)} raw")

    return G


def run_node2vec(
    G: nx.Graph,
    dimensions: int = 128,
    walk_length: int = 30,
    num_walks: int = 200,
    p: float = 1.0,
    q: float = 1.0,
) -> tuple:
    """Run node2vec and return embeddings."""
    # node2vec needs undirected for walks
    G_undirected = G.to_undirected()

    # Remove isolated nodes (no edges)
    isolates = list(nx.isolates(G_undirected))
    if isolates:
        print(f"Removing {len(isolates)} isolated nodes")
        G_undirected.remove_nodes_from(isolates)

    n2v = Node2Vec(
        G_undirected,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=4,
        quiet=True,
    )

    print(f"Training node2vec (d={dimensions}, walks={num_walks}, len={walk_length})...")
    model = n2v.fit(window=10, min_count=1, batch_words=4)

    nodes = sorted(G_undirected.nodes())
    embeddings = np.array([model.wv[str(n)] for n in nodes])
    print(f"Node2vec embeddings: {embeddings.shape}")

    return embeddings, nodes, isolates


def main():
    parser = argparse.ArgumentParser(description="Build Instagram graph + node2vec")
    parser.add_argument("--follow-dir", type=Path, required=True)
    parser.add_argument("--following-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("embeddings/image_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--num-walks", type=int, default=200)
    parser.add_argument("--walk-length", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get artist list from manifest
    manifest = pd.read_csv(args.manifest)
    artists = set(manifest["artist"].unique())
    print(f"Artists in dataset: {len(artists)}")

    # Load and build graph
    edges = load_follow_data(args.follow_dir, args.following_dir)
    G = build_artist_graph(edges, artists, artist_only=True)

    # Save graph
    nx.write_graphml(G, args.output_dir / "artist_graph.graphml")

    # Run node2vec
    embeddings, nodes, isolates = run_node2vec(
        G,
        dimensions=args.dimensions,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
    )

    np.save(args.output_dir / "node2vec_embeddings.npy", embeddings)
    pd.DataFrame({"artist": nodes}).to_csv(
        args.output_dir / "node2vec_artists.csv", index=False
    )
    if isolates:
        pd.DataFrame({"artist": isolates}).to_csv(
            args.output_dir / "isolated_artists.csv", index=False
        )

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
