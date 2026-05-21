"""Neighborhood mass probe: sum generation probability over N(i_f)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Set

from src.components.unlearning.neighborhood_sampler import (
    collect_items_in_shards,
    embedding_neighbors,
    load_codebook,
    load_dense_embeddings,
    prefix_neighbors,
)
from src.data.unlearning.deletion_spec import (
    load_forget_manifest,
    load_target_items,
    manifest_deletion_spec,
    resolve_neighborhood_centers,
    resolve_forget_manifest_path,
)


def _list_shards(directory: str) -> List[str]:
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".tfrecord.gz")
    ]


def compute_neighborhood_item_sets(
    *,
    data_dir: str,
    semantic_id_path: str,
    deletion_spec: str = "session",
    neighborhood_method: str = "prefix",
    sid_prefix_length: int = 2,
    embedding_path: Optional[str] = None,
    embedding_epsilon: Optional[float] = None,
    num_hierarchies: Optional[int] = None,
) -> Dict[str, object]:
    manifest_path = resolve_forget_manifest_path(data_dir)
    manifest = load_forget_manifest(manifest_path)
    spec = manifest_deletion_spec(manifest, deletion_spec)
    target_items = load_target_items(manifest)
    forget_dir = os.path.join(data_dir, "training_forget")
    forget_shard_items = collect_items_in_shards(_list_shards(forget_dir))
    centers = resolve_neighborhood_centers(
        deletion_spec=spec,
        forget_shard_items=forget_shard_items,
        target_items=target_items,
    )
    neighbors: Set[int] = set()
    if neighborhood_method == "embedding" and embedding_path and embedding_epsilon:
        emb = load_dense_embeddings(embedding_path)
        for cid in centers:
            neighbors.update(
                embedding_neighbors(
                    int(cid),
                    emb,
                    float(embedding_epsilon),
                    exclude_ids=centers,
                )
            )
    else:
        codebook = load_codebook(semantic_id_path, num_hierarchies=num_hierarchies)
        for cid in centers:
            neighbors.update(prefix_neighbors(codebook, [cid], sid_prefix_length))
    return {
        "deletion_spec": spec,
        "n_centers": len(centers),
        "n_neighbors": len(neighbors),
        "center_items": sorted(centers)[:100],
        "neighbor_items": sorted(neighbors)[:500],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compute neighborhood item sets for mass eval.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--semantic_id_path", required=True)
    p.add_argument("--deletion_spec", default="session", choices=["session", "item"])
    p.add_argument("--neighborhood_method", default="prefix", choices=["prefix", "embedding"])
    p.add_argument("--sid_prefix_length", type=int, default=2)
    p.add_argument("--embedding_path", default=None)
    p.add_argument("--embedding_epsilon", type=float, default=None)
    p.add_argument("--num_hierarchies", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    info = compute_neighborhood_item_sets(
        data_dir=args.data_dir,
        semantic_id_path=args.semantic_id_path,
        deletion_spec=args.deletion_spec,
        neighborhood_method=args.neighborhood_method,
        sid_prefix_length=args.sid_prefix_length,
        embedding_path=args.embedding_path,
        embedding_epsilon=args.embedding_epsilon,
        num_hierarchies=args.num_hierarchies,
    )
    out = args.out or os.path.join(args.data_dir, "neighborhood_mass_probe.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
