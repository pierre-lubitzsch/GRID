#!/usr/bin/env python
"""Step 2 of the minimal validation: learn the latent refiner, export ``z``.

    python -m scripts.train_latent_refiner \
        --ckpt        logs/train/runs/<...>/checkpoints/<...>.ckpt \
        --embedding_path    embeddings/beauty_merged_predictions_tensor_latest.pt \
        --semantic_id_path  embeddings/beauty/merged_predictions_tensor.pt \
        --num_hierarchies 4 \
        --out embeddings/beauty_latent/refined_latents.pt

Writes three files next to ``--out``:

``<out>``                      ``[N, latent_dim]`` float tensor, row-aligned with
                               the SID codebook. This is what
                               ``unlearning.coherence_latent_path`` consumes.
``<out>.refiner.pt``           refiner ``state_dict`` + config, so ``z`` can be
                               regenerated or the module reused at inference time
                               (the candidate-repair step, not part of this run).
``<out>.stats.json``           training curve + the neighbourhood-overlap
                               diagnostics printed below.

The diagnostics are the point of this script as much as the tensor is. If
``N_z`` overlaps ``N_emb`` at ~1.0 the refined space is a reparameterisation of
the embedding space and the ``latent`` arm cannot differ from the ``embedding``
arm; if it overlaps at ~0.0 the latent has drifted off the item geometry
entirely. Either extreme means the loss weights, not the unlearning, explain the
downstream result -- so they are recorded before any unlearning job is submitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.unlearning.latent_refiner import (  # noqa: E402
    LatentRefinerConfig,
    build_sid_embedding_matrix,
    export_latents,
    load_sid_table_from_checkpoint,
    topk_cosine_neighbors,
    train_latent_refiner,
)
from src.components.unlearning.neighborhood_sampler import (  # noqa: E402
    build_sorted_sid_index,
    closest_prefix_neighbors,
    load_codebook,
    load_dense_embeddings,
)

log = logging.getLogger("train_latent_refiner")


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="trained (pre-unlearning) recommender checkpoint")
    p.add_argument("--embedding_path", required=True, help="pre-quantization item embeddings x_i")
    p.add_argument("--semantic_id_path", required=True, help="merged_predictions_tensor.pt (SID codebook)")
    p.add_argument("--num_hierarchies", type=int, default=None)
    p.add_argument("--out", required=True, help="output path for the [N, latent_dim] z tensor")

    g = p.add_argument_group("refiner")
    g.add_argument("--latent_dim", type=int, default=128)
    g.add_argument("--hidden_dim", type=int, default=512)
    g.add_argument("--dropout", type=float, default=0.0)

    g = p.add_argument_group("objective weights")
    g.add_argument("--w_reconstruction", type=float, default=1.0)
    g.add_argument("--w_neighborhood", type=float, default=1.0)
    g.add_argument("--w_sid_consistency", type=float, default=0.1)

    g = p.add_argument_group("training")
    g.add_argument("--neighbor_k", type=int, default=8)
    g.add_argument("--temperature", type=float, default=0.07)
    g.add_argument("--sid_prefix_length", type=int, default=2)
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch_size", type=int, default=512)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=2)
    g.add_argument("--device", default=None)

    g = p.add_argument_group("diagnostics")
    g.add_argument(
        "--diag_k",
        type=int,
        default=8,
        help="k at which N_z / N_emb / N_prefix overlap is measured "
             "(match unlearning.neighborhood_count)",
    )
    g.add_argument(
        "--diag_sample",
        type=int,
        default=2000,
        help="items sampled for the prefix-overlap diagnostic (0 = all)",
    )
    return p.parse_args(argv)


def _jaccard_and_overlap(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Mean |A ∩ B| / k and Jaccard over row-aligned neighbour lists ``[N, k]``."""
    n, k = int(a.shape[0]), int(a.shape[1])
    inter = torch.zeros(n)
    union = torch.zeros(n)
    for i in range(n):
        sa = set(a[i].tolist())
        sb = set(b[i].tolist())
        inter[i] = len(sa & sb)
        union[i] = len(sa | sb)
    return {
        "mean_overlap_at_k": float((inter / max(k, 1)).mean()),
        "mean_jaccard": float((inter / union.clamp_min(1)).mean()),
        "frac_identical": float((inter == k).float().mean()),
        "frac_disjoint": float((inter == 0).float().mean()),
    }


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    codebook = load_codebook(args.semantic_id_path, num_hierarchies=args.num_hierarchies)
    num_items = int(codebook.shape[0])
    dense = load_dense_embeddings(args.embedding_path)
    item_emb = dense.tensor.float()
    if int(item_emb.shape[0]) != num_items:
        raise ValueError(
            f"embedding_path has {int(item_emb.shape[0])} items but the codebook "
            f"has {num_items}: the refined space is indexed BY ROW, so the two "
            "must be row-aligned (same catalog, same order). Check that "
            f"{args.embedding_path!r} is the tensor the SID codebook was built from."
        )

    sid_table = load_sid_table_from_checkpoint(args.ckpt)
    sid_emb = build_sid_embedding_matrix(codebook, sid_table)

    cfg = LatentRefinerConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        w_reconstruction=args.w_reconstruction,
        w_neighborhood=args.w_neighborhood,
        w_sid_consistency=args.w_sid_consistency,
        neighbor_k=args.neighbor_k,
        temperature=args.temperature,
        sid_prefix_length=args.sid_prefix_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info(
        "[latent-refiner] %d items | SIDEmb %d dims | x %d dims -> z %d dims | device=%s",
        num_items,
        int(sid_emb.shape[1]),
        int(item_emb.shape[1]),
        cfg.latent_dim,
        device,
    )

    model, stats = train_latent_refiner(
        sid_emb=sid_emb, item_emb=item_emb, codebook=codebook,
        config=cfg, device=device,
    )
    z = export_latents(model, sid_emb=sid_emb, item_emb=item_emb, device=device)

    # ---- diagnostics: is N_z actually a different neighbourhood? ----------
    k = int(args.diag_k)
    nz = topk_cosine_neighbors(z, k, device=device)
    nemb = topk_cosine_neighbors(item_emb, k, device=device)
    diag: Dict[str, object] = {
        "k": k,
        "num_items": num_items,
        "latent_vs_embedding": _jaccard_and_overlap(nz, nemb),
    }
    log.info(
        "[diag] N_z vs N_emb @k=%d: overlap=%.3f jaccard=%.3f identical=%.1f%% disjoint=%.1f%%",
        k,
        diag["latent_vs_embedding"]["mean_overlap_at_k"],
        diag["latent_vs_embedding"]["mean_jaccard"],
        100 * diag["latent_vs_embedding"]["frac_identical"],
        100 * diag["latent_vs_embedding"]["frac_disjoint"],
    )

    # Prefix neighbourhood, on a sample: closest_prefix_neighbors is a per-item
    # python call, so the full catalog would dominate the runtime for a number
    # that only needs to be indicative.
    sample_n = num_items if int(args.diag_sample) <= 0 else min(int(args.diag_sample), num_items)
    gen = torch.Generator().manual_seed(int(args.seed))
    sample = torch.randperm(num_items, generator=gen)[:sample_n]
    sorted_ids = build_sorted_sid_index(codebook)
    sorted_sids = codebook.numpy()[sorted_ids]
    pfx_hits, pfx_empty, lat_cov, emb_cov = 0, 0, 0.0, 0.0
    for i in sample.tolist():
        pn = closest_prefix_neighbors(
            codebook, i, k, int(args.sid_prefix_length),
            sorted_ids=sorted_ids, sorted_sids=sorted_sids,
        )
        if not pn:
            pfx_empty += 1
            continue
        pfx_hits += 1
        sp = set(int(x) for x in pn)
        lat_cov += len(sp & set(nz[i].tolist())) / len(sp)
        emb_cov += len(sp & set(nemb[i].tolist())) / len(sp)
    diag["prefix_sample"] = {
        "n_sampled": sample_n,
        "frac_with_prefix_neighbours": pfx_hits / max(sample_n, 1),
        "frac_prefix_empty": pfx_empty / max(sample_n, 1),
        "mean_prefix_recovered_by_latent": lat_cov / max(pfx_hits, 1),
        "mean_prefix_recovered_by_embedding": emb_cov / max(pfx_hits, 1),
    }
    log.info(
        "[diag] prefix(p=%d) on %d sampled items: %.1f%% have neighbours; of those, "
        "latent recovers %.3f and embedding recovers %.3f",
        int(args.sid_prefix_length), sample_n,
        100 * diag["prefix_sample"]["frac_with_prefix_neighbours"],
        diag["prefix_sample"]["mean_prefix_recovered_by_latent"],
        diag["prefix_sample"]["mean_prefix_recovered_by_embedding"],
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(z, args.out)
    torch.save(
        {"state_dict": model.state_dict(), "config": vars(cfg),
         "sid_dim": int(sid_emb.shape[1]), "item_dim": int(item_emb.shape[1]),
         "ckpt": args.ckpt, "embedding_path": args.embedding_path,
         "semantic_id_path": args.semantic_id_path},
        f"{args.out}.refiner.pt",
    )
    with open(f"{args.out}.stats.json", "w") as fh:
        json.dump({"config": vars(cfg), "training": stats.as_dict(),
                   "diagnostics": diag}, fh, indent=2)
    log.info("[latent-refiner] wrote %s %s", args.out, tuple(z.shape))
    log.info("[latent-refiner] wrote %s.refiner.pt and %s.stats.json", args.out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
