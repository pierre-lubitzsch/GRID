#!/usr/bin/env python
"""Markdown table of the latent-refiner neighbourhood diagnostics.

    python -m scripts.write_latent_diag_table [--root embeddings/latent] \
        [--out tables/latent_refiner_diag.md]

Reads every ``*.stats.json`` written by ``scripts/train_latent_refiner.py`` and
tabulates the numbers that decide whether the refined space is worth an
unlearning sweep at all:

``overlap@k``   mean |N_z ∩ N_emb| / k. Near 1.0 means z is a reparameterisation
                of the embedding space and the ``latent`` arm cannot differ from
                the ``embedding`` arm; near 0.0 means z left the item geometry
                and any downstream effect is noise. The interesting range is the
                middle.
``prefix rec.`` fraction of an item's SID-prefix neighbourhood recovered by the
                latent vs the embedding top-k. Latent > embedding is the design
                goal landing: SID structure preserved, item detail added.

Emits to stdout, and to ``--out`` when given.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import List


def _label(path: str) -> str:
    """`beauty_bw_tgtmid_n1_seed2_dz128.pt.stats.json` -> `mid`."""
    base = os.path.basename(path)
    m = re.search(r"_bw_tgt([a-z]+)_", base)
    return m.group(1) if m else base.replace(".pt.stats.json", "")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="embeddings/latent")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.root, "*.stats.json")))
    if not paths:
        raise SystemExit(f"no *.stats.json under {args.root!r} — run Step 2 first")

    # Keep the popularity strata in their conventional order.
    order = {"unpopular": 0, "mid": 1, "popular": 2}
    paths.sort(key=lambda q: order.get(_label(q), 99))

    rows = []
    for path in paths:
        with open(path) as fh:
            d = json.load(fh)
        diag, cfg = d["diagnostics"], d["config"]
        lve, pfx = diag["latent_vs_embedding"], diag["prefix_sample"]
        fin = d["training"]["final"]
        rows.append({
            "model": _label(path),
            "k": diag["k"],
            "overlap": lve["mean_overlap_at_k"],
            "jaccard": lve["mean_jaccard"],
            "identical": 100 * lve["frac_identical"],
            "disjoint": 100 * lve["frac_disjoint"],
            "pfx_cov": 100 * pfx["frac_with_prefix_neighbours"],
            "rec_lat": pfx["mean_prefix_recovered_by_latent"],
            "rec_emb": pfx["mean_prefix_recovered_by_embedding"],
            "rec": fin.get("rec", float("nan")),
            "nbr": fin.get("nbr", float("nan")),
            "sid": fin.get("sid", float("nan")),
            "dz": cfg["latent_dim"],
        })

    hdr = ("| model | d_z | overlap@k | Jaccard | identical | disjoint | "
           "prefix cov. | prefix rec. latent | prefix rec. emb | L_rec | L_nbr | L_sid |")
    sep = ("|-------|-----|-----------|---------|-----------|----------|"
           "-------------|--------------------|-----------------|-------|-------|-------|")
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dz']} | {r['overlap']:.3f} | {r['jaccard']:.3f} | "
            f"{r['identical']:.1f}% | {r['disjoint']:.1f}% | {r['pfx_cov']:.1f}% | "
            f"**{r['rec_lat']:.3f}** | {r['rec_emb']:.3f} | "
            f"{r['rec']:.4f} | {r['nbr']:.4f} | {r['sid']:.4f} |"
        )
    table = "\n".join(lines)
    print(table)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(table + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
