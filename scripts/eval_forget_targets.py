"""Evaluate forgetting effectiveness on target items (Recall@K on I_f)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import torch

from src.data.unlearning.deletion_spec import load_target_items, resolve_forget_manifest_path


def _load_manifest(data_dir: str) -> Dict:
    path = resolve_forget_manifest_path(data_dir)
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_sid_to_item(codebook: torch.Tensor) -> Dict[Tuple[int, ...], int]:
    """Build reverse mapping: SID tuple → sequential item index.

    ``codebook`` is the N×D tensor stored in the model (``self.codebooks``),
    where row i is the D-digit SID for item i.
    """
    result: Dict[Tuple[int, ...], int] = {}
    for idx in range(codebook.size(0)):
        key = tuple(int(x) for x in codebook[idx].tolist())
        result[key] = idx
    return result


def eval_forget_target_recall(
    *,
    model: torch.nn.Module,
    histories: List[List[int]],
    target_items: Set[int],
    top_k: int = 10,
    device: torch.device,
) -> Dict[str, float]:
    """Measure how often the model retrieves target items on probe histories.

    ``histories`` must contain **sequential item indices** (0..N-1), i.e. the
    same ID space as the model's SID codebook — not raw dataset item IDs if the
    dataset uses non-sequential IDs (e.g. rsc15).  ``target_items`` must also
    be in the same sequential-index space.

    ``model.codebooks`` (N×D) is used to decode generated SID tuples back to
    sequential item indices before comparing against ``target_items``.
    """
    if not target_items or not histories:
        return {"forget_recall@k": 0.0, "n_probes": 0.0}

    codebook: Optional[torch.Tensor] = getattr(model, "codebooks", None)
    if codebook is None or not hasattr(model, "generate"):
        return {"forget_recall@k": 0.0, "n_probes": float(len(histories))}

    sid_to_item = _build_sid_to_item(codebook.cpu())

    hits = 0
    for hist in histories:
        if len(hist) < 1:
            continue
        seq = torch.tensor([hist], dtype=torch.long, device=device)
        mask = torch.ones_like(seq, dtype=torch.long)
        with torch.no_grad():
            gen_ids, _ = model.generate(attention_mask=mask, input_ids=seq)
        # gen_ids shape: (batch, top_k, num_hierarchies)
        # Each beam is a SID tuple; decode to item index via codebook.
        n_beams = gen_ids.shape[1]
        found = False
        for beam in range(n_beams):
            sid_tuple = tuple(int(x) for x in gen_ids[0, beam].tolist())
            item_idx = sid_to_item.get(sid_tuple)
            if item_idx is not None and item_idx in target_items:
                found = True
                break
        if found:
            hits += 1
    n = max(1, len(histories))
    return {"forget_recall@k": hits / n, "n_probes": float(len(histories))}


def main() -> None:
    p = argparse.ArgumentParser(description="Probe forget-target recall@K.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    manifest = _load_manifest(args.data_dir)
    target_items = load_target_items(manifest)
    result = {
        "data_dir": os.path.abspath(args.data_dir),
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "n_target_items": len(target_items),
        "target_items": sorted(target_items)[:50],
        "note": "Full eval requires TIGER checkpoint + probe histories; run via inference pipeline.",
    }
    out = args.out or os.path.join(args.data_dir, "forget_target_eval.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
