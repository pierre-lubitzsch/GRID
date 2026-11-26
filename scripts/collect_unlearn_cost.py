"""Cost of unlearning runs, from their scif_info.json (no new runs needed).

Per run: wall seconds summed over requests, optimizer steps, parameters that receive
an update (lr_groups with a nonzero multiplier) and, where the update is restricted to
rows (item-memory rows, PKM value slots), the rows actually allowed to move.

Usage (from GRID/):
  python -m scripts.collect_unlearn_cost --match ITEMMEM_imsc_ --match SLOTSEL_ \
      --out logs/cost/unlearn_cost.csv
"""

import argparse
import csv
import glob
import json
import os


def summarize(run_dir: str):
    p = os.path.join(run_dir, "scif_info.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    reqs = d.get("requests") or []
    if not reqs:
        return None
    infos = [r.get("scif_info") or {} for r in reqs]
    first = infos[0]
    groups = first.get("lr_groups") or []
    upd = sum(int(g.get("n_params", 0)) for g in groups if float(g.get("mult", 1.0)) != 0.0)
    im = first.get("item_memory") or {}
    slot = first.get("slot_selection")
    row_info = ""
    if im.get("n_rows_selected") is not None:
        row_info = f"item rows {im['n_rows_selected']}"
    if isinstance(slot, dict) and slot.get("per_memory"):
        n = sum(int(v.get("selected", 0) or 0)
                for v in slot["per_memory"].values())
        row_info = (row_info + "; " if row_info else "") + f"pkm slots {n}"
    vs = d.get("val_selection") or {}
    return {
        "tag": d.get("unlearning_run_tag") or os.path.basename(run_dir),
        "requests": len(reqs),
        "wall_s_total": round(sum(float(i.get("wall_seconds", 0.0)) for i in infos), 1),
        "steps_total": sum(int(i.get("steps_run", i.get("steps", 0)) or 0) for i in infos),
        "params_updated": upd,
        "lr_groups": ";".join(f"{g.get('mult')}x:{g.get('n_params')}" for g in groups),
        "restricted_rows": row_info,
        "update_scope": first.get("update_scope", ""),
        "val_selected_n_epochs": vs.get("best_n_epochs", ""),
        "run_dir": run_dir,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="logs/unlearn/runs")
    ap.add_argument("--match", action="append", default=[],
                    help="substring of the run dir name (repeatable; any match)")
    ap.add_argument("--out", default=None, help="CSV path (stdout table if omitted)")
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if a.match and not any(m in os.path.basename(d) for m in a.match):
            continue
        r = summarize(d)
        if r:
            rows.append(r)
    if not rows:
        print("no runs matched")
        return 1
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} runs -> {a.out}")
    for r in rows:
        print(f"{r['tag'][:52]:52s} req={r['requests']:2d} wall={r['wall_s_total']:7.1f}s "
              f"steps={r['steps_total']:4d} updated={r['params_updated']/1e6:6.2f}M "
              f"{r['restricted_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
