"""RQ code-position restriction study: which hierarchy levels can carry unlearning?

Reads the `update_positions` arms (`_acp<pos>[bb]` tags) written by
`run_unlearn_positions_beauty.sh` and reports UR and PGR SIDE BY SIDE -- neither
number means anything alone. An arm that removes spam by destroying the model
scores well on PGR; an arm that preserves utility by doing nothing scores well on
UR. The pair is the result.

Levels use the repo's layout: SID table rows [h*K, (h+1)*K) and decoder head
decoder_mlp[h] for position h. NOTE the naming trap -- the repo calls levels 0,1
"stable" (coarse) and 2,3 "adaptive" (fine), so `adaptive_code_lr_scale` scales
levels 2,3, NOT 0,1.

The codes-only arms (backbone frozen) are the discriminating ones: with the
backbone free, every restriction reaches PGR ~ 1.0 and the levels cannot be told
apart.

    PYTHONPATH=. python scripts/extract_code_positions.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `extract_bandwagon_head2head` was rewritten to resolve references per
# (dataset, track, seed) and no longer exports the beauty constants directly; they
# live in LEGACY_REFERENCES, which is still the right source here because every arm
# in this table is beauty / tiger / RQ-KMeans / seed 2. Reuse it rather than
# re-hardcoding the numbers, so there stays ONE definition of the denominators.
from scripts.extract_bandwagon_head2head import LEGACY_REFERENCES
from scripts.extract_lambda_n_ablation import last_metrics

_REF = LEGACY_REFERENCES["beauty"]
CLEAN_N10 = _REF["clean_n10"]
CLEAN_SH = _REF["clean_sh"]
POISON_SH = _REF["poison_sh"]

STRATA = ["unpopular", "mid", "popular"]
SEEDS = [2, 3, 5]
LF, LS = "0.1", "0.1"

# label -> tag body (minus the bw_tgt<strat>_ prefix and the _useed<n> suffix).

# LEVEL NUMBERING. Labels in this table are 1-BASED (levels 1..4), matching the
# paper. The arm tags are NOT: `acp01` means code positions 0,1 in the repo's
# 0-based internal indexing, i.e. levels 1,2 here. The tags are run-dir
# identifiers, so renumbering them would orphan every recorded run; only the
# human-readable label shifts.
ARMS: List[Tuple[str, str]] = [
    ("all params (unrestricted)",  f"unified_lf{LF}_ls{LS}_ac0_ln0.0_sepgen"),
    ("levels 3,4 frozen (aclr=0)", f"unified_lf{LF}_ls{LS}_ac0_ln0.0_sepgen_aclr0.0"),
    ("levels 1,2 + backbone",      f"unified_lf{LF}_ls{LS}_acp01bb"),
    ("levels 3,4 + backbone",      f"unified_lf{LF}_ls{LS}_acp23bb"),
    ("levels 1,2 CODES ONLY",      f"unified_lf{LF}_ls{LS}_acp01"),
    ("levels 3,4 CODES ONLY",      f"unified_lf{LF}_ls{LS}_acp23"),
    # Per-level ablation: ONLY level i is updatable. This is the cleanest read of
    # "where does the unlearning signal live", because each arm touches exactly
    # one hierarchy. Expectation from the codebook stats (a level-0 code is
    # shared by ~47 items; c0..c2 is 90.4% singleton; level 3 uses 16 distinct
    # codes across the whole catalog): 0 > 1 >> 2 > 3 ~ nothing.
    ("level 1 ONLY (codes)",       f"unified_lf{LF}_ls{LS}_acp0"),
    ("level 2 ONLY (codes)",       f"unified_lf{LF}_ls{LS}_acp1"),
    ("level 3 ONLY (codes)",       f"unified_lf{LF}_ls{LS}_acp2"),
    ("level 4 ONLY (codes)",       f"unified_lf{LF}_ls{LS}_acp3"),
    ("level 3 + backbone",         f"unified_lf{LF}_ls{LS}_acp2bb"),
    # Code-lr scales ON TOP of a position restriction (requested 2026-08-18).
    # levels 0,1: aclr would scale the FROZEN tail, so code_lr_scale (uniform
    # over the trainable code params) is the well-defined knob there.
    # levels 0,1,2: level 2 stays trainable, so aclr genuinely damps it against
    # levels 0,1 -- the only position set where aclr means something.
    ("levels 1,2 +bb, clr 0.1",    f"unified_lf{LF}_ls{LS}_acp01bb_clr0.1"),
    ("levels 1,2,3 +bb",           f"unified_lf{LF}_ls{LS}_acp012bb"),
    ("levels 1,2,3 +bb, aclr 0.1", f"unified_lf{LF}_ls{LS}_acp012bb_aclr0.1"),
    # ac1 = adaptive_codes=true: the repo's ORIGINAL restriction, which freezes
    # the stable prefix (levels 0,1) and trains only the adaptive tail (2,3),
    # backbone frozen too (adaptive_update_backbone defaults false). It is the
    # exact complement of "levels 0,1 CODES ONLY", so the two together bracket
    # the question of which half of the identifier space carries the update.
    ("ac1: ONLY levels 3,4 (codes)", f"unified_lf{LF}_ls{LS}_ac1"),
    # The MATCHED ac0 control for the row above. Needed because the other ac0
    # arms in this table are not clean comparisons: `..._ac0_ln0.0_sepgen` also
    # sets sep_loss_type=generative, so it differs from `_ac1` in the separation
    # loss as well as in adaptive_codes, and the difference cannot be attributed.
    # This arm comes from the unified hyperparameter grid and matches `_ac1` on
    # lambda_f, lambda_s, lambda_n, sep_negatives and n_epochs, differing ONLY in
    # adaptive_codes (false vs true) -- i.e. no position restriction at all.
    # Coverage caveat: the hyperparameter grid ran on the MID stratum at seed 2,
    # so this row is a single cell rather than 3 strata x 3 seeds.
    ("ac0 MATCHED (no restriction)", f"unified_hp_lf{LF}_ls{LS}"),
]


def _pm(vals: List[float], fmt: str = "{:.4f}") -> str:
    """mean +- half-range, matching every other extractor in this repo."""
    if not vals:
        return "-"
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return fmt.format(m)
    return f"{fmt.format(m)}+-{fmt.format((max(vals) - min(vals)) / 2.0)}"


def cell(strat: str, body: str) -> Optional[Dict[str, Any]]:
    urs: List[float] = []
    pgrs: List[float] = []
    shs: List[float] = []
    seeds: List[int] = []
    den = POISON_SH[strat] - CLEAN_SH[strat]
    for sd in SEEDS:
        tag = f"bw_tgt{strat}_{body}" + ("" if sd == 2 else f"_useed{sd}")
        dirs = glob.glob(f"logs/unlearn/runs/*_bs1_{tag}")
        if not dirs:
            continue
        # newest job wins, mirroring the other extractors.
        newest = sorted(dirs, key=lambda d: int(
            re.search(r"job(\d+)", os.path.basename(d)).group(1)))[-1]
        # The borrowed last_metrics takes the metrics.csv PATH, not the run dir.
        csvs = glob.glob(os.path.join(newest, "**", "metrics.csv"), recursive=True)
        if not csvs:
            continue
        csvs.sort(key=lambda p: ("eval" not in p, len(p)))   # prefer the eval csv
        m = last_metrics(csvs[0])
        if not m:
            continue
        n10, sh = m.get("test/ndcg@10"), m.get("test/SH@10")
        if n10 is None or sh is None:
            continue
        urs.append(n10 / CLEAN_N10)
        shs.append(sh)
        # PGR = (SH_poison - SH) / (SH_poison - SH_clean). The clean floor is
        # load-bearing on `popular` (0.00125 of 0.00180), where dropping it
        # silently turns PGR 3.29 into 1.00.
        pgrs.append((POISON_SH[strat] - sh) / den if den else float("nan"))
        seeds.append(sd)
    if not seeds:
        return None
    return {"seeds": seeds, "UR": urs, "SH": shs, "PGR": pgrs}


def main() -> int:
    for strat in STRATA:
        den = POISON_SH[strat] - CLEAN_SH[strat]
        print(f"\n{'=' * 92}")
        print(f"target={strat}   clean N@10 {CLEAN_N10:.5f} / SH@10 "
              f"{CLEAN_SH[strat]:.5f}   poisoned SH@10 {POISON_SH[strat]:.5f}   "
              f"PGR den {den:.5f}")
        print(f"{'arm':<30}{'seeds':>8}{'UR':>18}{'PGR@10':>18}{'SH@10':>20}")
        for label, body in ARMS:
            c = cell(strat, body)
            if not c:
                print(f"{label:<30}{'-':>8}{'not run':>18}")
                continue
            print(f"{label:<30}{','.join(map(str, c['seeds'])):>8}"
                  f"{_pm(c['UR']):>18}{_pm(c['PGR'], '{:.3f}'):>18}"
                  f"{_pm(c['SH'], '{:.5f}'):>20}")
    print("\nUR and PGR must be read together: PGR alone rewards a destroyed "
          "model, UR alone rewards a no-op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
