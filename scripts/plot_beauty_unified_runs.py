"""Bar chart of recall@5 for the beauty runs recorded in WORKFLOW.md.

Includes the clean reference (Retrained), the poisoned model (Initial), and
the unified unlearning runs after the 'same but with right starting
checkpoint' comment (jobs 9175186-9184450).

Usage: python scripts/plot_beauty_unified_runs.py [out.png]
"""

import sys

import matplotlib.pyplot as plt


def unified_label(lambda_forget=1.0, lambda_sep=0.1, sep_negatives="neighbors"):
    return (
        f"Unified lambda_forget {lambda_forget} "
        f"lambda_sep {lambda_sep} sep_negatives {sep_negatives}"
    )


# (label, recall@5, job id) — values from WORKFLOW.md, beauty dataset
RUNS = [
    ("Retrained", 0.0453, 9096928),  # clean training
    ("Initial", 0.0444, 9096933),  # poisoned training
    (unified_label(lambda_forget=0.1), 0.04391, 9175186),
    (unified_label(lambda_forget=0.0, lambda_sep=0.1), 0.04467, 9175189),
    (unified_label(lambda_forget=0.0, lambda_sep=0.0), 0.04458, 9175190),
    (unified_label(lambda_forget=0.0, lambda_sep=1.0), 0.04494, 9175196),
    (
        unified_label(lambda_forget=0.0, lambda_sep=1.0, sep_negatives="random_retain"),
        0.04490,
        9184450,
    ),
]


def main(out_path="beauty_unified_recall5.png"):
    labels = [r[0] for r in RUNS]
    values = [r[1] for r in RUNS]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:green", "tab:red"] + ["tab:blue"] * (len(RUNS) - 2)
    bars = ax.bar(range(len(RUNS)), values, color=colors)

    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Recall@5")
    ax.set_title("Beauty: recall@5 by run")
    ax.margins(y=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
