"""
Generate eval artifacts for the README:
  - eval/eval_summary.csv : one row per eval question, all four metric scores
  - eval/charts/per_row_metrics.png : per-row bar chart of all four metrics
  - eval/charts/summary_bars.png : the four mean scores as a headline chart

Inputs 
  - recall and precision: computed deterministically (mechanical) on real chunks
  - faithfulness and answer_relevancy: scored by RAGAS
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = Path(__file__).parent
CHART_DIR = OUT_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

# Actual Scores from my run 
ROWS = [
    # id,         difficulty,         recall, precision, faithfulness, relevancy
    ("mh_001",    "easy_lookup",      1.00, 0.333, 1.000, 0.903),
    ("mh_002",    "easy_lookup",      1.00, 0.333, 1.000, 0.952),
    ("mh_003",    "specific_detail",  1.00, 0.333, 1.000, 0.991),
    ("mh_004",    "specific_detail",  1.00, 0.667, 1.000, 0.930),
    ("mh_005",    "specific_detail",  1.00, 0.333, 1.000, 0.896),
    ("mh_006",    "multi_hop",        1.00, 0.333, 1.000, 0.832),
    ("mh_007",    "specific_detail",  1.00, 0.333, 1.000, 0.924),
    ("mh_008",    "easy_lookup",      1.00, 0.333, 1.000, 0.865),
    ("mh_009",    "specific_detail",  1.00, 0.333, 1.000, 0.956),
    ("mh_010",    "multi_hop",        1.00, 0.333, 1.000, 0.933),
    ("cv_001",    "specific_detail",  1.00, 0.333, 1.000, 0.926),
    ("cv_002",    "specific_detail",  1.00, 0.333, 1.000, 0.955),
    ("cv_003",    "specific_detail",  1.00, 0.333, 0.667, 0.941),
    ("cv_004",    "easy_lookup",      1.00, 0.333, 1.000, 0.776),
    ("cv_005",    "easy_lookup",      1.00, 0.333, 1.000, 0.941),
    ("cv_006",    "specific_detail",  1.00, 0.333, 1.000, 0.844),
    ("cv_007",    "specific_detail",  1.00, 0.333, 1.000, 0.922),
    ("cross_001", "multi_hop",        0.50, 0.667, 1.000, 0.000),
    ("neg_001",   "specific_detail",  None, None,  1.000, 0.000),  # negative test: no gold
]
# fmt: on

# --- write CSV ---
csv_path = OUT_DIR / "eval_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "difficulty", "recall@k", "precision@k", "faithfulness", "answer_relevancy"])
    for rid, diff, r, p, fa, ar in ROWS:
        w.writerow([
            rid, diff,
            "" if r is None else f"{r:.3f}",
            "" if p is None else f"{p:.3f}",
            f"{fa:.3f}",
            f"{ar:.3f}",
        ])

# compute means (skipping None for recall/precision on neg_001) ---
def mean(vals): return sum(vals) / len(vals)
recall_mean    = mean([r for _,_,r,_,_,_ in ROWS if r is not None])
precision_mean = mean([p for _,_,_,p,_,_ in ROWS if p is not None])
faith_mean     = mean([fa for _,_,_,_,fa,_ in ROWS])
relev_mean     = mean([ar for _,_,_,_,_,ar in ROWS])

# also a "relevancy excluding refusals" mean — the more honest number
relev_non_refusal = mean([
    ar for rid,_,_,_,_,ar in ROWS if rid not in ("neg_001", "cross_001")
])

print(f"recall@k        mean = {recall_mean:.3f}")
print(f"precision@k     mean = {precision_mean:.3f}")
print(f"faithfulness    mean = {faith_mean:.3f}")
print(f"answer_relevancy mean (all) = {relev_mean:.3f}")
print(f"answer_relevancy mean (excl. refusals) = {relev_non_refusal:.3f}")

# CHART 1: per-row bars, all four metrics
ids        = [r[0] for r in ROWS]
recall     = [r[2] if r[2] is not None else 0 for r in ROWS]
precision  = [r[3] if r[3] is not None else 0 for r in ROWS]
faith      = [r[4] for r in ROWS]
relev      = [r[5] for r in ROWS]
# mark recall/precision N/A for neg_001 visually
recall_na_mask    = [r[2] is None for r in ROWS]
precision_na_mask = [r[3] is None for r in ROWS]

x = range(len(ids))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 6))
b1 = ax.bar([i - 1.5*width for i in x], recall,    width, label="recall@k",       color="#3b82f6")
b2 = ax.bar([i - 0.5*width for i in x], precision, width, label="precision@k",    color="#60a5fa")
b3 = ax.bar([i + 0.5*width for i in x], faith,     width, label="faithfulness",   color="#10b981")
b4 = ax.bar([i + 1.5*width for i in x], relev,     width, label="answer relevancy", color="#34d399")

# annotate N/A bars (neg_001 recall/precision)
for i, na in enumerate(recall_na_mask):
    if na:
        ax.text(i - 1.5*width, 0.02, "N/A", ha="center", fontsize=7, color="#666")
for i, na in enumerate(precision_na_mask):
    if na:
        ax.text(i - 0.5*width, 0.02, "N/A", ha="center", fontsize=7, color="#666")

ax.set_xticks(list(x))
ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 1.25)
ax.set_ylabel("Score (0\u20131)")
ax.set_title("DocuMind RAG Eval — Per-row metric scores (k=3, 19 rows)", pad=30)
ax.legend(loc="lower left", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# annotations on the interesting rows
def annotate(row_id, text):
    i = ids.index(row_id)
    ax.annotate(text, xy=(i, 1.05), xytext=(i, 1.16), ha="center",
                fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))

annotate("cv_003",    "ungrounded\nelaboration")
annotate("cross_001", "retrieval\ngap")
annotate("neg_001",   "refusal\n(metric limit)")

plt.tight_layout()
fig.savefig(CHART_DIR / "per_row_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# CHART 2: summary mean bars 
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["recall@k", "precision@k", "faithfulness", "answer relevancy\n(excl. refusals)"]
values  = [recall_mean, precision_mean, faith_mean, relev_non_refusal]
colors  = ["#3b82f6", "#60a5fa", "#10b981", "#34d399"]

bars = ax.bar(metrics, values, color=colors)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
            ha="center", fontsize=11, fontweight="bold")

ax.set_ylim(0, 1.15)
ax.set_ylabel("Mean score")
ax.set_title("DocuMind RAG Eval \u2014 Summary (19 rows, k=3)")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(CHART_DIR / "summary_bars.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nWrote:")
print(f"  {csv_path}")
print(f"  {CHART_DIR / 'per_row_metrics.png'}")
print(f"  {CHART_DIR / 'summary_bars.png'}")