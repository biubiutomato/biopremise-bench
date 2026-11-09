#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results" / "final_summary.json").read_text())
llms = summary["llms"]

fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.6))

settings = [
    ("gpt-4o-mini|default", "4o-mini\nDefault"),
    ("gpt-4o-mini|verify", "4o-mini\nVerify"),
    ("gpt-4o|default", "4o\nDefault"),
    ("gpt-4o|verify", "4o\nVerify"),
]
keys = [("PDR", "#4C78A8"), ("FPCR", "#E45756"), ("FAR", "#B279A2")]
ax = axes[0]
x = range(len(settings))
width = 0.24
for i, (key, color) in enumerate(keys):
    vals = [llms[s][key] for s, _ in settings]
    ax.bar([xi + (i - 1) * width for xi in x], vals, width=width, label=key, color=color, edgecolor="white", linewidth=0.4)
ax.set_xticks(list(x))
ax.set_xticklabels([lab for _, lab in settings])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Rate")
ax.set_title("Detection vs compliance vs false alarm")
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
order = [
    ("wrong_drug_class", "Drug class"),
    ("wrong_gene_function", "Gene fn."),
    ("wrong_drug_target", "Drug target"),
    ("wrong_genomic_location", "Locus"),
    ("nonexistent_pathway", "Fake path."),
    ("nonexistent_gene", "Fake gene"),
]
vals4 = [llms["gpt-4o|default"]["PDR_by_type"][k] for k, _ in order]
valsmini = [llms["gpt-4o-mini|default"]["PDR_by_type"][k] for k, _ in order]
idx = range(len(order))
ax.bar([i - 0.18 for i in idx], valsmini, width=0.36, label="4o-mini Def.", color="#9D755D", edgecolor="white")
ax.bar([i + 0.18 for i in idx], vals4, width=0.36, label="4o Def.", color="#54A24B", edgecolor="white")
ax.set_xticks(list(idx))
ax.set_xticklabels([lab for _, lab in order], rotation=20, ha="right")
ax.set_ylim(0, 1.08)
ax.set_ylabel("PDR (default prompt)")
ax.set_title("Detection by error type")
ax.legend(frameon=False, fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.35)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig_baselines.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures" / "fig_baselines.png", dpi=200, bbox_inches="tight")
print("updated figure")
