# %% [markdown]
# # Phase 3: target and split
# Run from the project root:  python notebooks/03_split.py

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.split import OUTLIER, paper_split  # noqa: E402

INK, MUTED = "#222222", "#8a8a85"
SET_COLORS = {"train": "#2a78d6", "test1": "#eb6834", "test2": "#1baf7a"}
SET_NAMES = {"train": "train", "test1": "primary test", "test2": "secondary test\n(batch 3)"}

cells = load_all()
split = paper_split(cells)
rng = np.random.default_rng(0)  # jitter only; fixed seed so the figure is reproducible

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6, axis="y")

# (a) Where each set's cells sit, on the log scale we model in.
ax = axes[0]
for i, (name, ids) in enumerate(split.items()):
    life = np.array([cells[k]["cycle_life"] for k in ids])
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(life)), life, s=26, color=SET_COLORS[name],
               edgecolor="white", linewidth=0.6)
    ax.hlines(np.median(life), i - 0.3, i + 0.3, color=INK, lw=2)
    ax.text(i + 0.33, np.median(life), f"median {np.median(life):.0f}", va="center", fontsize=8.5, color=INK)
ax.scatter([1], [cells[OUTLIER]["cycle_life"]], marker="x", s=50, color=INK)
ax.annotate(f"{OUTLIER} (148 cycles)\nremoved", xy=(1, 148), xytext=(1.35, 175), fontsize=8.5,
            color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_yscale("log")
ax.set_xticks(range(3), [f"{SET_NAMES[n]}\nn={len(v)}" for n, v in split.items()])
ax.set(ylabel="cycle life (log scale)", xlim=(-0.5, 2.9),
       title="(a) The three sets (secondary test cells live longer)")

# (b) Why a log target: the same *relative* error is a very different number of
# cycles for short- and long-lived cells; log turns relative error into a constant.
ax = axes[1]
life = np.array([300, 600, 1200, 2000])
ax.bar(np.arange(4) - 0.2, 0.15 * life, width=0.38, color=SET_COLORS["train"],
       label="a 15% error, in cycles")
ax.bar(np.arange(4) + 0.2, np.full(4, np.log10(1.15)) * 1000, width=0.38, color=SET_COLORS["test1"],
       label="the same 15% error, in log10 units (x1000)")
ax.set_xticks(range(4), [f"life = {L}" for L in life])
ax.set(ylabel="size of error", title="(b) Log target: equal relative errors weigh equally")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase3_split.png", dpi=150)
print("saved figures/phase3_split.png")
