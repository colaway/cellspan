# %% [markdown]
# # Phase 4: the Delta-Q(V) feature
# Run from the project root:  python notebooks/04_features.py
# Only TRAINING cells are plotted: looking at test cells while designing features
# would be a (small) leak.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.features import build_feature_table, delta_q, early_window  # noqa: E402
from src.split import paper_split  # noqa: E402

INK, MUTED, BLUE = "#222222", "#8a8a85", "#2a78d6"
LIFE_CMAP = LinearSegmentedColormap.from_list("life", ["#b7d3f6", "#3987e5", "#0d366b"])

cells = load_all()
train = paper_split(cells)["train"]
df = build_feature_table(cells, train)
norm = LogNorm(df.cycle_life.min(), df.cycle_life.max())
V = cells[train[0]]["Vdlin"]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)

# (a) Delta-Q curves: short-lived cells (light) dip much deeper.
ax = axes[0]
for k in df.sort_values("cycle_life").index:
    ax.plot(V, delta_q(early_window(cells[k])) * 1000, color=LIFE_CMAP(norm(df.cycle_life[k])), lw=1)
ax.axhline(0, color=MUTED, lw=0.8)
ax.invert_xaxis()
ax.set(xlabel="voltage (V)", ylabel="Q$_{100}$(V) - Q$_{10}$(V)  (mAh)",
       title="(a) ΔQ(V) for the 41 training cells")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=LIFE_CMAP), ax=ax, pad=0.01).set_label("cycle life")

# (b) The headline relationship: one number per cell, nearly a straight line.
ax = axes[1]
x, y = df.dq_log_var.values, df.cycle_life.values
fit = stats.linregress(x, np.log10(y))
ax.scatter(x, y, s=34, color=BLUE, edgecolor="white", linewidth=0.6)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, 10 ** (fit.intercept + fit.slope * xs), color=INK, lw=1.5)
ax.set_yscale("log")
ax.set(xlabel="log10 var(ΔQ$_{100-10}$(V))", ylabel="cycle life (log scale)",
       title=f"(b) log variance vs life: r = {fit.rvalue:.2f}")
ax.text(0.97, 0.95, f"slope = {fit.slope:.2f}\n(life ∝ var$^{{{fit.slope:.2f}}}$)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)

# (c) For contrast: the capacity feature from Phase 2 on the same cells.
ax = axes[2]
qd100 = np.array([cells[k]["summary"]["QD"][99] for k in df.index])
r = stats.pearsonr(qd100, np.log10(y)).statistic
ax.scatter(qd100, y, s=34, color=MUTED, edgecolor="white", linewidth=0.6)
ax.set_yscale("log")
ax.set(xlabel="discharge capacity at cycle 100 (Ah)", ylabel="cycle life (log scale)",
       title=f"(c) For contrast, capacity at cycle 100: r = {r:.2f}")

fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase4_delta_q.png", dpi=150)
print(f"saved; log-var fit on train: slope {fit.slope:.3f}, intercept {fit.intercept:.3f}, r {fit.rvalue:.3f}")
dq_min_v = [V[np.argmin(delta_q(early_window(cells[k])))] for k in df.index]
print(f"voltage of the deepest dip: median {np.median(dq_min_v):.2f} V, range {min(dq_min_v):.2f}-{max(dq_min_v):.2f} V")
