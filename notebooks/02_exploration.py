# %% [markdown]
# # Phase 2: exploration
# Run from the project root:  python notebooks/02_exploration.py

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import EOL_CAPACITY, load_all  # noqa: E402

FIG = ROOT / "figures"
INK, MUTED = "#222222", "#8a8a85"
BATCH_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
# One hue, light -> dark: magnitude (cycle life) should read as "more", not as a category.
LIFE_CMAP = LinearSegmentedColormap.from_list("life", ["#b7d3f6", "#3987e5", "#0d366b"])

cells = load_all()
life = np.array([c["cycle_life"] for c in cells.values()])
batch = np.array([c["batch"] for c in cells.values()])
norm = LogNorm(life.min(), life.max())


def capacity(cell):
    """Cycle numbers and discharge capacity, with the QD=0 'missing cycle'
    placeholders turned into NaN so they don't plot as a drop to zero."""
    qd = cell["summary"]["QD"].astype(float).copy()
    qd[qd <= 0] = np.nan
    return cell["summary"]["cycle"], qd


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


# %% 1. Capacity vs cycle for every cell
fig, ax = plt.subplots(figsize=(10, 5.5))
style(ax)
# Draw long-lived (dark) cells last so the short-lived ones don't get buried.
for k in sorted(cells, key=lambda k: cells[k]["cycle_life"]):
    cyc, qd = capacity(cells[k])
    ax.plot(cyc, qd, color=LIFE_CMAP(norm(cells[k]["cycle_life"])), lw=0.9, alpha=0.9)
ax.axhline(EOL_CAPACITY, color=INK, ls="--", lw=1)
ax.text(2250, EOL_CAPACITY + 0.004, "end of life: 0.88 Ah", ha="right", fontsize=9, color=INK)
ax.set(xlim=(0, 2300), ylim=(0.86, 1.10), xlabel="cycle number",
       ylabel="discharge capacity (Ah)",
       title="Discharge capacity of all 124 cells (4 single-cycle glitches above 1.10 Ah not shown)")
sm = plt.cm.ScalarMappable(norm=norm, cmap=LIFE_CMAP)
cb = fig.colorbar(sm, ax=ax, pad=0.01)
cb.set_label("cycle life")
fig.tight_layout()
fig.savefig(FIG / "phase2_capacity_all.png", dpi=150)

# %% 2. Distribution of cycle life (linear and log scale), by batch
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, use_log in zip(axes, (False, True)):
    style(ax)
    vals = np.log10(life) if use_log else life
    bins = np.linspace(vals.min(), vals.max(), 21)
    ax.hist([vals[batch == b] for b in (1, 2, 3)], bins=bins, stacked=True,
            color=[BATCH_COLORS[b] for b in (1, 2, 3)], edgecolor="white", linewidth=1,
            label=[f"batch {b} (n={np.sum(batch == b)}, median {np.median(life[batch == b]):.0f})"
                   for b in (1, 2, 3)])
    ax.set(ylabel="number of cells",
           xlabel="log10(cycle life)" if use_log else "cycle life (cycles)",
           title=("(b) log10 scale: more symmetric" if use_log
                  else "(a) Linear scale: long right tail"))
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "phase2_life_histogram.png", dpi=150)

# %% 3. What the model actually sees: the first 100 cycles
qd2 = np.array([capacity(c)[1][1] for c in cells.values()])      # cycle 2 (index 1)
qd100 = np.array([capacity(c)[1][99] for c in cells.values()])   # cycle 100 (index 99)
fade = qd100 - qd2

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax in axes:
    style(ax)
ax = axes[0]
for k in sorted(cells, key=lambda k: cells[k]["cycle_life"]):
    cyc, qd = capacity(cells[k])
    ax.plot(cyc[:100], qd[:100], color=LIFE_CMAP(norm(cells[k]["cycle_life"])), lw=0.9)
ax.set(xlim=(1, 100), ylim=(0.95, 1.11), xlabel="cycle number", ylabel="discharge capacity (Ah)",
       title="(a) First 100 cycles: almost flat\n(b1c18's 2.88 Ah spike at cycle 40 off-scale)")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=LIFE_CMAP), ax=ax, pad=0.01).set_label("cycle life")

for ax, x, xlabel, title in [
    (axes[1], qd100, "capacity at cycle 100 (Ah)", "(b) Capacity at cycle 100 vs life"),
    (axes[2], fade * 1000, "capacity change, cycle 2 to 100 (mAh)", "(c) Early fade vs life"),
]:
    for b in (1, 2, 3):
        m = batch == b
        ax.scatter(x[m], life[m], s=28, color=BATCH_COLORS[b], edgecolor="white",
                   linewidth=0.6, label=f"batch {b}")
    r = stats.pearsonr(x, np.log10(life)).statistic
    ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel="cycle life (log scale)", title=f"{title}\nr = {r:.2f}")
axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "phase2_first100.png", dpi=150)

# %% Summary numbers
print(f"cycle life: min {life.min():.0f}, median {np.median(life):.0f}, mean {life.mean():.0f}, max {life.max():.0f}")
print(f"  skewness of life: {stats.skew(life):.2f}   of log10(life): {stats.skew(np.log10(life)):.2f}")
print(f"  max/min ratio: {life.max() / life.min():.1f}x")
for b in (1, 2, 3):
    m = batch == b
    print(f"  batch {b}: n={m.sum()}, median {np.median(life[m]):.0f}, range {life[m].min():.0f}-{life[m].max():.0f}")
print(f"capacity change cycle 2->100: median {np.median(fade)*1000:.1f} mAh "
      f"({np.median(fade / qd2) * 100:.2f}% of capacity); range {fade.min()*1000:.1f} to {fade.max()*1000:.1f} mAh")
print(f"cells whose capacity INCREASED over cycles 2-100: {(fade > 0).sum()} of {len(fade)}")
print(f"Pearson r with log10(life): QD100 {stats.pearsonr(qd100, np.log10(life)).statistic:.2f}, "
      f"fade {stats.pearsonr(fade, np.log10(life)).statistic:.2f}")
