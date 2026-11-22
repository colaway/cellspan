# %% [markdown]
# # Phase 1: what one cell's data looks like
# Run from the project root:  python notebooks/01_data_structure.py
# (The "# %%" markers make this runnable cell-by-cell in VS Code / Jupyter too.)

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"

cells = load_all()
cell = cells["b1c5"]
V_grid = cell["Vdlin"]

# Cycle N lives at key str(N-1): key '0' is cycle 1.
c10, c100 = cell["cycles"]["9"], cell["cycles"]["99"]

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)

# (a) One full cycle: fast charge, rest, then 4C discharge down to 2.0 V.
ax = axes[0]
def longest_run(mask):
    """Boolean mask of the longest contiguous True stretch.
    'Current < 0' alone is not the discharge: the internal-resistance test
    (~12 min) fires short negative pulses too. The real 4C discharge is the
    one long unbroken stretch of negative current."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    i = np.argmax(ends - starts)
    out = np.zeros_like(mask)
    out[starts[i]:ends[i]] = True
    return out


ax.plot(c10["t"], c10["V"], color=BLUE, lw=2)
dis = longest_run(c10["I"] < 0)
ax.axvspan(c10["t"][dis].min(), c10["t"][dis].max(), color=MUTED, alpha=0.15)
for x, label in [(5.5, "fast charge\n4.4C to 80%"), (17, "1C\nto 3.6 V"), (26.5, "hold\n3.6 V"),
                 (c10["t"][dis].mean(), "discharge\n4C to 2.0 V")]:
    ax.text(x, 2.25, label, ha="center", va="bottom", color=INK, fontsize=8)
ax.annotate("IR test pulses", xy=(12, 3.3), xytext=(14, 2.9), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set(xlabel="time within cycle (min)", ylabel="voltage (V)",
       title="(a) Cycle 10 of b1c5: voltage over one cycle")

# (b) Raw discharge samples (uneven, cell-specific) vs. Qdlin (fixed grid).
ax = axes[1]
ax.plot(c10["Qd"][dis], c10["V"][dis], "o", ms=3, color=MUTED, label="raw samples (Qd, V)")
ax.plot(c10["Qdlin"], V_grid, color=BLUE, lw=2, label="Qdlin on 1000-pt voltage grid")
ax.set(xlabel="discharge capacity (Ah)", ylabel="voltage (V)",
       title="(b) Qdlin = Q(V) on a shared voltage grid")
ax.legend(frameon=False, fontsize=9, loc="lower left")

# (c) Because every cycle shares the grid, curves can be subtracted point by point.
ax = axes[2]
ax.plot(V_grid, c10["Qdlin"], color=BLUE, lw=2, label="cycle 10")
ax.plot(V_grid, c100["Qdlin"], color=ORANGE, lw=2, label="cycle 100")
ax.set(xlabel="voltage (V)", ylabel="Qdlin (Ah)",
       title="(c) Cycles 10 and 100 nearly overlap")
ax.invert_xaxis()  # discharge runs from high voltage to low
ax.legend(frameon=False, fontsize=9, loc="upper left")
ins = ax.inset_axes([0.5, 0.12, 0.45, 0.4])
ins.plot(V_grid, c100["Qdlin"] - c10["Qdlin"], color=INK, lw=1.5)
ins.axhline(0, color=MUTED, lw=0.8)
ins.invert_xaxis()
ins.set_title("difference: Q100 - Q10 (Ah)", fontsize=8)
ins.tick_params(labelsize=7)

fig.tight_layout()
out = ROOT / "figures" / "phase1_cell_structure.png"
fig.savefig(out, dpi=150)
print("saved", out)

# %% Summary numbers
dq = c100["Qdlin"] - c10["Qdlin"]
print(f"raw discharge samples in cycle 10: {dis.sum()}")
print(f"Qdlin length: {len(c10['Qdlin'])}, V grid {V_grid[0]:.2f} -> {V_grid[-1]:.2f} V")
print(f"capacity at cycle 10: {c10['Qdlin'].max():.4f} Ah, at 100: {c100['Qdlin'].max():.4f} Ah")
print(f"Delta Q(V): min {dq.min():.4f} Ah, max {dq.max():.4f} Ah, variance {dq.var():.2e}")
