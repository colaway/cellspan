# %% [markdown]
# # Phase 5: baseline
# Run from the project root:  python notebooks/05_baseline.py
# Uses TRAINING cells only. Test sets are evaluated once, in Phase 7.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.evaluate import loocv_predict, score  # noqa: E402
from src.split import from_target, paper_split  # noqa: E402

INK, MUTED, BLUE, ORANGE = "#222222", "#8a8a85", "#2a78d6", "#eb6834"

cells = load_all()
train_ids = paper_split(cells)["train"]
df = pd.read_csv(ROOT / "data" / "processed" / "features.csv", index_col=0).loc[train_ids]
X, y, life = df[["dq_log_var"]].values, df.y.values, df.cycle_life.values

models = {
    "predict the mean": DummyRegressor(strategy="mean"),  # ignores X entirely
    "linear on log var(ΔQ)": LinearRegression(),
}
rows, loo_preds = [], {}
for name, model in models.items():
    model.fit(X, y)
    loo_preds[name] = loocv_predict(model, X, y)
    tr, cv = score(life, model.predict(X)), score(life, loo_preds[name])
    rows.append({"model": name, "train RMSE": tr["rmse"], "train MAPE %": tr["mape"],
                 "LOOCV RMSE": cv["rmse"], "LOOCV MAPE %": cv["mape"]})
results = pd.DataFrame(rows).set_index("model")
print(results.round(1).to_string())

lin = models["linear on log var(ΔQ)"]
print(f"\nfitted line: log10(life) = {lin.intercept_:.3f} + ({lin.coef_[0]:.3f}) * log10 var(ΔQ)")
results.to_csv(ROOT / "data" / "processed" / "phase5_baseline.csv")

# %% Diagnostics
loo = from_target(loo_preds["linear on log var(ΔQ)"])
pct_err = (loo - life) / life * 100
worst = np.argsort(-np.abs(pct_err))[:4]
print("\nworst LOOCV predictions:")
for i in worst:
    print(f"  {df.index[i]}: true {life[i]:.0f}, predicted {loo[i]:.0f} ({pct_err[i]:+.0f}%), "
          f"policy {cells[df.index[i]]['charge_policy']}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)

# (a) Parity plot: predicted vs true. Perfect predictions lie on the diagonal.
ax = axes[0]
lims = [250, 2500]
ax.plot(lims, lims, color=INK, lw=1)
ax.fill_between(lims, [l * 0.85 for l in lims], [l * 1.15 for l in lims], color=MUTED, alpha=0.15,
                label="within ±15%")
ax.scatter(life, from_target(loo_preds["predict the mean"]), s=26, marker="s", color=MUTED,
           label=f"predict the mean (LOOCV MAPE {results.iloc[0]['LOOCV MAPE %']:.0f}%)")
ax.scatter(life, loo, s=34, color=BLUE, edgecolor="white", linewidth=0.6,
           label=f"linear on log var (LOOCV MAPE {results.iloc[1]['LOOCV MAPE %']:.0f}%)")
ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, xlabel="true cycle life",
       ylabel="predicted cycle life (LOOCV)", title="(a) Parity plot, training cells (leave-one-out)")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

# (b) Residuals vs true life: any trend means the model is systematically biased somewhere.
ax = axes[1]
ax.axhline(0, color=INK, lw=1)
# Colour by batch: the residual pattern turned out to be a batch effect.
for b, color in ((1, BLUE), (2, ORANGE)):
    m = df.batch.values == b
    ax.scatter(life[m], pct_err[m], s=34, color=color, edgecolor="white", linewidth=0.6,
               label=f"batch {b}: mean error {pct_err[m].mean():+.0f}%")
ax.legend(frameon=False, fontsize=9)
ax.set(xscale="log", xlabel="true cycle life", ylabel="prediction error (%)",
       title="(b) Residuals: batch 1 under-, batch 2 over-predicted")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase5_baseline.png", dpi=150)
print("\nsaved figures/phase5_baseline.png")
