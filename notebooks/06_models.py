# %% [markdown]
# # Phase 6: ElasticNet models
# Run from the project root:  python notebooks/06_models.py
# Fits the final models on the 41 TRAINING cells and freezes them to
# data/processed/models.pkl. Phase 7 only loads and evaluates them: no refitting
# after seeing test data.

# %%
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, enet_path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.features import FEATURE_SETS  # noqa: E402
from src.models import describe, make_elastic_net  # noqa: E402
from src.split import paper_split  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
INK, MUTED, BLUE, ORANGE = "#222222", "#8a8a85", "#2a78d6", "#eb6834"
# Categorical slots, fixed order, for the regularization-path lines.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cd6", "#c43b3b", "#5b6770"]

cells = load_all()
train_ids = paper_split(cells)["train"]
df = pd.read_csv(ROOT / "data" / "processed" / "features.csv", index_col=0).loc[train_ids]
y = df.y.values

# %% 1. Fit and freeze the final models
models = {}
for name, feats in FEATURE_SETS.items():
    models[name] = {"features": feats, "pipeline": make_elastic_net(seed=0).fit(df[feats].values, y)}
    d = describe(models[name]["pipeline"], feats)
    print(f"{name:9} alpha={d['alpha']:.4f} l1_ratio={d['l1_ratio']:.2f} "
          f"kept {d['n_nonzero']}/{len(feats)}: {[f for f, c in d['coefs'].items() if c != 0]}")
with open(ROOT / "data" / "processed" / "models.pkl", "wb") as fp:
    pickle.dump({"models": models, "train_ids": train_ids}, fp)
print("froze models -> data/processed/models.pkl")

# %% 2. Stability: does the choice depend on how the CV folds were shuffled?
print("\nstability over 10 CV shuffles (full model):")
stab = []
for seed in range(10):
    p = make_elastic_net(seed=seed).fit(df[FEATURE_SETS["full"]].values, y)
    d = describe(p, FEATURE_SETS["full"])
    stab.append({"seed": seed, "alpha": d["alpha"], "l1_ratio": d["l1_ratio"],
                 "kept": ", ".join(f for f, c in d["coefs"].items() if c != 0)})
stab = pd.DataFrame(stab)
print(stab.groupby(["l1_ratio", "kept"]).alpha.agg(["count", "min", "max"]).round(4).to_string())

# %% 3. Figure: coefficients, regularization path, CV curve (full model)
feats = FEATURE_SETS["full"]
pipe = models["full"]["pipeline"]
enet = pipe[-1]
Xs = StandardScaler().fit_transform(df[feats].values)
ols = LinearRegression().fit(Xs, y)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)

# (a) Coefficients: OLS vs ElasticNet (standardized, so magnitudes are comparable).
ax = axes[0]
pos = np.arange(len(feats))
ax.barh(pos + 0.2, ols.coef_, height=0.38, color=MUTED, label="OLS (no penalty)")
ax.barh(pos - 0.2, enet.coef_, height=0.38, color=BLUE, label="ElasticNet")
for i, c in enumerate(enet.coef_):
    if c == 0:
        ax.text(0.002, i - 0.2, "0", va="center", fontsize=8, color=INK)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(pos, feats)
ax.invert_yaxis()
ax.set(xlabel="coefficient (per 1 std of feature, in log10 cycles)",
       title="(a) Full model: penalty shrinks or zeroes coefficients")
ax.legend(frameon=False, fontsize=9, loc="lower right")

# (b) Regularization path at the chosen l1_ratio: how each coefficient grows as the
# penalty is relaxed (right to left = weaker penalty).
ax = axes[1]
alphas, coefs, _ = enet_path(Xs, y - y.mean(), l1_ratio=enet.l1_ratio_, alphas=enet.alphas_[
    list(enet.l1_ratio).index(enet.l1_ratio_)] if np.ndim(enet.alphas_) > 1 else enet.alphas_)
for i, f in enumerate(feats):
    ax.plot(alphas, coefs[i], color=CAT[i], lw=1.8, label=f)
ax.axvline(enet.alpha_, color=INK, ls="--", lw=1)
ax.text(enet.alpha_ * 0.9, 0.02, f"chosen α = {enet.alpha_:.3f}", fontsize=8.5,
        transform=ax.get_xaxis_transform())  # x in data units, y in axes fraction
ax.set(xscale="log", xlabel="penalty strength α (log scale)", ylabel="coefficient",
       title=f"(b) Regularization path (l1_ratio = {enet.l1_ratio_})")
ax.invert_xaxis()
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)

# (c) CV error vs alpha for each l1_ratio: the curve CV minimised to pick alpha.
ax = axes[2]
mse = enet.mse_path_.mean(axis=-1)  # (n_l1, n_alpha)
for j, l1 in enumerate(enet.l1_ratio):
    if l1 in (0.01, 0.3, 0.7, 1.0):
        ax.plot(enet.alphas_[j], np.sqrt(mse[j]), lw=1.6, label=f"l1_ratio = {l1}",
                color=CAT[[0.01, 0.3, 0.7, 1.0].index(l1)])
ax.axvline(enet.alpha_, color=INK, ls="--", lw=1)
ax.set(xscale="log", xlabel="penalty strength α (log scale)",
       ylabel="CV RMSE (log10 cycles)", title="(c) CV error: too little vs too much penalty")
ax.invert_xaxis()
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase6_regularization.png", dpi=150)
print("\nsaved figures/phase6_regularization.png")
