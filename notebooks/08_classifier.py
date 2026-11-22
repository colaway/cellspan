# %% [markdown]
# # Phase 8a: classify short vs long life from the first 5 cycles
# Run from the project root:  python notebooks/08_classifier.py
#
# The paper's third model (Table 2): instead of predicting the exact cycle life, answer
# a yes/no question, "will this cell last fewer than 550 cycles?", using cycles 1-5 only.
# That is the screening use case: reject bad cells after minutes of testing, not weeks.
#
# Honesty note: the test sets were used in Phase 7. These are NEW models, so their
# test scores are a first look, but we chose the features and settings below BEFORE
# running anything on the test cells, and we don't revise them afterwards.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.models import make_classifier  # noqa: E402
from src.split import OUTLIER, paper_split  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
INK, MUTED, BLUE, ORANGE = "#222222", "#8a8a85", "#2a78d6", "#eb6834"
THRESHOLD = 550  # cycles, as in the paper

# Paper Table 2 (nature.com/articles/s41560-019-0356-8/tables/2), accuracy in %.
PAPER = {"variance": {"train": 82.1, "test1": 78.6, "test2": 97.5},
         "full": {"train": 97.4, "test1": 92.7, "test2": 97.5}}

# Features from cycles 2-5 only. Delta-Q is now Q_5(V) - Q_4(V): the latest pair of
# cycles available (cycle 1 is missing in batch 1, so we can't reach further back
# without losing the window's last cycle as reference).
# The paper's exact classifier feature list is in its paywalled Methods; this is OUR
# choice: the full-model features that still make sense over 4 cycles. Fade slope /
# intercept are dropped: a line through 4 nearly flat points is mostly noise.
SETS = {"variance": ["dq_log_var"],
        "full": ["dq_log_var", "dq_log_min", "q2", "chargetime_mean", "tmax", "ir_min", "ir_diff"]}

cells = load_all()
split = paper_split(cells, drop_outlier=False)
df = build_feature_table(cells, first_cycle=4, last_cycle=5)
df["short"] = (df.cycle_life < THRESHOLD).astype(int)  # 1 = short-lived (the "bad" class)
df.to_csv(ROOT / "data" / "processed" / "features_cycle5.csv")
sets = {"train": [k for k in split["train"] if k != OUTLIER], "test1": split["test1"],
        "test1 -b2c1": [k for k in split["test1"] if k != OUTLIER], "test2": split["test2"]}
for s, ids in sets.items():
    print(f"{s:12} n={len(ids):2}  short (<{THRESHOLD}): {df.loc[ids, 'short'].sum()}")

tr = df.loc[sets["train"]]
print("\ntrain correlation with log10 life (cycles 2-5 features):")
print(tr[SETS["full"]].corrwith(tr.y).round(2).to_string())

# %% 1. Fit on train, nested CV on train, then score the test sets once
rows, probs = [], pd.DataFrame(index=df.index)
for name, feats in SETS.items():
    model = make_classifier().fit(tr[feats].values, tr.short.values)
    probs[name] = model.predict_proba(df[feats].values)[:, 1]  # P(short)
    nested = cross_val_predict(make_classifier(), tr[feats].values, tr.short.values,
                               cv=LeaveOneOut())  # outer LOO, inner CV picks C
    row = {"model": name, "C": float(model[-1].C_), "nested CV": 100 * accuracy_score(tr.short, nested)}
    for s, ids in sets.items():
        pred = (probs.loc[ids, name] > 0.5).astype(int)
        row[s] = 100 * accuracy_score(df.loc[ids, "short"], pred)
        if s in ("test1", "test2"):
            tn, fp, fn, tp = confusion_matrix(df.loc[ids, "short"], pred, labels=[0, 1]).ravel()
            row[f"{s} errors"] = f"{fp} long called short, {fn} short called long"
    rows.append(row)
res = pd.DataFrame(rows).set_index("model")
res.to_csv(ROOT / "data" / "processed" / "phase8_classifier.csv")
pd.set_option("display.width", 200)
print("\n" + res.round(1).to_string())
print("\npaper (Table 2):", PAPER)

# Which cells are misclassified, and how far are they from the threshold?
for s in ("test1", "test2"):
    ids = sets[s]
    wrong = [k for k in ids if (probs.loc[k, "full"] > 0.5) != df.loc[k, "short"]]
    print(f"full, {s} misclassified:",
          {k: f"life {df.loc[k, 'cycle_life']:.0f}, P(short)={probs.loc[k, 'full']:.2f}" for k in wrong})

# %% 2. Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
style = {"train": (MUTED, "o", "train"), "test1": (BLUE, "s", "primary test"),
         "test2": (ORANGE, "^", "secondary test")}
for s, (c, mk, lab) in style.items():
    ids = sets[s]
    axes[0].scatter(df.loc[ids, "cycle_life"], df.loc[ids, "dq_log_var"], color=c, marker=mk,
                    s=26, alpha=0.85, edgecolor="white", linewidth=0.5, label=lab)
    axes[1].scatter(df.loc[ids, "cycle_life"], probs.loc[ids, "full"], color=c, marker=mk,
                    s=26, alpha=0.85, edgecolor="white", linewidth=0.5, label=lab)
for ax in axes:
    ax.axvline(THRESHOLD, color=INK, ls="--", lw=1)
    ax.set(xscale="log", xlabel="true cycle life")
axes[0].set(ylabel="log10 var ΔQ$_{5-4}$(V)",
            title="(a) ΔQ from cycles 4→5: weaker but real signal (train r = −0.63)")
axes[1].axhline(0.5, color=INK, lw=0.8)
axes[1].set(ylabel="P(short-lived), full classifier",
            title="(b) Correct = top-left or bottom-right of the crosshair")
axes[1].legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase8_classifier.png", dpi=150)
print("\nsaved figures/phase8_classifier.png")

# %% 3. Diagnostic: how much of this is just the charging policy?
# chargetime_mean has r = 0.77 with life: faster charging = shorter life. If charge time
# alone classifies nearly as well, the "full" classifier is mostly reading the test
# protocol, not the cell's health. (Diagnostic only; not used to change the models above.)
ct = make_classifier().fit(tr[["chargetime_mean"]].values, tr.short.values)
for s in ("train", "test1", "test2"):
    ids = sets[s]
    acc = 100 * accuracy_score(df.loc[ids, "short"], ct.predict(df.loc[ids, ["chargetime_mean"]].values))
    print(f"charge time only, {s}: {acc:.1f}%")
print("always predict 'long', test2:", f"{100 * (1 - df.loc[sets['test2'], 'short'].mean()):.1f}%")
