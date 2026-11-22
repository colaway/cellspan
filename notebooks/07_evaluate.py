# %% [markdown]
# # Phase 7: first (and only) evaluation on the test sets
# Run from the project root:  python notebooks/07_evaluate.py
#
# Rules for this phase:
# - Load the models frozen in Phase 6. NO refitting, NO changing features or alphas.
#   If we tweaked anything after seeing these numbers, they would stop being an honest
#   estimate of performance on new cells (the test set would become training data).
# - Report every number, good or bad. Picking the flattering ones is also a leak.

# %%
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.evaluate import score  # noqa: E402
from src.split import OUTLIER, from_target, paper_split  # noqa: E402

INK, MUTED, BLUE, ORANGE, GREEN = "#222222", "#8a8a85", "#2a78d6", "#eb6834", "#1baf7a"

# Paper's Table 1 (read from nature.com/articles/s41560-019-0356-8/tables/1).
# Parentheses in the paper = primary test WITHOUT the outlier cell.
PAPER = pd.DataFrame(
    {"train RMSE": [103, 76, 51], "test1 RMSE": [138, 91, 118], "test1 -b2c1 RMSE": [138, 86, 100],
     "test2 RMSE": [196, 173, 214],
     "train MAPE %": [14.1, 9.8, 5.6], "test1 MAPE %": [14.7, 13.0, 14.1],
     "test1 -b2c1 MAPE %": [13.2, 10.1, 7.5], "test2 MAPE %": [11.4, 8.6, 10.7]},
    index=["variance", "discharge", "full"])

cells = load_all()
split = paper_split(cells, drop_outlier=False)  # keep b2c1 so we can report both ways
with open(ROOT / "data" / "processed" / "models.pkl", "rb") as fp:
    frozen = pickle.load(fp)
# Sanity check: the frozen models were trained on exactly the cells we call "train" now.
assert frozen["train_ids"] == [k for k in split["train"] if k != OUTLIER]
df = pd.read_csv(ROOT / "data" / "processed" / "features.csv", index_col=0)

sets = {
    "train": frozen["train_ids"],
    "test1": split["test1"],                                   # 43 cells, with b2c1
    "test1 -b2c1": [k for k in split["test1"] if k != OUTLIER],  # 42 cells
    "test2": split["test2"],                                   # 40 cells (batch 3)
}

# %% 1. Predict every cell with each frozen model, then score each set
preds = pd.DataFrame(index=df.index)
for name, m in frozen["models"].items():
    preds[name] = from_target(m["pipeline"].predict(df[m["features"]].values))
preds["true"] = df.cycle_life

rows = []
for name in frozen["models"]:
    row = {"model": name}
    for s, ids in sets.items():
        sc = score(df.loc[ids, "cycle_life"], np.log10(preds.loc[ids, name]))
        row[f"{s} RMSE"], row[f"{s} MAPE %"] = sc["rmse"], sc["mape"]
    rows.append(row)
ours = pd.DataFrame(rows).set_index("model")
ours.to_csv(ROOT / "data" / "processed" / "phase7_results.csv")
preds.to_csv(ROOT / "data" / "processed" / "phase7_predictions.csv")

for metric in ["RMSE", "MAPE %"]:
    cols = [c for c in ours.columns if c.endswith(metric)]
    side = pd.concat({"ours": ours[cols], "paper": PAPER[cols]}, axis=1)
    print(f"\n{metric}:\n{side.round(1).to_string()}")

# %% 2. Where do the errors come from? (diagnostics, NOT used to change the model)
print(f"\n{OUTLIER}: true {df.loc[OUTLIER, 'cycle_life']:.0f} cycles, predicted",
      {k: round(preds.loc[OUTLIER, k]) for k in frozen["models"]})
for s in ["test1 -b2c1", "test2"]:
    ids = sets[s]
    err = (preds.loc[ids, "full"] / preds.loc[ids, "true"] - 1) * 100
    print(f"\nfull model, {s}: mean signed error {err.mean():+.1f}%, "
          f"median {err.median():+.1f}%; by batch:",
          err.groupby(df.loc[ids, "batch"]).mean().round(1).to_dict())
    worst = err.abs().sort_values(ascending=False).head(4)
    print("  worst cells:", {k: f"{err[k]:+.0f}% (true {df.loc[k, 'cycle_life']:.0f})"
                             for k in worst.index})
# Is batch 3 outside the training range of the features? (extrapolation check)
tr = df.loc[sets["train"]]
for f in frozen["models"]["full"]["features"]:
    t2 = df.loc[sets["test2"], f]
    outside = ((t2 < tr[f].min()) | (t2 > tr[f].max())).sum()
    if outside:
        print(f"  test2 cells outside the train range of {f}: {outside}/40")
print(f"  cycle life: train max {tr.cycle_life.max():.0f}, "
      f"test2 cells above it: {(df.loc[sets['test2'], 'cycle_life'] > tr.cycle_life.max()).sum()}/40")

# Why is batch 3 under-predicted? A linear model's mean prediction for a group moves by
# sum(coef * mean standardized feature of that group). Splitting that sum by feature shows
# which features pushed batch 3's predictions up or down relative to the training average.
# (x 100 ln10 converts a log10 shift into an approximate % shift.)
full = frozen["models"]["full"]
Z = full["pipeline"][0].transform(df.loc[sets["test2"], full["features"]].values)
shift = pd.Series(Z.mean(axis=0) * full["pipeline"][-1].coef_ * 100 * np.log(10),
                  index=full["features"])
actual = (np.log10(df.loc[sets["test2"], "cycle_life"]).mean()
          - np.log10(df.loc[sets["train"], "cycle_life"]).mean()) * 100 * np.log(10)
print("\nfull model, test2 vs train: predicted shift by feature (%):",
      shift[shift != 0].round(1).to_dict(),
      f"total {shift.sum():+.1f}% vs actual {actual:+.1f}%")
for name in frozen["models"]:
    e = preds.loc[sets["test1 -b2c1"], name] - preds.loc[sets["test1 -b2c1"], "true"]
    print(f"{name:9} test1 -b2c1 RMSE without b1c2 (true 2237): "
          f"{np.sqrt((e.drop('b1c2') ** 2).mean()):.1f}  (b1c2 error {e['b1c2']:+.0f} cycles)")

# %% 3. Parity plots: one panel per model, all three sets
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharex=True, sharey=True)
style = {"train": (MUTED, "o", "train"), "test1": (BLUE, "s", "primary test"),
         "test2": (ORANGE, "^", "secondary test")}
lim = (80, 2600)
for ax, name in zip(axes, frozen["models"]):
    for s, (c, mk, lab) in style.items():
        ids = [k for k in sets[s] if k != OUTLIER]
        ax.scatter(preds.loc[ids, "true"], preds.loc[ids, name], s=26, color=c, marker=mk,
                   alpha=0.85, edgecolor="white", linewidth=0.5,
                   label=f"{lab} ({ours.loc[name, (s if s != 'test1' else 'test1 -b2c1') + ' MAPE %']:.1f}%)")
    ax.scatter(preds.loc[OUTLIER, "true"], preds.loc[OUTLIER, name], s=60, facecolor="none",
               edgecolor=INK, marker="s", label=f"{OUTLIER} (outlier)")
    ax.plot(lim, lim, color=INK, lw=0.8)
    ax.fill_between(lim, [v * 0.85 for v in lim], [v * 1.15 for v in lim], color=INK,
                    alpha=0.06, lw=0, label="±15%")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="true cycle life",
           title=f"'{name}' model")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
axes[0].set_ylabel("predicted cycle life")
fig.suptitle("Phase 7: frozen models on the test sets (log axes; MAPE in legend)", x=0.01,
             ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase7_parity.png", dpi=150)
print("\nsaved figures/phase7_parity.png")
