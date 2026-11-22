# %% [markdown]
# # Phase 8b: how early can we predict? (~15-20 min)
# Run from the project root:  python notebooks/08_earliness.py
#
# The paper's headline is "predict life from the first 100 cycles". Why 100? Here we
# rebuild every feature from the first N cycles (N = 20 ... 100), with Delta-Q defined as
# Q_N(V) - Q_10(V), and re-run the same modelling procedure for each N.
# Main yardstick: nested CV on the training cells (honest, never sees test cells).
# Test-set scores are shown as a secondary, exploratory check (see Phase 7 notes).

# %%
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.evaluate import loocv_predict, score  # noqa: E402
from src.features import FEATURE_SETS, build_feature_table  # noqa: E402
from src.models import make_elastic_net  # noqa: E402
from src.split import paper_split  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
INK, MUTED, BLUE, ORANGE, GREEN = "#222222", "#8a8a85", "#2a78d6", "#eb6834", "#1baf7a"
WINDOWS = [20, 30, 40, 60, 80, 100]
# 5 inner repeats instead of 25 (Phase 6): 6 windows x 3 models x 41 outer folds would
# otherwise take hours. The N = 100 row lets us check this against Phase 6's numbers.
N_REPEATS = 5

cells = load_all()
split = paper_split(cells)  # outlier b2c1 dropped from the primary test
factories = {"variance": lambda: make_pipeline(StandardScaler(), LinearRegression()),
             "discharge": lambda: make_elastic_net(n_repeats=N_REPEATS),
             "full": lambda: make_elastic_net(n_repeats=N_REPEATS)}

rows = []
for n in WINDOWS:
    df = build_feature_table(cells, first_cycle=10, last_cycle=n)
    tr = df.loc[split["train"]]
    for name, make in factories.items():
        t = time.time()
        feats = FEATURE_SETS[name]
        nested = score(tr.cycle_life, loocv_predict(make(), tr[feats].values, tr.y.values))
        model = make().fit(tr[feats].values, tr.y.values)
        row = {"N": n, "model": name, "nested CV MAPE %": nested["mape"],
               "nested CV RMSE": nested["rmse"]}
        for s in ("test1", "test2"):
            te = df.loc[split[s]]
            sc = score(te.cycle_life, model.predict(te[feats].values))
            row[f"{s} MAPE %"], row[f"{s} RMSE"] = sc["mape"], sc["rmse"]
        rows.append(row)
        print(f"N={n:3} {name:9} nested CV {nested['mape']:5.1f}%  test1 {row['test1 MAPE %']:5.1f}%"
              f"  test2 {row['test2 MAPE %']:5.1f}%  ({time.time() - t:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "data" / "processed" / "phase8_earliness.csv", index=False)
print(res.pivot(index="N", columns="model", values="nested CV MAPE %").round(1).to_string())

# %% Figure: error vs number of cycles used
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
colors = {"variance": MUTED, "discharge": GREEN, "full": BLUE}
for ax, (col, title) in zip(axes, [("nested CV MAPE %", "(a) nested CV on train (honest)"),
                                   ("test1 MAPE %", "(b) primary test (exploratory)"),
                                   ("test2 MAPE %", "(c) secondary test (exploratory)")]):
    for name, c in colors.items():
        d = res[res.model == name]
        ax.plot(d.N, d[col], marker="o", color=c, lw=1.8, label=name)
    ax.set(xlabel="cycles used (N): ΔQ = Q$_N$ − Q$_{10}$", title=title, xticks=WINDOWS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
axes[0].set_ylabel("MAPE (%)")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "phase8_earliness.png", dpi=150)
print("saved figures/phase8_earliness.png")
