# %% [markdown]
# # Phase 8b diagnosis: why does the discharge model spike to 32% on batch 3 at N = 60?
# Run from the project root:  python notebooks/08_earliness_diagnosis.py  (~1 min)
#
# Steps we took (kept here because the debugging path is part of the lesson):
#   1. Is it one bad cell? No: the median batch-3 error at N = 60 is 28% (12% at N = 40/80).
#   2. Which feature? Standardize batch-3 features with the TRAINING mean/std and count
#      extreme values (|z| > 3). At N = 60, dq_log_kurt is extreme for 17 of 40 cells.
#   3. A glitch in the cycle-60 curves? No: the largest point-to-point jump is the same
#      size at cycles 59, 60 and 61 (checked by hand on b3c16, see the log).
#   4. The real cause: shape statistics of ΔQ (below).

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.features import FEATURE_SETS, build_feature_table  # noqa: E402
from src.split import paper_split  # noqa: E402

warnings.filterwarnings("ignore")
cells = load_all()
split = paper_split(cells)
feats = FEATURE_SETS["discharge"]

for n in (40, 60, 80):
    df = build_feature_table(cells, first_cycle=10, last_cycle=n)
    tr, te = df.loc[split["train"]], df.loc[split["test2"]]
    z = (te[feats] - tr[feats].mean()) / tr[feats].std()
    print(f"N={n}: batch-3 cells with |z|>3:", (z.abs() > 3).sum().to_dict(),
          f"| train std of dq_log_kurt {tr.dq_log_kurt.std():.3f}")

# Shape of ΔQ = Q_N - Q_10: share of voltage points where it is POSITIVE (capacity still
# rising), and its kurtosis (tail-heaviness), median over cells.
for grp in ("train", "test2"):
    for n in (40, 50, 60, 70, 80):
        pos, ku = [], []
        for k in split[grp]:
            dq = cells[k]["cycles"][str(n - 1)]["Qdlin"] - cells[k]["cycles"]["9"]["Qdlin"]
            pos.append((dq > 0).mean())
            ku.append(stats.kurtosis(dq, fisher=False))
        print(f"{grp:5} N={n}: share ΔQ>0 {np.median(pos):.2f}, kurtosis median "
              f"{np.median(ku):.1f}, max {np.max(ku):.1f}")
