# %% [markdown]
# # Phase 6: nested cross-validation (slow, ~10 min)
# Run from the project root:  python notebooks/06_nested_cv.py
#
# Why nested: ElasticNetCV picks alpha/l1_ratio with CV on the training cells. Quoting
# that CV score would be optimistic (we kept whichever setting scored best on those very
# folds). Nested CV wraps the WHOLE procedure in an outer leave-one-out loop: for each
# training cell, re-run scaling + hyperparameter search on the other 40, then predict it.

# %%
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.evaluate import loocv_predict, score  # noqa: E402
from src.features import FEATURE_SETS  # noqa: E402
from src.models import make_elastic_net  # noqa: E402
from src.split import paper_split  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

cells = load_all()
train_ids = paper_split(cells)["train"]
df = pd.read_csv(ROOT / "data" / "processed" / "features.csv", index_col=0).loc[train_ids]
y, life = df.y.values, df.cycle_life.values

rows, preds = [], {}
for name, feats in FEATURE_SETS.items():
    X = df[feats].values
    for label, factory in [("ElasticNet", make_elastic_net),
                           ("OLS", lambda: make_pipeline(StandardScaler(), LinearRegression()))]:
        t = time.time()
        p = loocv_predict(factory(), X, y)  # outer LOO; ElasticNetCV does the inner CV
        preds[f"{name}/{label}"] = p
        train = score(life, factory().fit(X, y).predict(X))
        nested = score(life, p)
        rows.append({"feature set": name, "model": label, "n features": len(feats),
                     "train RMSE": train["rmse"], "train MAPE %": train["mape"],
                     "CV RMSE": nested["rmse"], "CV MAPE %": nested["mape"]})
        print(f"{name:9} {label:10} train {train['mape']:5.1f}%  CV {nested['mape']:5.1f}%  "
              f"({time.time() - t:.0f}s)", flush=True)

out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "processed" / "phase6_nested_cv.csv", index=False)
pd.DataFrame(preds, index=df.index).to_csv(ROOT / "data" / "processed" / "phase6_cv_predictions.csv")
print(out.round(1).to_string(index=False))
