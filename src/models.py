"""Model definitions for Phase 6.

Each model is a Pipeline(StandardScaler -> ElasticNetCV). Putting the scaler INSIDE
the pipeline matters: whenever the pipeline is fit (including inside each
cross-validation fold) the scaler learns its mean/std from that fit's training
rows only, so held-out rows never influence the scaling.
"""

import numpy as np
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Candidate L1/L2 mixes. 1.0 = pure lasso (L1); small values ~ ridge-like (L2).
# (ElasticNetCV can't go all the way to 0; very small values are effectively ridge.)
L1_RATIOS = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

# 4-fold CV repeated 25 times with different shuffles. With 41 cells, one random
# 4-fold split is noisy (which cells land together changes the chosen alpha);
# averaging over many shuffles stabilises the choice. The paper similarly used
# 4-fold CV with repeated random sampling.
N_SPLITS, N_REPEATS = 4, 25


def make_elastic_net(seed=0, n_repeats=N_REPEATS):
    # n_repeats can be lowered for big sweeps (Phase 8) where 25 repeats x many
    # refits would take hours; fewer repeats = a slightly noisier alpha choice.
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=seed)
    # alphas=100: try 100 penalty strengths on a log grid, from "so strong every
    # coefficient is 0" down to 1/1000 of that. (sklearn >= 1.7 spelling.)
    enet = ElasticNetCV(l1_ratio=L1_RATIOS, alphas=100, cv=cv, n_jobs=-1,
                        max_iter=200_000, tol=1e-6)
    return make_pipeline(StandardScaler(), enet)


def describe(pipe, feature_names):
    """Chosen hyperparameters and coefficients (in standardized units)."""
    enet = pipe[-1]
    return {
        "alpha": float(enet.alpha_),
        "l1_ratio": float(enet.l1_ratio_),
        "coefs": dict(zip(feature_names, np.round(enet.coef_, 4))),
        "n_nonzero": int(np.sum(enet.coef_ != 0)),
    }


def make_classifier(seed=0):
    """Phase 8: L2-penalised logistic regression, penalty strength picked by CV.

    - Stratified folds keep the short/long ratio the same in every fold; with ~40 cells
      a plain split could leave a fold with almost no cells of one class.
    - Scored by log-loss, not accuracy: accuracy only changes when a cell crosses the
      0.5 line, so on 40 cells many penalty values tie; log-loss rewards confident-and-right
      probabilities and gives CV a smoother curve to minimise.
    - Pure L2 (l1_ratios=(0,)) with lbfgs: few features, so no need for L1 selection.
    """
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=10, random_state=seed)
    clf = LogisticRegressionCV(Cs=20, l1_ratios=(0.0,), cv=cv, scoring="neg_log_loss",
                               max_iter=10_000, use_legacy_attributes=False)
    return make_pipeline(StandardScaler(), clf)
