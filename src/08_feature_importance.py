"""
Step 8 - Feature importance on the DEVELOPMENT period only (Part 5, explainability).

For every walk-forward fold of 07_model.py, the frozen HGB configuration (results/hgb_best_params.json)
is fitted on the fold's training block and permutation importance is measured on the fold's validation
block as the drop in PR-AUC (average precision) when one feature is shuffled. The test period is never
loaded into the computation, so this analysis cannot influence or be influenced by the test evaluation.

    python src/08_feature_importance.py
writes results/feature_importance_dev.csv and results/figures/feature_importance_dev.png
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import importlib
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from common import FIG, RESULTS, SEED
from features import FEATURES

sys.stdout.reconfigure(encoding="utf-8")
M = importlib.import_module("07_model")
N_REPEATS = 3


def main() -> None:
    t0 = time.perf_counter()
    df = M.load()
    dev = M.period(df, M.DEV_START, M.DEV_END)          # development period only
    del df
    params = json.loads((RESULTS / "hgb_best_params.json").read_text())
    model = clone(M.make_models()["hgb"]).set_params(**{f"clf__{k}": v for k, v in params.items()})

    rows = []
    for tr_end, va_start, va_end in M.CV_FOLDS:
        tr, va = M.period(dev, M.DEV_START, tr_end), M.period(dev, va_start, va_end)
        fit = clone(model).fit(tr[FEATURES], tr.y)
        r = permutation_importance(fit, va[FEATURES], va.y, scoring="average_precision",
                                   n_repeats=N_REPEATS, random_state=SEED, n_jobs=1)
        for f, m, s in zip(FEATURES, r.importances_mean, r.importances_std):
            rows.append({"fold": f"val {va_start[:7]}..{va_end[:7]}", "feature": f,
                         "delta_pr_auc": m, "std": s, "val_positives": int(va.y.sum())})
        print(f"fold val {va_start[:7]}..{va_end[:7]}: done ({time.perf_counter() - t0:.0f}s)", flush=True)

    per_fold = pd.DataFrame(rows)
    imp = (per_fold.groupby("feature")
           .agg(mean_delta_pr_auc=("delta_pr_auc", "mean"), sd_across_folds=("delta_pr_auc", "std"),
                folds_positive=("delta_pr_auc", lambda s: int((s > 0).sum())))
           .sort_values("mean_delta_pr_auc", ascending=False).reset_index())
    # economic direction from the development-period feature dictionary (median of failing vs other rows)
    fd = pd.read_csv(RESULTS / "feature_dictionary.csv").set_index("feature")
    imp["group"] = imp.feature.map(fd["group"])
    imp["median_failing"] = imp.feature.map(fd["median_failing (y=1)"])
    imp["median_others"] = imp.feature.map(fd["median_others (y=0)"])
    imp["direction"] = np.where(imp.median_failing > imp.median_others, "higher in failing banks",
                                np.where(imp.median_failing < imp.median_others, "lower in failing banks", "equal medians"))
    imp.to_csv(RESULTS / "feature_importance_dev.csv", index=False, float_format="%.5f")
    per_fold.to_csv(RESULTS / "feature_importance_dev_by_fold.csv", index=False, float_format="%.5f")
    print(imp.head(12).to_string(index=False))

    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top.feature, top.mean_delta_pr_auc, xerr=top.sd_across_folds, color="#2a6f97", capsize=2)
    ax.set_xlabel("drop in PR-AUC when the feature is permuted (mean over 4 dev folds)")
    ax.set_title("Permutation importance, development period only (HGB)")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(), fig.savefig(FIG / "feature_importance_dev.png", dpi=150), plt.close(fig)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
