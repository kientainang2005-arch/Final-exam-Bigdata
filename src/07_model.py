"""
Step 7 - Modelling & validation (Part 5).

Protocol (decided before looking at the test period):
  * development period  DEV_START .. DEV_END        (all tuning, selection, threshold)
  * gap of 4 quarters    DEV_END+1q .. DEV_END+4q    (no rows used: labels of dev rows look into it)
  * test period          TEST_START .. 2025Q2        (evaluated ONCE at the end)
  * expanding-window walk-forward CV inside dev, every validation block preceded by a 4-quarter gap
  * all preprocessing (winsorising, imputation, scaling) inside an sklearn Pipeline fitted on the
    training block only
  * metrics for rare events: PR-AUC (average precision), ROC-AUC, recall/precision in the top-k%
    of bank-quarters, Brier score; accuracy is reported only to show it is meaningless here
  * alarm threshold minimises expected cost on the pooled out-of-fold dev predictions, with
    costs of a missed failure and a false alarm estimated from the data (see cost_parameters)

    python src/07_model.py
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import CURATED, DATA, FIG, RESULTS, SEED, save_json
from features import FEATURES

sys.stdout.reconfigure(encoding="utf-8")

DEV_START, DEV_END = "1986-03-31", "2006-12-31"     # 1985 lost to 4-quarter lags
TEST_START, TEST_END = "2008-03-31", "2025-06-30"
CV_FOLDS = [  # (train_end, val_start, val_end): 4-quarter gap between train_end and val_start
    ("1987-12-31", "1989-03-31", "1990-12-31"),
    ("1989-12-31", "1991-03-31", "1992-12-31"),
    ("1991-12-31", "1993-03-31", "1995-12-31"),
    ("1994-12-31", "1996-03-31", "2006-12-31"),
]
BASELINE_FEATURES = ["eq_ta", "npa_ta", "roa_q", "liq_ta", "log_assets"]   # classic CAMELS-lite logit


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data only."""

    def __init__(self, q: float = 0.005):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.q, axis=0)
        self.hi_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class DropEmpty(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing in the TRAINING block (e.g. fields not yet collected
    in the 1980s); the same columns are dropped when transforming validation/test data."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = ~np.isnan(X).all(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def make_models() -> dict[str, Pipeline]:
    pre = [("drop_empty", DropEmpty()), ("winsor", Winsorizer(0.005)),
           ("impute", SimpleImputer(strategy="median", add_indicator=True))]
    return {
        "baseline_logit_5": Pipeline(pre + [("scale", StandardScaler()),
                                            ("clf", LogisticRegression(C=1.0, max_iter=2000))]),
        "logit_all": Pipeline(pre + [("scale", StandardScaler()),
                                     ("clf", LogisticRegression(C=0.1, max_iter=4000))]),
        "hgb": Pipeline([("drop_empty", DropEmpty()), ("winsor", Winsorizer(0.005)),  # HGB handles NaN natively
                         ("clf", HistGradientBoostingClassifier(random_state=SEED, early_stopping=False))]),
    }


HGB_GRID = [dict(learning_rate=lr, max_leaf_nodes=l, min_samples_leaf=m, max_iter=it, l2_regularization=1.0)
            for lr, it in ((0.05, 300), (0.1, 150)) for l in (15, 31) for m in (50, 200)]


def cols_for(name: str) -> list[str]:
    return BASELINE_FEATURES if name.startswith("baseline") else FEATURES


def load() -> pd.DataFrame:
    df = (pl.read_parquet(DATA / "features.parquet")
          .filter(pl.col("REPDTE").is_between(pl.lit(DEV_START).str.to_date(), pl.lit(TEST_END).str.to_date())))
    return df.to_pandas()


def period(df, a, b):
    return df[(df.REPDTE >= pd.Timestamp(a)) & (df.REPDTE <= pd.Timestamp(b))]


def topk_metrics(y, p, frac):
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-p)[:k]
    return {"recall": y[idx].sum() / max(1, y.sum()), "precision": y[idx].mean()}


def metrics(y, p, thr=None) -> dict:
    y = np.asarray(y)
    out = {"n": len(y), "positives": int(y.sum()), "pr_auc": average_precision_score(y, p),
           "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
           "base_rate": y.mean()}
    for f in (0.01, 0.05):
        t = topk_metrics(y, p, f)
        out[f"recall_top{int(f * 100)}pct"], out[f"precision_top{int(f * 100)}pct"] = t["recall"], t["precision"]
    if thr is not None:
        pred = p >= thr
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        out.update(threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn, recall=tp / max(1, tp + fn),
                   precision=tp / max(1, tp + fp), accuracy=(tp + tn) / len(y))
    return out


def walk_forward(df: pd.DataFrame, model: Pipeline, cols: list[str]) -> tuple[list[dict], pd.DataFrame]:
    res, oof = [], []
    for tr_end, va_start, va_end in CV_FOLDS:
        tr, va = period(df, DEV_START, tr_end), period(df, va_start, va_end)
        m = clone(model).fit(tr[cols], tr.y)
        p = m.predict_proba(va[cols])[:, 1]
        r = metrics(va.y.values, p)
        r.update(fold=f"train<= {tr_end[:7]} | val {va_start[:7]}..{va_end[:7]}")
        res.append(r)
        oof.append(pd.DataFrame({"CERT": va.CERT.values, "REPDTE": va.REPDTE.values, "y": va.y.values, "p": p}))
    return res, pd.concat(oof)


def cost_parameters(dev: pd.DataFrame) -> dict:
    """Costs estimated from the data of the development period only."""
    f = pd.read_parquet(CURATED / "failures.parquet")
    f["FAILDATE"] = pd.to_datetime(f["FAILDATE"])
    f = f[(f.RESTYPE.str.upper() == "FAILURE") & f.FAILDATE.between(pd.Timestamp(DEV_START), pd.Timestamp(DEV_END))]
    f = f[f.COST.notna() & (f.QBFASSET > 0)]
    loss_rate = (f.COST / f.QBFASSET).median()                 # DIF loss per $ of failed-bank assets
    median_cost = f.COST.median()                              # $ thousands per failure
    # False alarm = an extra targeted on-site examination of a healthy bank. Its cost is not in the FDIC
    # data; we anchor it to the median size of an alarmed bank: c_exam x assets, c_exam = 0.1% of assets
    # (examination + management time), capped below by $50k. Sensitivity is reported for 0.5x..4x.
    early_avoidable = 0.25                                     # share of loss avoidable by early action
    c_fn = early_avoidable * median_cost
    c_fp = max(50.0, 0.001 * dev.ASSET.median())
    return {"failures_used": len(f), "median_loss_rate": loss_rate, "median_cost_kusd": median_cost,
            "avoidable_share": early_avoidable, "C_FN_kusd": c_fn, "C_FP_kusd": c_fp, "ratio_FN_FP": c_fn / c_fp}


def choose_threshold(oof: pd.DataFrame, c_fn: float, c_fp: float) -> tuple[float, pd.DataFrame]:
    y, p = oof.y.values, oof.p.values
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 400)))
    rows = []
    for t in grid:
        pred = p >= t
        fn, fp = ((~pred) & (y == 1)).sum(), (pred & (y == 0)).sum()
        rows.append({"threshold": t, "FN": fn, "FP": fp, "alarms": pred.sum(), "cost": c_fn * fn + c_fp * fp})
    tab = pd.DataFrame(rows)
    return float(tab.loc[tab.cost.idxmin(), "threshold"]), tab


def main() -> None:
    t0 = time.perf_counter()
    df = load()
    dev, test = period(df, DEV_START, DEV_END), period(df, TEST_START, TEST_END)
    gap = period(df, pd.Timestamp(DEV_END) + pd.offsets.QuarterEnd(1), pd.Timestamp(TEST_START) - pd.Timedelta(days=1))
    split = {"dev": [DEV_START, DEV_END, len(dev), int(dev.y.sum()), int(dev[dev.y == 1].CERT.nunique())],
             "gap_rows_unused": len(gap),
             "test": [TEST_START, TEST_END, len(test), int(test.y.sum()), int(test[test.y == 1].CERT.nunique())]}
    print("split [start, end, rows, positive rows, failing banks]:", split)

    # ---------------- 1. model selection by walk-forward CV on dev
    models = make_models()
    cv_rows, oofs = [], {}
    for name, m in models.items():
        if name == "hgb":
            best = None
            for g in HGB_GRID:
                mm = clone(m).set_params(**{f"clf__{k}": v for k, v in g.items()})
                res, oof = walk_forward(dev, mm, cols_for(name))
                score = np.mean([r["pr_auc"] for r in res])
                print(f"  hgb {g} -> mean PR-AUC {score:.4f}")
                if best is None or score > best[0]:
                    best = (score, g, res, oof, mm)
            _, g, res, oof, mm = best
            models[name] = mm
            save_json(g, RESULTS / "hgb_best_params.json")
        else:
            res, oof = walk_forward(dev, m, cols_for(name))
        for r in res:
            r["model"] = name
        cv_rows += res
        oofs[name] = oof
        print(f"{name:<18s} mean PR-AUC {np.mean([r['pr_auc'] for r in res]):.4f}  "
              f"ROC-AUC {np.mean([r['roc_auc'] for r in res]):.4f}")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(RESULTS / "cv_walk_forward.csv", index=False, float_format="%.4f")
    cv_mean = cv.groupby("model")[["pr_auc", "roc_auc", "recall_top1pct", "recall_top5pct", "brier"]].mean()
    print(cv_mean)
    chosen = cv_mean.pr_auc.idxmax()

    # ---------------- 2. threshold from costs (dev OOF predictions only)
    costs = cost_parameters(dev)
    thr, ctab = choose_threshold(oofs[chosen], costs["C_FN_kusd"], costs["C_FP_kusd"])
    sens = []
    for mult in (0.25, 0.5, 1, 2, 4):
        t_m, _ = choose_threshold(oofs[chosen], costs["C_FN_kusd"], costs["C_FP_kusd"] * mult)
        sens.append({"C_FP multiplier": mult, "threshold": t_m,
                     **{k: v for k, v in metrics(oofs[chosen].y.values, oofs[chosen].p.values, t_m).items()
                        if k in ("recall", "precision", "tp", "fp")}})
    pd.DataFrame(sens).to_csv(RESULTS / "threshold_sensitivity.csv", index=False, float_format="%.4f")
    ctab.to_csv(RESULTS / "threshold_cost_curve.csv", index=False, float_format="%.6f")

    frozen = {"chosen_model": chosen, "threshold": thr, "costs": costs, "split": split,
              "features": cols_for(chosen), "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(frozen, RESULTS / "frozen_choices.json")       # everything fixed BEFORE the test
    print("frozen:", json.dumps({k: frozen[k] for k in ("chosen_model", "threshold", "costs")}, default=str))

    # ---------------- 3. single evaluation on the test period
    final = {}
    for name, m in models.items():
        cols = cols_for(name)
        fit = clone(m).fit(dev[cols], dev.y)
        p = fit.predict_proba(test[cols])[:, 1]
        thr_m = thr if name == chosen else choose_threshold(oofs[name], costs["C_FN_kusd"], costs["C_FP_kusd"])[0]
        final[name] = metrics(test.y.values, p, thr_m)
        if name == chosen:
            test_pred = test[["CERT", "REPDTE", "y", "q_to_fail", "ASSET"]].assign(p=p)
            chosen_fit = fit
    tm = pd.DataFrame(final).T
    tm.to_csv(RESULTS / "test_metrics.csv", float_format="%.4f")
    print("\nTEST (evaluated once):\n", tm[["pr_auc", "roc_auc", "recall_top1pct", "recall_top5pct", "recall",
                                           "precision", "accuracy", "tp", "fp", "fn"]].round(4))

    # early-warning view: of banks that failed in the test period, how many were flagged and how early?
    fb = test_pred[test_pred.y == 1].copy()
    fb["flag"] = fb.p >= thr
    lead = fb[fb.flag].groupby("CERT").q_to_fail.max()
    ew = {"failed_banks_in_test": int(fb.CERT.nunique()), "flagged_at_least_once": int(lead.size),
          "median_lead_quarters": float(lead.median()) if len(lead) else None,
          "alarms_total": int((test_pred.p >= thr).sum()), "bank_quarters": len(test_pred)}
    save_json(ew, RESULTS / "test_early_warning.json")
    print(ew)
    test_pred.to_parquet(DATA / "test_predictions.parquet")
    plots(oofs, test_pred, chosen, ctab, thr, chosen_fit)
    print(f"done in {time.perf_counter() - t0:.0f}s")


def plots(oofs, test_pred, chosen, ctab, thr, fit) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, o in oofs.items():
        pr, rc, _ = precision_recall_curve(o.y, o.p)
        axes[0].plot(rc, pr, label=f"{name} (AP {average_precision_score(o.y, o.p):.3f})")
    axes[0].set(xlabel="recall", ylabel="precision", title="Dev walk-forward (out-of-fold) PR curves")
    axes[0].legend(fontsize=8), axes[0].grid(alpha=.3)
    axes[1].plot(ctab.threshold, ctab.cost / 1e3, color="#2a6f97")
    axes[1].axvline(thr, ls="--", color="#c0392b")
    axes[1].set(xscale="log", xlabel="threshold (log)", ylabel="expected cost ($ millions)",
                title=f"Cost-based threshold ({chosen}) = {thr:.3f}")
    axes[1].grid(alpha=.3)
    fig.tight_layout(), fig.savefig(FIG / "model_pr_and_cost.png", dpi=150), plt.close(fig)

    g = test_pred.groupby("REPDTE").agg(alarms=("p", lambda s: (s >= thr).sum()), failing=("y", "sum"))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(g.index, g.alarms, label="alarms (p >= threshold)", color="#e67e22")
    ax.plot(g.index, g.failing, label="banks failing within 4 quarters", color="#c0392b")
    ax.set_title("Test period: alarms vs. actual failing banks per quarter"), ax.legend(), ax.grid(alpha=.3)
    fig.tight_layout(), fig.savefig(FIG / "test_alarms_over_time.png", dpi=150), plt.close(fig)


if __name__ == "__main__":
    main()
