"""
Step 9 - Post-hoc breakdown of the single test evaluation by sub-period (Part 5).

Reads the FROZEN test predictions written once by 07_model.py (data/test_predictions.parquet) and the
frozen threshold (results/frozen_choices.json). Nothing is refitted and no choice is changed: the
script only splits the already-computed scores by period. It first checks that the overall metrics it
recomputes equal results/test_metrics.csv, so the breakdown provably uses the same single evaluation.

    python src/09_test_subperiods.py
writes results/test_subperiods.csv
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import DATA, RESULTS

sys.stdout.reconfigure(encoding="utf-8")

PERIODS = [("Full test period", "2008-03-31", "2025-06-30"),
           ("Crisis wave 2008-2014", "2008-03-31", "2014-12-31"),
           ("Calm period 2015-2025Q2", "2015-03-31", "2025-06-30")]


def block(d: pd.DataFrame, thr: float) -> dict:
    y, p = d.y.values, d.p.values
    flag = p >= thr
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fb = d[d.y == 1]
    caught = fb[fb.p >= thr].CERT.nunique()
    return {"bank_quarters": len(d), "positive_rows": int(y.sum()), "failing_banks": int(fb.CERT.nunique()),
            "base_rate_pct": 100 * y.mean(), "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p), "recall_at_thr": tp / max(1, y.sum()),
            "precision_at_thr": tp / max(1, tp + fp), "alarms": int(flag.sum()),
            "failing_banks_flagged": int(caught)}


def main() -> None:
    thr = json.loads((RESULTS / "frozen_choices.json").read_text())["threshold"]
    tp = pd.read_parquet(DATA / "test_predictions.parquet")
    tp["REPDTE"] = pd.to_datetime(tp["REPDTE"])

    # consistency check against the single evaluation reported in results/test_metrics.csv
    ref = pd.read_csv(RESULTS / "test_metrics.csv", index_col=0).loc["hgb"]
    full = block(tp, thr)
    for k_ours, k_ref in (("pr_auc", "pr_auc"), ("roc_auc", "roc_auc"), ("recall_at_thr", "recall"),
                          ("precision_at_thr", "precision")):
        assert abs(full[k_ours] - ref[k_ref]) < 5e-4, f"{k_ours} differs from test_metrics.csv"
    print("consistency check passed: predictions equal the frozen single test evaluation")

    rows = []
    for name, a, b in PERIODS:
        d = tp[(tp.REPDTE >= pd.Timestamp(a)) & (tp.REPDTE <= pd.Timestamp(b))]
        rows.append({"period": name, "from": a, "to": b, **block(d, thr)})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "test_subperiods.csv", index=False, float_format="%.4f")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
