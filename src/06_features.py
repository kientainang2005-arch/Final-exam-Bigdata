"""
Step 6 - Leak-free feature table joined with the label (Part 4).

    python src/06_features.py
writes data/features.parquet, results/feature_dictionary.csv and runs a point-in-time test:
features recomputed from data TRUNCATED at a cut-off quarter T must equal the features computed
from the full data for every row with REPDTE <= T. If any feature peeked at the future, the two
would differ and the script stops with an error.
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import sys
import time

import numpy as np
import pandas as pd
import polars as pl

from common import DATA, RESULTS, save_json
from features import FEATURES, RATIOS, build_features
from transforms import build_panel, failure_events, scan_financials

sys.stdout.reconfigure(encoding="utf-8")


def features_from(fin: pl.LazyFrame, fails: pl.LazyFrame) -> pl.LazyFrame:
    return build_features(build_panel(fin, fails), fails)


def pit_test(cutoffs=("1991-12-31", "2009-06-30", "2023-03-31")) -> list[dict]:
    """Truncation test for point-in-time correctness of every feature."""
    out = []
    fin_all, fails_all = scan_financials(), failure_events()
    full = features_from(fin_all, fails_all).collect()
    for c in cutoffs:
        T = pl.lit(c).str.to_date()
        trunc = features_from(fin_all.filter(pl.col("REPDTE") <= T),
                              fails_all.filter(pl.col("FAILDATE") <= T)).collect()
        a = full.filter(pl.col("REPDTE") <= T).sort(["CERT", "REPDTE"])
        b = trunc.sort(["CERT", "REPDTE"])
        # rows of banks that fail AFTER T exist in both; their features must not change
        assert a.height == b.height, f"row count differs at {c}: {a.height} vs {b.height}"
        worst, worst_col = 0.0, None
        for f in FEATURES:
            x, y = a[f].cast(pl.Float64).to_numpy(), b[f].cast(pl.Float64).to_numpy()
            if (np.isnan(x) != np.isnan(y)).any():
                raise AssertionError(f"null pattern of {f} changes when future data is removed ({c})")
            d = np.nanmax(np.abs(x - y)) if np.isfinite(x).any() else 0.0
            if d > worst:
                worst, worst_col = float(d), f
        assert worst < 1e-9, f"feature {worst_col} changes by {worst} when data after {c} is removed"
        out.append({"cutoff": c, "rows_compared": a.height, "features": len(FEATURES), "max_abs_diff": worst})
        print(f"  PIT test cut-off {c}: {a.height:,} rows x {len(FEATURES)} features identical (max diff {worst:.1e})")
    return out


def main() -> None:
    t0 = time.perf_counter()
    fails = failure_events()
    feat = features_from(scan_financials(), fails).collect()
    feat.write_parquet(DATA / "features.parquet", compression="zstd")
    dt = time.perf_counter() - t0
    print(f"feature table: {feat.height:,} rows x {len(FEATURES)} features in {dt:.1f}s")

    groups = {n: g for n, (_, g, _) in RATIOS.items()}
    why = {n: r for n, (_, _, r) in RATIOS.items()}
    extra = {"d4_": ("dyn", "change over 4 quarters (deterioration speed)"),
             "g4_": ("dyn", "4-quarter growth (rapid growth precedes failures)"),
             "roa_vol8": ("dyn", "earnings volatility, trailing 8 quarters"),
             "roa_4q": ("dyn", "trailing 4-quarter mean ROA"),
             "n_reports": ("dyn", "length of reporting history (de novo risk)"),
             "state_": ("ctx", "state-level stress, same quarter"),
             "npa_vs_state": ("ctx", "bank NPA relative to state median"),
             "fails_trailing4q_us": ("ctx", "US failures in the past 4 quarters (public)")}
    rows = []
    lab = feat.filter(pl.col("REPDTE") <= pl.date(2025, 6, 30))
    pos, neg = lab.filter(pl.col("y") == 1), lab.filter(pl.col("y") == 0)
    for f in FEATURES:
        g, r = groups.get(f), why.get(f)
        if g is None:
            for k, (gg, rr) in extra.items():
                if f.startswith(k):
                    g, r = gg, rr
                    break
        rows.append({"feature": f, "group": g, "rationale": r,
                     "missing_pct": round(100 * lab[f].null_count() / lab.height, 2),
                     "median_failing (y=1)": pos[f].median(), "median_others (y=0)": neg[f].median()})
    fd = pd.DataFrame(rows)
    fd.to_csv(RESULTS / "feature_dictionary.csv", index=False, float_format="%.4f")
    print(fd.to_string(index=False))

    print("\npoint-in-time truncation test:")
    pit = pit_test()
    save_json({"rows": feat.height, "n_features": len(FEATURES), "seconds": round(dt, 1), "pit_test": pit},
              RESULTS / "features_summary.json")


if __name__ == "__main__":
    main()
