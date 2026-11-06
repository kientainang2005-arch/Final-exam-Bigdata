"""
Step 4 - Clean bank-quarter panel with exit types and the failure label (Part 2).

    python src/04_panel.py
writes data/panel.parquet and results/panel_*.csv / figures.
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from common import DATA, FIG, LAST_MODEL_QUARTER, RESULTS, save_json
from transforms import build_panel, failure_events, scan_financials

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    lf = build_panel()
    (RESULTS / "polars_plan_panel_optimized.txt").write_text(lf.explain(), encoding="utf-8")
    (RESULTS / "polars_plan_panel_naive.txt").write_text(lf.explain(optimized=False), encoding="utf-8")
    print("Optimised plan (excerpt):\n", lf.explain()[:1500], "...\n")
    panel = lf.collect()
    panel.write_parquet(DATA / "panel.parquet", compression="zstd")
    dt = time.perf_counter() - t0

    raw_rows = scan_financials().select(pl.len()).collect().item()
    fails = failure_events().collect()
    n_in_scope = fails.filter(pl.col("FAILDATE") > pl.date(1984, 3, 31)).height
    banks = panel.group_by("CERT").agg(pl.col("exit_type").first())
    exit_ct = banks.group_by("exit_type").agg(pl.len().alias("banks")).sort("banks", descending=True)
    labelled = panel.filter(pl.col("REPDTE") <= pl.lit(LAST_MODEL_QUARTER).str.to_date())
    failed_with_label = labelled.filter(pl.col("y") == 1)["CERT"].n_unique()

    per_year = (labelled.group_by(pl.col("REPDTE").dt.year().alias("year"))
                .agg(pl.len().alias("bank_quarters"), pl.col("y").sum().alias("positive_rows"),
                     pl.col("CERT").filter(pl.col("q_to_fail") == 1).n_unique().alias("failures_next_q"))
                .sort("year"))
    fail_year = (fails.group_by(pl.col("FAILDATE").dt.year().alias("year")).agg(pl.len().alias("failures"))
                 .sort("year"))
    per_year = per_year.join(fail_year, on="year", how="left").fill_null(0)
    per_year.write_csv(RESULTS / "panel_by_year.csv")

    summary = {
        "raw_financial_rows": raw_rows, "panel_rows": panel.height,
        "rows_removed": raw_rows - panel.height,
        "banks": panel["CERT"].n_unique(),
        "failure_events_FAILURE_in_scope": n_in_scope,
        "failed_banks_with_a_positive_label_row": failed_with_label,
        "labelled_rows_upto_2025Q2": labelled.height,
        "positive_rows_upto_2025Q2": int(labelled["y"].sum()),
        "positive_rate_pct": round(100 * labelled["y"].mean(), 3),
        "exit_types": dict(zip(exit_ct["exit_type"], exit_ct["banks"])),
        "build_seconds": round(dt, 1),
    }
    save_json(summary, RESULTS / "panel_summary.json")
    for k, v in summary.items():
        print(f"{k:<42s}: {v}")
    print(per_year.to_pandas().to_string(index=False))

    pdf = per_year.to_pandas()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(pdf.year, pdf.failures, color="#c0392b")
    ax.set_title("Bank failures per year (FDIC, RESTYPE = FAILURE)"), ax.set_ylabel("failures")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(), fig.savefig(FIG / "failures_per_year.png", dpi=150), plt.close(fig)


if __name__ == "__main__":
    main()
