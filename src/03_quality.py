"""
Step 3 - Automated data-quality report (Part 2).

Runs a fixed battery of checks on the curated layer and writes
    results/dq_report.md          human-readable report (regenerated on every run)
    results/dq_*.csv              the underlying tables
    results/figures/dq_*.png      figures
Every check returns numbers, so the report is evidence, not opinion.

    python src/03_quality.py
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from common import CURATED, FIG, RAW, RESULTS

sys.stdout.reconfigure(encoding="utf-8")

FIN = pl.scan_parquet(CURATED / "financials" / "**" / "*.parquet", hive_partitioning=True)
FAIL = pl.read_parquet(CURATED / "failures.parquet")
INST = pl.read_parquet(CURATED / "institutions.parquet")
MD: list[str] = []          # markdown lines of the report
SUMMARY: list[dict] = []    # one row per issue for the summary table


def h(txt: str, level: int = 2) -> None:
    MD.append(f"\n{'#' * level} {txt}\n")


def p(txt: str) -> None:
    MD.append(txt + "\n")


def table(df: pd.DataFrame, name: str | None = None, floatfmt: str = ".3f", max_rows: int = 40) -> None:
    if name:
        df.to_csv(RESULTS / f"dq_{name}.csv", index=False)
    MD.append(df.head(max_rows).to_markdown(index=False, floatfmt=floatfmt) + "\n")


def issue(check: str, finding: str, n_affected, handling: str) -> None:
    SUMMARY.append({"check": check, "finding": finding, "rows/banks affected": n_affected,
                    "handling in pipeline": handling})


ERA = (pl.when(pl.col("REPDTE").dt.year() < 1990).then(pl.lit("1984-89"))
       .when(pl.col("REPDTE").dt.year() < 2000).then(pl.lit("1990-99"))
       .when(pl.col("REPDTE").dt.year() < 2010).then(pl.lit("2000-09"))
       .when(pl.col("REPDTE").dt.year() < 2020).then(pl.lit("2010-19"))
       .otherwise(pl.lit("2020-26")).alias("era"))


# ------------------------------------------------------------------ 1. completeness
def check_completeness() -> None:
    h("1. Completeness of the download")
    man = json.loads((RAW / "manifest_financials.json").read_text())
    per_q = (FIN.group_by("REPDTE").agg(pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks"))
             .sort("REPDTE").collect().to_pandas())
    per_q["api_total"] = per_q["REPDTE"].dt.strftime("%Y%m%d").map(lambda q: man[q]["api_total"])
    mism = int((per_q.rows != per_q.api_total).sum())
    small = json.loads((RAW / "manifest_small_tables.json").read_text())
    p(f"* Quarters: **{len(per_q)}** ({per_q.REPDTE.min():%Y-%m-%d} to {per_q.REPDTE.max():%Y-%m-%d}); "
      f"rows: **{per_q.rows.sum():,}**; quarters where rows != API total: **{mism}**.")
    p(f"* Failures: {small['failures']['rows']:,} rows (API total {small['failures']['api_total']:,}); "
      f"Institutions: {small['institutions']['rows']:,} rows (API total {small['institutions']['api_total']:,}).")
    per_q.to_csv(RESULTS / "dq_rows_per_quarter.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_q.REPDTE, per_q.banks, lw=1.5, color="#2a6f97")
    ax.set_ylabel("reporting banks"), ax.set_title("Banks filing a Call Report per quarter")
    ax.grid(alpha=.3)
    fig.tight_layout(), fig.savefig(FIG / "dq_banks_per_quarter.png", dpi=150), plt.close(fig)
    p("![](figures/dq_banks_per_quarter.png)")
    issue("Completeness", f"{mism} quarters with row count != API total", mism, "download re-run until 0 (resumable manifest)")


# ------------------------------------------------------------------ 2. keys
def check_keys() -> None:
    h("2. Primary key (CERT, REPDTE)")
    k = FIN.select(pl.len().alias("rows"), pl.struct("CERT", "REPDTE").n_unique().alias("unique"),
                   pl.col("CERT").null_count().alias("null_cert")).collect()
    dup = k["rows"][0] - k["unique"][0]
    p(f"* rows {k['rows'][0]:,}, unique keys {k['unique'][0]:,}, duplicates **{dup}**, null CERT {k['null_cert'][0]}.")
    issue("Primary key", f"{dup} duplicate (CERT, REPDTE)", dup, "assert in panel build (pipeline fails if > 0)")


# ------------------------------------------------------------------ 3. missingness by era
def check_missing() -> None:
    h("3. Missing values by era")
    cols = [c for c in FIN.collect_schema().names() if c not in ("CERT", "REPDTE", "year")]
    m = (FIN.with_columns(ERA).group_by("era")
         .agg([pl.col(c).is_null().mean().alias(c) for c in cols]).sort("era").collect().to_pandas()
         .set_index("era").T)
    m.index.name = "column"
    m.reset_index().to_csv(RESULTS / "dq_missing_by_era.csv", index=False)
    show = m[(m > 0.001).any(axis=1)]
    table((show * 100).round(1).reset_index(), floatfmt=".1f")
    p("Values are % of rows missing.")
    fig, ax = plt.subplots(figsize=(6, max(3, .22 * len(show))))
    im = ax.imshow(show.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(show.shape[1]), show.columns), ax.set_yticks(range(len(show)), show.index, fontsize=7)
    fig.colorbar(im, label="% missing"), ax.set_title("Missingness by era")
    fig.tight_layout(), fig.savefig(FIG / "dq_missing_by_era.png", dpi=150), plt.close(fig)
    p("![](figures/dq_missing_by_era.png)")
    structural = show[(show["1984-89"] > 0.5) & (show["2010-19"] < 0.05)].index.tolist()
    issue("Structural missingness", f"fields not collected in early years: {', '.join(structural)}",
          len(structural), "never imputed with future info: missing-indicator + training-block median; "
                           "columns empty in a training block are dropped for that block")


# ------------------------------------------------------------------ 4. domain / range
def check_domain() -> None:
    h("4. Domain and range violations")
    nonneg = ["ASSET", "DEP", "DEPDOM", "CHBAL", "SC", "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON",
              "LNATRES", "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "ORE", "BRO", "INTINC"]
    rules = {f"{c} < 0": pl.col(c) < 0 for c in nonneg}
    rules.update({
        "ASSET == 0": pl.col("ASSET") == 0,
        "EQ < 0 (balance-sheet insolvent)": pl.col("EQ") < 0,
        "|ROA| > 100 %": pl.col("ROA").abs() > 100,
        "EQV > 100 % or < -100 %": pl.col("EQV").abs() > 100,
        "RBCRWAJ > 100 % (tiny RWA)": pl.col("RBCRWAJ") > 100,
        "RBC1AAJ > 100 %": pl.col("RBC1AAJ") > 100,
        "ASSET < 1 000 (under $1m)": pl.col("ASSET") < 1000,
    })
    out = FIN.select([r.sum().alias(k) for k, r in rules.items()]).collect().to_pandas().T.reset_index()
    out.columns = ["rule", "rows"]
    out = out[out.rows > 0].sort_values("rows", ascending=False)
    table(out, "domain_violations")
    neg = int(out[out.rule.str.contains("< 0") & ~out.rule.str.startswith("EQ")].rows.sum())
    issue("Negative stock amounts", "negative values in fields that cannot be negative", neg,
          "set to null (treated as a reporting error)")
    issue("Extreme ratios", "ratios far outside economic range (tiny denominators)",
          int(out[out.rule.str.contains("%")].rows.sum()),
          "ratios recomputed from amounts with safe division; winsorised at train-period 0.5/99.5 pct")
    # zeros used as "not reported": risk-based capital ratios did not exist before 1990-92
    z = (FIN.with_columns(ERA).group_by("era")
         .agg([(pl.col(c) == 0).mean().mul(100).round(2).alias(f"{c} == 0 (%)") for c in ("RBCRWAJ", "IDT1RWAJR", "RBC1AAJ")])
         .sort("era").collect().to_pandas())
    p("Share of rows where a capital ratio is exactly 0 (a bank cannot operate with 0% capital, so 0 = not reported). "
      "Before 1990 risk-based ratios did not exist; from 2020 small banks that opt into the Community Bank "
      "Leverage Ratio framework stop reporting risk-based ratios:")
    table(z, "zero_as_missing", floatfmt=".2f")
    issue("Zero used as missing", "risk-based capital ratios are 0 instead of null (pre-1990, and CBLR banks from 2020)",
          f"{z.iloc[0, 1]:.0f}% of 1984-89 rows", "capital ratios not used as model features; equity/assets "
          "(eq_ta), which exists in all eras, is used instead")
    eqneg = out.loc[out.rule.str.startswith("EQ"), "rows"]
    issue("Negative equity", "insolvent banks report EQ < 0", int(eqneg.iloc[0]) if len(eqneg) else 0,
          "kept: genuine (and highly informative) signal, not an error")


# ------------------------------------------------------------------ 5. identities
def check_identities() -> None:
    h("5. Accounting identities (cross-field consistency)")
    tol = 0.01
    rel = lambda a, b: ((a - b).abs() / pl.max_horizontal(b.abs(), pl.lit(1))) > tol
    rules = {
        "LNLSNET != LNLSGR - LNATRES (>1%)": rel(pl.col("LNLSNET"), pl.col("LNLSGR") - pl.col("LNATRES")),
        "EQV != 100*EQ/ASSET (>1% rel.)": rel(pl.col("EQV"), 100 * pl.col("EQ") / pl.col("ASSET")),
        "DEP > ASSET": pl.col("DEP") > pl.col("ASSET"),
        "DEPDOM > DEP": pl.col("DEPDOM") > pl.col("DEP") * 1.0001,
        "LNLSGR > ASSET": pl.col("LNLSGR") > pl.col("ASSET"),
        "RE components > LNRE": (pl.sum_horizontal("LNRECONS", "LNREMULT", "LNRENRES", "LNRERES")
                                 > pl.col("LNRE") * 1.01),
        "NAASSET > NCLNLS + ORE (nonaccrual > noncurrent)": pl.col("NAASSET") > (pl.col("NCLNLS") + pl.col("ORE")) * 1.01,
    }
    n = FIN.select(pl.len()).collect().item()
    out = FIN.select([r.sum().alias(k) for k, r in rules.items()]).collect().to_pandas().T.reset_index()
    out.columns = ["identity violated", "rows"]
    out["% rows"] = 100 * out.rows / n
    table(out, "identities", floatfmt=".3f")
    issue("Accounting identities", "small share of rows break balance-sheet identities",
          int(out.rows.sum()), "rows kept: DEP > ASSET coincides with negative equity (insolvency), the rest is <0.05% of rows; ratios use safe division and winsorising")


# ------------------------------------------------------------------ 6. YTD flows
def check_ytd() -> None:
    h("6. Income-statement flows are year-to-date (YTD)")
    p("Call Report income items (NETINC, INTINC, EINTEXP, NONII, NONIX, ELNATR, NTLNLS, EQCDIV) accumulate "
      "from January. If used raw, Q4 values look ~4x Q1 values, and a feature like NETINC/ASSET would be "
      "seasonal noise.")
    s = (FIN.filter(pl.col("REPDTE").dt.year() >= 1985)
         .with_columns(pl.col("REPDTE").dt.month().alias("m"))
         .group_by("m").agg((pl.col("INTINC") / pl.col("ASSET")).median().alias("median INTINC/ASSET"),
                            pl.col("ROA").median().alias("median ROA (%)"))
         .sort("m").collect().to_pandas())
    s["quarter"] = s.m.map({3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"})
    table(s[["quarter", "median INTINC/ASSET", "median ROA (%)"]], "ytd_evidence", floatfmt=".4f")
    p("INTINC/ASSET rises linearly Q1->Q4 (YTD) while ROA is flat (already annualised by the FDIC).")
    issue("YTD flows", "income items are cumulative within the calendar year",
          "all rows", "de-cumulated: flow_q = YTD_q - YTD_{q-1} (same year, consecutive quarter), Q1 = YTD; "
                      "annualised x4; null when previous quarter missing")


# ------------------------------------------------------------------ 7. gaps
def check_gaps() -> None:
    h("7. Reporting gaps inside a bank's history")
    qidx = (pl.col("REPDTE").dt.year() * 4 + (pl.col("REPDTE").dt.month() - 1) // 3)
    g = (FIN.select("CERT", "REPDTE").with_columns(qidx.alias("qi")).sort(["CERT", "qi"])
         .with_columns((pl.col("qi") - pl.col("qi").shift(1).over("CERT")).alias("step"))
         .filter(pl.col("step") > 1).collect())
    banks = g["CERT"].n_unique()
    p(f"* {g.height:,} gaps (a missing quarter between two reports) in **{banks:,}** banks; "
      f"median gap {g['step'].median() - 1:.0f} quarter(s), max {g['step'].max() - 1} quarters.")
    issue("Reporting gaps", "missing quarters inside a bank's reporting history", f"{banks} banks",
          "lags/changes computed on the quarter index, not on row order; null when gap")


# ------------------------------------------------------------------ 8. failures table
def check_failures() -> pd.DataFrame:
    h("8. Failures table")
    last = FIN.group_by("CERT").agg(pl.col("REPDTE").max().alias("last_rep"),
                                    pl.col("REPDTE").min().alias("first_rep")).collect()
    f = FAIL.join(last, on="CERT", how="left")
    in_scope = f.filter(pl.col("FAILDATE") >= pl.date(1984, 4, 1))
    res = in_scope.group_by("RESTYPE").agg(pl.len().alias("events"),
                                           pl.col("last_rep").is_null().sum().alias("no financials"))
    table(res.to_pandas(), "failures_by_restype", floatfmt=".0f")
    null_in_scope = in_scope.filter(pl.col("CERT").is_null()).height
    dup = FAIL.filter(pl.col("CERT").is_not_null() & pl.col("CERT").is_duplicated())
    combos = (dup.group_by("CERT").agg(pl.col("RESTYPE").sort().str.join("+").alias("events"))
              .group_by("events").agg(pl.len().alias("banks")))
    p(f"* Null CERT: {FAIL['CERT'].null_count()} rows, of which **{null_in_scope}** inside the 1984-2026 scope "
      f"(all null-CERT events are pre-1984 resolutions). Events before scope: "
      f"{FAIL.filter(pl.col('FAILDATE') < pl.date(1984, 4, 1)).height:,}.")
    p(f"* {combos['banks'].sum()} CERTs appear more than once (e.g. open-bank assistance followed later by "
      "failure). The label uses the first event with RESTYPE = FAILURE:")
    table(combos.to_pandas(), "failures_duplicate_cert", floatfmt=".0f")
    fe = in_scope.filter(pl.col("last_rep").is_not_null()).with_columns(
        ((pl.col("FAILDATE") - pl.col("last_rep")).dt.total_days()).alias("days"))
    lag = fe.with_columns(pl.col("days").cut([0, 92, 183, 365], labels=["<=0 (reported after failure)", "1-92",
                                                                       "93-183", "184-365", ">365"]).alias("bucket"))
    lt = lag.group_by("RESTYPE", "bucket").agg(pl.len().alias("n")).sort("RESTYPE", "bucket").to_pandas()
    p("Days between the last Call Report and FAILDATE:")
    table(lt, "failure_report_lag", floatfmt=".0f")
    after = fe.filter((pl.col("days") <= 0) & (pl.col("RESTYPE") == "FAILURE"))
    by = (after.group_by("SAVR", "FAILYR").agg(pl.len().alias("n"), pl.col("days").median().alias("median_days"))
          .sort("n", descending=True).head(8))
    p(f"{after.height} FAILURE events have Call Reports dated on/after FAILDATE. They are concentrated in:")
    table(by.to_pandas(), "failures_reported_after", floatfmt=".0f")
    p("These are mainly savings institutions placed in RTC conservatorship in 1989: FAILDATE is the date of "
      "takeover and the institution kept filing until it was resolved. The event is public at FAILDATE, so "
      "all reports dated on/after FAILDATE are dropped from the panel.")
    nores = in_scope.filter(pl.col("last_rep").is_null() & pl.col("CERT").is_not_null()).height
    issue("Failures without Call Reports", "in-scope failure events whose CERT never appears in Financials", nores,
          "cannot be labelled; reported as a (tiny) coverage limitation")
    issue("Reports after FAILDATE", "failed institution keeps filing after FAILDATE (1989 RTC conservatorships)",
          after.height, "rows with REPDTE >= FAILDATE are dropped; label uses the first FAILURE date")
    issue("Repeated events per CERT", "assistance followed by failure, or two failures",
          int(combos['banks'].sum()), "first RESTYPE = FAILURE event defines the label; assistance is not a failure")
    return in_scope.to_pandas()


# ------------------------------------------------------------------ 9. exits / institutions
def check_exits() -> None:
    h("9. Exits: failure vs. other exits (Institutions table)")
    last = FIN.group_by("CERT").agg(pl.col("REPDTE").max().alias("last_rep")).collect()
    maxq = last["last_rep"].max()
    failed = set(FAIL.filter(pl.col("RESTYPE") == "FAILURE")["CERT"].drop_nulls().to_list())
    ex = (last.filter(pl.col("last_rep") < maxq).join(INST.select("CERT", "CHANGEC1", "ACTIVE"), on="CERT", how="left")
          .with_columns(pl.col("CERT").is_in(list(failed)).alias("in_failures")))
    ct = (ex.group_by("CHANGEC1").agg(pl.len().alias("exited_banks"),
                                      pl.col("in_failures").sum().alias("in_Failures_table"))
          .with_columns((100 * pl.col("in_Failures_table") / pl.col("exited_banks")).round(1).alias("% failed"))
          .sort("exited_banks", descending=True).head(12))
    table(ct.to_pandas(), "exit_codes", floatfmt=".1f")
    c211 = ex.filter(pl.col("CHANGEC1") == 211)
    nf = ex.filter(pl.col("in_failures"))
    p(f"Code 211 is the usual 'failure' code, but only {100 * nf.filter(pl.col('CHANGEC1') == 211).height / nf.height:.1f}% "
      f"of failed banks carry it (others carry merger-type codes 213/217/235 that also occur for healthy mergers), "
      f"and {c211.height - c211['in_failures'].sum()} banks with 211 are not in the Failures table. CHANGEC is "
      "therefore NOT used to define failure: the Failures table (RESTYPE = FAILURE) is the authoritative source; "
      "every other exit (merger, voluntary liquidation, charter conversion) has label 0.")
    no_inst = last.join(INST.select("CERT"), on="CERT", how="anti").height
    issue("Exit classification", "banks that stop reporting: failure vs. merger/other",
          f"{ex.height} exited banks", "failure iff CERT in Failures with RESTYPE=FAILURE; other exits keep label 0 "
                                       "and leave the panel after their last report")
    if no_inst:
        issue("Orphan CERTs", "CERTs in Financials missing from Institutions", no_inst, "kept (Institutions not needed for features)")


# ------------------------------------------------------------------ 10. point-in-time
def check_pit() -> None:
    h("10. Point-in-time traps")
    ch = (FIN.group_by("CERT").agg(pl.col("BKCLASS").n_unique().alias("k"), pl.col("STNAME").n_unique().alias("s"))
          .collect())
    nb = ch.height
    p(f"* {int((ch['k'] > 1).sum()):,} of {nb:,} banks ({100 * (ch['k'] > 1).mean():.1f}%) changed charter class "
      f"(BKCLASS) and {int((ch['s'] > 1).sum()):,} changed state over time. The Institutions table stores only the "
      "**latest** value, so joining it to old quarters would leak today's information into the past.")
    p("* Institutions fields ACTIVE, ENDEFYMD, CHANGEC*, INACTIVE describe the bank's *end* of life: using them as "
      "features is direct look-ahead. The pipeline uses Institutions only to *classify exits for the label*.")
    p("* The API serves the latest revised (amended) value of each Call Report, not the vintage first published "
      "~30-45 days after quarter end. Restatements cannot be detected from this source (limitation).")
    issue("Static attributes", "Institutions holds current, not historical, BKCLASS/STNAME/SPECGRP",
          f"{int((ch['k'] > 1).sum())} banks", "use quarter-specific BKCLASS/STNAME from Financials")
    issue("Exit fields", "ACTIVE/ENDEFYMD/CHANGEC are future information", "all", "never used as features")


def check_integral() -> None:
    h("11. Type drift in the raw files")
    d = pd.read_csv(RESULTS / "dq_raw_dtype_drift.csv")
    p(f"* {len(d)} columns change physical type between raw quarter files (e.g. int64 in one quarter, double "
      "when a null appears). The curated layer enforces one schema (amounts Int64, ratios Float64, REPDTE Date).")
    table(d.rename(columns={d.columns[0]: "column"}), floatfmt=".0f", max_rows=15)
    issue("Schema drift", "column dtypes differ between quarter files", len(d), "explicit schema cast in 02_storage.py")


def main() -> None:
    MD.append("# Automated data-quality report\n")
    MD.append("Generated by `src/03_quality.py` from the curated layer. All numbers are recomputed on every run.\n")
    check_completeness(); check_keys(); check_missing(); check_domain(); check_identities()
    check_ytd(); check_gaps(); check_failures(); check_exits(); check_pit(); check_integral()
    summ = pd.DataFrame(SUMMARY)
    summ.to_csv(RESULTS / "dq_summary.csv", index=False)
    MD.insert(2, "\n## Summary of issues and handling\n\n" + summ.to_markdown(index=False) + "\n")
    (RESULTS / "dq_report.md").write_text("".join(m if m.endswith("\n") else m + "\n" for m in MD), encoding="utf-8")
    print(summ.to_string())


if __name__ == "__main__":
    main()
