"""
Core transformations, written once in Polars (lazy) and reused by the panel, the feature
pipeline and the benchmark. Everything is expressed on LazyFrames so the optimiser can
push projections/predicates into the Parquet scan and fuse the window expressions.

Design rule for point-in-time correctness: any value attached to quarter q is computed
only from Call Reports with REPDTE <= q. Lags are taken on the *quarter index* (qi), not on
row position, so a missing quarter yields null instead of silently using an older report.
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import polars as pl

from common import CURATED

YTD_FLOWS = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS", "EQCDIV"]
NONNEG_STOCKS = ["ASSET", "DEP", "DEPDOM", "CHBAL", "SC", "LNLSGR", "LNLSNET", "LNRE", "LNRECONS",
                 "LNREMULT", "LNRENRES", "LNRERES", "LNCI", "LNCON", "LNAG", "LNATRES", "NCLNLS",
                 "P3ASSET", "P9ASSET", "NAASSET", "ORE", "BRO", "COREDEP", "VOLIAB", "FREPP", "OTHBFHLB"]


def qidx(col: str = "REPDTE") -> pl.Expr:
    """Quarter index: consecutive integers, 1984Q1 -> 1984*4+0."""
    return pl.col(col).dt.year().cast(pl.Int32) * 4 + (pl.col(col).dt.month().cast(pl.Int32) - 1) // 3


def scan_financials(path=None) -> pl.LazyFrame:
    path = path or CURATED / "financials"
    return pl.scan_parquet(path / "**" / "*.parquet", hive_partitioning=True).drop("year", strict=False)


# ---------------------------------------------------------------- step A: clean
def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop unusable rows, null impossible negatives, add quarter index."""
    return (lf.filter(pl.col("ASSET") > 0)                                  # ratios need assets
              .with_columns([pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
                             for c in NONNEG_STOCKS])
              .with_columns(qidx().alias("qi")))


# ---------------------------------------------------------------- step B: YTD -> quarterly
def decumulate(lf: pl.LazyFrame) -> pl.LazyFrame:
    """flow_q = YTD_q - YTD_{q-1} if previous quarter of the same year exists; Q1 = YTD."""
    lf = lf.sort(["CERT", "qi"])
    prev_qi = pl.col("qi").shift(1).over("CERT")
    is_q1 = pl.col("REPDTE").dt.month() == 3
    consecutive = (prev_qi == pl.col("qi") - 1)
    return lf.with_columns([
        pl.when(is_q1).then(pl.col(c))
          .when(consecutive).then(pl.col(c) - pl.col(c).shift(1).over("CERT"))
          .otherwise(None).alias(f"{c}_Q")
        for c in YTD_FLOWS])


# ---------------------------------------------------------------- step C: labels
def failure_events(path=None, restypes=("FAILURE",)) -> pl.LazyFrame:
    path = path or CURATED / "failures.parquet"
    return (pl.scan_parquet(path)
              .filter(pl.col("RESTYPE").str.to_uppercase().is_in(list(restypes)),
                      pl.col("CERT").is_not_null())
              .group_by("CERT").agg(pl.col("FAILDATE").min())               # first event per bank
              .with_columns(qidx("FAILDATE").alias("fail_qi"))
              .select("CERT", "FAILDATE", "fail_qi"))


def add_label(lf: pl.LazyFrame, fails: pl.LazyFrame, horizon: int = 4) -> pl.LazyFrame:
    """y = 1{bank fails in (q, q+horizon]}. Rows on/after the failure date are removed."""
    return (lf.join(fails, on="CERT", how="left")
              .filter(pl.col("FAILDATE").is_null() | (pl.col("REPDTE") < pl.col("FAILDATE")))
              .with_columns(((pl.col("fail_qi") > pl.col("qi")) &
                             (pl.col("fail_qi") <= pl.col("qi") + horizon)).fill_null(False)
                            .cast(pl.Int8).alias("y"),
                            (pl.col("fail_qi") - pl.col("qi")).alias("q_to_fail")))


def add_exit_info(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Per-bank exit type known only at the END of the sample - used for labels/analysis only."""
    last_qi = pl.col("qi").max().over("CERT")
    data_end = pl.col("qi").max()
    return lf.with_columns(
        pl.when(pl.col("FAILDATE").is_not_null()).then(pl.lit("failure"))
          .when(last_qi < data_end).then(pl.lit("other exit (merger/closure)"))
          .otherwise(pl.lit("active")).alias("exit_type"),
        (last_qi == pl.col("qi")).alias("is_last_report"))


def build_panel(fin: pl.LazyFrame | None = None, fails: pl.LazyFrame | None = None) -> pl.LazyFrame:
    fin = fin if fin is not None else scan_financials()
    fails = fails if fails is not None else failure_events()
    return add_exit_info(add_label(decumulate(clean(fin)), fails))
