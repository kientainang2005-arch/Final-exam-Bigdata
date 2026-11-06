"""
Feature definitions (Part 4). Pure Polars expressions on the bank-quarter panel.

Point-in-time contract: a feature for (bank i, quarter q) uses only
  * bank i's reports with REPDTE <= q (lags via self-join on the quarter index), and
  * cross-sectional aggregates of OTHER banks' reports at quarters <= q, and
  * failures with FAILDATE <= REPDTE_q (public at that time).
No statistic of the full sample (mean, quantile, scaler...) is computed here: anything
fitted lives in the sklearn Pipeline and is fitted on the training period only.
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import polars as pl


def sdiv(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Safe division: null when the denominator is 0 or null."""
    return pl.when(b.is_null() | (b == 0)).then(None).otherwise(a / b)


# name -> (expression, CAMELS group, rationale)
RATIOS: dict[str, tuple[pl.Expr, str, str]] = {
    # Capital
    "eq_ta":        (sdiv(pl.col("EQ"), pl.col("ASSET")), "C", "equity cushion against losses"),
    # Asset quality
    "npl_ln":       (sdiv(pl.col("NCLNLS"), pl.col("LNLSGR")), "A", "non-current loans share"),
    "npa_ta":       (sdiv(pl.col("NCLNLS") + pl.col("ORE").fill_null(0), pl.col("ASSET")), "A", "non-performing assets incl. repossessed RE"),
    "pd30_ta":      (sdiv(pl.col("P3ASSET"), pl.col("ASSET")), "A", "early delinquency, leads NPLs"),
    "ore_ta":       (sdiv(pl.col("ORE"), pl.col("ASSET")), "A", "foreclosed real estate"),
    "texas":        (sdiv(pl.col("NCLNLS") + pl.col("ORE").fill_null(0), pl.col("EQ") + pl.col("LNATRES")), "A", "Texas ratio: bad assets / loss-absorbing capital"),
    "alll_ln":      (sdiv(pl.col("LNATRES"), pl.col("LNLSGR")), "A", "loan-loss reserve level"),
    "nco_ln":       (sdiv(4 * pl.col("NTLNLS_Q"), pl.col("LNLSGR")), "A", "annualised net charge-offs"),
    "prov_ta":      (sdiv(4 * pl.col("ELNATR_Q"), pl.col("ASSET")), "A", "annualised provisions"),
    # Management / efficiency
    "efficiency":   (sdiv(pl.col("NONIX_Q"), pl.col("INTINC_Q") - pl.col("EINTEXP_Q") + pl.col("NONII_Q")), "M", "cost / income"),
    # Earnings
    "roa_q":        (sdiv(4 * pl.col("NETINC_Q"), pl.col("ASSET")), "E", "annualised quarterly ROA (de-cumulated)"),
    "nim_q":        (sdiv(4 * (pl.col("INTINC_Q") - pl.col("EINTEXP_Q")), pl.col("ASSET")), "E", "annualised net interest margin"),
    "div_payer":    ((pl.col("EQCDIV_Q").fill_null(0) > 0).cast(pl.Int8), "E", "pays dividends this quarter"),
    # Liquidity / funding
    "liq_ta":       (sdiv(pl.col("CHBAL") + pl.col("SC"), pl.col("ASSET")), "L", "liquid assets"),
    "loans_ta":     (sdiv(pl.col("LNLSGR"), pl.col("ASSET")), "L", "loan-to-assets"),
    "coredep_ta":   (sdiv(pl.col("COREDEP"), pl.col("ASSET")), "L", "stable funding"),
    "brokered_dep": (sdiv(pl.col("BRO"), pl.col("DEP")), "L", "hot money"),
    "volliab_ta":   (sdiv(pl.col("VOLIAB"), pl.col("ASSET")), "L", "volatile liabilities"),
    "fhlb_ta":      (sdiv(pl.col("OTHBFHLB"), pl.col("ASSET")), "L", "wholesale borrowings"),
    "cof_ta":       (sdiv(4 * pl.col("EINTEXP_Q"), pl.col("ASSET")), "S", "cost of funds (rate sensitivity)"),
    # Loan mix / concentration
    "cre_ta":       (sdiv(pl.col("LNRECONS") + pl.col("LNREMULT") + pl.col("LNRENRES"), pl.col("ASSET")), "S", "commercial RE concentration"),
    "constr_ta":    (sdiv(pl.col("LNRECONS"), pl.col("ASSET")), "S", "construction & land development"),
    "resid_ln":     (sdiv(pl.col("LNRERES"), pl.col("LNLSGR")), "S", "1-4 family mortgages"),
    "ci_ln":        (sdiv(pl.col("LNCI"), pl.col("LNLSGR")), "S", "C&I loans"),
    "cons_ln":      (sdiv(pl.col("LNCON"), pl.col("LNLSGR")), "S", "consumer loans"),
    "ag_ln":        (sdiv(pl.col("LNAG"), pl.col("LNLSGR")), "S", "agricultural loans"),
    # Size
    "log_assets":   (pl.col("ASSET").cast(pl.Float64).log(), "size", "size (too-big-to-fail, diversification)"),
}
LAG_BASE = ["eq_ta", "npa_ta", "npl_ln", "roa_q"]      # 4-quarter changes
GROWTH_BASE = ["ASSET", "LNLSGR"]                       # 4-quarter growth


def ratio_features(panel: pl.LazyFrame) -> pl.LazyFrame:
    return panel.with_columns([e.alias(n) for n, (e, _, _) in RATIOS.items()])


def dynamic_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Changes vs. 4 quarters earlier via a self-join on (CERT, qi-4): gap-safe and leak-free."""
    lag = lf.select(["CERT", (pl.col("qi") + 4).alias("qi"),
                     *[pl.col(c).alias(f"{c}_l4") for c in LAG_BASE + GROWTH_BASE]])
    lf = lf.join(lag, on=["CERT", "qi"], how="left")
    lf = lf.with_columns(
        *[(pl.col(c) - pl.col(f"{c}_l4")).alias(f"d4_{c}") for c in LAG_BASE],
        *[(sdiv(pl.col(c).cast(pl.Float64), pl.col(f"{c}_l4").cast(pl.Float64)) - 1).alias(f"g4_{c.lower()}")
          for c in GROWTH_BASE],
    ).drop([f"{c}_l4" for c in LAG_BASE + GROWTH_BASE])
    # rolling statistics over the trailing 8 quarters (window defined on qi, so gaps shrink it)
    lf = lf.sort(["CERT", "qi"]).with_columns(
        pl.col("roa_q").rolling_std_by("qi", window_size="8i", min_samples=4).over("CERT").alias("roa_vol8"),
        pl.col("roa_q").rolling_mean_by("qi", window_size="4i", min_samples=2).over("CERT").alias("roa_4q"),
        pl.col("qi").cum_count().over("CERT").alias("n_reports"),   # history length (de novo banks)
    )
    return lf


def context_features(lf: pl.LazyFrame, fails: pl.LazyFrame) -> pl.LazyFrame:
    """System-wide/state stress at quarter q from data public at q."""
    state = (lf.group_by(["STNAME", "qi"])
               .agg(pl.col("npa_ta").median().alias("state_npa_med"),
                    pl.col("roa_q").median().alias("state_roa_med"),
                    pl.len().alias("state_banks")))
    lf = lf.join(state, on=["STNAME", "qi"], how="left").with_columns(
        (pl.col("npa_ta") - pl.col("state_npa_med")).alias("npa_vs_state"))
    # failures in the trailing 4 quarters nationally (FAILDATE <= quarter end -> already public)
    f = (fails.group_by("fail_qi").agg(pl.len().alias("n")).sort("fail_qi")
              .rename({"fail_qi": "qi"}))
    qs = lf.select("qi").unique()
    trailing = (qs.join(f, how="cross", suffix="_f")
                  .filter((pl.col("qi_f") <= pl.col("qi")) & (pl.col("qi_f") > pl.col("qi") - 4))
                  .group_by("qi").agg(pl.col("n").sum().alias("fails_trailing4q_us")))
    return (lf.join(trailing, on="qi", how="left")
              .with_columns(pl.col("fails_trailing4q_us").fill_null(0)))


FEATURES = (list(RATIOS) + [f"d4_{c}" for c in LAG_BASE] + [f"g4_{c.lower()}" for c in GROWTH_BASE]
            + ["roa_vol8", "roa_4q", "n_reports", "state_npa_med", "state_roa_med", "npa_vs_state",
               "fails_trailing4q_us"])


def build_features(panel: pl.LazyFrame, fails: pl.LazyFrame) -> pl.LazyFrame:
    lf = context_features(dynamic_features(ratio_features(panel)), fails)
    keep = ["CERT", "REPDTE", "qi", "STNAME", "BKCLASS", "ASSET", "exit_type", "FAILDATE", "q_to_fail", "y"]
    return lf.select(keep + FEATURES).sort(["REPDTE", "CERT"])
