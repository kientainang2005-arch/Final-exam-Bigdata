"""
Step 5 - Benchmark: pandas baseline vs Polars eager / lazy / streaming (Part 3).

The SAME core pipeline (clean -> de-cumulate YTD -> label -> ratio features -> 4-quarter
lags -> trailing rolling stats -> state medians) is run by each engine in a FRESH subprocess,
so peak memory (Windows peak working set / Linux maxrss) is measured per run and no engine
benefits from another's warm caches inside the process. Each configuration runs 3 times
(median reported). Outputs are cross-checked for equality.

A scaling experiment replicates the panel k times (new CERTs) to see how time and memory
grow, which is the evidence for the "when would a cluster be needed" argument.

    python src/05_benchmark.py                 # full benchmark (~15-25 min)
    python src/05_benchmark.py --quick         # 1 repeat, no scaling
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CURATED, DATA, FIG, RESULTS  # noqa: E402  (first: sets POLARS env)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import psutil  # noqa: E402

FIN_DIR = CURATED / "financials"
FAIL_PATH = CURATED / "failures.parquet"
OUT_COLS = ["CERT", "qi", "y", "eq_ta", "npa_ta", "npl_ln", "roa_q", "nim_q", "texas", "cre_ta",
            "brokered_dep", "d4_eq_ta", "d4_npa_ta", "g4_asset", "roa_vol8", "roa_4q", "state_npa_med"]


# =============================================================== engines
def run_polars(mode: str, src: Path) -> pl.DataFrame:
    import transforms as T
    import features as F
    fin = pl.scan_parquet(src / "**" / "*.parquet", hive_partitioning=True).drop("year", strict=False)
    fails = T.failure_events(FAIL_PATH)
    if mode == "eager":
        # same expressions, but materialise after every step (what an "eager" user would write)
        df = fin.collect()
        df = T.clean(df.lazy()).collect()
        df = T.decumulate(df.lazy()).collect()
        df = T.add_label(df.lazy(), fails.collect().lazy()).collect()
        df = F.ratio_features(df.lazy()).collect()
        df = F.dynamic_features(df.lazy()).collect()
        df = F.context_features(df.lazy(), fails).collect()
        return df.select(OUT_COLS)
    lf = F.context_features(F.dynamic_features(F.ratio_features(T.add_label(T.decumulate(T.clean(fin)), fails))), fails)
    lf = lf.select(OUT_COLS)
    return lf.collect(engine="streaming") if mode == "streaming" else lf.collect()


def run_pandas(src: Path) -> pd.DataFrame:
    """Idiomatic pandas implementation of the identical logic."""
    files = sorted(src.rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop(columns="year", errors="ignore")
    # A clean
    df = df[df["ASSET"] > 0].copy()
    nonneg = ["ASSET", "DEP", "DEPDOM", "CHBAL", "SC", "LNLSGR", "LNLSNET", "LNRE", "LNRECONS", "LNREMULT",
              "LNRENRES", "LNRERES", "LNCI", "LNCON", "LNAG", "LNATRES", "NCLNLS", "P3ASSET", "P9ASSET",
              "NAASSET", "ORE", "BRO", "COREDEP", "VOLIAB", "FREPP", "OTHBFHLB"]
    for c in nonneg:
        df[c] = df[c].astype("float64").mask(df[c] < 0)
    d = pd.to_datetime(df["REPDTE"])
    df["qi"] = d.dt.year * 4 + (d.dt.month - 1) // 3
    # B de-cumulate
    df = df.sort_values(["CERT", "qi"], kind="stable").reset_index(drop=True)
    g = df.groupby("CERT", sort=False)
    prev_qi = g["qi"].shift(1)
    q1 = pd.to_datetime(df["REPDTE"]).dt.month.eq(3)
    consec = prev_qi.eq(df["qi"] - 1)
    for c in ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS", "EQCDIV"]:
        v = df[c].astype("float64")
        df[f"{c}_Q"] = np.where(q1, v, np.where(consec, v - g[c].shift(1).astype("float64"), np.nan))
    # C label
    f = pd.read_parquet(FAIL_PATH)
    f = f[f["RESTYPE"].str.upper().eq("FAILURE") & f["CERT"].notna()].groupby("CERT", as_index=False)["FAILDATE"].min()
    fd = pd.to_datetime(f["FAILDATE"])
    f["fail_qi"] = fd.dt.year * 4 + (fd.dt.month - 1) // 3
    df = df.merge(f, on="CERT", how="left")
    df = df[df["FAILDATE"].isna() | (pd.to_datetime(df["REPDTE"]) < pd.to_datetime(df["FAILDATE"]))]
    df["y"] = ((df["fail_qi"] > df["qi"]) & (df["fail_qi"] <= df["qi"] + 4)).astype("int8")

    def sdiv(a, b):
        a, b = pd.Series(a, index=df.index).astype("float64"), pd.Series(b, index=df.index).astype("float64")
        return (a / b).where(b.ne(0) & b.notna())
    # D ratio features (subset used in the comparison + those needed downstream)
    ore0 = df["ORE"].fillna(0)
    df["eq_ta"] = sdiv(df.EQ, df.ASSET)
    df["npl_ln"] = sdiv(df.NCLNLS, df.LNLSGR)
    df["npa_ta"] = sdiv(df.NCLNLS + ore0, df.ASSET)
    df["texas"] = sdiv(df.NCLNLS + ore0, df.EQ + df.LNATRES)
    df["roa_q"] = sdiv(4 * df.NETINC_Q, df.ASSET)
    df["nim_q"] = sdiv(4 * (df.INTINC_Q - df.EINTEXP_Q), df.ASSET)
    df["cre_ta"] = sdiv(df.LNRECONS + df.LNREMULT + df.LNRENRES, df.ASSET)
    df["brokered_dep"] = sdiv(df.BRO, df.DEP)
    # E lags via self-merge on (CERT, qi-4)
    lag = df[["CERT", "qi", "eq_ta", "npa_ta", "ASSET"]].copy()
    lag["qi"] += 4
    df = df.merge(lag, on=["CERT", "qi"], how="left", suffixes=("", "_l4"))
    df["d4_eq_ta"] = df.eq_ta - df.eq_ta_l4
    df["d4_npa_ta"] = df.npa_ta - df.npa_ta_l4
    df["g4_asset"] = sdiv(df.ASSET, df.ASSET_l4) - 1
    # F rolling on the quarter index (time-based window: fake 1 quarter = 1 day)
    df = df.sort_values(["CERT", "qi"], kind="stable").reset_index(drop=True)
    df["_t"] = pd.to_datetime(df["qi"], unit="D")
    r = df.groupby("CERT", sort=False).rolling("8D", on="_t", min_periods=4)["roa_q"].std()
    df["roa_vol8"] = r.reset_index(level=0, drop=True).values
    r = df.groupby("CERT", sort=False).rolling("4D", on="_t", min_periods=2)["roa_q"].mean()
    df["roa_4q"] = r.reset_index(level=0, drop=True).values
    # G state medians per quarter
    df["state_npa_med"] = df.groupby(["STNAME", "qi"], observed=True)["npa_ta"].transform("median")
    return df[OUT_COLS]


# =============================================================== subprocess harness
def child(engine: str, src: str, out: str | None) -> None:
    src = Path(src)
    t0 = time.perf_counter()
    res = run_pandas(src) if engine == "pandas" else run_polars(engine.split("-")[1], src)
    dt = time.perf_counter() - t0
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None) or __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024
    if out:
        (pl.from_pandas(res) if isinstance(res, pd.DataFrame) else res).write_parquet(out)
    print(json.dumps({"seconds": dt, "peak_mb": peak / 2**20, "rows": len(res)}))


def launch(engine: str, src: Path, out: Path | None = None) -> dict:
    cmd = [sys.executable, __file__, "--child", engine, "--src", str(src)] + (["--out", str(out)] if out else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def compare(a: Path, b: Path) -> dict:
    x = pl.read_parquet(a).sort(["CERT", "qi"])
    y = pl.read_parquet(b).sort(["CERT", "qi"])
    res = {"rows_a": x.height, "rows_b": y.height}
    worst = 0.0
    for c in OUT_COLS[2:]:
        xa, ya = x[c].cast(pl.Float64).to_numpy(), y[c].cast(pl.Float64).to_numpy()
        nan_mismatch = int((np.isnan(xa) != np.isnan(ya)).sum())
        diff = np.nanmax(np.abs(xa - ya)) if (~np.isnan(xa) & ~np.isnan(ya)).any() else 0.0
        worst = max(worst, float(diff))
        res[c] = {"null_mismatch": nan_mismatch, "max_abs_diff": float(diff)}
    res["max_abs_diff_all"] = worst
    return res


def replicate(k: int) -> Path:
    """Panel replicated k times with disjoint CERTs (synthetic scale-up)."""
    dst = DATA / f"bench_x{k}"
    if dst.exists():
        return dst
    base = pl.scan_parquet(FIN_DIR / "**" / "*.parquet", hive_partitioning=True)
    # write year by year so the replicated data never has to fit in memory at once
    years = base.select(pl.col("year").unique()).collect()["year"].sort().to_list()
    for yr in years:
        part = base.filter(pl.col("year") == yr).drop("year")
        out = dst / f"year={yr}"
        out.mkdir(parents=True, exist_ok=True)
        pl.concat([part.with_columns((pl.col("CERT") + i * 1_000_000).cast(pl.Int32)) for i in range(k)]) \
          .sink_parquet(out / "part-0.parquet", compression="zstd")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child"), ap.add_argument("--src"), ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.src, a.out)

    repeat = 1 if a.quick else 3
    engines = ["pandas", "polars-eager", "polars-lazy", "polars-streaming"]
    tmp = DATA / "bench_out"
    tmp.mkdir(exist_ok=True)
    rows = []
    print(f"machine: {psutil.cpu_count(logical=True)} logical cores, "
          f"{psutil.virtual_memory().total / 2**30:.1f} GB RAM, polars {pl.__version__}, pandas {pd.__version__}")
    for e in engines:
        runs = [launch(e, FIN_DIR, tmp / f"{e}.parquet" if i == 0 else None) for i in range(repeat)]
        r = {"engine": e, "scale": 1, "rows_out": runs[0]["rows"],
             "seconds_median": statistics.median(x["seconds"] for x in runs),
             "seconds_min": min(x["seconds"] for x in runs),
             "peak_mb_median": statistics.median(x["peak_mb"] for x in runs)}
        rows.append(r)
        print(f"  {e:<18s} {r['seconds_median']:7.2f} s   peak {r['peak_mb_median']:7.0f} MB", flush=True)
    eq = {e: compare(tmp / "pandas.parquet", tmp / f"{e}.parquet") for e in engines[1:]}
    (RESULTS / "benchmark_equality.json").write_text(json.dumps(eq, indent=1))
    print("max |pandas - polars| over all compared columns:",
          {e: v["max_abs_diff_all"] for e, v in eq.items()})

    if not a.quick:
        for k in (2, 4, 8):
            src = replicate(k)
            for e in (["pandas"] if k <= 4 else []) + ["polars-lazy", "polars-streaming"]:
                try:
                    r0 = launch(e, src)
                    rows.append({"engine": e, "scale": k, "rows_out": r0["rows"], "seconds_median": r0["seconds"],
                                 "seconds_min": r0["seconds"], "peak_mb_median": r0["peak_mb"]})
                    print(f"  x{k} {e:<18s} {r0['seconds']:7.2f} s   peak {r0['peak_mb']:7.0f} MB", flush=True)
                except RuntimeError as err:
                    print(f"  x{k} {e}: FAILED ({str(err)[-200:]})")
                    rows.append({"engine": e, "scale": k, "rows_out": None, "seconds_median": None,
                                 "seconds_min": None, "peak_mb_median": None})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "benchmark_engines.csv", index=False, float_format="%.3f")
    print(res.to_string(index=False))
    plot(res)


def plot(res: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    colors = {"pandas": "#7f8c8d", "polars-eager": "#e67e22", "polars-lazy": "#2a6f97", "polars-streaming": "#16a085"}
    for e, g in res.dropna(subset=["seconds_median"]).groupby("engine"):
        g = g.sort_values("scale")
        rows_m = g["rows_out"] / 1e6
        axes[0].plot(rows_m, g["seconds_median"], "o-", label=e, color=colors.get(e))
        axes[1].plot(rows_m, g["peak_mb_median"] / 1024, "o-", label=e, color=colors.get(e))
    axes[0].set(xlabel="output rows (millions)", ylabel="wall time (s)", title="Time vs. data size")
    axes[1].set(xlabel="output rows (millions)", ylabel="peak memory (GB)", title="Peak memory vs. data size")
    for ax in axes:
        ax.grid(alpha=.3)
    axes[1].axhline(psutil.virtual_memory().total / 2**30, ls="--", color="k", lw=.8)
    axes[1].text(0.02, psutil.virtual_memory().total / 2**30 * .95, "laptop RAM", fontsize=8, va="top",
                 transform=axes[1].get_yaxis_transform())
    axes[0].legend(fontsize=8)
    fig.tight_layout(), fig.savefig(FIG / "benchmark_scaling.png", dpi=150), plt.close(fig)


if __name__ == "__main__":
    main()
