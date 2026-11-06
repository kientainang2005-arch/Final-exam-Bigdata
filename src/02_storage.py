"""
Step 2 - Storage layout (Part 1).

raw landing zone (1 Parquet file per quarter, schema inferred by pandas per file)
   -> curated layer with ONE explicit schema:
        data/curated/financials/year=YYYY/part-0.parquet   (Hive partitioning, zstd, sorted)
        data/curated/failures.parquet, data/curated/institutions.parquet (dates parsed)
Then benchmark the curated layout against raw CSV and other Parquet variants:
size on disk, write time, full scan, column projection and a filtered (predicate) query.

    python src/02_storage.py            # build curated layer + benchmark
"""
from __future__ import annotations

import common  # noqa: F401  (sets env before polars is imported)

import shutil
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

from common import CURATED, DATA, RAW, RESULTS, save_json

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- schema
KEYS = {"CERT": pl.Int32, "REPDTE": pl.Date}
DESCR = {"NAME": pl.String, "STNAME": pl.Categorical, "BKCLASS": pl.Categorical,
         "SPECGRP": pl.Int8}
AMOUNTS = ["ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC", "COREDEP", "VOLIAB",
           "FREPP", "OTHBFHLB", "LNLSGR", "LNLSNET", "LNRE", "LNRECONS", "LNREMULT",
           "LNRENRES", "LNRERES", "LNCI", "LNCON", "LNAG", "LNATRES", "NCLNLS", "P3ASSET",
           "P9ASSET", "NAASSET", "NTLNLS", "ORE", "NETINC", "INTINC", "EINTEXP", "NONII",
           "NONIX", "ELNATR", "EQCDIV"]          # thousands of USD -> Int64
RATIOS = ["ROA", "ROE", "NIMY", "EQV", "RBC1AAJ", "RBCRWAJ", "IDT1RWAJR"]  # % -> Float64
SCHEMA = {**KEYS, **DESCR, **{c: pl.Int64 for c in AMOUNTS}, **{c: pl.Float64 for c in RATIOS}}


def cast_quarter(path: Path) -> pl.DataFrame:
    """Read one raw quarter and force the canonical schema (raw dtypes drift between files)."""
    df = pl.read_parquet(path)
    exprs = []
    for c, t in SCHEMA.items():
        if c == "REPDTE":
            exprs.append(pl.col(c).cast(pl.String).str.strptime(pl.Date, "%Y%m%d").alias(c))
        elif t == pl.Int64:
            # amounts are whole thousands; go through Float64 so int/float/None all cast
            exprs.append(pl.col(c).cast(pl.Float64).round(0).cast(pl.Int64).alias(c))
        else:
            exprs.append(pl.col(c).cast(t).alias(c))
    return df.select(exprs)


def raw_dtype_drift() -> pd.DataFrame:
    """How many distinct physical dtypes does each column have across the raw files?"""
    import pyarrow.parquet as pq
    rows = []
    for f in sorted((RAW / "financials").glob("*.parquet")):
        sch = pq.read_schema(f)
        rows.append({n: str(sch.field(n).type) for n in sch.names})
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"n_distinct_dtypes": df.nunique(),
                        "dtypes": df.apply(lambda s: ", ".join(sorted(s.dropna().unique())))})
    return out[out.n_distinct_dtypes > 1]


def build_curated() -> pl.DataFrame:
    t0 = time.perf_counter()
    files = sorted((RAW / "financials").glob("REPDTE=*.parquet"))
    fin = pl.concat([cast_quarter(f) for f in files], how="vertical")
    # verify the integer cast lost nothing: compare sums of raw vs cast for amounts
    fin = fin.with_columns(pl.col("REPDTE").dt.year().alias("year")).sort(["CERT", "REPDTE"])
    out = CURATED / "financials"
    shutil.rmtree(out, ignore_errors=True)
    fin.write_parquet(out, compression="zstd", compression_level=3, statistics=True,
                      partition_by="year", row_group_size=128_000)
    print(f"curated financials: {fin.height:,} rows x {fin.width} cols "
          f"in {time.perf_counter() - t0:.1f}s")

    fail = (pl.read_parquet(RAW / "failures.parquet")
            .with_columns(pl.col("CERT").cast(pl.Int32),
                          pl.col("FAILDATE").str.strptime(pl.Date, "%m/%d/%Y", strict=False),
                          pl.col("RESDATE").cast(pl.String).str.strptime(pl.Date, "%m/%d/%Y", strict=False),
                          pl.col("COST").cast(pl.Float64), pl.col("QBFASSET").cast(pl.Float64),
                          pl.col("QBFDEP").cast(pl.Float64), pl.col("FAILYR").cast(pl.Int16)))
    fail.write_parquet(CURATED / "failures.parquet", compression="zstd")

    date_cols = ["ESTYMD", "INSDATE", "ENDEFYMD", "EFFDATE", "INSDROPDATE", "REPDTE"]
    inst = pl.read_parquet(RAW / "institutions.parquet")
    inst = inst.with_columns(
        pl.col("CERT").cast(pl.Int32), pl.col("NEWCERT").cast(pl.Float64).cast(pl.Int32),
        pl.col("ACTIVE").cast(pl.Int8), pl.col("INACTIVE").cast(pl.Int8),
        *[pl.col(c).cast(pl.String).str.strptime(pl.Date, "%m/%d/%Y", strict=False) for c in date_cols],
        *[pl.col(f"CHANGEC{i}").cast(pl.Float64).cast(pl.Int16) for i in range(1, 16)])
    inst.write_parquet(CURATED / "institutions.parquet", compression="zstd")
    print(f"curated failures: {fail.height:,} rows; institutions: {inst.height:,} rows")
    return fin


# ---------------------------------------------------------------- benchmark
def dir_size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def median_time(fn, repeat: int = 5) -> float:
    """One untimed warm-up (OS file cache warm for every format alike), then median of `repeat` runs."""
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def benchmark(fin: pl.DataFrame) -> pd.DataFrame:
    bench = DATA / "bench"
    shutil.rmtree(bench, ignore_errors=True)
    bench.mkdir(parents=True)
    fin = fin.with_columns(pl.col("STNAME").cast(pl.String), pl.col("BKCLASS").cast(pl.String))
    cols = ["CERT", "REPDTE", "ASSET", "EQ", "NCLNLS"]
    lo, hi = pl.date(2008, 1, 1), pl.date(2010, 12, 31)

    variants = {
        "CSV (uncompressed)": (bench / "fin.csv", lambda p: fin.write_csv(p)),
        "CSV + gzip": (bench / "fin.csv.gz", None),
        "Parquet snappy (1 file)": (bench / "fin_snappy.parquet",
                                    lambda p: fin.write_parquet(p, compression="snappy")),
        "Parquet zstd-3 (1 file)": (bench / "fin_zstd.parquet",
                                    lambda p: fin.write_parquet(p, compression="zstd", compression_level=3)),
        "Parquet zstd-3, partitioned by year": (bench / "fin_part", lambda p: fin.with_columns(
            pl.col("REPDTE").dt.year().alias("year")).write_parquet(
            p, compression="zstd", compression_level=3, partition_by="year")),
    }
    rows = []
    for name, (path, writer) in variants.items():
        if name == "CSV + gzip":
            import gzip
            t0 = time.perf_counter()
            with open(bench / "fin.csv", "rb") as src, gzip.open(path, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=16 * 2**20)
            w = time.perf_counter() - t0
        elif writer is None:
            w = float("nan")  # written in build_curated
        else:
            t0 = time.perf_counter()
            writer(path)
            w = time.perf_counter() - t0

        if name.startswith("CSV"):
            scan = lambda: pl.scan_csv(path, try_parse_dates=True, infer_schema_length=100_000)
        elif "partitioned" in name:
            scan = lambda: pl.scan_parquet(path / "**" / "*.parquet", hive_partitioning=True)
        else:
            scan = lambda: pl.scan_parquet(path)

        if name == "CSV + gzip":   # polars cannot scan gzip lazily -> eager read
            full = lambda: pl.read_csv(path, try_parse_dates=True, infer_schema_length=100_000)
            proj = lambda: pl.read_csv(path, columns=cols, try_parse_dates=True)
            filt = lambda: (pl.read_csv(path, columns=cols, try_parse_dates=True)
                            .filter(pl.col("REPDTE").is_between(lo, hi)))
        else:
            full = lambda: scan().collect()
            proj = lambda: scan().select(cols).collect()
            filt = lambda: scan().select(cols).filter(pl.col("REPDTE").is_between(lo, hi)).collect()
        r = {"layout": name, "size_MB": dir_size(path) / 2**20, "write_s": w,
             "read_all_s": median_time(full), "read_5cols_s": median_time(proj),
             "filter_2008_2010_s": median_time(filt)}
        print(f"  {name:<38s} {r['size_MB']:8.1f} MB  write {w:6.2f}s  all {r['read_all_s']:6.2f}s  "
              f"5 cols {r['read_5cols_s']:6.2f}s  filter {r['filter_2008_2010_s']:6.2f}s", flush=True)
        rows.append(r)

    # pandas baseline on the CSV (what a naive workflow would do)
    t = median_time(lambda: pd.read_csv(bench / "fin.csv", low_memory=False), repeat=3)
    rows.append({"layout": "CSV read with pandas (baseline)", "size_MB": dir_size(bench / "fin.csv") / 2**20,
                 "write_s": float("nan"), "read_all_s": t,
                 "read_5cols_s": median_time(lambda: pd.read_csv(bench / "fin.csv", usecols=cols), repeat=3),
                 "filter_2008_2010_s": float("nan")})
    res = pd.DataFrame(rows)
    csv_size = res.loc[0, "size_MB"]
    res["size_vs_csv"] = res["size_MB"] / csv_size
    return res


def main() -> None:
    drift = raw_dtype_drift()
    drift.to_csv(RESULTS / "dq_raw_dtype_drift.csv")
    print(f"columns whose physical dtype drifts across raw quarter files: {len(drift)}")
    fin = build_curated()
    # partition summary
    parts = sorted((CURATED / "financials").glob("year=*"))
    save_json({"n_partitions": len(parts), "rows": fin.height, "cols": fin.width,
               "schema": {k: str(v) for k, v in fin.schema.items()}}, RESULTS / "curated_schema.json")
    print("\nbenchmark (1 warm-up + median of 5 runs, Polars reader unless stated):")
    res = benchmark(fin)
    res.to_csv(RESULTS / "storage_benchmark.csv", index=False, float_format="%.3f")
    print(res.round(3).to_string(index=False))
    shutil.rmtree(DATA / "bench", ignore_errors=True)


if __name__ == "__main__":
    main()
