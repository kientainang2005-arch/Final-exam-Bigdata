"""
Step 10 - Spark vs. Polars at equal thread counts and several data sizes (Part 3).

The same job - the key pipeline steps of src/spark_pipeline.py / src/transforms.py + src/features.py,
ending in the sum of the 15 output columns (which forces every window, join and median to be computed) -
is run by PySpark (local[4], local[8]) and by Polars lazy (4 and 8 threads) on the real panel (x1) and on
the replicated panels of 05_benchmark.py (x2, x4). Every run is a fresh subprocess; the Spark session
(JVM) start-up is timed separately from the computation; peak memory is the sampled peak RSS of the whole
process tree (Python + JVM). The column sums of Spark and Polars are compared at every scale.

    python src/10_spark_benchmark.py                   (needs Java 17; x1, x2, x4: about 40 minutes)
    python src/10_spark_benchmark.py --scales 1        (real panel only)
writes results/spark_benchmark.csv and results/spark_benchmark_parity.csv

    python src/10_spark_benchmark.py --summarize-logs logs/spark_benchmark/*.log
rebuilds results/spark_benchmark.csv from the per-run lines printed in benchmark logs.

On the author's laptop (x64 Python and JVM emulated on an ARM CPU) no VALID session got past x1. Two earlier
sessions reached x2/x4 but were discarded for timing (the laptop slept during one, the other ran throttled on
battery); their parity checks are still valid and are kept in logs/spark_benchmark/discarded/. Of the two
sessions on mains power with sleep blocked, the first ended when the JVM crashed (access violation in its
garbage collector, see logs/spark_benchmark/jvm_crash_session1_excerpt.txt) and the second when Windows itself
crashed. The report uses their completed x1 runs, summarised with --summarize-logs. Session 1 ran an earlier
version of this script without the automatic repeat of crashed Spark runs; the measurement code was the same.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CURATED, DATA, RESULTS, spark_env  # noqa: E402  (first: sets POLARS env)

import psutil  # noqa: E402

OUT_SUM = ["y", "eq_ta", "npa_ta", "npl_ln", "roa_q", "nim_q", "texas", "cre_ta", "brokered_dep",
           "d4_eq_ta", "d4_npa_ta", "g4_asset", "roa_vol8", "roa_4q", "state_npa_med"]
SCALES = {1: CURATED / "financials", 2: DATA / "bench_x2", 4: DATA / "bench_x4"}
SPARK_REPEATS = {1: 3, 2: 2, 4: 2}
POLARS_REPEATS = 3
THREADS = (4, 8)


class PeakTreeRSS:
    """Samples the RSS of this process and all its children (e.g. the Spark JVM) every 0.2 s."""

    def __init__(self):
        self.peak, self._stop = 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        me = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
                self.peak = max(self.peak, rss)
            except psutil.Error:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def child_spark(threads: int, src: Path) -> dict:
    spark_env()
    from pyspark.sql import SparkSession, functions as F
    import spark_pipeline as SP
    with PeakTreeRSS() as mem:
        t0 = time.perf_counter()
        spark = (SparkSession.builder.master(f"local[{threads}]").appName("bench")
                 .config("spark.driver.memory", "6g").config("spark.sql.shuffle.partitions", "16")
                 .config("spark.sql.adaptive.enabled", "true").config("spark.sql.session.timeZone", "UTC")
                 .config("spark.ui.enabled", "false").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        t1 = time.perf_counter()
        out = SP.build(spark, SP.read_partitioned(spark, src), CURATED / "failures.parquet")
        row = out.agg(F.count("*").alias("rows"), *[F.sum(c).alias(c) for c in OUT_SUM]).collect()[0].asDict()
        t2 = time.perf_counter()
        spark.stop()
    return {"session_s": t1 - t0, "compute_s": t2 - t1, "total_s": t2 - t0, "peak_tree_mb": mem.peak / 2**20,
            "rows": row.pop("rows"), "sums": row}


def child_polars(src: Path) -> dict:
    import polars as pl
    import features as FE
    import transforms as T
    with PeakTreeRSS() as mem:
        t0 = time.perf_counter()
        fin = pl.scan_parquet(src / "**" / "*.parquet", hive_partitioning=True).drop("year", strict=False)
        fails = T.failure_events()
        lf = FE.context_features(FE.dynamic_features(FE.ratio_features(T.add_label(T.decumulate(T.clean(fin)), fails))), fails)
        row = lf.select(pl.len().alias("rows"), *[pl.col(c).cast(pl.Float64).sum().alias(c) for c in OUT_SUM]).collect().row(0, named=True)
        t1 = time.perf_counter()
    return {"session_s": 0.0, "compute_s": t1 - t0, "total_s": t1 - t0, "peak_tree_mb": mem.peak / 2**20,
            "rows": row.pop("rows"), "sums": row, "threads": pl.thread_pool_size()}


JVM_CRASH_MARKERS = ("ConnectionRefusedError", "Py4JNetworkError", "Answer from Java side is empty",
                     "fatal error has been detected by the Java Runtime")


def launch(engine: str, threads: int, scale: int, max_attempts: int = 4) -> dict:
    """Run one measurement in a fresh process. On this laptop the x64 JVM, emulated on an ARM CPU, occasionally
    dies with EXCEPTION_ACCESS_VIOLATION inside its garbage collector; such a run is discarded, counted and repeated."""
    env = {**os.environ}
    if engine == "polars":
        env["POLARS_MAX_THREADS"] = str(threads)
    cmd = [sys.executable, __file__, "--child", engine, "--threads", str(threads), "--scale", str(scale)]
    crashes = 0
    for _ in range(max_attempts):
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        if p.returncode == 0 and lines:
            # Windows prints "SUCCESS: The process ... terminated" when the JVM is stopped: take the JSON line
            res = json.loads(lines[-1])
            res["jvm_crashes_before"] = crashes
            return res
        if engine == "spark" and any(m in (p.stderr + p.stdout) for m in JVM_CRASH_MARKERS):
            crashes += 1
            print(f"    (JVM crashed, run discarded and repeated; crashes so far for this run: {crashes})", flush=True)
            continue
        raise RuntimeError(p.stderr[-2000:])
    raise RuntimeError(f"{engine} x{scale} {threads} threads: JVM crashed {crashes} times in a row")


RUN_LINE = re.compile(r"^\s*x(\d+) (polars|spark)\s+(\d+) threads: total\s+([\d.]+)s \(session\s+([\d.]+)s, "
                      r"compute\s+([\d.]+)s\) peak\s+(\d+) MB")


def summarize_logs(paths: list[str]) -> None:
    """Aggregate the per-run lines of one or more benchmark logs (completed runs only)."""
    import pandas as pd
    runs = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
            m = RUN_LINE.match(line)
            if m:
                sc, eng, th, tot, ses, comp, peak = m.groups()
                runs.append({"scale": int(sc), "engine": eng, "threads": int(th), "total_s": float(tot),
                             "session_s": float(ses), "compute_s": float(comp), "peak_tree_mb": float(peak),
                             "log": Path(p).name})
    df = pd.DataFrame(runs)
    df.to_csv(RESULTS / "spark_benchmark_runs.csv", index=False)
    out = (df.groupby(["scale", "engine", "threads"])
             .agg(runs=("total_s", "size"), session_s=("session_s", "median"), compute_s=("compute_s", "median"),
                  total_s=("total_s", "median"), total_s_min=("total_s", "min"), total_s_max=("total_s", "max"),
                  peak_tree_mb=("peak_tree_mb", "max"), logs=("log", lambda s: " + ".join(sorted(set(s)))))
             .reset_index())
    out.insert(0, "config", [f"{e} {'local[' + str(t) + ']' if e == 'spark' else str(t) + ' threads'}"
                             for e, t in zip(out.engine, out.threads)])
    out.to_csv(RESULTS / "spark_benchmark.csv", index=False, float_format="%.3f")
    print(out.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child"), ap.add_argument("--threads", type=int), ap.add_argument("--scale", type=int)
    ap.add_argument("--scales", default="1,2,4", help="comma-separated data sizes to run, e.g. 1 or 1,2")
    ap.add_argument("--summarize-logs", nargs="+", metavar="LOG", help="summarise existing benchmark logs instead")
    a = ap.parse_args()
    if a.child:
        res = child_spark(a.threads, SCALES[a.scale]) if a.child == "spark" else child_polars(SCALES[a.scale])
        print(json.dumps(res))
        return
    if a.summarize_logs:
        return summarize_logs(a.summarize_logs)
    scales = [int(s) for s in a.scales.split(",")]

    import pandas as pd
    # A laptop that goes to sleep during a run inflates wall-clock times (a first run recorded a 21-hour
    # "measurement"). On Windows, ask the OS not to sleep while this process runs; the request ends with
    # the process and does not change any power setting.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    print(f"machine: {psutil.cpu_count(logical=True)} logical cores, {psutil.virtual_memory().total / 2**30:.1f} GB RAM",
          flush=True)
    rows, parity = [], []
    for scale in scales:
        sums = {}
        for engine in ("polars", "spark"):
            for th in THREADS:
                reps = POLARS_REPEATS if engine == "polars" else SPARK_REPEATS[scale]
                runs = []
                for _ in range(reps):
                    r = launch(engine, th, scale)
                    runs.append(r)
                    print(f"  x{scale} {engine:<6s} {th} threads: total {r['total_s']:7.1f}s "
                          f"(session {r['session_s']:5.1f}s, compute {r['compute_s']:7.1f}s) "
                          f"peak {r['peak_tree_mb']:7.0f} MB", flush=True)
                if engine == "polars":
                    assert runs[0]["threads"] == th, f"Polars used {runs[0]['threads']} threads, expected {th}"
                sums[(engine, th)] = runs[0]["sums"]
                rows.append({"config": f"{engine} {'local[' + str(th) + ']' if engine == 'spark' else str(th) + ' threads'}",
                             "scale": scale, "engine": engine, "threads": th, "rows": runs[0]["rows"], "runs": reps,
                             "session_s": statistics.median(r["session_s"] for r in runs),
                             "compute_s": statistics.median(r["compute_s"] for r in runs),
                             "total_s": statistics.median(r["total_s"] for r in runs),
                             "total_s_min": min(r["total_s"] for r in runs),
                             "total_s_max": max(r["total_s"] for r in runs),
                             "peak_tree_mb": max(r["peak_tree_mb"] for r in runs),
                             "jvm_crashes_discarded": sum(r.get("jvm_crashes_before", 0) for r in runs)})
        base = sums[("polars", 8)]
        worst = max(abs(s[c] - base[c]) / max(abs(base[c]), 1e-12)
                    for s in sums.values() for c in OUT_SUM)
        parity.append({"scale": scale, "configurations_compared": len(sums), "max_relative_sum_difference": worst})
        print(f"  x{scale} parity: max relative difference of column sums across all runs = {worst:.1e}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "spark_benchmark.csv", index=False, float_format="%.3f")
    pd.DataFrame(parity).to_csv(RESULTS / "spark_benchmark_parity.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
