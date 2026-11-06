"""Shared paths, constants and small helpers for the BFT65B pipeline."""
from __future__ import annotations

import os

# Polars' import-time CPU-feature probe cannot read CPUID when x64 Python runs under emulation
# on Windows-on-ARM (this laptop: Snapdragon X). The binary itself works; skip only the probe.
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

import json
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "fdic"            # landing zone written by 01_download.py
DATA = ROOT / "data"                        # curated / derived layers (git-ignored)
CURATED = DATA / "curated"
RESULTS = ROOT / "results"                  # small tables & figures that ARE committed
FIG = RESULTS / "figures"
for p in (DATA, CURATED, RESULTS, FIG):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
SCOPE_START, SCOPE_END = "1984-03-31", "2026-06-30"
LABEL_HORIZON_Q = 4                         # y = 1{fail in (q, q+4]}
LAST_MODEL_QUARTER = "2025-06-30"           # later labels are incomplete


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


@contextmanager
def timer(label: str, store: dict | None = None):
    """Wall-clock timer that also records the RSS memory delta of this process."""
    proc = psutil.Process(os.getpid())
    m0, t0 = proc.memory_info().rss, time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    dm = (proc.memory_info().rss - m0) / 2**20
    print(f"  {label:<45s} {dt:8.3f} s   dRSS {dm:+8.1f} MB", flush=True)
    if store is not None:
        store[label] = dt


def spark_env() -> None:
    """Point PySpark at a JDK 17 if JAVA_HOME is not already set (see README)."""
    if not os.environ.get("JAVA_HOME"):
        cand = sorted(Path.home().joinpath("tools").glob("jdk-17*"))
        if cand:
            os.environ["JAVA_HOME"] = str(cand[-1])
    os.environ.setdefault("PYSPARK_PYTHON", str(Path(os.sys.executable)))
