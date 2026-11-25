"""Run the whole pipeline end to end:  python run_all.py  [--skip-download] [--skip-spark] [--quick-bench]"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--skip-download", action="store_true", help="reuse data_raw/ (download is resumable anyway)")
ap.add_argument("--skip-spark", action="store_true", help="skip the PySpark notebook (needs Java 17)")
ap.add_argument("--quick-bench", action="store_true", help="1 repeat, no scaling experiment")
ap.add_argument("--spark-benchmark", action="store_true",
                help="also re-measure Spark vs Polars at equal threads (src/10_spark_benchmark.py, ~40 min)")
a = ap.parse_args()


def run(step: list[str]) -> None:
    t0 = time.time()
    print(f"\n=== {' '.join(step)} ===", flush=True)
    subprocess.run([sys.executable, "-u", *step], cwd=ROOT, check=True)
    print(f"=== done in {time.time() - t0:.0f}s ===", flush=True)


steps = []
if not a.skip_download:
    steps.append(["src/01_download.py"])
steps += [["src/02_storage.py"], ["src/03_quality.py"], ["src/04_panel.py"],
          ["src/05_benchmark.py"] + (["--quick"] if a.quick_bench else []),
          ["src/06_features.py"], ["src/07_model.py"],
          ["src/08_feature_importance.py"],     # development period only
          ["src/09_test_subperiods.py"]]        # post-hoc split of the frozen test predictions
for s in steps:
    run(s)
if not a.skip_spark:
    # register a kernel inside the venv so the notebook runs with the pinned packages
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--sys-prefix", "--name", "bft65b",
                    "--display-name", "Python (bft65b)"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "notebooks/make_spark_notebook.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.kernel_name=bft65b",
                    "notebooks/08_spark_demo.ipynb"], cwd=ROOT, check=True)
if a.spark_benchmark:
    # measure Spark vs Polars at equal threads (needs Java 17; ~40 min for x1/x2/x4) and keep the log
    with open(ROOT / "logs" / "spark_benchmark" / "rerun.log", "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "src/10_spark_benchmark.py"], cwd=ROOT, check=True, stdout=log)
# summarise the benchmark logs in logs/spark_benchmark/ into results/spark_benchmark.csv (fast, no Spark)
run(["src/10_spark_benchmark.py", "--summarize-logs",
     *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "logs" / "spark_benchmark").glob("*.log"))])
run(["report/build_report.py"])
print("\nAll steps finished. Results in results/, figures in results/figures/, report in report/.")
