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
a = ap.parse_args()

steps = []
if not a.skip_download:
    steps.append(["src/01_download.py"])
steps += [["src/02_storage.py"], ["src/03_quality.py"], ["src/04_panel.py"],
          ["src/05_benchmark.py"] + (["--quick"] if a.quick_bench else []),
          ["src/06_features.py"], ["src/07_model.py"]]
for s in steps:
    t0 = time.time()
    print(f"\n=== {' '.join(s)} ===", flush=True)
    subprocess.run([sys.executable, "-u", *s], cwd=ROOT, check=True)
    print(f"=== done in {time.time() - t0:.0f}s ===", flush=True)
if not a.skip_spark:
    # register a kernel inside the venv so the notebook runs with the pinned packages
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--sys-prefix", "--name", "bft65b",
                    "--display-name", "Python (bft65b)"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "notebooks/make_spark_notebook.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.kernel_name=bft65b", "notebooks/08_spark_demo.ipynb"], cwd=ROOT, check=True)
print("\nAll steps finished. Results in results/, figures in results/figures/.")
