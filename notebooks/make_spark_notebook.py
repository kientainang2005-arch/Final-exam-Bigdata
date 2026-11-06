"""Generates notebooks/08_spark_demo.ipynb (kept as code so the notebook is reviewable in Git).
    python notebooks/make_spark_notebook.py
    python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=bft65b notebooks/08_spark_demo.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# PySpark demonstration (Part 3 and Part 5)

Goal: reproduce the key Polars transformations in Spark, check that the results match row by row,
read the physical plan, time both engines on the same laptop, and run a short MLlib model.

Runs locally (`local[*]`). On Windows Spark needs a JDK 17 (`JAVA_HOME`). Without Hadoop's native
`winutils.exe`/`hadoop.dll`, Spark on Windows cannot *list directories* or write files, so the notebook
passes the partition files explicitly with `basePath` (Spark still discovers the `year` partition column
and prunes partitions) and returns results to Python. On Linux/Colab the plain directory read works.
Polars results are produced by the same functions as the main pipeline (`src/transforms.py`,
`src/features.py`).""")

code("""import sys, os, time
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import CURATED, RESULTS, spark_env, SEED
spark_env()
import polars as pl, numpy as np, pandas as pd
from pyspark.sql import SparkSession, functions as F, Window
spark = (SparkSession.builder.master("local[*]").appName("bft65b")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.session.timeZone", "UTC")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("Spark", spark.version, "| cores", os.cpu_count())""")

md("## 1. Read the curated, year-partitioned Parquet layer\nPartition pruning: a filter on `year` touches only matching folders.")
code("""root = (CURATED / "financials").as_posix()
files = sorted(p.as_posix() for p in (CURATED / "financials").glob("year=*/*.parquet"))
fin = spark.read.option("basePath", root).parquet(*files)   # = spark.read.parquet(root) on Linux
print(len(files), "partition files")
print(fin.count(), "rows,", len(fin.columns), "columns")
fin.filter(F.col("year").between(2008, 2010)).select("CERT", "REPDTE", "ASSET").explain("formatted")""")

md("## 2. Clean -> de-cumulate YTD flows -> label (same logic as `transforms.py`)")
code("""NONNEG = ["ASSET","DEP","DEPDOM","CHBAL","SC","LNLSGR","LNLSNET","LNRE","LNRECONS","LNREMULT","LNRENRES",
          "LNRERES","LNCI","LNCON","LNAG","LNATRES","NCLNLS","P3ASSET","P9ASSET","NAASSET","ORE","BRO",
          "COREDEP","VOLIAB","FREPP","OTHBFHLB"]
YTD = ["NETINC","INTINC","EINTEXP","NONII","NONIX","ELNATR","NTLNLS","EQCDIV"]
t0 = time.perf_counter()
df = fin.filter(F.col("ASSET") > 0)
for c in NONNEG:
    df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))
df = df.withColumn("qi", F.year("REPDTE") * 4 + F.floor((F.month("REPDTE") - 1) / 3).cast("int"))
w = Window.partitionBy("CERT").orderBy("qi")
prev_qi = F.lag("qi").over(w)
for c in YTD:
    df = df.withColumn(c + "_Q", F.when(F.month("REPDTE") == 3, F.col(c))
                                   .when(prev_qi == F.col("qi") - 1, F.col(c) - F.lag(c).over(w)))
fails = (spark.read.parquet((CURATED / "failures.parquet").as_posix())
         .filter((F.upper("RESTYPE") == "FAILURE") & F.col("CERT").isNotNull())
         .groupBy("CERT").agg(F.min("FAILDATE").alias("FAILDATE"))
         .withColumn("fail_qi", F.year("FAILDATE") * 4 + F.floor((F.month("FAILDATE") - 1) / 3).cast("int")))
df = (df.join(F.broadcast(fails), "CERT", "left")
        .filter(F.col("FAILDATE").isNull() | (F.col("REPDTE") < F.col("FAILDATE")))
        .withColumn("y", F.coalesce(((F.col("fail_qi") > F.col("qi")) &
                                      (F.col("fail_qi") <= F.col("qi") + 4)).cast("int"), F.lit(0))))""")

md("## 3. Features: ratios, 4-quarter lags (self-join on the quarter index), trailing windows, state medians")
code("""def sdiv(a, b):
    return F.when(b.isNull() | (b == 0), None).otherwise(a / b)
ore0 = F.coalesce(F.col("ORE"), F.lit(0))
df = (df.withColumn("eq_ta", sdiv(F.col("EQ"), F.col("ASSET")))
        .withColumn("npl_ln", sdiv(F.col("NCLNLS"), F.col("LNLSGR")))
        .withColumn("npa_ta", sdiv(F.col("NCLNLS") + ore0, F.col("ASSET")))
        .withColumn("texas", sdiv(F.col("NCLNLS") + ore0, F.col("EQ") + F.col("LNATRES")))
        .withColumn("roa_q", sdiv(4 * F.col("NETINC_Q"), F.col("ASSET")))
        .withColumn("nim_q", sdiv(4 * (F.col("INTINC_Q") - F.col("EINTEXP_Q")), F.col("ASSET")))
        .withColumn("cre_ta", sdiv(F.col("LNRECONS") + F.col("LNREMULT") + F.col("LNRENRES"), F.col("ASSET")))
        .withColumn("brokered_dep", sdiv(F.col("BRO"), F.col("DEP"))))
lag = df.select("CERT", (F.col("qi") + 4).alias("qi"), F.col("eq_ta").alias("eq_ta_l4"),
                F.col("npa_ta").alias("npa_ta_l4"), F.col("ASSET").alias("ASSET_l4"))
df = (df.join(lag, ["CERT", "qi"], "left")
        .withColumn("d4_eq_ta", F.col("eq_ta") - F.col("eq_ta_l4"))
        .withColumn("d4_npa_ta", F.col("npa_ta") - F.col("npa_ta_l4"))
        .withColumn("g4_asset", sdiv(F.col("ASSET").cast("double"), F.col("ASSET_l4").cast("double")) - 1))
# range window on the quarter index = gap-safe trailing window (same semantics as Polars rolling_*_by)
w8 = Window.partitionBy("CERT").orderBy("qi").rangeBetween(-7, 0)
w4 = Window.partitionBy("CERT").orderBy("qi").rangeBetween(-3, 0)
df = (df.withColumn("roa_vol8", F.when(F.count("roa_q").over(w8) >= 4, F.stddev_samp("roa_q").over(w8)))
        .withColumn("roa_4q", F.when(F.count("roa_q").over(w4) >= 2, F.avg("roa_q").over(w4))))
ws = Window.partitionBy("STNAME", "qi")
df = df.withColumn("state_npa_med", F.median("npa_ta").over(ws))
OUT = ["CERT","qi","y","eq_ta","npa_ta","npl_ln","roa_q","nim_q","texas","cre_ta","brokered_dep",
       "d4_eq_ta","d4_npa_ta","g4_asset","roa_vol8","roa_4q","state_npa_med"]
spark_out = df.select(OUT)""")

md("## 4. Physical plan\nRead it bottom-up: `FileScan parquet` (with `ReadSchema` = only the needed columns), "
   "`Exchange hashpartitioning(CERT)` = shuffle for the window/join, `Sort`, `Window`, `BroadcastHashJoin` "
   "for the small failures table, `SortMergeJoin` for the self-join.")
code("""spark_out.explain("formatted")""")

md("## 5. Run, time, and compare with Polars row by row")
code("""t0 = time.perf_counter()
sp = pl.from_arrow(spark_out.toArrow()) if hasattr(spark_out, "toArrow") else pl.from_pandas(spark_out.toPandas())
t_spark = time.perf_counter() - t0

import transforms as T, features as FE
t0 = time.perf_counter()
fl = T.failure_events()
po = FE.context_features(FE.dynamic_features(FE.ratio_features(
        T.add_label(T.decumulate(T.clean(T.scan_financials())), fl))), fl).select(OUT).collect()
t_polars = time.perf_counter() - t0
print(f"Spark local[*]: {t_spark:.1f}s  | Polars lazy: {t_polars:.1f}s  | rows spark {sp.height:,} polars {po.height:,}")

a = po.sort(["CERT","qi"]); b = sp.sort(["CERT","qi"])
assert a.height == b.height and (a["CERT"] == b["CERT"]).all() and (a["qi"] == b["qi"]).all()
rows = []
for c in OUT[2:]:
    x = a[c].cast(pl.Float64).to_numpy(); y = b[c].cast(pl.Float64).to_numpy()
    both = ~np.isnan(x) & ~np.isnan(y)
    rows.append({"column": c, "null_mismatch": int((np.isnan(x) != np.isnan(y)).sum()),
                 "max_abs_diff": float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0,
                 "sum_polars": float(np.nansum(x)), "sum_spark": float(np.nansum(y))})
cmp = pd.DataFrame(rows); cmp.to_csv(RESULTS / "spark_vs_polars.csv", index=False)
pd.DataFrame([{"engine":"spark local[*]","seconds":t_spark},{"engine":"polars lazy","seconds":t_polars}]) \\
  .to_csv(RESULTS / "spark_timing.csv", index=False)
cmp""")

md("## 6. Spark MLlib demo: logistic regression\nTrained on the development period up to 1994 and "
   "validated on 1996-2006 (the last walk-forward fold). The test period is **not** touched here, "
   "so the single-test rule of Part 5 is respected.")
code("""from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
feats = ["eq_ta","npa_ta","npl_ln","roa_q","nim_q","texas","cre_ta","brokered_dep","d4_eq_ta","d4_npa_ta",
         "g4_asset","roa_vol8","roa_4q","state_npa_med"]
# Spark is lazy: without cache() every action below (quantiles, Imputer/Scaler fits, each LR iteration)
# would re-run the whole lineage from the Parquet scan. Materialise the feature table once in memory.
data = spark_out.withColumn("year", F.floor(F.col("qi") / 4)).cache()
t0 = time.perf_counter(); n_cached = data.count(); print(f"cached {n_cached:,} rows in {time.perf_counter() - t0:.1f}s")
# winsorise with quantiles from the TRAINING block only
train = data.filter(F.col("year").between(1986, 1994))
valid = data.filter(F.col("year").between(1996, 2006))
q = train.approxQuantile(feats, [0.005, 0.995], 0.0001)
def clip(d):
    for c, (lo, hi) in zip(feats, q):
        d = d.withColumn(c, F.when(F.col(c) < lo, lo).when(F.col(c) > hi, hi).otherwise(F.col(c)))
    return d
train, valid = clip(train), clip(valid)
pipe = Pipeline(stages=[
    Imputer(inputCols=feats, outputCols=[f + "_i" for f in feats], strategy="median"),
    VectorAssembler(inputCols=[f + "_i" for f in feats], outputCol="raw"),
    StandardScaler(inputCol="raw", outputCol="features"),
    LogisticRegression(labelCol="y", featuresCol="features", regParam=0.01, maxIter=100)])
t0 = time.perf_counter(); model = pipe.fit(train); t_fit = time.perf_counter() - t0
pred = model.transform(valid)
ev_pr = BinaryClassificationEvaluator(labelCol="y", metricName="areaUnderPR").evaluate(pred)
ev_roc = BinaryClassificationEvaluator(labelCol="y", metricName="areaUnderROC").evaluate(pred)
print(f"MLlib logit: fit {t_fit:.1f}s | validation 1996-2006 PR-AUC {ev_pr:.4f} ROC-AUC {ev_roc:.4f}")
pd.DataFrame([{"model":"MLlib logistic (14 features)","train":"1986-1994","valid":"1996-2006",
               "pr_auc":ev_pr,"roc_auc":ev_roc,"fit_seconds":t_fit}]).to_csv(RESULTS / "mllib_demo.csv", index=False)""")
code("""spark.stop()""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "bft65b", "display_name": "Python (bft65b)", "language": "python"}})
Path(__file__).with_name("08_spark_demo.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print("written notebooks/08_spark_demo.ipynb")
