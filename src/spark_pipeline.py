"""
The key steps of the pipeline in the PySpark DataFrame API (same logic as notebooks/08_spark_demo.ipynb,
src/transforms.py and src/features.py), packaged as a function so the Spark benchmark can run it on any
input size. Only reads Parquet (no Hadoop winutils needed on Windows).
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

NONNEG = ["ASSET", "DEP", "DEPDOM", "CHBAL", "SC", "LNLSGR", "LNLSNET", "LNRE", "LNRECONS", "LNREMULT",
          "LNRENRES", "LNRERES", "LNCI", "LNCON", "LNAG", "LNATRES", "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET",
          "ORE", "BRO", "COREDEP", "VOLIAB", "FREPP", "OTHBFHLB"]
YTD = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS", "EQCDIV"]
OUT = ["CERT", "qi", "y", "eq_ta", "npa_ta", "npl_ln", "roa_q", "nim_q", "texas", "cre_ta", "brokered_dep",
       "d4_eq_ta", "d4_npa_ta", "g4_asset", "roa_vol8", "roa_4q", "state_npa_med"]


def read_partitioned(spark: SparkSession, root: Path) -> DataFrame:
    """Explicit file list + basePath: keeps Hive partition discovery without directory listing (Windows)."""
    files = sorted(p.as_posix() for p in root.glob("year=*/*.parquet"))
    return spark.read.option("basePath", root.as_posix()).parquet(*files)


def sdiv(a, b):
    return F.when(b.isNull() | (b == 0), None).otherwise(a / b)


def build(spark: SparkSession, fin: DataFrame, failures_path: Path) -> DataFrame:
    df = fin.filter(F.col("ASSET") > 0)
    for c in NONNEG:
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))
    df = df.withColumn("qi", F.year("REPDTE") * 4 + F.floor((F.month("REPDTE") - 1) / 3).cast("int"))
    w = Window.partitionBy("CERT").orderBy("qi")
    prev_qi = F.lag("qi").over(w)
    for c in YTD:
        df = df.withColumn(c + "_Q", F.when(F.month("REPDTE") == 3, F.col(c))
                                       .when(prev_qi == F.col("qi") - 1, F.col(c) - F.lag(c).over(w)))
    fails = (spark.read.parquet(failures_path.as_posix())
             .filter((F.upper("RESTYPE") == "FAILURE") & F.col("CERT").isNotNull())
             .groupBy("CERT").agg(F.min("FAILDATE").alias("FAILDATE"))
             .withColumn("fail_qi", F.year("FAILDATE") * 4 + F.floor((F.month("FAILDATE") - 1) / 3).cast("int")))
    df = (df.join(F.broadcast(fails), "CERT", "left")
            .filter(F.col("FAILDATE").isNull() | (F.col("REPDTE") < F.col("FAILDATE")))
            .withColumn("y", F.coalesce(((F.col("fail_qi") > F.col("qi")) &
                                         (F.col("fail_qi") <= F.col("qi") + 4)).cast("int"), F.lit(0))))
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
    w8 = Window.partitionBy("CERT").orderBy("qi").rangeBetween(-7, 0)
    w4 = Window.partitionBy("CERT").orderBy("qi").rangeBetween(-3, 0)
    df = (df.withColumn("roa_vol8", F.when(F.count("roa_q").over(w8) >= 4, F.stddev_samp("roa_q").over(w8)))
            .withColumn("roa_4q", F.when(F.count("roa_q").over(w4) >= 2, F.avg("roa_q").over(w4))))
    df = df.withColumn("state_npa_med", F.median("npa_ta").over(Window.partitionBy("STNAME", "qi")))
    return df.select(OUT)
