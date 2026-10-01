"""Seccion 9.10.4.7 (LIME): prepara los datos de fondo que necesita LimeTabularExplainer para
estimar las estadisticas de cada variable (media/desvio de las numericas, frecuencias de las
categoricas ya one-hot-encoded). No hace falta el 1.8M de filas completo para esto -- una
submuestra aleatoria estratificada (semilla fija) de 30.000 filas de TRAIN es mas que suficiente y
evita convertir a denso el dataset completo (car0 para sklearn, y para Spark implica materializar
el vector "features" completo, que hoy vive solo como Vector de Spark)."""
import json
import numpy as np
import pandas as pd
from scipy import sparse

N_SUB = 30000
SEED = 42

# ---------------- scikit-learn ----------------
print("[sklearn] preparando fondo LIME...", flush=True)
X_train = sparse.load_npz("data/sklearn_X_train.npz")
y_train = np.load("data/sklearn_y_train.npy")
rng = np.random.default_rng(SEED)
idx_pos = np.where(y_train == 1)[0]
idx_neg = np.where(y_train == 0)[0]
frac = N_SUB / len(y_train)
sub_idx_sk = np.concatenate([
    rng.choice(idx_pos, size=int(round(len(idx_pos) * frac)), replace=False),
    rng.choice(idx_neg, size=int(round(len(idx_neg) * frac)), replace=False),
])
fondo_sklearn = X_train[sub_idx_sk].toarray()
np.savez_compressed("data/lime_background_sklearn.npz", X=fondo_sklearn, y=y_train[sub_idx_sk])
print(f"  guardado data/lime_background_sklearn.npz: {fondo_sklearn.shape}, "
      f"tasa default={y_train[sub_idx_sk].mean()*100:.2f}%", flush=True)

# ---------------- PySpark ----------------
print("\n[spark] preparando fondo LIME (necesita materializar el vector 'features')...", flush=True)
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel

spark = (SparkSession.builder.appName("LimeBackgroundSpark").master("local[2]")
         .config("spark.driver.memory", "4g").config("spark.executor.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8").config("spark.default.parallelism", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

train = spark.read.parquet("data/spark_train_prepared.parquet")

# nombres de columna reales del vector "features" (107 dims): las 14 numericas vienen genericas
# ("numeric_scaled_0..13") en los metadatos de ML porque son la salida de un VectorAssembler+Scaler
# -- se traducen a sus nombres reales con NUMERIC_VARS (mismo orden que en preprocess_pyspark.py);
# las 93 categoricas ya tienen nombres legibles en los metadatos ("sub_grade_ohe_C1", etc.)
NUMERIC_VARS = ["loan_amnt", "int_rate", "dti", "fico_range_high", "open_acc",
                "revol_util", "total_acc", "pub_rec", "delinq_2yrs", "mort_acc",
                "inq_last_6mths", "emp_length_num", "annual_inc_log", "revol_bal_log"]
meta = train.schema["features"].metadata["ml_attr"]["attrs"]
n_total = meta_n = sum(len(v) for v in meta.values())
nombres_spark = [None] * n_total
for a in meta["numeric"]:
    nombres_spark[a["idx"]] = NUMERIC_VARS[a["idx"]]
for a in meta["binary"]:
    nombres_spark[a["idx"]] = a["name"].replace("_ohe_", "_")
assert all(n is not None for n in nombres_spark)
with open("data/lime_feature_names_spark.json", "w") as f:
    json.dump(nombres_spark, f, ensure_ascii=False, indent=1)
print(f"  {len(nombres_spark)} nombres de variables de Spark guardados en data/lime_feature_names_spark.json")

frac_sp = N_SUB / train.count()
fondo_df = (train.sample(withReplacement=False, fraction=min(frac_sp * 1.5, 1.0), seed=SEED)
                  .limit(N_SUB)
                  .withColumn("features_arr", vector_to_array(F.col("features"))))
fondo_df.select("id", "default", "features_arr").write.mode("overwrite").parquet("data/_lime_bg_spark_tmp.parquet")
spark.stop()

tmp = pd.read_parquet("data/_lime_bg_spark_tmp.parquet")
X_spark_bg = np.stack(tmp["features_arr"].values)
np.savez_compressed("data/lime_background_spark.npz", X=X_spark_bg, y=tmp["default"].values)
print(f"  guardado data/lime_background_spark.npz: {X_spark_bg.shape}, "
      f"tasa default={tmp['default'].values.mean()*100:.2f}%", flush=True)

import shutil
shutil.rmtree("data/_lime_bg_spark_tmp.parquet")
print("\nListo.")
