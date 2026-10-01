import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 2. Preprocesamiento

Sección 9.10.4.2. Se define **una sola partición común** (80/20, estratificada, `random_state=42`)
que usan tanto scikit-learn como PySpark, guardada como `id` + `split` en Parquet — esto es lo que
permite que la prueba de DeLong compare AUC calculados sobre exactamente las mismas observaciones.
""")

add_md(nb, "## 2.1. Partición común de los datos")

add_code(nb, """import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("../data/accepted_raw.parquet", columns=["id","loan_status"])
df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")

id_train, id_test = train_test_split(
    df["id"], test_size=0.20, stratify=df["default"], random_state=42,
)

split_df = pd.DataFrame({
    "id": pd.concat([id_train, id_test]).values,
    "split": ["train"]*len(id_train) + ["test"]*len(id_test),
}).astype({"id": "string", "split": "string"})
split_df.to_parquet("../data/split_assignment.parquet", index=False)

print(f"Train: {len(id_train):,} ({len(id_train)/len(df)*100:.2f}%)")
print(f"Test:  {len(id_test):,} ({len(id_test)/len(df)*100:.2f}%)")

merged = split_df.merge(df, on="id")
print("\\nTasa de default por partición (verifica la estratificación):")
print((merged.groupby("split", observed=True)["default"].mean()*100).round(2))
""")

add_md(nb, """La partición queda guardada en `data/split_assignment.parquet` con solo dos columnas
(`id`, `split`). Tanto el pipeline de scikit-learn como el de PySpark parten del CSV original y
hacen un *join* contra este archivo — así garantizamos exactamente las mismas 1.808.560 filas de
entrenamiento y 452.141 de prueba en ambos entornos, sin usar `randomSplit` de Spark (que no
reproduciría esta misma partición).
""")

add_md(nb, """## 2.2. scikit-learn

**Selección de variables.** A partir de las decisiones del EDA (sección 1.5), se usan variables
disponibles al momento de originación (sin fuga de datos), se descarta `fico_range_low`
(redundante con `fico_range_high`, r=1.0), se usa `sub_grade` en vez de `grade` (lo contiene por
completo) y se excluye `installment` (r=0.95 con `loan_amnt`).

**Transformaciones aplicadas** (todas ajustadas únicamente con el conjunto de entrenamiento):
- `dti`: los valores centinela -1 y ≥999 (detectados en el EDA) se tratan como faltantes.
- `annual_inc` y `revol_bal`: transformación `log1p` por su asimetría extrema (494 y 13.2).
- `emp_length`: se convierte de texto ordinal a numérico (`10+ years`→10, `< 1 year`→0, etc.).
- `addr_state`: categorías con <1% de frecuencia en train se agrupan en `"OTHER"`.
- Imputación numérica con mediana, escalado con `StandardScaler`, codificación con `OneHotEncoder`
  (`handle_unknown="ignore"`) — **sin usar `Pipeline`**, tal como pide el enunciado: cada
  transformador se ajusta y aplica manualmente.
""")

add_code(nb, """import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib, json

NUMERIC_VARS = ["loan_amnt","int_rate","dti","fico_range_high","open_acc",
                 "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc",
                 "inq_last_6mths","emp_length_num","annual_inc_log","revol_bal_log"]
CATEGORICAL_VARS = ["term","sub_grade","home_ownership","verification_status",
                     "purpose","addr_state_grouped","initial_list_status","application_type"]
RARE_THRESHOLD = 0.01

def parse_emp_length(x):
    if pd.isna(x): return np.nan
    x = str(x)
    if "10+" in x: return 10.0
    if "< 1" in x: return 0.0
    digits = "".join(ch for ch in x if ch.isdigit())
    return float(digits) if digits else np.nan

def engineer_features(df):
    df = df.copy()
    df["emp_length_num"] = df["emp_length"].apply(parse_emp_length)
    df["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    df.loc[(df["dti"] == -1) | (df["dti"] >= 999), "dti"] = np.nan
    df["annual_inc_log"] = np.log1p(pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=0))
    df["revol_bal_log"] = np.log1p(pd.to_numeric(df["revol_bal"], errors="coerce").clip(lower=0))
    return df

cols_needed = ["id","loan_status","loan_amnt","int_rate","dti","fico_range_high","open_acc",
               "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc","inq_last_6mths",
               "emp_length","annual_inc","revol_bal","term","sub_grade","home_ownership",
               "verification_status","purpose","addr_state","initial_list_status","application_type"]
df = pd.read_parquet("../data/accepted_raw.parquet", columns=cols_needed)
df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")
df = engineer_features(df)

split = pd.read_parquet("../data/split_assignment.parquet")
df = df.merge(split, on="id", how="inner")
train_mask = df["split"] == "train"
test_mask = df["split"] == "test"
print(f"Train: {train_mask.sum():,}  Test: {test_mask.sum():,}")
""")

add_code(nb, """freq_train = df.loc[train_mask, "addr_state"].value_counts(normalize=True)
keep_states = set(freq_train[freq_train >= RARE_THRESHOLD].index)
print(f"addr_state: se mantienen {len(keep_states)} categorías, el resto -> 'OTHER'")
df["addr_state_grouped"] = df["addr_state"].where(df["addr_state"].isin(keep_states), "OTHER")

X_train_raw = df.loc[train_mask, NUMERIC_VARS + CATEGORICAL_VARS]
X_test_raw  = df.loc[test_mask,  NUMERIC_VARS + CATEGORICAL_VARS]
y_train = df.loc[train_mask, "default"].values
y_test  = df.loc[test_mask,  "default"].values
id_train = df.loc[train_mask, "id"].values
id_test  = df.loc[test_mask,  "id"].values

num_imputer = SimpleImputer(strategy="median")
X_train_num = num_imputer.fit_transform(X_train_raw[NUMERIC_VARS])
X_test_num  = num_imputer.transform(X_test_raw[NUMERIC_VARS])

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled  = scaler.transform(X_test_num)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_cat = ohe.fit_transform(X_train_raw[CATEGORICAL_VARS].astype(str))
X_test_cat  = ohe.transform(X_test_raw[CATEGORICAL_VARS].astype(str))
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_VARS).tolist()

X_train = sparse.hstack([sparse.csr_matrix(X_train_num_scaled), X_train_cat]).tocsr()
X_test  = sparse.hstack([sparse.csr_matrix(X_test_num_scaled),  X_test_cat]).tocsr()
feature_names = NUMERIC_VARS + cat_feature_names

print(f"Dimensión final: X_train={X_train.shape}, X_test={X_test.shape}")

sparse.save_npz("../data/sklearn_X_train.npz", X_train)
sparse.save_npz("../data/sklearn_X_test.npz", X_test)
np.save("../data/sklearn_y_train.npy", y_train)
np.save("../data/sklearn_y_test.npy", y_test)
np.save("../data/sklearn_id_train.npy", id_train)
np.save("../data/sklearn_id_test.npy", id_test)
with open("../data/sklearn_feature_names.json","w") as f:
    json.dump(feature_names, f)
joblib.dump({"imputer": num_imputer, "scaler": scaler, "ohe": ohe, "keep_states": keep_states},
            "../data/sklearn_preprocessors.joblib")
print("Guardado en data/sklearn_*")
""")

add_md(nb, """## 2.3. PySpark

**Configuración de Spark — desviación justificada de la especificación "obligatoria".**
El enunciado pide `executor.memory=8g` + `driver.memory=8g` (16 GB en total). Esta máquina tiene
**7.8 GB de RAM y 2 núcleos** — 16 GB de heap simplemente no caben. Se usa una configuración
reducida y proporcional al hardware real:

| Parámetro | Valor "obligatorio" | Valor usado aquí | Justificación |
|---|---|---|---|
| `driver.memory` | 8g | **4g** | deja margen para el SO y el proceso Python driver en una máquina de 7.8 GB |
| `executor.memory` | 8g | **4g** | en `local[2]` no hay un proceso executor separado: todo corre en la misma JVM del driver, así que este valor es más documental que efectivo aquí |
| `shuffle.partitions` | 400 | **8** | 400 particiones diminutas en 2 núcleos generan overhead de scheduling sin ningún beneficio; 400 tiene sentido en un clúster con decenas de núcleos, no en un portátil |
| `default.parallelism` | 400 | **8** | mismo argumento |
| `memory.fraction` / `memory.storageFraction` | 0.8 / 0.3 | igual | son fracciones, no memoria absoluta — no dependen del tamaño de RAM disponible |

Esto se retoma en la reflexión final: la comparación de velocidad entre PySpark y scikit-learn en
este proyecto está condicionada por correr Spark en modo local con recursos reducidos, no en un
clúster real.

**Nota técnica encontrada al implementar:** el CSV de Lending Club tiene campos de texto libre
(p. ej. `desc`) con comas y saltos de línea dentro de comillas. `pandas` maneja esto por defecto,
pero el lector CSV de Spark no, salvo que se indique `multiLine=true` — sin esa opción, algunas
filas quedan desalineadas (un código de estado como `"WI"` terminaba intentándose convertir a
`double`). Se agregó `.option("multiLine", "true")`.
""")

add_code(nb, """import time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder as SparkOHE, VectorAssembler, StandardScaler as SparkScaler, Imputer
from pyspark.ml import Pipeline
from pyspark import StorageLevel

t_start = time.time()

spark = (SparkSession.builder
         .appName("LendingClub_Optimized")
         .master("local[2]")
         .config("spark.driver.memory", "4g")
         .config("spark.executor.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.default.parallelism", "8")
         .config("spark.memory.fraction", 0.8)
         .config("spark.memory.storageFraction", 0.3)
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("SparkSession creada con configuración adaptada al hardware disponible.")
""")

add_code(nb, """cols_needed = ["id","loan_status","loan_amnt","int_rate","dti","fico_range_high","open_acc",
               "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc","inq_last_6mths",
               "emp_length","annual_inc","revol_bal","term","sub_grade","home_ownership",
               "verification_status","purpose","addr_state","initial_list_status","application_type"]

t0 = time.time()
raw = spark.read.option("multiLine", "true").option("escape", '"').csv(
    "../data/accepted_2007_to_2018Q4.csv", header=True, inferSchema=False)
raw = raw.select(cols_needed)
double_cols = ["loan_amnt","int_rate","dti","fico_range_high","open_acc","revol_util",
               "total_acc","pub_rec","delinq_2yrs","mort_acc","inq_last_6mths","annual_inc","revol_bal"]
for c in double_cols:
    raw = raw.withColumn(c, F.col(c).cast("double"))
print(f"Tiempo de lectura del CSV completo con Spark: {time.time()-t0:.2f} s")
""")

add_md(nb, """*(Nota: no se usa `.toPandas()` ni `.collect()` sobre el dataset completo en ningún punto de
este pipeline — la única llamada a `.collect()` más adelante es sobre una tabla de frecuencias ya
agregada de `addr_state`, del tamaño de 51 estados, no sobre las filas del dataset.)*
""")

add_code(nb, """NUMERIC_VARS = ["loan_amnt","int_rate","dti","fico_range_high","open_acc",
                "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc",
                "inq_last_6mths","emp_length_num","annual_inc_log","revol_bal_log"]
CATEGORICAL_VARS = ["term","sub_grade","home_ownership","verification_status",
                     "purpose","addr_state_grouped","initial_list_status","application_type"]
RARE_THRESHOLD = 0.01

df_spark = raw.withColumn("default", F.when(F.col("loan_status") == "Charged Off", 1).otherwise(0).cast("int"))

df_spark = df_spark.withColumn("emp_length_num",
    F.when(F.col("emp_length").contains("10+"), 10.0)
     .when(F.col("emp_length").contains("< 1"), 0.0)
     .when(F.col("emp_length").isNull(), None)
     .otherwise(F.regexp_extract(F.col("emp_length"), r"(\\d+)", 1).cast("double")))

df_spark = df_spark.withColumn("dti", F.when((F.col("dti") == -1) | (F.col("dti") >= 999), None).otherwise(F.col("dti")))
df_spark = df_spark.withColumn("annual_inc_log", F.log1p(F.greatest(F.col("annual_inc"), F.lit(0.0))))
df_spark = df_spark.withColumn("revol_bal_log", F.log1p(F.greatest(F.col("revol_bal"), F.lit(0.0))))

split_df_spark = spark.read.parquet("../data/split_assignment.parquet")
df_spark = df_spark.join(split_df_spark, on="id", how="inner")

train_df = df_spark.filter(F.col("split") == "train")
test_df = df_spark.filter(F.col("split") == "test")

n_train = train_df.count()
freq = (train_df.groupBy("addr_state").count().withColumn("freq", F.col("count")/F.lit(n_train)))
keep_states_spark = [r["addr_state"] for r in freq.filter(F.col("freq") >= RARE_THRESHOLD).collect()]
print(f"addr_state (Spark, fit en train): se mantienen {len(keep_states_spark)} categorías")

def group_state(dframe):
    return dframe.withColumn("addr_state_grouped",
        F.when(F.col("addr_state").isin(keep_states_spark), F.col("addr_state")).otherwise(F.lit("OTHER")))

train_df = group_state(train_df)
test_df = group_state(test_df)
print(f"Train: {train_df.count():,}  Test: {test_df.count():,}")
""")

add_code(nb, """imputer = Imputer(strategy="median", inputCols=NUMERIC_VARS, outputCols=[c+"_imp" for c in NUMERIC_VARS])
indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in CATEGORICAL_VARS]
encoders = [SparkOHE(inputCol=c+"_idx", outputCol=c+"_ohe") for c in CATEGORICAL_VARS]
num_assembler = VectorAssembler(inputCols=[c+"_imp" for c in NUMERIC_VARS], outputCol="numeric_vec")
scaler_spark = SparkScaler(inputCol="numeric_vec", outputCol="numeric_scaled", withMean=True, withStd=True)
final_assembler = VectorAssembler(inputCols=["numeric_scaled"] + [c+"_ohe" for c in CATEGORICAL_VARS], outputCol="features")

pipeline = Pipeline(stages=[imputer] + indexers + encoders + [num_assembler, scaler_spark, final_assembler])

t0 = time.time()
fitted_pipeline = pipeline.fit(train_df)  # fit SOLO con train
print(f"Tiempo de fit del pipeline: {time.time()-t0:.2f} s")

train_prep = fitted_pipeline.transform(train_df).select("id", "default", "features")
test_prep  = fitted_pipeline.transform(test_df).select("id", "default", "features")

# Cache OBLIGATORIO despues del VectorAssembler y antes del modelo
train_prep = train_prep.persist(StorageLevel.MEMORY_AND_DISK)
test_prep = test_prep.persist(StorageLevel.MEMORY_AND_DISK)
train_prep.count(); test_prep.count()

print(f"Dimensión del vector de features (Spark): {len(train_prep.first()['features'])}")
""")

add_code(nb, """train_prep.write.mode("overwrite").parquet("../data/spark_train_prepared.parquet")
test_prep.write.mode("overwrite").parquet("../data/spark_test_prepared.parquet")
fitted_pipeline.write().overwrite().save("../data/spark_preprocessing_pipeline")
print(f"Tiempo total del pipeline de PySpark: {time.time()-t_start:.2f} s")
spark.stop()
""")

add_md(nb, """**Nota de equivalencia:** el vector final tiene 107 columnas en PySpark frente a 114 en
scikit-learn. La diferencia viene de cómo cada librería maneja categorías: `OneHotEncoder` de
scikit-learn con `handle_unknown="ignore"` no reserva slot para "desconocido", mientras que
`StringIndexer` de Spark con `handleInvalid="keep"` sí agrega una categoría adicional por variable
para valores no vistos, y Spark además *descarta* por defecto la última categoría de cada
`OneHotEncoder` (evita colinealidad perfecta / *dummy variable trap*), algo que scikit-learn no
hace a menos que se pida `drop="first"`. Ninguno de los dos pipelines es "incorrecto"; son
implementaciones equivalentes pero no idénticas — igual que las demás equivalencias aproximadas
que documenta el enunciado (regParam vs. C, hinge vs. squared hinge, maxBins).
""")

save(nb, "book/02_preprocesamiento.ipynb")
print("Notebook 02_preprocesamiento.ipynb creado.")
