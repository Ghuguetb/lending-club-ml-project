import time
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml import Pipeline
from pyspark import StorageLevel

t_start = time.time()

# Configuracion de Spark ajustada al hardware real de esta maquina (2 nucleos, 7.8 GB RAM),
# no a la especificacion "obligatoria" de 8g+8g (16 GB), que no cabe aqui.
# Se documenta esta desviacion tal como permite el enunciado ("u otros equivalentes
# debidamente justificados"). shuffle.partitions/default.parallelism se bajan de 400 a 8:
# 400 particiones diminutas en 2 nucleos generan overhead de scheduling sin beneficio real;
# 400 tiene sentido en un cluster con decenas de cores, no en una laptop de 2 nucleos.
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

NUMERIC_VARS = ["loan_amnt","int_rate","dti","fico_range_high","open_acc",
                "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc",
                "inq_last_6mths","emp_length_num","annual_inc_log","revol_bal_log"]
CATEGORICAL_VARS = ["term","sub_grade","home_ownership","verification_status",
                     "purpose","addr_state_grouped","initial_list_status","application_type"]
RARE_THRESHOLD = 0.01

cols_needed = ["id","loan_status","loan_amnt","int_rate","dti","fico_range_high","open_acc",
               "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc","inq_last_6mths",
               "emp_length","annual_inc","revol_bal","term","sub_grade","home_ownership",
               "verification_status","purpose","addr_state","initial_list_status","application_type"]

print("Leyendo CSV completo con Spark...")
t0 = time.time()
raw = spark.read.option("multiLine", "true").option("escape", '"').csv(
    "../data/accepted_2007_to_2018Q4.csv", header=True, inferSchema=False)
raw = raw.select(cols_needed)
double_cols = ["loan_amnt","int_rate","dti","fico_range_high","open_acc","revol_util",
               "total_acc","pub_rec","delinq_2yrs","mort_acc","inq_last_6mths","annual_inc","revol_bal"]
for c in double_cols:
    raw = raw.withColumn(c, F.col(c).cast("double"))
print(f"Tiempo de lectura del CSV: {time.time()-t0:.2f} s")

# ---- Ingenieria de variables (equivalente a la version de scikit-learn) ----
df = raw.withColumn("default", F.when(F.col("loan_status") == "Charged Off", 1).otherwise(0).cast("int"))

# emp_length: "10+ years"->10, "< 1 year"->0, "n years"->n, null->null
df = df.withColumn("emp_length_num",
    F.when(F.col("emp_length").contains("10+"), 10.0)
     .when(F.col("emp_length").contains("< 1"), 0.0)
     .when(F.col("emp_length").isNull(), None)
     .otherwise(F.regexp_extract(F.col("emp_length"), r"(\d+)", 1).cast("double")))

# dti: -1 y >=999 son centinelas invalidos (detectado en el EDA) -> nulo
df = df.withColumn("dti", F.when((F.col("dti") == -1) | (F.col("dti") >= 999), None).otherwise(F.col("dti")))

# transformaciones log (mismo criterio que en scikit-learn)
df = df.withColumn("annual_inc_log", F.log1p(F.greatest(F.col("annual_inc"), F.lit(0.0))))
df = df.withColumn("revol_bal_log", F.log1p(F.greatest(F.col("revol_bal"), F.lit(0.0))))

# ---- Cargar la particion comun (NO randomSplit) ----
split_df = spark.read.parquet("../data/split_assignment.parquet")
df = df.join(split_df, on="id", how="inner")

train_df = df.filter(F.col("split") == "train")
test_df = df.filter(F.col("split") == "test")

# ---- addr_state: agrupar categorias raras, calculado SOLO con train ----
n_train = train_df.count()
freq = (train_df.groupBy("addr_state").count()
        .withColumn("freq", F.col("count") / F.lit(n_train)))
keep_states = [r["addr_state"] for r in freq.filter(F.col("freq") >= RARE_THRESHOLD).collect()]
print(f"\naddr_state: se mantienen {len(keep_states)} categorias (fit en train), el resto -> 'OTHER'")

def group_state(dframe):
    return dframe.withColumn("addr_state_grouped",
        F.when(F.col("addr_state").isin(keep_states), F.col("addr_state")).otherwise(F.lit("OTHER")))

train_df = group_state(train_df)
test_df = group_state(test_df)

print(f"Train: {train_df.count():,}  Test: {test_df.count():,}")

# ---- Imputer numerico (mediana), StringIndexer+OHE categoricas, ajustados SOLO con train ----
imputer = Imputer(strategy="median", inputCols=NUMERIC_VARS,
                   outputCols=[c+"_imp" for c in NUMERIC_VARS])

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in CATEGORICAL_VARS]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_ohe") for c in CATEGORICAL_VARS]

num_assembler = VectorAssembler(inputCols=[c+"_imp" for c in NUMERIC_VARS], outputCol="numeric_vec")
scaler = StandardScaler(inputCol="numeric_vec", outputCol="numeric_scaled", withMean=True, withStd=True)

final_assembler = VectorAssembler(
    inputCols=["numeric_scaled"] + [c+"_ohe" for c in CATEGORICAL_VARS],
    outputCol="features")

pipeline = Pipeline(stages=[imputer] + indexers + encoders + [num_assembler, scaler, final_assembler])

print("\nAjustando pipeline de preprocesamiento (fit SOLO con train)...")
t0 = time.time()
fitted_pipeline = pipeline.fit(train_df)
print(f"Tiempo de fit: {time.time()-t0:.2f} s")

train_prep = fitted_pipeline.transform(train_df).select("id", "default", "features")
test_prep  = fitted_pipeline.transform(test_df).select("id", "default", "features")

# Cache OBLIGATORIO despues del VectorAssembler y antes del modelo
train_prep = train_prep.persist(StorageLevel.MEMORY_AND_DISK)
test_prep = test_prep.persist(StorageLevel.MEMORY_AND_DISK)
train_prep.count()  # materializa el cache
test_prep.count()

print(f"\nDimension del vector de features: {len(train_prep.first()['features'])}")

# Guardar a Parquet para reutilizar en el notebook de modelado (evita recomputar el pipeline)
train_prep.write.mode("overwrite").parquet("../data/spark_train_prepared.parquet")
test_prep.write.mode("overwrite").parquet("../data/spark_test_prepared.parquet")
fitted_pipeline.write().overwrite().save("../data/spark_preprocessing_pipeline")

print(f"\nTiempo total del script: {time.time()-t_start:.2f} s")
spark.stop()
