"""Smoke test: valida que los 6 modelos de PySpark corren sobre una submuestra pequena
y da una estimacion de tiempo por ajuste antes de lanzar la malla completa sobre 1.8M filas."""
import time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier,
                                        RandomForestClassifier, GBTClassifier,
                                        LinearSVC, NaiveBayes)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import StorageLevel

spark = (SparkSession.builder
         .appName("SmokeTest")
         .master("local[2]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.default.parallelism", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("Cargando y submuestreando...", flush=True)
train_full = spark.read.parquet("data/spark_train_prepared.parquet")
n_full = train_full.count()
frac = 20000 / n_full
train = train_full.sample(withReplacement=False, fraction=frac, seed=0).persist(StorageLevel.MEMORY_AND_DISK)
n = train.count()
print(f"Submuestra: {n} filas (de {n_full})", flush=True)

evaluator = BinaryClassificationEvaluator(labelCol="default", metricName="areaUnderROC")

def probar(nombre, estimator, grid, col_score="probability"):
    t0 = time.time()
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid, evaluator=evaluator,
                         numFolds=3, parallelism=1, seed=42)
    model = cv.fit(train)
    t_fit = time.time() - t0
    best_auc = max(model.avgMetrics)
    print(f"OK {nombre}: fit={t_fit:.1f}s  mejores_metricas_cv={model.avgMetrics}", flush=True)
    return t_fit

print("\n[1/6] LogisticRegression...", flush=True)
lr = LogisticRegression(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(lr.regParam, [1e-6, 1e-5, 1e-4]).build()
probar("LogisticRegression", lr, grid)

print("\n[2/6] DecisionTreeClassifier...", flush=True)
dt = DecisionTreeClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).build()
probar("DecisionTree", dt, grid)

print("\n[3/6] RandomForestClassifier...", flush=True)
rf = RandomForestClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(rf.numTrees, [10, 50]).addGrid(rf.maxDepth, [5, 10]).build()
probar("RandomForest", rf, grid)

print("\n[4/6] GBTClassifier...", flush=True)
gbt = GBTClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(gbt.maxIter, [10, 20]).addGrid(gbt.maxDepth, [3, 5]).build()
probar("GBT", gbt, grid)

print("\n[5/6] LinearSVC...", flush=True)
svc = LinearSVC(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(svc.regParam, [1e-6, 1e-5, 1e-4]).build()
probar("LinearSVC", svc, grid, col_score="rawPrediction")

print("\n[6/6] NaiveBayes (gaussian)...", flush=True)
nb = NaiveBayes(featuresCol="features", labelCol="default", modelType="gaussian")
grid = ParamGridBuilder().build()  # sin busqueda de hiperparametros, igual que en sklearn
probar("NaiveBayes", nb, grid)

print("\nTODO OK", flush=True)
spark.stop()
