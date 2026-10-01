"""Puntajes de ENTRENAMIENTO de los 6 modelos de PySpark (necesarios para elegir el umbral de
clasificacion usando solo train, seccion 9.10.4.6.2). Para GBT se reusa el modelo ya guardado
(data/spark_best_model_GBT). Los otros 5 se reentrenan con SUS HIPERPARAMETROS YA CONOCIDOS
(spark_results.csv) -- un solo ajuste cada uno, SIN CrossValidator ni grilla -- mucho mas rapido
que la corrida original de ~4h que si tuvo que probar todas las combinaciones."""
import json, time
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier,
                                        RandomForestClassifier, GBTClassifier,
                                        LinearSVC, NaiveBayes,
                                        LogisticRegressionModel, DecisionTreeClassificationModel,
                                        RandomForestClassificationModel, GBTClassificationModel,
                                        LinearSVCModel, NaiveBayesModel)
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel

spark = (SparkSession.builder.appName("TrainScores").master("local[2]")
         .config("spark.driver.memory", "4g").config("spark.executor.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8").config("spark.default.parallelism", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

train = spark.read.parquet("data/spark_train_prepared.parquet").persist(StorageLevel.MEMORY_AND_DISK)
test = spark.read.parquet("data/spark_test_prepared.parquet").persist(StorageLevel.MEMORY_AND_DISK)
train.count(); test.count()

resultados = pd.read_csv("outputs/tables/spark_results.csv").set_index("modelo")
bp = {m: json.loads(resultados.loc[m, "best_params"]) for m in resultados.index}
print("Mejores hiperparametros ya conocidos:", bp, flush=True)

train_scores = train.select("id", "default")
test_scores_check = test.select("id", "default")


def anotar(nombre, modelo_ajustado, train_df, test_df):
    global train_scores, test_scores_check
    for etiqueta, df in [("train", train_df), ("test", test_df)]:
        pred = modelo_ajustado.transform(df)
        score_source = "probability" if "probability" in pred.columns else "rawPrediction"
        pred = pred.withColumn(f"score_{nombre}", vector_to_array(F.col(score_source))[1])
        if etiqueta == "train":
            train_scores = train_scores.join(pred.select("id", f"score_{nombre}"), on="id")
        else:
            test_scores_check = test_scores_check.join(pred.select("id", f"score_{nombre}"), on="id")


# ---- GBT: reusar el modelo ya guardado ----
print("\n[GBT] cargando modelo ya guardado...", flush=True)
t0 = time.time()
gbt_model = GBTClassificationModel.load("data/spark_best_model_GBT")
anotar("GBT", gbt_model, train, test)
print(f"  listo en {time.time()-t0:.1f}s", flush=True)

# ---- LogisticRegression: un solo fit con regParam ya conocido ----
print("\n[LogisticRegression] un solo fit...", flush=True)
t0 = time.time()
lr = LogisticRegression(featuresCol="features", labelCol="default", regParam=bp["LogisticRegression"]["regParam"])
lr_model = lr.fit(train)
anotar("LogisticRegression", lr_model, train, test)
print(f"  listo en {time.time()-t0:.1f}s", flush=True)

# ---- LinearSVC: un solo fit con regParam ya conocido ----
print("\n[LinearSVC] un solo fit...", flush=True)
t0 = time.time()
svc = LinearSVC(featuresCol="features", labelCol="default", regParam=bp["LinearSVC"]["regParam"])
svc_model = svc.fit(train)
anotar("LinearSVC", svc_model, train, test)
print(f"  listo en {time.time()-t0:.1f}s", flush=True)

# ---- NaiveBayes: sin hiperparametros ----
print("\n[NaiveBayes] un solo fit...", flush=True)
t0 = time.time()
nb = NaiveBayes(featuresCol="features", labelCol="default", modelType="gaussian")
nb_model = nb.fit(train)
anotar("NaiveBayes", nb_model, train, test)
print(f"  listo en {time.time()-t0:.1f}s", flush=True)

# ---- DecisionTree: un solo fit con maxDepth ya conocido (corregido: 15) ----
print("\n[DecisionTree] un solo fit...", flush=True)
t0 = time.time()
dt = DecisionTreeClassifier(featuresCol="features", labelCol="default", seed=42,
                             maxDepth=int(bp["DecisionTree"]["maxDepth"]))
dt_model = dt.fit(train)
anotar("DecisionTree", dt_model, train, test)
print(f"  listo en {time.time()-t0:.1f}s", flush=True)

# ---- RandomForest: un solo fit con numTrees/maxDepth ya conocidos (el mas caro) ----
print("\n[RandomForest] un solo fit (el mas costoso)...", flush=True)
t0 = time.time()
rf = RandomForestClassifier(featuresCol="features", labelCol="default", seed=42,
                             numTrees=int(bp["RandomForest"]["numTrees"]),
                             maxDepth=int(bp["RandomForest"]["maxDepth"]))
rf_model = rf.fit(train)
anotar("RandomForest", rf_model, train, test)
print(f"  listo en {time.time()-t0:.1f}s", flush=True)

print("\nEscribiendo puntajes de train y test (verificacion) a Parquet...", flush=True)
train_scores.write.mode("overwrite").parquet("data/spark_train_scores_raw.parquet")
test_scores_check.write.mode("overwrite").parquet("data/_spark_test_scores_check.parquet")
print("Listo.", flush=True)
spark.stop()
