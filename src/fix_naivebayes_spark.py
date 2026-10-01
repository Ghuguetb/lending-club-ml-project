"""Bug encontrado al preparar McNemar/bootstrap: igual que con DecisionTree (seccion 4.2.2), el
NaiveBayes de Spark fue seleccionado/evaluado durante la corrida original (model_pyspark.py) con
BinaryClassificationEvaluator(rawPredictionCol="rawPrediction"), mientras que el puntaje GUARDADO
(y usado para la matriz de confusion) viene de la columna "probability". Para NaiveBayesModel esas
dos columnas NO son equivalentes en ranking (igual que para el arbol de decision individual), asi
que el test_auc/cv_auc reportado (0.5643) esta mal -- la matriz de confusion y el puntaje guardado
ya eran correctos. Aqui se recalcula cv_auc y test_auc con el evaluador correcto (consistente con
"probability"), sin volver a la grilla completa (NaiveBayes no tiene hiperparametros: bp={})."""
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark import StorageLevel

spark = (SparkSession.builder.appName("FixNaiveBayes").master("local[2]")
         .config("spark.driver.memory", "4g").config("spark.executor.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8").config("spark.default.parallelism", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

train = spark.read.parquet("data/spark_train_prepared.parquet").persist(StorageLevel.MEMORY_AND_DISK)
test = spark.read.parquet("data/spark_test_prepared.parquet").persist(StorageLevel.MEMORY_AND_DISK)
train.count(); test.count()

nb = NaiveBayes(featuresCol="features", labelCol="default", modelType="gaussian")
grid = ParamGridBuilder().build()  # sin hiperparametros que probar, igual que la corrida original
evaluador_correcto = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="probability",
                                                    metricName="areaUnderROC")

print("Recalculando NaiveBayes con CrossValidator (evaluador correcto: probability)...", flush=True)
cv = CrossValidator(estimator=nb, estimatorParamMaps=grid, evaluator=evaluador_correcto,
                     numFolds=3, parallelism=1, seed=42)
cv_model = cv.fit(train)
cv_auc_correcto = max(cv_model.avgMetrics)
best = cv_model.bestModel

pred = best.transform(test)
test_auc_correcto = evaluador_correcto.evaluate(pred)
print(f"cv_auc correcto  = {cv_auc_correcto:.4f}  (antes: 0.5644)")
print(f"test_auc correcto = {test_auc_correcto:.4f}  (antes: 0.5643)")

# Verificacion: el puntaje de test recalculado debe coincidir EXACTO con el ya guardado en
# data/spark_test_scores.parquet (que ya usaba la columna "probability")
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
pred_chk = pred.withColumn("score_check", vector_to_array(F.col("probability"))[1])
pred_chk.select("id", "score_check").write.mode("overwrite").parquet("data/_nb_check.parquet")
spark.stop()

chk = pd.read_parquet("data/_nb_check.parquet")
guardado = pd.read_parquet("data/spark_test_scores.parquet")[["id", "score_NaiveBayes"]]
chk["id"] = chk["id"].astype(str); guardado["id"] = guardado["id"].astype(str)
comp = chk.merge(guardado, on="id")
diff_max = (comp["score_check"] - comp["score_NaiveBayes"]).abs().max()
print(f"\nDiferencia maxima vs. puntaje ya guardado: {diff_max:.2e} (debe ser ~0.0)")
assert diff_max < 1e-9, "el puntaje recalculado no coincide con el ya guardado -- no actualizar el CSV"

import shutil, os
shutil.rmtree("data/_nb_check.parquet")

# Actualizar SOLO las columnas cv_auc/test_auc de la fila NaiveBayes en spark_results.csv
resultados = pd.read_csv("outputs/tables/spark_results.csv")
idx = resultados[resultados["modelo"] == "NaiveBayes"].index[0]
print(f"\nAntes: {resultados.loc[idx].to_dict()}")
resultados.loc[idx, "cv_auc"] = cv_auc_correcto
resultados.loc[idx, "test_auc"] = test_auc_correcto
resultados.to_csv("outputs/tables/spark_results.csv", index=False)
print(f"Despues: {resultados.loc[idx].to_dict()}")
print("\nGuardado: outputs/tables/spark_results.csv (fila NaiveBayes corregida)")
