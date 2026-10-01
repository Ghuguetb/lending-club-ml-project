"""Corrige un bug real: en model_pyspark.py, el evaluador de AUC usado tanto para la seleccion de
hiperparametros (CrossValidator) como para el AUC final se configuro con rawPredictionCol=
"rawPrediction". Para LogisticRegression/RandomForest/GBT/LinearSVC/NaiveBayes esto no cambia nada
(su "rawPrediction" y su "probability" preservan el mismo orden relativo por fila -- son la misma
puntuacion salvo una transformacion monotona global). Pero en un DecisionTreeClassificationModel
INDIVIDUAL, "rawPrediction" son los conteos de clase SIN normalizar de la hoja donde cae cada fila
(distintas hojas tienen distintos tamanos totales), mientras que "probability" es ese conteo
normalizado por el tamano de la hoja -- dos hojas distintas pueden invertir el orden entre ambas
columnas. Esto se detecto al notar que el AUC guardado en outputs/tables/spark_results.csv (0.598,
calculado sobre "rawPrediction") no coincidia con el AUC recalculado sobre el score realmente
usado en todo el analisis posterior ("probability", guardado en data/spark_test_scores.parquet,
AUC=0.699). Se corrige re-entrenando SOLO el arbol de decision con un evaluador consistente que usa
"probability" tanto para la seleccion de hiperparametros como para el AUC final.
"""
import time, json
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel

spark = (SparkSession.builder.appName("FixDecisionTree").master("local[2]")
         .config("spark.driver.memory", "4g").config("spark.executor.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8").config("spark.default.parallelism", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

train = spark.read.parquet("data/spark_train_prepared.parquet").persist(StorageLevel.MEMORY_AND_DISK)
test = spark.read.parquet("data/spark_test_prepared.parquet").persist(StorageLevel.MEMORY_AND_DISK)
train.count(); test.count()

# Evaluador consistente: usa la columna "probability" (normalizada por hoja), NO "rawPrediction"
evaluator_prob = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="probability",
                                                metricName="areaUnderROC")

dt = DecisionTreeClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]).build()
cv = CrossValidator(estimator=dt, estimatorParamMaps=grid, evaluator=evaluator_prob,
                     numFolds=3, parallelism=1, seed=42)

print("Re-entrenando DecisionTreeClassifier con evaluador corregido...", flush=True)
t0 = time.time()
cv_model = cv.fit(train)
t_fit = time.time() - t0
best = cv_model.bestModel
best_cv_auc = max(cv_model.avgMetrics)
best_max_depth = best.getOrDefault(best.getParam("maxDepth"))
print(f"Mejor maxDepth: {best_max_depth}  CV AUC (corregido): {best_cv_auc:.4f}  fit={t_fit:.1f}s", flush=True)

t0 = time.time()
pred = best.transform(test)
pred = pred.withColumn("score_DecisionTree", vector_to_array(F.col("probability"))[1])
test_auc = evaluator_prob.evaluate(pred)
pred = pred.withColumn("pred_label", F.when(F.col("score_DecisionTree") >= 0.5, 1).otherwise(0))
conteos = {(r["default"], r["pred_label"]): r["count"]
           for r in pred.groupBy("default", "pred_label").count().collect()}
tn = conteos.get((0,0),0); fp = conteos.get((0,1),0); fn = conteos.get((1,0),0); tp = conteos.get((1,1),0)
t_predict = time.time() - t0

acc = (tn+tp)/(tn+fp+fn+tp)
prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
print(f"Test AUC (corregido, sobre 'probability'): {test_auc:.4f}", flush=True)
print(f"tn={tn} fp={fp} fn={fn} tp={tp} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}", flush=True)

# --- Actualizar outputs/tables/spark_results.csv ---
results = pd.read_csv("outputs/tables/spark_results.csv")
idx = results.index[results["modelo"] == "DecisionTree"][0]
results.loc[idx, "best_params"] = json.dumps({"maxDepth": int(best_max_depth)})
results.loc[idx, "cv_auc"] = best_cv_auc
results.loc[idx, "test_accuracy"] = acc
results.loc[idx, "test_precision"] = prec
results.loc[idx, "test_recall"] = rec
results.loc[idx, "test_f1"] = f1
results.loc[idx, "test_auc"] = test_auc
results.loc[idx, "tn"] = tn; results.loc[idx, "fp"] = fp
results.loc[idx, "fn"] = fn; results.loc[idx, "tp"] = tp
results.loc[idx, "tiempo_entrenamiento_s"] = t_fit
results.loc[idx, "tiempo_prediccion_s"] = t_predict
results.to_csv("outputs/tables/spark_results.csv", index=False)
print("\noutputs/tables/spark_results.csv actualizado.", flush=True)

# --- Actualizar data/spark_test_scores.parquet (reemplazar solo la columna score_DecisionTree) ---
# Escritura distribuida (no toPandas/collect), igual que en model_pyspark.py; se lee con pandas
# recien despues de que Spark ya la materializo en disco.
pred.select("id", "score_DecisionTree").write.mode("overwrite").parquet("data/_fix_dt_scores.parquet")
scores_pd = pd.read_parquet("data/_fix_dt_scores.parquet")
scores_full = pd.read_parquet("data/spark_test_scores.parquet")
scores_full["id"] = scores_full["id"].astype(str)
scores_pd["id"] = scores_pd["id"].astype(str)
scores_full = scores_full.drop(columns=["score_DecisionTree"]).merge(scores_pd, on="id", how="left")
scores_full.to_parquet("data/spark_test_scores.parquet", index=False)
print("data/spark_test_scores.parquet actualizado (columna score_DecisionTree corregida).", flush=True)

spark.stop()
