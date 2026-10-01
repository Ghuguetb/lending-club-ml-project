"""Modelado con PySpark: 6 modelos + CrossValidator (ParamGridBuilder). Seccion 9.10.4.5.

Regla clave respetada en todo el script: NUNCA se llama .toPandas()/.collect() sobre el dataset
completo (1.8M filas x vector de features). Las unicas veces que se trae algo al driver son:
  (a) conteos agregados pequenos (matriz de confusion: 4 numeros por modelo), y
  (b) al final, un archivo Parquet ya escrito a disco por Spark (no un DataFrame en memoria) con
      solo 3 columnas (id, default, score) por modelo -- eso se lee despues con pandas para las
      pruebas estadisticas (DeLong/McNemar), exactamente igual que en el lado de scikit-learn.
"""
import time, json
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier,
                                        RandomForestClassifier, GBTClassifier,
                                        LinearSVC, NaiveBayes)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel

t_script_start = time.time()

# Misma desviacion justificada de la configuracion "obligatoria" (8g+8g, 400 particiones) que en
# el preprocesamiento (seccion 2.3): esta maquina tiene 7.8 GB de RAM y 2 nucleos reales.
spark = (SparkSession.builder
         .appName("LendingClub_Modeling")
         .master("local[2]")
         .config("spark.driver.memory", "4g")
         .config("spark.executor.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.default.parallelism", "8")
         .config("spark.memory.fraction", 0.8)
         .config("spark.memory.storageFraction", 0.3)
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("Cargando datos preprocesados (ya vectorizados en la seccion 2)...", flush=True)
train = spark.read.parquet("data/spark_train_prepared.parquet")
test = spark.read.parquet("data/spark_test_prepared.parquet")

# Cache OBLIGATORIO de los vectores de features ya ensamblados, antes de entrenar cualquier modelo.
train = train.persist(StorageLevel.MEMORY_AND_DISK)
test = test.persist(StorageLevel.MEMORY_AND_DISK)
n_train = train.count()
n_test = test.count()
print(f"train={n_train:,}  test={n_test:,}", flush=True)

HARDWARE = {"cores": 2, "ram_gb": 7.8, "nota": "misma maquina fisica que scikit-learn; Spark en local[2]"}
evaluator_auc = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="rawPrediction",
                                               metricName="areaUnderROC")

results = []
scores_wide = test.select("id", "default")  # se le van uniendo columnas score_<modelo>


def mejor_parametros(best_model, nombres_params):
    return {p: best_model.getOrDefault(best_model.getParam(p)) for p in nombres_params}


def evaluar_entrenar(nombre, estimator, grid, params_a_reportar, umbral=0.5):
    global scores_wide
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid, evaluator=evaluator_auc,
                         numFolds=3, parallelism=1, seed=42)
    t0 = time.time()
    cv_model = cv.fit(train)
    t_fit = time.time() - t0
    best = cv_model.bestModel
    best_cv_auc = max(cv_model.avgMetrics)
    best_params = mejor_parametros(best, params_a_reportar)

    t0 = time.time()
    pred = best.transform(test)
    score_source = "probability" if "probability" in pred.columns else "rawPrediction"
    pred = pred.withColumn(f"score_{nombre}", vector_to_array(F.col(score_source))[1])
    test_auc = evaluator_auc.evaluate(pred)

    pred = pred.withColumn("pred_label", F.when(F.col(f"score_{nombre}") >= F.lit(umbral), 1).otherwise(0))
    # Unico .collect(): 4 numeros (matriz de confusion), no el dataset completo.
    conteos = {(r["default"], r["pred_label"]): r["count"]
               for r in pred.groupBy("default", "pred_label").count().collect()}
    tn = conteos.get((0, 0), 0); fp = conteos.get((0, 1), 0)
    fn = conteos.get((1, 0), 0); tp = conteos.get((1, 1), 0)
    t_predict = time.time() - t0

    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    results.append({
        "modelo": nombre, "best_params": json.dumps(best_params), "cv_auc": best_cv_auc,
        "test_accuracy": acc, "test_precision": prec, "test_recall": rec, "test_f1": f1,
        "test_auc": test_auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "tiempo_entrenamiento_s": t_fit, "tiempo_prediccion_s": t_predict,
    })
    scores_wide = scores_wide.join(pred.select("id", f"score_{nombre}"), on="id")
    print(f"  {nombre}: AUC test={test_auc:.4f} | CV AUC={best_cv_auc:.4f} | params={best_params} "
          f"| fit={t_fit:.1f}s pred={t_predict:.2f}s", flush=True)
    return best


best_models = {}

# ============ 1. Regresion logistica ============
# regParam de Spark ya es analogo al "alpha" (1/(C*n)) de scikit-learn: Spark minimiza la perdida
# PROMEDIO + regParam*penalizacion, mientras que sklearn minimiza la perdida SUMA + (1/C)*penalizacion;
# como 1/C = alpha*n, ambas formulaciones son equivalentes usando los mismos valores de alpha=regParam.
print("\n[1/6] LogisticRegression...", flush=True)
lr = LogisticRegression(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(lr.regParam, [1e-6, 1e-5, 1e-4]).build()
best_models["LogisticRegression"] = evaluar_entrenar("LogisticRegression", lr, grid, ["regParam"])

# ============ 2. Arbol de decision ============
print("\n[2/6] DecisionTreeClassifier...", flush=True)
dt = DecisionTreeClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]).build()
best_models["DecisionTree"] = evaluar_entrenar("DecisionTree", dt, grid, ["maxDepth"])

# ============ 3. Bosque aleatorio ============
print("\n[3/6] RandomForestClassifier (mismo grid que en scikit-learn)...", flush=True)
rf = RandomForestClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(rf.numTrees, [10, 50, 100]).addGrid(rf.maxDepth, [5, 10, 15]).build()
best_models["RandomForest"] = evaluar_entrenar("RandomForest", rf, grid, ["numTrees", "maxDepth"])

# ============ 4. Gradient boosting (GBTClassifier nativo, sin sustitucion) ============
print("\n[4/6] GBTClassifier...", flush=True)
gbt = GBTClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(gbt.maxIter, [50, 100]).addGrid(gbt.maxDepth, [3, 5]).build()
best_models["GBT"] = evaluar_entrenar("GBT", gbt, grid, ["maxIter", "maxDepth"])

# ============ 5. SVM lineal ============
# LinearSVC de Spark SIEMPRE usa perdida hinge (no hay opcion de squared_hinge) -- por eso, del
# lado de scikit-learn, se forzo loss="hinge" explicitamente para poder comparar de forma justa.
print("\n[5/6] LinearSVC (hinge, unica opcion en Spark)...", flush=True)
svc = LinearSVC(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(svc.regParam, [1e-6, 1e-5, 1e-4]).build()
best_models["LinearSVC"] = evaluar_entrenar("LinearSVC", svc, grid, ["regParam"], umbral=0.0)

# ============ 6. Naive Bayes (gaussiano, sin busqueda de hiperparametros) ============
print("\n[6/6] NaiveBayes (gaussian, sin busqueda de hiperparametros)...", flush=True)
nb = NaiveBayes(featuresCol="features", labelCol="default", modelType="gaussian")
grid = ParamGridBuilder().build()
best_models["NaiveBayes"] = evaluar_entrenar("NaiveBayes", nb, grid, [])

# ============ Guardar todo ============
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/tables/spark_results.csv", index=False)
with open("outputs/tables/hardware_spark.json", "w") as f:
    json.dump(HARDWARE, f)

print("\nEscribiendo puntajes de prueba a Parquet (escritura distribuida, no toPandas)...", flush=True)
scores_wide.write.mode("overwrite").parquet("data/spark_test_scores.parquet")

# Mejor modelo por AUC de prueba, guardado en formato nativo de Spark (no es picklable con joblib;
# se necesita para el LIME de la seccion 9.10.4.7)
mejor_nombre = results_df.loc[results_df["test_auc"].idxmax(), "modelo"]
best_models[mejor_nombre].write().overwrite().save(f"data/spark_best_model_{mejor_nombre}")
with open("outputs/tables/spark_mejor_modelo.json", "w") as f:
    json.dump({"mejor_modelo": mejor_nombre}, f)
print(f"\nMejor modelo por AUC de prueba: {mejor_nombre} (guardado en data/spark_best_model_{mejor_nombre})", flush=True)

print("\n=== RESUMEN FINAL ===", flush=True)
print(results_df[["modelo", "cv_auc", "test_auc", "test_accuracy", "test_f1",
                   "tiempo_entrenamiento_s"]].to_string(index=False), flush=True)
print(f"\nTiempo total del script: {time.time()-t_script_start:.1f}s", flush=True)
print("Guardado: outputs/tables/spark_results.csv, data/spark_test_scores.parquet", flush=True)

spark.stop()
