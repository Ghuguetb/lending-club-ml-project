"""Valida rapido, sobre una submuestra, que evaluar_entrenar() de model_pyspark.py corre sin errores
de principio a fin (incluye el join y la escritura a Parquet) antes de lanzar la corrida completa."""
import time, json
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel

spark = (SparkSession.builder.appName("Validate").master("local[2]")
         .config("spark.driver.memory", "4g").config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

train_full = spark.read.parquet("data/spark_train_prepared.parquet")
test_full = spark.read.parquet("data/spark_test_prepared.parquet")
train = train_full.sample(False, 20000/train_full.count(), seed=0).persist(StorageLevel.MEMORY_AND_DISK)
test = test_full.sample(False, 5000/test_full.count(), seed=0).persist(StorageLevel.MEMORY_AND_DISK)
print("train:", train.count(), "test:", test.count(), flush=True)

evaluator_auc = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
results = []
scores_wide = test.select("id", "default")

def mejor_parametros(best_model, nombres_params):
    return {p: best_model.getOrDefault(best_model.getParam(p)) for p in nombres_params}

def evaluar_entrenar(nombre, estimator, grid, params_a_reportar, umbral=0.5):
    global scores_wide
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid, evaluator=evaluator_auc,
                         numFolds=3, parallelism=1, seed=42)
    t0 = time.time(); cv_model = cv.fit(train); t_fit = time.time()-t0
    best = cv_model.bestModel
    best_cv_auc = max(cv_model.avgMetrics)
    best_params = mejor_parametros(best, params_a_reportar)
    t0 = time.time()
    pred = best.transform(test)
    score_source = "probability" if "probability" in pred.columns else "rawPrediction"
    pred = pred.withColumn(f"score_{nombre}", vector_to_array(F.col(score_source))[1])
    test_auc = evaluator_auc.evaluate(pred)
    pred = pred.withColumn("pred_label", F.when(F.col(f"score_{nombre}") >= F.lit(umbral), 1).otherwise(0))
    conteos = {(r["default"], r["pred_label"]): r["count"] for r in pred.groupBy("default", "pred_label").count().collect()}
    tn = conteos.get((0,0),0); fp = conteos.get((0,1),0); fn = conteos.get((1,0),0); tp = conteos.get((1,1),0)
    t_predict = time.time()-t0
    acc = (tn+tp)/(tn+fp+fn+tp)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    results.append({"modelo": nombre, "best_params": json.dumps(best_params), "cv_auc": best_cv_auc,
                     "test_accuracy": acc, "test_f1": f1, "test_auc": test_auc,
                     "tn": tn, "fp": fp, "fn": fn, "tp": tp, "t_fit": t_fit, "t_predict": t_predict})
    scores_wide = scores_wide.join(pred.select("id", f"score_{nombre}"), on="id")
    print(f"  OK {nombre}: AUC={test_auc:.4f} CV_AUC={best_cv_auc:.4f} params={best_params} fit={t_fit:.1f}s tn={tn} fp={fp} fn={fn} tp={tp}", flush=True)
    return best

lr = LogisticRegression(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(lr.regParam, [1e-4]).build()
evaluar_entrenar("LogisticRegression", lr, grid, ["regParam"])

svc = LinearSVC(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(svc.regParam, [1e-4]).build()
evaluar_entrenar("LinearSVC", svc, grid, ["regParam"], umbral=0.0)

nb = NaiveBayes(featuresCol="features", labelCol="default", modelType="gaussian")
evaluar_entrenar("NaiveBayes", nb, ParamGridBuilder().build(), [])

print("\nEscribiendo parquet de validacion...", flush=True)
scores_wide.write.mode("overwrite").parquet("/tmp/validate_spark_scores.parquet")
check = pd.read_parquet("/tmp/validate_spark_scores.parquet")
print(check.shape, check.columns.tolist(), flush=True)
print("\nVALIDACION OK", flush=True)
spark.stop()
