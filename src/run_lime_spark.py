"""Seccion 9.10.4.7: LIME sobre el mejor modelo de PySpark (GBTClassifier, AUC=0.733, seccion 4).
Mismo espiritu que run_lime_sklearn.py, pero el predict_fn que LIME necesita (una funcion Python
que reciba un array numpy y devuelva probabilidades) tiene que envolver, por debajo, una llamada a
`GBTClassificationModel.transform()` sobre un DataFrame de Spark -- LIME no sabe nada de Spark, asi
que cada lote de muestras perturbadas (num_samples=5000 por instancia) se convierte a DataFrame,
se pasa por el modelo, y el resultado se trae de vuelta a numpy."""
import json
import time
import numpy as np
import pandas as pd
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

CATEGORICAL_START = 14

spark = (SparkSession.builder.appName("LimeSpark").master("local[2]")
         .config("spark.driver.memory", "4g").config("spark.executor.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8").config("spark.default.parallelism", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("Cargando modelo GBT ya guardado...", flush=True)
gbt_model = GBTClassificationModel.load("data/spark_best_model_GBT")

nombres = json.load(open("data/lime_feature_names_spark.json"))
bg = np.load("data/lime_background_spark.npz")
X_bg = bg["X"]

# ---- elegir 2 instancias de test mal clasificadas (umbral natural 0.5) ----
sp_test = pd.read_parquet("data/spark_test_scores.parquet")[["id", "default", "score_GBT"]]
sp_test["id"] = sp_test["id"].astype(str)
sp_test["pred"] = (sp_test["score_GBT"] >= 0.5).astype(int)
fp_ids = sp_test.loc[(sp_test.pred == 1) & (sp_test.default == 0), "id"].values
fn_ids = sp_test.loc[(sp_test.pred == 0) & (sp_test.default == 1), "id"].values
print(f"Falsos positivos disponibles: {len(fp_ids):,}  |  Falsos negativos: {len(fn_ids):,}")

rng = np.random.default_rng(7)
id_fp = fp_ids[rng.integers(0, len(fp_ids))]
id_fn = fn_ids[rng.integers(0, len(fn_ids))]

test_prep = spark.read.parquet("data/spark_test_prepared.parquet")
filas = (test_prep.filter(test_prep.id.isin([id_fp, id_fn]))
                   .withColumn("features_arr", vector_to_array("features"))
                   .select("id", "features_arr").toPandas())  # 2 filas: no viola la regla de "nada de .toPandas() sobre datos completos"
filas = filas.set_index("id")
x_fp = np.array(filas.loc[id_fp, "features_arr"])
x_fn = np.array(filas.loc[id_fn, "features_arr"])


def predict_fn_spark(X):
    """X: array (n_muestras, 107) generado por LIME al perturbar una instancia. Devuelve
    (n_muestras, 2) de probabilidades, pasando el lote completo UNA sola vez por Spark."""
    filas = [(int(i), Vectors.dense(row.tolist())) for i, row in enumerate(X)]
    df = spark.createDataFrame(filas, ["_orden", "features"])
    pred = gbt_model.transform(df).withColumn("proba_arr", vector_to_array("probability"))
    pdf = pred.select("_orden", "proba_arr").toPandas().sort_values("_orden")  # lote pequeno (num_samples), no el dataset completo
    return np.stack(pdf["proba_arr"].values)


explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_bg,
    feature_names=nombres,
    class_names=["No default", "Default"],
    categorical_features=list(range(CATEGORICAL_START, len(nombres))),
    discretize_continuous=True,
    random_state=42,
)

resultados = {}
for etiqueta, id_, x in [("falso_positivo", id_fp, x_fp), ("falso_negativo", id_fn, x_fn)]:
    fila = sp_test.set_index("id").loc[id_]
    print(f"\n=== {etiqueta} (id={id_}) ===")
    print(f"  y_real={int(fila['default'])}  prob_predicha={fila['score_GBT']:.4f}  pred={int(fila['pred'])}")
    t0 = time.time()
    exp = explainer.explain_instance(x, predict_fn_spark, num_features=10, num_samples=5000)
    print(f"  explain_instance: {time.time()-t0:.1f}s", flush=True)
    lista = exp.as_list()
    for feat, peso in lista:
        print(f"    {feat:45s} {peso:+.4f}")
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    fig.savefig(f"outputs/figures/eda/lime_spark_{etiqueta}.png", dpi=110)
    plt.close(fig)
    resultados[etiqueta] = {
        "id": str(id_), "y_real": int(fila["default"]), "prob_predicha": float(fila["score_GBT"]),
        "pred_label": int(fila["pred"]), "explicacion": [(f, float(w)) for f, w in lista],
        "intercepto_local": float(exp.intercept[1]), "r2_local": float(exp.score),
    }

with open("outputs/tables/lime_spark_resultados.json", "w") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
spark.stop()
print("\nGuardado: outputs/tables/lime_spark_resultados.json y outputs/figures/eda/lime_spark_*.png")
