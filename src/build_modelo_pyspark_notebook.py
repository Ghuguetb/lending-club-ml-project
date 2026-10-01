import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 4. Modelado con PySpark (9.10.4.5)

Se entrenan los mismos 6 modelos, sobre la misma partición 80/20 (`data/split_assignment.parquet`)
y las mismas variables (sección 2.3), usando `pyspark.ml.tuning.CrossValidator` +
`ParamGridBuilder` (`numFolds=3`, `BinaryClassificationEvaluator(metricName="areaUnderROC")`).

**Configuración de Spark** — la misma desviación justificada que en el preprocesamiento (sección
2.3): `driver.memory=4g`, `executor.memory=4g`, `shuffle.partitions=8`, `default.parallelism=8`, en
vez de los 8g+8g / 400 particiones "obligatorios", porque esta máquina tiene 7.8 GB de RAM y 2
núcleos reales — la misma máquina física que corrió scikit-learn, lo que permite comparar tiempos
de cómputo de forma justa (no es una comparación entre hardware distinto).

**`CrossValidator(parallelism=1)`**: se somete cada combinación de pliegue/hiperparámetro de forma
secuencial, nunca en paralelo. En un clúster real con muchos núcleos valdría la pena subir este
número; en una máquina de 2 núcleos, cada *fit* individual ya usa el paralelismo interno de Spark
(particiones distribuidas entre los 2 núcleos), así que paralelizar además entre modelos de la
grilla solo generaría contención por los mismos 2 núcleos, sin ninguna ganancia neta.

**Cómo se evitó traer el dataset completo al *driver*:** en ningún punto del script se llama
`.toPandas()` ni `.collect()` sobre las 1.8M/452K filas. La matriz de confusión de cada modelo se
calcula con una agregación distribuida (`groupBy("default","pred_label").count()`) que solo trae 4
números al *driver*; los puntajes de prueba de los 6 modelos se escriben con
`DataFrame.write.parquet(...)` (escritura distribuida en 8 particiones) y **después** se leen con
`pandas` desde ese archivo ya materializado en disco — nunca directamente desde el DataFrame de
Spark en memoria.

**Nota sobre cómo se ejecutó esta sección.** El script completo (`src/model_pyspark.py`) tardó
**~4 h 9 min** — dominado por `RandomForestClassifier` (177 min) y `GBTClassifier` (49 min). Por la
misma razón que en la sección 3, este notebook documenta el código exacto usado (como bloques de
referencia) y carga los resultados/puntajes ya calculados para el análisis.
""")

add_md(nb, "## 4.1. Carga de resultados y puntajes guardados")

add_code(nb, """import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

results_spark = pd.read_csv("../outputs/tables/spark_results.csv")
scores_spark = pd.read_parquet("../data/spark_test_scores.parquet")
results_sklearn = pd.read_csv("../outputs/tables/sklearn_results.csv")
with open("../outputs/tables/hardware_spark.json") as f:
    hardware_spark = json.load(f)

y_test = scores_spark["default"].values
print("Hardware usado (misma maquina fisica que scikit-learn):", hardware_spark)
print(f"Conjunto de prueba: {len(y_test):,} filas")

# Verificacion critica: mismo conjunto de prueba (mismas 452,141 filas) que en scikit-learn --
# indispensable para que la prueba de DeLong (seccion 5) compare pares validos.
scores_sklearn = pd.read_parquet("../data/sklearn_test_scores.parquet")
mismas_ids = set(scores_spark["id"].astype(str)) == set(scores_sklearn["id"].astype(str))
print(f"Mismo conjunto de test que scikit-learn (mismos ids): {mismas_ids}")
""")

add_md(nb, """## 4.2. Modelo por modelo: malla de hiperparámetros y equivalencias con scikit-learn

Código de referencia usado por `src/model_pyspark.py` (no se re-ejecuta aquí por el tiempo total
de ~4 h; ver nota de la introducción).
""")

add_md(nb, """### 4.2.1. Regresión logística

`regParam` de Spark ya es directamente análogo al `alpha` de la formulación `C=1/(alpha·n)` de
scikit-learn: Spark minimiza *pérdida promedio* + `regParam`·penalización, mientras que sklearn
minimiza *pérdida suma* + `(1/C)`·penalización; como `1/C = alpha·n`, ambas formulaciones son
matemáticamente equivalentes usando los mismos valores de `alpha = regParam`. Por eso se usa la
misma malla de valores (`1e-6, 1e-5, 1e-4`), sin necesidad de reescalarla por `n_train`.

```python
lr = LogisticRegression(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(lr.regParam, [1e-6, 1e-5, 1e-4]).build()
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator_auc,
                     numFolds=3, parallelism=1, seed=42)
cv_model = cv.fit(train)
```
""")

add_md(nb, """### 4.2.2. Árbol de decisión

```python
dt = DecisionTreeClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]).build()
```

**Corrección aplicada.** La primera corrida de este modelo usó, por error, la columna
`rawPrediction` (conteos de clase *sin normalizar* de la hoja donde cae cada fila) tanto para
seleccionar el mejor `maxDepth` por validación cruzada como para el AUC de prueba — en vez de
`probability` (esos mismos conteos normalizados por el tamaño de la hoja), que es la columna que sí
se guardó y se usa en todo el resto del análisis (curvas ROC, DeLong, McNemar, bootstrap). Para los
otros 5 modelos esto no cambia nada, porque su `rawPrediction` y su `probability` conservan el
mismo orden relativo entre filas (son la misma puntuación salvo una transformación monótona común
a todas las filas: dividir por `numTrees`, o aplicar una función logística). Pero en un
**árbol de decisión individual**, dos hojas distintas del árbol tienen distinto número total de
observaciones de entrenamiento, así que un conteo alto en una hoja grande puede corresponder a una
probabilidad *menor* que un conteo bajo en una hoja pequeña — `rawPrediction` y `probability` no
son intercambiables aquí. Se corrigió re-entrenando este modelo con un evaluador que usa
`probability` de forma consistente en ambas etapas (`src/fix_decisiontree_spark.py`); el resultado
cambió de forma sustancial: la validación cruzada ya no elige `maxDepth=10` sino
**`maxDepth=15`**, y el AUC de prueba sube de 0.598 a **0.720**. Esto es en sí mismo un hallazgo
relevante para la reflexión final (9.10.4.8): la elección de la columna de puntaje usada para
optimizar hiperparámetros no es un detalle cosmético — puede cambiar qué modelo termina eligiéndose
como "mejor".
""")

add_md(nb, """### 4.2.3. Bosque aleatorio — el más costoso también en Spark

```python
rf = RandomForestClassifier(featuresCol="features", labelCol="default", seed=42)
grid = (ParamGridBuilder().addGrid(rf.numTrees, [10, 50, 100])
                           .addGrid(rf.maxDepth, [5, 10, 15]).build())
```

Mismo grid exacto que en scikit-learn (9 combinaciones × 3 pliegues + reentrenamiento final = 28
ajustes). Tardó **10.597 s (~177 minutos)** — más que los 6.739 s (~112 min) que tomó en
scikit-learn con `n_jobs=1`. Se retoma en la sección 4.7: el paralelismo distribuido de Spark tiene
un costo fijo de coordinación (miles de *stages* internos para árboles profundos) que, en un
clúster de solo 2 núcleos, termina pesando más que el beneficio de paralelizar.
""")

add_md(nb, """### 4.2.4. Gradient Boosting (`GBTClassifier` nativo, sin sustitución)

A diferencia de scikit-learn, aquí **no** hace falta ninguna sustitución: `GBTClassifier` de Spark
ya implementa un algoritmo de *boosting* con particionamiento eficiente de forma nativa.

```python
gbt = GBTClassifier(featuresCol="features", labelCol="default", seed=42)
grid = ParamGridBuilder().addGrid(gbt.maxIter, [50, 100]).addGrid(gbt.maxDepth, [3, 5]).build()
```

`maxIter` es el análogo directo de `max_iter` de `HistGradientBoostingClassifier`. Curiosamente, la
combinación ganadora (`maxIter=100, maxDepth=5`) es exactamente la misma que escogió
`HistGradientBoostingClassifier` en scikit-learn.
""")

add_md(nb, """### 4.2.5. SVM lineal

`LinearSVC` de Spark **siempre** usa pérdida *hinge* (no existe la opción `squared_hinge`) — esa es
precisamente la razón por la que, del lado de scikit-learn, se forzó `loss="hinge"` explícitamente
en vez de usar el valor por defecto, para que la comparación fuera justa.

```python
svc = LinearSVC(featuresCol="features", labelCol="default")
grid = ParamGridBuilder().addGrid(svc.regParam, [1e-6, 1e-5, 1e-4]).build()
```

El resultado es uno de los hallazgos más interesantes de todo el proyecto (sección 4.6): con el
**mismo** tipo de pérdida y una malla de regularización equivalente, Spark **no** reproduce el
colapso a clasificador constante que sí ocurrió en scikit-learn.
""")

add_md(nb, """### 4.2.6. Naive Bayes (gaussiano)

```python
nb_model = NaiveBayes(featuresCol="features", labelCol="default", modelType="gaussian")
grid = ParamGridBuilder().build()  # sin busqueda de hiperparametros, igual que en sklearn
```

Se usa `modelType="gaussian"` (no `"multinomial"` ni `"bernoulli"`) por la misma razón que en
scikit-learn: las variables numéricas escaladas con `StandardScaler` toman valores negativos, algo
que los otros dos tipos de Naive Bayes de Spark no admiten.
""")

add_md(nb, "## 4.3. Tabla comparativa final (PySpark)")

add_code(nb, """cols_show = ["modelo","best_params","cv_auc","test_auc","test_accuracy",
             "test_precision","test_recall","test_f1","tiempo_entrenamiento_s","tiempo_prediccion_s"]
tabla_spark = results_spark[cols_show].sort_values("test_auc", ascending=False).reset_index(drop=True)
tabla_spark_fmt = tabla_spark.copy()
for c in ["cv_auc","test_auc","test_accuracy","test_precision","test_recall","test_f1"]:
    tabla_spark_fmt[c] = tabla_spark_fmt[c].round(4)
tabla_spark_fmt["tiempo_entrenamiento_s"] = tabla_spark_fmt["tiempo_entrenamiento_s"].round(1)
tabla_spark_fmt["tiempo_prediccion_s"] = tabla_spark_fmt["tiempo_prediccion_s"].round(2)
tabla_spark_fmt
""")

add_md(nb, """**Lectura rápida:** por AUC de prueba, el orden en Spark es
`GBT (0.733) > LogisticRegression (0.723) > DecisionTree (0.720) ≈ RandomForest (0.719) >
LinearSVC (0.704) > NaiveBayes (0.699)`. El mejor modelo — igual que en scikit-learn — es el de
*boosting* con histogramas, aunque aquí con una ventaja menos amplia sobre los demás. (Los valores
de `DecisionTree` y `NaiveBayes` en esta tabla ya incluyen las dos correcciones descritas en la
sección 4.6 — ver esa sección para el porqué.)
""")

add_md(nb, "## 4.4. Curvas ROC (los 6 modelos superpuestos)")

add_code(nb, """score_cols_spark = {
    "LogisticRegression": "score_LogisticRegression",
    "DecisionTree": "score_DecisionTree",
    "RandomForest": "score_RandomForest",
    "GBT": "score_GBT",
    "LinearSVC": "score_LinearSVC",
    "NaiveBayes": "score_NaiveBayes",
}

fig, ax = plt.subplots(figsize=(7,7))
for nombre, col in score_cols_spark.items():
    fpr, tpr, _ = roc_curve(y_test, scores_spark[col])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc:.3f})", linewidth=1.8)
ax.plot([0,1],[0,1], linestyle="--", color="gray", linewidth=1, label="Azar (AUC=0.5)")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curvas ROC — PySpark (conjunto de prueba)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/figures/eda/roc_spark.png", dpi=110)
plt.show()
""")

add_md(nb, "## 4.5. Matrices de confusión (umbral natural: 0.5 probabilidad / 0 margen)")

add_code(nb, """fig, axes = plt.subplots(2, 3, figsize=(13,8))
for ax, (nombre, col) in zip(axes.flat, score_cols_spark.items()):
    if nombre == "LinearSVC":
        y_pred = (scores_spark[col] >= 0).astype(int)
    else:
        y_pred = (scores_spark[col] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=9)
    ax.set_xticks([0,1]); ax.set_xticklabels(["No default","Default"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["No default","Default"])
    ax.set_title(nombre, fontsize=10)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/confusion_spark.png", dpi=110)
plt.show()
""")

add_md(nb, """## 4.6. Tres hallazgos que solo aparecen al comparar con scikit-learn

### (a) El árbol de decisión sigue rindiendo algo peor en Spark, incluso ya corregido (AUC 0.720 vs. 0.737)

Después de corregir el bug de la sección 4.2.2 (usar `probability` en vez de `rawPrediction`), la
diferencia se reduce mucho (de 14 puntos de AUC a menos de 2), y además cada entorno elige una
profundidad distinta como "mejor" (`maxDepth=15` en Spark, `maxDepth=10` en scikit-learn) — otra
pista de que las dos implementaciones no son idénticas. La diferencia residual que queda es
plausible que venga de `maxBins` (por defecto 32 en `DecisionTreeClassifier` de Spark): las
variables continuas (`int_rate`, `dti`, `annual_inc_log`, etc.) se discretizan en como máximo 32
puntos de corte candidatos antes de buscar la mejor división, mientras que scikit-learn evalúa
cualquier punto de corte real entre valores consecutivos. Pero a diferencia de la hipótesis inicial,
esto ya **no** explica una brecha grande — la mayor parte de la diferencia observada originalmente
era, de hecho, el bug de la columna de puntaje, no una diferencia real de algoritmo. Vale la pena
dejar ambas lecciones por separado en la reflexión final (9.10.4.8): (i) un error de implementación
propio puede parecer, a primera vista, una diferencia legítima entre librerías, y solo se distingue
comparando explícitamente contra el score que realmente se usa en el resto del pipeline; y (ii)
incluso corregido, sigue habiendo una diferencia pequeña y genuina atribuible a `maxBins`.

### (b) LinearSVC no colapsa en Spark (AUC 0.704 vs. 0.559 en scikit-learn)

Ambos usan pérdida *hinge* y una malla de regularización equivalente, pero:
""")

add_code(nb, """print("scikit-learn LinearSVC — mejor C:", results_sklearn.loc[results_sklearn.modelo=='LinearSVC','best_params'].values[0])
print("PySpark      LinearSVC — mejor regParam:", results_spark.loc[results_spark.modelo=='LinearSVC','best_params'].values[0])
print()
print("scikit-learn LinearSVC — score en prueba: min=%.4f max=%.4f std=%.6f" % (
    scores_sklearn['score_LinearSVC'].min(), scores_sklearn['score_LinearSVC'].max(), scores_sklearn['score_LinearSVC'].std()))
print("PySpark      LinearSVC — score en prueba: min=%.4f max=%.4f std=%.6f" % (
    scores_spark['score_LinearSVC'].min(), scores_spark['score_LinearSVC'].max(), scores_spark['score_LinearSVC'].std()))
""")

add_md(nb, """scikit-learn escogió la regularización **más fuerte disponible en su malla** (la que, según el
diagnóstico de la sección 3.7, empuja al modelo a colapsar hacia "no default" para todo el mundo);
Spark, en cambio, escogió la regularización **más débil** de la misma malla, y sus puntajes de
`rawPrediction` mantienen una dispersión real (no un valor casi constante). Esto indica que el
optimizador de Spark (OWL-QN, basado en gradiente cuasi-Newton) explora la superficie de pérdida de
forma distinta al `liblinear` de scikit-learn (descenso de coordenadas dual) y, para este problema
particular —fuertemente desbalanceado—, no queda atrapado en la misma solución degenerada. **Ambos
entornos igual predicen 0 casos positivos al umbral natural de margen=0** (ver matrices de
confusión): la diferencia real está en la calidad del *ranking* subyacente (AUC), no en el
comportamiento a ese umbral por defecto.

### (c) Naive Bayes: el mismo bug que en (a), no una diferencia de implementación

La primera versión de esta sección reportaba Naive Bayes de Spark muy por debajo de scikit-learn
(AUC 0.564 vs. 0.694) y proponía diferencias de suavizado de varianza como posible causa. Esa
comparación estaba **mal por la misma razón que el árbol de decisión en (a)**: `NaiveBayesModel`
no tiene hiperparámetros, así que `CrossValidator` no tenía nada que seleccionar entre candidatos
— pero el `test_auc`/`cv_auc` reportados igual se calcularon con
`BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")`, mientras que el puntaje
realmente guardado (y usado en la matriz de confusión) viene de la columna `probability`. Para
`NaiveBayesModel` esas dos columnas **tampoco** son equivalentes en *ranking* — igual que para el
árbol de decisión individual, y al revés de lo que se había asumido en la sección 4.2.2 para "los
modelos de conjunto o con transformación monótona global" (la lista debería haber excluido a Naive
Bayes también). Al recalcular con el evaluador correcto (`src/fix_naivebayes_spark.py`, verificado
contra el puntaje ya guardado con diferencia máxima de `0.0`):

```
cv_auc:   0.5644  ->  0.6998
test_auc: 0.5643  ->  0.6989
```

Con el número correcto, Naive Bayes en Spark queda prácticamente **empatado** con `GaussianNB` de
scikit-learn (0.699 vs. 0.694) — no hay ninguna diferencia real de implementación que explicar aquí.
La lección de (a) se repite y se refuerza: conviene verificar empíricamente, modelo por modelo, que
el evaluador usado para seleccionar/reportar coincide con la columna de puntaje realmente usada en
el resto del pipeline, en vez de asumirlo a partir del tipo de modelo.
""")

add_md(nb, "## 4.7. Costo computacional: scikit-learn vs. PySpark, modelo por modelo")

add_code(nb, """comparacion_tiempo = results_sklearn[["modelo","tiempo_entrenamiento_s","test_auc"]].rename(
    columns={"tiempo_entrenamiento_s":"tiempo_sklearn_s","test_auc":"auc_sklearn"})
mapa_nombres = {"GBT_HistGB": "GBT"}  # normalizar nombre para el cruce
tmp = results_spark[["modelo","tiempo_entrenamiento_s","test_auc"]].rename(
    columns={"tiempo_entrenamiento_s":"tiempo_spark_s","test_auc":"auc_spark"})
comparacion_tiempo["modelo_join"] = comparacion_tiempo["modelo"].replace(mapa_nombres)
comparacion_tiempo = comparacion_tiempo.merge(tmp, left_on="modelo_join", right_on="modelo", suffixes=("","_2"))
comparacion_tiempo["mas_rapido"] = np.where(comparacion_tiempo["tiempo_sklearn_s"] < comparacion_tiempo["tiempo_spark_s"],
                                              "scikit-learn", "PySpark")
comparacion_tiempo["razon_veces"] = (comparacion_tiempo[["tiempo_sklearn_s","tiempo_spark_s"]].max(axis=1) /
                                      comparacion_tiempo[["tiempo_sklearn_s","tiempo_spark_s"]].min(axis=1))
tabla_final = comparacion_tiempo[["modelo","tiempo_sklearn_s","tiempo_spark_s","mas_rapido","razon_veces","auc_sklearn","auc_spark"]]
tabla_final.round(2)
""")

add_code(nb, """total_sklearn = results_sklearn["tiempo_entrenamiento_s"].sum()
total_spark = results_spark["tiempo_entrenamiento_s"].sum()
print(f"Tiempo TOTAL de entrenamiento (6 modelos): scikit-learn={total_sklearn:,.1f}s (~{total_sklearn/60:.1f} min) "
      f"| PySpark={total_spark:,.1f}s (~{total_spark/60:.1f} min)")
print(f"scikit-learn fue {total_spark/total_sklearn:.2f}x mas rapido en total.")
""")

add_md(nb, """**scikit-learn resultó más rápido en el total** (~3 h 33 min vs. ~4 h 4 min), y gana claramente en
4 de los 6 modelos individuales (regresión logística, bosque aleatorio, *boosting*, Naive Bayes).
Esto tiene sentido: en una máquina de **solo 2 núcleos**, el paralelismo distribuido de Spark no
tiene margen para compensar su propio costo fijo de coordinación (miles de *stages*,
serialización entre tareas, planificación del *scheduler*) — ese costo se amortiza en un clúster
real con decenas o cientos de núcleos, no aquí. `LinearSVC` es la única excepción clara y
contundente (Spark 8.5 veces más rápido *y* con mejor AUC), precisamente porque en scikit-learn ese
modelo colapsó a una solución degenerada que, irónicamente, seguía siendo cara de calcular
(`liblinear` iteró 633 veces igual). `DecisionTree` (ya con el bug de la sección 4.2.2 corregido)
también fue cerca de 2 veces más rápido en Spark, con un AUC apenas 1.7 puntos por debajo del de
scikit-learn — aquí sí es una victoria razonablemente limpia para Spark, aunque más modesta que la
de `LinearSVC`.

Esto responde directamente a una de las preguntas de la reflexión final: **en este proyecto, con
este hardware, PySpark no llegó a superar a scikit-learn en velocidad total** — la ventaja de Spark
solo aparecería con datasets sustancialmente más grandes que no entren en la memoria de una sola
máquina, o con un clúster de varios nodos donde sí exista paralelismo real que explotar.
""")

add_md(nb, "## 4.8. Síntesis: mejor modelo del entorno PySpark")

add_md(nb, """Por AUC de prueba, **`GBTClassifier` (0.733)** es el mejor modelo de este entorno — igual que en
scikit-learn, un método de *boosting* con histogramas es el que mejor equilibra desempeño y costo.
Este modelo (guardado en `data/spark_best_model_GBT`) es el candidato para la interpretabilidad con
LIME de la sección 6, y los puntajes de los 6 modelos en `data/spark_test_scores.parquet` —sobre
exactamente las mismas 452.141 filas de prueba que scikit-learn, verificado en la sección 4.1— son
la base de las comparaciones estadísticas de las secciones 5 y 6.
""")

save(nb, "book/04_modelado_pyspark.ipynb")
print("Notebook 04_modelado_pyspark.ipynb creado.")
