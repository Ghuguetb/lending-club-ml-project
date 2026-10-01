import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 3. Modelado con scikit-learn (9.10.4.4)

Se entrenan los 6 modelos pedidos por el enunciado usando `GridSearchCV` (cv=3, `scoring="roc_auc"`),
**sin envolverlos en un `Pipeline`** (los datos ya llegan preprocesados desde la sección 2 — matrices
dispersas `X_train`/`X_test` con 114 columnas, ya imputadas, escaladas y codificadas, ajustadas
únicamente con el conjunto de entrenamiento).

**Hardware real de esta máquina:** 2 núcleos, 7.8 GB de RAM, sin GPU (guardado en
`outputs/tables/hardware_sklearn.json` para usarlo también en la comparación de tiempos con PySpark).

**Nota sobre cómo se ejecutó esta sección.** El entrenamiento completo de los 6 modelos
(`src/model_sklearn.py`) tardó **~3 h 35 min** en esta máquina — dominado por `RandomForestClassifier`
(112 min) y `LinearSVC` (80 min), ambos explicados más abajo. Ese tiempo hace impráctico (y
arriesgado) volver a ejecutar el entrenamiento completo cada vez que se reconstruye este Jupyter
Book. Por eso este notebook:

1. Documenta el **código exacto** que se usó para entrenar cada modelo (como bloques de código de
   referencia, no como celdas ejecutables), para que el proceso sea completamente verificable.
2. **Carga los resultados y modelos ya entrenados** (`outputs/tables/sklearn_results.csv`,
   `data/sklearn_test_scores.parquet`, `data/sklearn_fitted_models.joblib`) para construir las
   tablas, gráficos y diagnósticos de esta sección — estas celdas sí se ejecutan de verdad.

Esta separación entre "script de cómputo pesado" y "notebook de análisis" es una práctica estándar
cuando el entrenamiento toma horas: computar una vez, documentar y analizar cuantas veces haga falta.
""")

add_md(nb, "## 3.1. Carga de resultados, modelos y puntajes guardados")

add_code(nb, """import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

results_df = pd.read_csv("../outputs/tables/sklearn_results.csv")
scores_test = pd.read_parquet("../data/sklearn_test_scores.parquet")
models_fitted = joblib.load("../data/sklearn_fitted_models.joblib")
with open("../outputs/tables/hardware_sklearn.json") as f:
    hardware = json.load(f)

y_test = scores_test["default"].values
print("Hardware usado:", hardware)
print(f"Conjunto de prueba: {len(y_test):,} filas, tasa de default = {y_test.mean()*100:.2f}%")
print(f"\\nModelos entrenados: {results_df['modelo'].tolist()}")
""")

add_md(nb, """## 3.2. Modelo por modelo: malla de hiperparámetros y razonamiento

A continuación, el código de referencia usado por `src/model_sklearn.py` para cada uno de los 6
modelos (no se re-ejecuta aquí por las razones de tiempo ya explicadas).
""")

add_md(nb, """### 3.2.1. Regresión logística

La malla de `C` se define a partir de `n_train` (1.808.560 filas), siguiendo la convención de que
`C = 1/(alpha·n)` en scikit-learn equivale al parámetro de regularización `alpha` de otras
librerías. Se usan tres órdenes de magnitud de `alpha` (`1e-6, 1e-5, 1e-4`) para no acotar la
búsqueda a un único vecindario de regularización:

```python
C_vals = [1/(rp*n_train) for rp in [1e-6, 1e-5, 1e-4]]
gs = GridSearchCV(LogisticRegression(max_iter=500, random_state=42),
                   {"C": C_vals}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
```
""")

add_md(nb, """### 3.2.2. Árbol de decisión

```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                   {"max_depth": [5, 10, 15]}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
```
""")

add_md(nb, """### 3.2.3. Bosque aleatorio (RandomForest) — el más costoso

```python
gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1),
                   {"n_estimators": [10, 50, 100], "max_depth": [5, 10, 15]},
                   cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
```

Nótese `n_jobs=1` **dentro** del estimador y `n_jobs=-1` en `GridSearchCV`: se paraleliza a través
de las 27 combinaciones de la malla (9 combinaciones × 3 pliegues), no dentro de cada bosque — si se
paralelizaran ambos niveles a la vez, los procesos competirían por los mismos 2 núcleos sin ganancia
neta (paralelismo anidado). Aun así, este modelo tomó **6.739 s (~112 minutos)** solo en el ajuste
final: la combinación ganadora fue la más pesada de la malla (`n_estimators=100, max_depth=15`), y al
correr con `n_jobs=1`, sus 100 árboles se entrenan de forma secuencial sobre 1.8 millones de filas.
""")

add_md(nb, """### 3.2.4. Gradient Boosting → `HistGradientBoostingClassifier` (sustitución justificada)

El enunciado permite explícitamente sustituir `GradientBoostingClassifier` por
`HistGradientBoostingClassifier` cuando el costo computacional resulte prohibitivo.
`GradientBoostingClassifier` no tiene paralelismo nativo entre árboles (cada árbol depende del
residuo del anterior) y, para 1.8 millones de filas, el tiempo estimado era del orden de varias
horas. `HistGradientBoostingClassifier` usa el mismo algoritmo de histogramas que `GBTClassifier`
de Spark, lo cual además hace la comparación entre entornos más justa (ambos usan una estrategia de
binning equivalente).

```python
X_train_dense = X_train.toarray()   # HistGradientBoosting no admite entrada dispersa
X_test_dense = X_test.toarray()
gs = GridSearchCV(HistGradientBoostingClassifier(learning_rate=0.1, random_state=42),
                   {"max_iter": [50, 100], "max_depth": [3, 5]}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train_dense, y_train)
```
""")

add_md(nb, """### 3.2.5. SVM lineal (`LinearSVC`, pérdida *hinge*)

Se usa `loss="hinge", dual=True` — no la configuración por defecto de scikit-learn
(`squared_hinge`, `dual="auto"`) — para que la función de pérdida coincida con la de
`LinearSVC` de Spark (que solo implementa *hinge* estándar). Como `LinearSVC` no tiene
`predict_proba`, se usa `decision_function()` para obtener un puntaje continuo (necesario más
adelante para la prueba de DeLong).

```python
gs = GridSearchCV(LinearSVC(loss="hinge", dual=True, max_iter=5000, random_state=42),
                   {"C": C_vals}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
y_score = gs.decision_function(X_test)
```

Este modelo tardó **4.839 s (~81 minutos)** — y, como se ve en la sección 3.7, el resultado tiene un
problema serio que merece un diagnóstico propio.
""")

add_md(nb, """### 3.2.6. Naive Bayes (`GaussianNB`)

El enunciado no pide búsqueda de hiperparámetros para este modelo (no tiene ninguno relevante que
ajustar por grilla). Como `GaussianNB` tampoco admite entrada dispersa, se reutiliza la versión
densa ya creada para `HistGradientBoostingClassifier`.

```python
nb_model = GaussianNB()
nb_model.fit(X_train_dense, y_train)
cv_auc_nb = cross_val_score(GaussianNB(), X_train_dense, y_train, cv=3,
                             scoring="roc_auc", n_jobs=-1).mean()
```
""")

add_md(nb, "## 3.3. Tabla comparativa final")

add_code(nb, """cols_show = ["modelo","best_params","cv_auc","test_auc","test_accuracy",
             "test_precision","test_recall","test_f1","tiempo_entrenamiento_s","tiempo_prediccion_s"]
tabla = results_df[cols_show].sort_values("test_auc", ascending=False).reset_index(drop=True)
tabla_fmt = tabla.copy()
for c in ["cv_auc","test_auc","test_accuracy","test_precision","test_recall","test_f1"]:
    tabla_fmt[c] = tabla_fmt[c].round(4)
tabla_fmt["tiempo_entrenamiento_s"] = tabla_fmt["tiempo_entrenamiento_s"].round(1)
tabla_fmt["tiempo_prediccion_s"] = tabla_fmt["tiempo_prediccion_s"].round(2)
tabla_fmt
""")

add_md(nb, """**Lectura rápida:** por AUC de prueba, el orden es
`GBT_HistGB (0.739) > DecisionTree (0.737) > RandomForest (0.722) ≈ LogisticRegression (0.722) >
GaussianNB (0.693) >> LinearSVC (0.559)`. Sorprende que un árbol de decisión individual (poco
profundo, `max_depth=10`) supere en AUC de prueba al bosque aleatorio de 100 árboles — se retoma en
la sección 3.8 (costo computacional vs. beneficio).

**Los valores de `test_f1` son casi todos cercanos a 0** salvo `GaussianNB`. Esto **no** significa
que los modelos sean malos — es un artefacto del umbral fijo de 0.5 combinado con el fuerte
desbalance de clases (11.88% de `default=1`). Se explica en la sección 3.6.
""")

add_md(nb, "## 3.4. Curvas ROC (los 6 modelos superpuestos)")

add_code(nb, """score_cols = {
    "LogisticRegression": "score_LogisticRegression",
    "DecisionTree": "score_DecisionTree",
    "RandomForest": "score_RandomForest",
    "GBT_HistGB": "score_GBT_HistGB",
    "LinearSVC": "score_LinearSVC",
    "GaussianNB": "score_GaussianNB",
}

fig, ax = plt.subplots(figsize=(7,7))
for nombre, col in score_cols.items():
    fpr, tpr, _ = roc_curve(y_test, scores_test[col])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc:.3f})", linewidth=1.8)
ax.plot([0,1],[0,1], linestyle="--", color="gray", linewidth=1, label="Azar (AUC=0.5)")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curvas ROC — scikit-learn (conjunto de prueba)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/figures/eda/roc_sklearn.png", dpi=110)
plt.show()
""")

add_md(nb, """`LinearSVC` es visiblemente la única curva que se pega a la diagonal de azar — consistente con el
AUC de 0.559 y con el diagnóstico de la sección 3.7.
""")

add_md(nb, "## 3.5. Matrices de confusión (umbral = 0.5)")

add_code(nb, """fig, axes = plt.subplots(2, 3, figsize=(13,8))
for ax, (nombre, col) in zip(axes.flat, score_cols.items()):
    if nombre == "LinearSVC":
        y_pred = (scores_test[col] >= 0).astype(int)
    else:
        y_pred = (scores_test[col] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=9)
    ax.set_xticks([0,1]); ax.set_xticklabels(["No default","Default"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["No default","Default"])
    ax.set_title(nombre, fontsize=10)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/confusion_sklearn.png", dpi=110)
plt.show()
""")

add_md(nb, """## 3.6. Por qué el F1 es tan bajo: umbral de 0.5 + desbalance de clases

Con solo 11.88% de casos positivos (`default=1`), un modelo puede tener un AUC razonable
(buena capacidad de **ordenar** los casos de mayor a menor riesgo) y aun así predecir **casi nunca
"default"** si se usa un umbral fijo de 0.5 sobre una probabilidad o puntaje que rara vez supera ese
valor cuando la clase minoritaria es tan rara. Es justamente lo que pasa aquí:
""")

add_code(nb, """resumen_umbral = []
for nombre, col in score_cols.items():
    if nombre == "LinearSVC":
        pred_pos_rate = (scores_test[col] >= 0).mean()
    else:
        pred_pos_rate = (scores_test[col] >= 0.5).mean()
    resumen_umbral.append({"modelo": nombre, "tasa_predicha_positiva": pred_pos_rate})
resumen_umbral = pd.DataFrame(resumen_umbral)
resumen_umbral["tasa_real_positiva"] = y_test.mean()
resumen_umbral.round(4)
""")

add_md(nb, """`RandomForest` y `LinearSVC` predicen **0% de casos positivos** en todo el conjunto de prueba
(452.141 filas) — literalmente nunca cruzan el umbral. `DecisionTree` y `GBT_HistGB` casi nunca lo
hacen (recall < 1%). Solo `GaussianNB` predice positivo en una fracción alta (83%), lo que **invierte**
el problema: gana mucho recall (0.96) a costa de una precisión muy baja (0.14) y una exactitud que
cae a 27.7% — exactamente lo opuesto al resto de los modelos.

Esto es exactamente el motivo por el que el enunciado pide, más adelante (9.10.4.6.2), elegir el
**umbral de clasificación usando solo el conjunto de entrenamiento** para la prueba de McNemar, en
vez de usar 0.5 a ciegas: con un umbral calibrado al desbalance real, estos mismos modelos
mostrarían un comportamiento de clasificación mucho más razonable. El AUC (que no depende de ningún
umbral) sigue siendo la métrica más informativa para comparar modelos en esta sección.
""")

add_md(nb, "## 3.7. Diagnóstico: el colapso de `LinearSVC`")

add_code(nb, """svc = models_fitted["LinearSVC"]
print("Iteraciones hasta convergencia (n_iter_):", svc.n_iter_, "de max_iter =", svc.max_iter)
print("C seleccionado:", svc.C)
print(f"\\nCoeficientes: min={svc.coef_.min():.4g}  max={svc.coef_.max():.4g}  "
      f"media={svc.coef_.mean():.4g}  std={svc.coef_.std():.4g}")
print("Intercepto:", svc.intercept_[0])
print("Coeficientes con valor > 0:", (svc.coef_ > 0).sum(), "de", svc.coef_.shape[1])

score_svc = scores_test["score_LinearSVC"].values
print(f"\\ndecision_function en prueba: min={score_svc.min():.6g} max={score_svc.max():.6g} "
      f"media={score_svc.mean():.6g} std={score_svc.std():.4g}")
print("Valores únicos de decision_function:", len(np.unique(score_svc)), "de", len(score_svc))
""")

add_md(nb, """**Lo que revela el diagnóstico:** el modelo sí convergió (`n_iter_=633` de un máximo de 5.000 —
no es un problema de falta de iteraciones). El problema es **dónde** convergió: prácticamente todos
los coeficientes son negativos o esencialmente cero (el máximo es del orden de 1e-13), y el puntaje
`decision_function` en el conjunto de prueba está **pegado a -1 para el 100% de las filas**
(desviación estándar ≈ 5×10⁻¹²  — ruido numérico, no señal real). El modelo colapsó a un
clasificador casi constante que predice "no default" para todo el mundo.

**Por qué ocurre esto y no le pasa a la regresión logística con el mismo `C`.** La malla de `C`
usada (heredada del mismo criterio que la regresión logística, `C = 1/(alpha·n_train)`) produce
valores de regularización muy pequeños (entre ≈0.0055 y ≈0.56) para un dataset de 1.8 millones de
filas — es decir, una regularización **fuerte** en términos absolutos. Con pérdida logística (como
en `LogisticRegression`), el gradiente suave sigue empujando el modelo hacia estimaciones de
probabilidad informativas incluso bajo regularización fuerte. Con pérdida *hinge* (como en
`LinearSVC`) y **sin ponderar las clases** (`class_weight=None`), el objetivo que minimiza
`liblinear` puede satisfacerse mejor —dado el desbalance de 88%/12%— encogiendo los coeficientes
hacia cero y dejando que el intercepto niegue todo: así se paga la penalización de margen solo sobre
el 12% de casos positivos, en vez de arriesgar margen en el 88% de negativos. Es una patología
conocida de las SVM de margen duro/blando sin ponderación de clases sobre datos muy desbalanceados,
agravada aquí por una regularización más fuerte de lo que este modelo en particular puede tolerar.

Esto se documenta como un **hallazgo real del proyecto**, no como un error a esconder: es
exactamente el tipo de comparación entre implementaciones que pide la reflexión final
(9.10.4.8) — Spark implementa `LinearSVC` con un solver distinto (OWLQN), y es una pregunta abierta
legítima si mostrará el mismo colapso bajo la misma malla de `C`.
""")

add_md(nb, "## 3.8. Costo computacional vs. beneficio")

add_code(nb, """fig, ax = plt.subplots(figsize=(8,5))
orden = tabla.sort_values("tiempo_entrenamiento_s", ascending=True)
bars = ax.barh(orden["modelo"], orden["tiempo_entrenamiento_s"], color="steelblue")
ax.set_xlabel("Tiempo de entrenamiento (s, escala log)")
ax.set_xscale("log")
ax.set_title("Tiempo de entrenamiento por modelo — scikit-learn")
for bar, t in zip(bars, orden["tiempo_entrenamiento_s"]):
    ax.text(bar.get_width()*1.05, bar.get_y()+bar.get_height()/2,
            f"{t:,.0f} s", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/figures/eda/tiempos_sklearn.png", dpi=110)
plt.show()
""")

add_md(nb, """`RandomForest` (6.739 s) y `LinearSVC` (4.839 s) dominan por completo el tiempo total —juntos son
**más del 96%** del tiempo de cómputo de las 6 modelos combinados— y ninguno de los dos ofrece el
mejor desempeño: `RandomForest` empata esencialmente con `LogisticRegression` (que tarda 70 s, casi
100 veces menos) y `LinearSVC` directamente colapsa. `HistGradientBoostingClassifier`, en cambio,
logra el **mejor AUC de los seis modelos** (0.739) en 600 s — menos de una décima parte del tiempo de
`RandomForest`. Esto es evidencia concreta a favor de que, en datasets de este tamaño con hardware
limitado, un método de boosting basado en histogramas ofrece la mejor relación desempeño/costo, y
que el paralelismo entre árboles independientes (bagging) no compensa por sí solo la falta de
núcleos disponibles.
""")

add_md(nb, """## 3.9. Síntesis: mejor modelo del entorno scikit-learn

Por AUC de prueba, **`HistGradientBoostingClassifier` (0.739)** es el mejor modelo de este entorno,
seguido de cerca por `DecisionTreeClassifier` (0.737). Este resultado —y los puntajes continuos de
los 6 modelos guardados en `data/sklearn_test_scores.parquet`— son la base de las comparaciones
estadísticas de las secciones 4 (DeLong) y 5 (McNemar / bootstrap) más adelante, junto con los
resultados equivalentes que se obtendrán con PySpark en la siguiente sección.
""")

save(nb, "book/03_modelado_sklearn.ipynb")
print("Notebook 03_modelado_sklearn.ipynb creado.")
