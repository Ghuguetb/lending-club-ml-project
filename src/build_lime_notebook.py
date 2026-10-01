import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 7. Interpretabilidad con LIME (9.10.4.7)

**Qué hace LIME (Ribeiro, Singh y Guestrin, 2016).** Para explicar *una sola predicción* de un
modelo de caja negra, LIME genera muchas variantes perturbadas de esa instancia, les pide al modelo
sus probabilidades, y ajusta un modelo **lineal local** (ponderado por qué tan cerca está cada
variante de la instancia original) sobre esas perturbaciones. Los coeficientes de ese modelo local
son la "explicación": qué tanto empujó cada variable la predicción hacia una clase o la otra, **en
la vecindad de este caso particular** — no es una explicación global del modelo.

Se aplica al **mejor modelo de cada entorno** por AUC de prueba:
- scikit-learn: `HistGradientBoostingClassifier` (AUC=0.739, sección 3)
- PySpark: `GBTClassifier` (AUC=0.733, sección 4)

En cada entorno se explican dos instancias del conjunto de prueba **mal clasificadas al umbral
natural de 0.5** (no el umbral de Youden de la sección 6 — aquí el objetivo es puramente ilustrar
el comportamiento del modelo en un error, no simular una política de decisión real): un **falso
positivo** (predijo incumplimiento y no lo hubo) y un **falso negativo** (predijo que pagaría y
incumplió).
""")

add_code(nb, """import sys
sys.path.insert(0, "../src")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import sparse
import joblib
import lime.lime_tabular

pd.set_option("display.width", 160)
print("Librerias cargadas.")
""")

add_md(nb, "## 7.1. Datos de fondo para LIME")

add_md(nb, """`LimeTabularExplainer` necesita una muestra representativa del conjunto de entrenamiento para
estimar la distribución de cada variable (media/desvío de las numéricas, frecuencia de cada
categoría ya codificada en one-hot). No hace falta el 1.8 millones de filas completo para esto: una
submuestra aleatoria estratificada de 30.000 filas (semilla fija, `src/prepare_lime_backgrounds.py`)
es más que suficiente y evita convertir a denso el conjunto de entrenamiento completo.
""")

add_code(nb, """bg_sklearn = np.load("../data/lime_background_sklearn.npz")["X"]
bg_spark = np.load("../data/lime_background_spark.npz")["X"]
nombres_sklearn = json.load(open("../data/sklearn_feature_names.json"))
nombres_spark = json.load(open("../data/lime_feature_names_spark.json"))
print(f"Fondo scikit-learn: {bg_sklearn.shape}  ({len(nombres_sklearn)} variables)")
print(f"Fondo PySpark:      {bg_spark.shape}  ({len(nombres_spark)} variables)")
""")

add_md(nb, """### Un hallazgo inesperado, encontrado al preparar esto: **las dos representaciones NO son idénticas**

Ambos entornos parten de las mismas 14 variables numéricas (mismo orden, mismo escalado) y las
mismas 8 variables categóricas — pero el número de columnas one-hot resultante **no coincide**:
**114 en scikit-learn vs. 107 en PySpark**. La razón: `OneHotEncoder` de scikit-learn
(`handle_unknown="ignore"`) le da una columna explícita a cada categoría observada, **incluido
`NaN`** cuando `NaN` aparece como valor en los datos de entrenamiento (por eso existen columnas como
`term_nan`, `application_type_nan`); el `OneHotEncoder` de Spark, en cambio, usa por defecto
`dropLast=True` (deja caer la última categoría de cada variable, típicamente la que `StringIndexer`
le asigna a valores nulos/no vistos con `handleInvalid="keep"`), así que esa categoría queda
implícita en el vector de ceros en vez de tener su propia columna. Dos pipelines *conceptualmente*
iguales (mismo imputador, mismo escalador, mismo tipo de codificador) terminan produciendo espacios
de variables de **dimensión distinta** por una diferencia de configuración de librería, no de
método. Vale la pena dejar esto anotado para la reflexión final (9.10.4.8): significa que, aunque
las 452.141 filas de prueba son las mismas en ambos entornos, **no existe un vector de variables
"canónico" compartido** al que se puedan mapear uno a uno las explicaciones de LIME de un entorno y
del otro — cada explicación se interpreta dentro de su propio espacio de variables.
""")

add_md(nb, "## 7.2. LIME sobre scikit-learn (`HistGradientBoostingClassifier`)")

add_code(nb, """models = joblib.load("../data/sklearn_fitted_models.joblib")
gbt_sklearn = models["GBT_HistGB"]

X_test_sk = sparse.load_npz("../data/sklearn_X_test.npz").toarray()
y_test_sk = np.load("../data/sklearn_y_test.npy")

explainer_sklearn = lime.lime_tabular.LimeTabularExplainer(
    training_data=bg_sklearn,
    feature_names=nombres_sklearn,
    class_names=["No default", "Default"],
    categorical_features=list(range(14, len(nombres_sklearn))),  # las primeras 14 son numericas
    discretize_continuous=True,
    random_state=42,
)
print("Explainer de scikit-learn listo.")
""")

add_md(nb, """Las dos instancias a explicar se eligieron con una semilla fija en `src/run_lime_sklearn.py`
(que usa exactamente el `explainer_sklearn` de arriba); aquí se cargan esos resultados ya
calculados en vez de volver a correr `explain_instance` en esta celda, para que el notebook no
dependa de que la explicación aleatoria coincida exactamente cada vez que se re-ejecuta.
""")

add_code(nb, """resultados_sklearn = json.load(open("../outputs/tables/lime_sklearn_resultados.json"))
for etiqueta, r in resultados_sklearn.items():
    print(f"{etiqueta}: id={r['id']}  y_real={r['y_real']}  prob_predicha={r['prob_predicha']:.4f}  "
          f"pred={r['pred_label']}  R²_local={r['r2_local']:.3f}")
""")

add_code(nb, """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, etiqueta in zip(axes, ["falso_positivo", "falso_negativo"]):
    img = mpimg.imread(f"../outputs/figures/eda/lime_sklearn_{etiqueta}.png")
    ax.imshow(img); ax.axis("off"); ax.set_title(etiqueta.replace("_", " ").capitalize())
plt.tight_layout()
plt.show()
""")

add_md(nb, """**Falso positivo** (id=1412565, `prob_predicha=0.508`, apenas por encima de 0.5): la variable que
más empuja hacia "default" es `int_rate > 0.60` (una tasa de interés en el percentil más alto de la
distribución escalada) — coherente con que la tasa de interés es, para todo prestamista, la señal
de riesgo más directa que existe (créditos más riesgosos se les cobra más). El resto son columnas
`sub_grade_*`/`application_type_*` en 0, con pesos pequeños: son la contribución (marginal, porque
*no* están presentes) de las docenas de categorías que el modelo "descarta" al explicar este caso.
El $R^2$ local de 0.44 dice que el modelo lineal local captura una porción razonable —aunque no
completa— del comportamiento de `HistGradientBoostingClassifier` en esta vecindad.

**Falso negativo** (id=17232974, `prob_predicha=0.128`, bastante lejos de 0.5): la tasa de interés
esta vez juega en la dirección **opuesta** (`int_rate` en un rango bajo/medio, empujando hacia "no
default") pese a que la persona sí incumplió — el modelo, con la información disponible, no tenía
señal fuerte para anticipar este caso. El $R^2$ local aquí es de solo **0.09**: la aproximación
lineal explica muy poco de lo que hace el modelo en esta vecindad, así que esta explicación
puntual debería tomarse con más cautela que la anterior — es, en sí mismo, un diagnóstico útil de
cuándo confiar (o no) en una explicación de LIME.
""")

add_md(nb, "## 7.3. LIME sobre PySpark (`GBTClassifier`): el reto de envolver un modelo distribuido")

add_md(nb, """LIME espera una función Python (`predict_fn`) que reciba un array de NumPy y devuelva
probabilidades — exactamente lo que es `predict_proba` en scikit-learn. Un `GBTClassificationModel`
de Spark no tiene ese método: solo sabe transformar un `DataFrame` de Spark. Hace falta un
*adaptador* que, cada vez que LIME llama a `predict_fn` con un lote de muestras perturbadas
(`num_samples=5000` para cada instancia explicada), lo convierta a `DataFrame`, lo pase por
`.transform()`, y traiga el resultado de vuelta a NumPy:

```python
def predict_fn_spark(X):
    filas = [(i, Vectors.dense(fila.tolist())) for i, fila in enumerate(X)]
    df = spark.createDataFrame(filas, ["_orden", "features"])
    pred = gbt_model.transform(df).withColumn("proba_arr", vector_to_array("probability"))
    pdf = pred.select("_orden", "proba_arr").toPandas().sort_values("_orden")
    return np.stack(pdf["proba_arr"].values)
```

Dos detalles importantes: (i) el `_orden` explícito es necesario porque Spark **no** garantiza que
las filas salgan de `.transform()` en el mismo orden en que entraron; sin reordenar, LIME asociaría
cada probabilidad con la perturbación equivocada. (ii) el `.toPandas()` aquí es sobre un lote de
**5.000 filas** (una instancia perturbada a la vez), no sobre el dataset completo — no viola la
restricción de la sección 4 sobre recolectar datos completos a memoria local.

En la práctica, cada llamada a `explain_instance` (que dispara un solo lote de 5.000 filas por
Spark) tomó entre **1 y 3 segundos** en esta máquina — rápido, porque es un Spark local de 2
núcleos sin overhead real de red/cluster. Esto es importante matizarlo en la sección 7.5.
""")

add_code(nb, """resultados_spark = json.load(open("../outputs/tables/lime_spark_resultados.json"))
for etiqueta, r in resultados_spark.items():
    print(f"{etiqueta}: id={r['id']}  y_real={r['y_real']}  prob_predicha={r['prob_predicha']:.4f}  "
          f"pred={r['pred_label']}  R²_local={r['r2_local']:.3f}")
""")

add_code(nb, """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, etiqueta in zip(axes, ["falso_positivo", "falso_negativo"]):
    img = mpimg.imread(f"../outputs/figures/eda/lime_spark_{etiqueta}.png")
    ax.imshow(img); ax.axis("off"); ax.set_title(etiqueta.replace("_", " ").capitalize())
plt.tight_layout()
plt.show()
""")

add_md(nb, """Los dos casos elegidos en PySpark son instancias **distintas** a las de scikit-learn (cada modelo
se equivoca en casos distintos, y no hay garantía de que el mismo `id` sea un error en los dos
entornos) — pero el patrón que emerge es notablemente parecido: en el falso positivo (id=60981901,
`prob=0.559`), `int_rate > 0.60` vuelve a ser la señal dominante hacia "default" ($R^2$ local=0.36,
razonable); en el falso negativo (id=9776027, `prob=0.147`), el modelo tampoco tenía señal fuerte —
y el $R^2$ local vuelve a desplomarse a **0.09**, casi idéntico al 0.09 del falso negativo de
scikit-learn. Esta coincidencia no parece casual: **el ajuste lineal local de LIME es
sistemáticamente peor para los falsos negativos que para los falsos positivos**, en ambos entornos
— probablemente porque los casos de incumplimiento "sorpresa" (probabilidad baja, pero sí
incumplen) están, casi por definición, en zonas del espacio de variables donde el comportamiento
real del modelo es más no lineal/idiosincrático, justo donde una aproximación lineal local es menos
fiel.
""")

add_md(nb, "## 7.4. Limitaciones de LIME en general, y de aplicarlo sobre un modelo de Spark en particular")

add_md(nb, """**Generales (aplican a los dos entornos):**
- **Inestabilidad:** LIME perturba y muestrea aleatoriamente; con otra semilla, el conjunto de
  variables "top-10" puede cambiar, sobre todo cuando (como aquí) hay decenas de columnas one-hot
  con pesos pequeños y parecidos entre sí compitiendo por los últimos lugares del ranking.
- **Fidelidad local variable:** el $R^2$ del modelo lineal local no es constante — en esta sección
  fue bueno (0.36–0.44) para los falsos positivos y pobre (0.09) para los falsos negativos. Una
  explicación con $R^2$ bajo no es necesariamente "incorrecta", pero sí es una aproximación más
  floja de lo que realmente hace el modelo, y conviene reportar siempre ese número junto con la
  explicación, no solo la lista de variables.
- **Ruido de muchas variables *dummy*:** con 93–107 columnas categóricas ya expandidas en one-hot,
  buena parte del "top-10" queda ocupado por columnas en 0 con peso marginal — informativo hasta un
  punto, pero menos legible que si LIME pudiera tratar cada variable categórica original (p.ej.
  `sub_grade`, con sus ~35 niveles) como una sola entidad en vez de docenas de columnas binarias.

**Específicas de aplicar LIME sobre un modelo entrenado en Spark:**
- **No existe un LIME "nativo" para Spark ML** — hubo que escribir a mano el adaptador
  `predict_fn_spark` de la sección 7.3, con el riesgo de bugs de por medio (el reordenamiento por
  `_orden`, por ejemplo, es fácil de pasar por alto y produciría explicaciones completamente
  erróneas sin ningún error visible).
- **Costo por instancia explicada:** cada llamada implica crear un `DataFrame` nuevo, un
  `.transform()` y un `.toPandas()` — aquí, ~1–3 segundos por instancia en un Spark local de 2
  núcleos. Para explicar un puñado de casos (como aquí) es perfectamente viable; para explicar
  miles de instancias de forma rutinaria (p. ej., generar una explicación por cada solicitud
  rechazada) ese costo por-instancia se vuelve el cuello de botella, y en un clúster real con
  overhead de red/planificación por cada micro-lote sería aún peor que en esta máquina.
- **Los dos entornos no comparten el mismo espacio de variables** (114 vs. 107 columnas, sección
  7.1) — las explicaciones de un entorno y del otro no se pueden alinear término a término aunque
  describan, en esencia, el mismo tipo de modelo sobre los mismos datos originales.
""")

save(nb, "book/07_lime.ipynb")
print("Notebook 07_lime.ipynb creado.")
