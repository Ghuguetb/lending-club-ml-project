import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 6. McNemar + bootstrap pareado, con umbral elegido solo con datos de entrenamiento (9.10.4.6.2)

**Qué agregan estas dos pruebas frente a DeLong (sección 5).** DeLong compara la capacidad de
*ranking* de dos modelos — el AUC — con una fórmula analítica exacta que tiene en cuenta la
correlación entre los dos clasificadores. Eso es exactamente lo que hace falta para responder "¿cuál
modelo ordena mejor los casos?", pero **no** dice nada sobre lo que pasaría si alguno de estos
modelos se usara realmente para aprobar o rechazar solicitudes con un punto de corte fijo. Esta
sección responde una pregunta distinta y complementaria: **a un umbral de decisión real, ¿un
modelo se equivoca con más frecuencia que el otro, y en qué dirección?**

Dos herramientas para eso:
- **Prueba de McNemar**: compara las tasas de *acierto* de dos clasificadores en los mismos casos,
  a través de una tabla de concordancia 2×2 (¿en qué casos acierta A y falla B, y viceversa?).
- **Bootstrap pareado**: un intervalo de confianza no paramétrico para ΔAUC, ΔAUC-PR y ΔF1, que no
  depende de la aproximación asintótica normal que usa DeLong — útil como chequeo independiente, y
  necesario para dos métricas (AUC-PR, F1) que no tienen una fórmula de varianza tan simple como la
  de DeLong.

**Requisito metodológico clave:** cualquier umbral de clasificación usado aquí se elige **solo con
datos de entrenamiento**. Elegirlo mirando el conjunto de prueba (por ejemplo, buscando el umbral
que maximiza el F1 *en el test*) filtraría información del test hacia una decisión que después se
evalúa en ese mismo test — un sesgo optimista clásico. Con el umbral fijado de antemano en
entrenamiento, la evaluación en test es honesta.
""")

add_code(nb, """import sys
sys.path.insert(0, "../src")
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mcnemar_bootstrap import umbral_youden, mcnemar_test, bootstrap_pareado
from delong import holm_correction

pd.set_option("display.width", 160)
print("Modulo mcnemar_bootstrap.py cargado correctamente.")
""")

add_md(nb, "## 6.1. Umbral de clasificación: índice de Youden, calculado solo con datos de entrenamiento")

add_md(nb, r"""El **índice J de Youden** es $J = \text{sensibilidad} + \text{especificidad} - 1 = TPR - FPR$,
evaluado en cada punto de la curva ROC. El umbral que maximiza $J$ es el que balancea mejor ambos
tipos de error (falsos positivos y falsos negativos) — a diferencia del umbral por defecto de 0.5
(o de margen=0 para modelos lineales), que no tiene ninguna razón particular para ser óptimo en un
problema tan desbalanceado como este (~12% de incumplimiento).

```python
def umbral_youden(y_train, score_train):
    fpr, tpr, thresholds = roc_curve(y_train, score_train)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])
```

Se calcula **una vez por modelo, por entorno, usando únicamente `y_train`/`score_train`** — nunca
se toca el conjunto de prueba en este paso.
""")

add_code(nb, """sk_train = pd.read_parquet("../data/sklearn_train_scores.parquet")
sk_test = pd.read_parquet("../data/sklearn_test_scores.parquet")
sp_train = pd.read_parquet("../data/spark_train_scores_raw.parquet")
sp_test = pd.read_parquet("../data/spark_test_scores.parquet")
for df in (sk_train, sk_test, sp_train, sp_test):
    df["id"] = df["id"].astype(str)

# Alineacion por id (igual que en la seccion 5): todo lo que compare ambos entornos se lee de aqui.
test_merged = sk_test.merge(sp_test, on="id", suffixes=("_sk", "_sp"))
assert (test_merged["default_sk"] == test_merged["default_sp"]).all()
y_test = test_merged["default_sk"].values
print(f"Test alineado: {len(test_merged):,} filas (tasa de default: {y_test.mean()*100:.2f}%)")
""")

add_md(nb, """**Demostración rápida** de que la funcion reproduce el umbral ya guardado, con un solo modelo
(regresión logística, scikit-learn):""")

add_code(nb, """u_demo = umbral_youden(sk_train["default"].values, sk_train["score_LogisticRegression"].values)
print(f"Umbral Youden (recalculado ahora): {u_demo:.6f}")
""")

add_code(nb, """import json
umbrales = json.load(open("../outputs/tables/umbrales_youden.json"))
tabla_umbrales = pd.DataFrame([
    {"modelo": k.rsplit("_", 1)[0], "entorno": k.rsplit("_", 1)[1], "umbral": v}
    for k, v in umbrales.items()
]).pivot(index="modelo", columns="entorno", values="umbral")
tabla_umbrales
""")

add_md(nb, """La mayoría son razonables y cercanos entre entornos (regresión logística, árbol, bosque, GBT: todos
entre 0.12 y 0.13 — coherente con que ~12% de la base es la clase positiva). Dos casos se salen de
ese patrón y merecen mirarse con cuidado antes de usarlos.
""")

add_md(nb, "### 6.1.1. Los dos umbrales atípicos: por qué están ahí (no son un error)")

add_code(nb, """for nombre, col in [("sklearn LinearSVC", ("score_LinearSVC", sk_train)),
                    ("spark LinearSVC",   ("score_LinearSVC", sp_train)),
                    ("sklearn GaussianNB (NaiveBayes)", ("score_GaussianNB", sk_train)),
                    ("spark NaiveBayes",  ("score_NaiveBayes", sp_train))]:
    c, df = col
    s = df[c].values
    print(f"{nombre:34s} min={s.min():9.4f}  max={s.max():9.4f}  mean={s.mean():8.4f}  std={s.std():.4f}")
""")

add_md(nb, """- **`LinearSVC` en scikit-learn**: `std=0.0`, `min=max=-1.0000` exacto. Esto **no** es un umbral
  elegido entre alternativas: es el síntoma, en los datos de entrenamiento, del colapso ya
  diagnosticado en la sección 3.7 — el modelo asigna literalmente el **mismo** puntaje a las
  1.808.560 filas de entrenamiento. Con un puntaje constante no existe ninguna curva ROC real (todo
  punto de corte por encima o por debajo de -1 da la misma predicción para todo el mundo), así que
  el valor que devuelve `argmax` sobre esa meseta plana es esencialmente arbitrario. Es un umbral
  **degenerado**, y así se reporta en las conclusiones más abajo.
- **`LinearSVC` en Spark** (no colapsado): el puntaje sí varía (`std=0.059`), pero está *todo*
  concentrado en valores negativos, entre -1.34 y -1.00. Con el umbral "natural" de margen=0 este
  modelo predeciría "no default" para el 100% de los casos (igual que scikit-learn a ese umbral,
  como ya se notó en la sección 4.6b) — pero el índice de Youden encuentra, dentro de ese rango
  angosto, el punto real donde la separación entre clases es mejor (~-1.0001), y ese punto sí
  recupera capacidad predictiva genuina (de ahí el AUC=0.704 de la sección 4). Es un umbral extremo,
  pero no degenerado: es la prueba de que elegir el punto de corte con datos de entrenamiento, en
  vez de usar 0 por defecto, importa.
- **`GaussianNB` en scikit-learn**: `mean=0.83`, con **369.821 de 1.808.560 filas** (~20%) en
  exactamente `1.0000`. La probabilidad se satura en el techo para una fracción enorme de los datos
  — el reflejo, otra vez, de la mala calibración ya señalada en la sección 3.6 (recall=0.96,
  precisión=0.24: el modelo básicamente dice "default" casi siempre). El umbral óptimo termina en el
  borde mismo de esa saturación.
- **`NaiveBayes` en Spark**: mucho menos saturado (solo ~3% de las filas en el máximo, mediana en
  torno a `4×10⁻⁵`) — una distribución de probabilidad más extendida, consistente con su AUC
  (0.699) apenas por encima del de scikit-learn (0.694, ya corregido en la sección 4.6c).

En síntesis: de los cuatro casos atípicos, **solo uno (`LinearSVC` en scikit-learn) es realmente
degenerado**; los otros tres son umbrales extremos pero genuinos, reflejo de qué tan bien (o mal)
calibrado está cada modelo.
""")

add_md(nb, "## 6.2. Las 8 comparaciones: 6 entre entornos + el par top-2-AUC dentro de cada entorno")

add_code(nb, """res_sk = pd.read_csv("../outputs/tables/sklearn_results.csv").set_index("modelo")
res_sp = pd.read_csv("../outputs/tables/spark_results.csv").set_index("modelo")
top2_sk = res_sk["test_auc"].sort_values(ascending=False).index[:2].tolist()
top2_sp = res_sp["test_auc"].sort_values(ascending=False).index[:2].tolist()
print("Top-2 AUC en scikit-learn:", top2_sk)
print("Top-2 AUC en PySpark     :", top2_sp)
print()
print("Las 8 comparaciones: 6 pares entre entornos (misma familia de modelo, scikit-learn vs. PySpark)")
print("                    + top2_dentro_sklearn (los 2 mejores AUC de scikit-learn entre si)")
print("                    + top2_dentro_spark    (los 2 mejores AUC de PySpark entre si)")
""")

add_md(nb, """La corrección de Holm se aplica **dentro de este único grupo de 8 comparaciones** (no junto con
las 36 de la sección 5): son preguntas distintas — aquí interesa controlar la tasa de falsos
descubrimientos entre las 8 decisiones de McNemar que se van a tomar en este análisis, no mezclarlas
con las comparaciones de AUC de la sección anterior.
""")

add_md(nb, "## 6.3. Prueba de McNemar: tabla de concordancia + corrección de Holm")

add_md(nb, r"""Para cada par de modelos, con predicciones binarias `pred_a`, `pred_b` ya aplicado el umbral de
Youden correspondiente, se construye una tabla de **concordancia de aciertos**:

|                     | B acierta        | B falla          |
|---------------------|------------------|-------------------|
| **A acierta**       | $n_{11}$         | $n_{10}$ (A gana) |
| **A falla**         | $n_{01}$ (B gana)| $n_{00}$          |

McNemar prueba $H_0: n_{10} = n_{01}$ (ningún modelo se equivoca sistemáticamente más que el otro,
en los casos donde justo difieren). Se usa la versión **exacta** (binomial) cuando $n_{10}+n_{01}<25$
y la versión **asintótica con corrección de continuidad** en caso contrario — el mismo criterio que
usa `statsmodels.stats.contingency_tables.mcnemar`.

```python
def mcnemar_test(y_true, pred_a, pred_b):
    correct_a, correct_b = (pred_a == y_true), (pred_b == y_true)
    tabla = np.array([[ (correct_a & correct_b).sum(),  (correct_a & ~correct_b).sum() ],
                       [ (~correct_a & correct_b).sum(), (~correct_a & ~correct_b).sum() ]])
    b, c = tabla[0,1], tabla[1,0]
    return mcnemar(tabla, exact=(b+c < 25), correction=True)
```

**Importante: McNemar se calcula siempre sobre las 452.141 filas completas del test** (a diferencia
del bootstrap de la sección 6.4, que usa una submuestra por costo computacional) — es una prueba
barata, no hace falta reducir nada.
""")

add_code(nb, """mcnemar_resultados = pd.read_csv("../outputs/tables/mcnemar_resultados.csv")
cols_mc = ["par","tipo","n10_A_acierta_B_falla","n01_A_falla_B_acierta","b_mas_c","exacta",
           "statistic","p_value","p_holm","rechaza_h0_holm_0.05"]
mcnemar_resultados[cols_mc].round(4)
""")

add_md(nb, """**Lectura:** con hasta 452.141 pares de observaciones, incluso una diferencia modesta en la tasa de
error se vuelve estadísticamente detectable — **las 8 comparaciones rechazan $H_0$** después de
Holm, igual que ocurrió con las 36 comparaciones de DeLong en la sección 5. El caso más extremo es
`LinearSVC` entre entornos ($n_{10}+n_{01}=193{,}719$, un 43% de los casos donde los dos modelos
*difieren*) — coherente con que uno de los dos está colapsado a predecir "no default" casi siempre y
el otro no. El caso con menor desacuerdo es `NaiveBayes` entre entornos (apenas 78 pares
discordantes), lo cual es un poco engañoso: ambos umbrales (1.0 y 0.0003) son tan extremos que casi
ningún caso se clasifica como "default" en ninguno de los dos entornos — hay poco desacuerdo porque
hay poquísimas predicciones positivas en general, no porque los modelos sean parecidos.
""")

add_md(nb, "## 6.4. Bootstrap pareado: ΔAUC, ΔAUC-PR, ΔF1 con IC95%")

add_md(nb, """El bootstrap pareado remuestrea **los mismos índices** para ambos modelos en cada iteración
(preserva la correlación entre A y B, igual que DeLong lo hace analíticamente), y con eso construye
un intervalo de confianza percentil 95% para la diferencia observada en tres métricas: AUC, AUC-PR
(más informativa que AUC bajo desbalance de clases) y F1 (a diferencia de las otras dos, sí depende
del umbral de Youden elegido en 6.1).

```python
def bootstrap_pareado(y_true, score_a, score_b, pred_a, pred_b, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    deltas_auc, deltas_auc_pr, deltas_f1 = [], [], []
    for _ in range(B):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yb = y_true[idx]
        if yb.sum() in (0, len(yb)):
            continue  # remuestreo degenerado, se descarta
        deltas_auc.append(roc_auc_score(yb, score_a[idx]) - roc_auc_score(yb, score_b[idx]))
        # (analogo para AUC-PR y F1)
    ...  # IC95% percentil de cada lista de deltas
```

### Por qué se usa una submuestra de 50.000 filas (y no las 452.141 completas)

Perfilando una sola iteración de bootstrap sobre las 452.141 filas completas: **0.73 s/iteración**
→ con B=2000 y 8 comparaciones, **≈194 minutos** solo para este chequeo — impráctico para lo que es,
por diseño, un chequeo **complementario** a DeLong (que sí corrió sobre el test completo en la
sección 5). Con una submuestra aleatoria **estratificada** (semilla fija=42, tasa de incumplimiento
preservada) de 50.000 filas, el costo por iteración baja a **0.084 s** → **≈22 minutos** en total,
sin comprometer la validez del intervalo de confianza (50.000 sigue siendo una muestra grande para
un bootstrap). McNemar, en cambio, sí usa las 452.141 filas completas (sección 6.3) porque ahí el
costo por sí es insignificante.
""")

add_code(nb, """bootstrap_resultados = pd.read_csv("../outputs/tables/bootstrap_resultados.csv")
cols_bt = ["par","tipo","delta_auc","auc_ci_low","auc_ci_high","auc_excluye_cero",
           "delta_auc_pr","auc_pr_ci_low","auc_pr_ci_high","delta_f1","f1_ci_low","f1_ci_high"]
bootstrap_resultados[cols_bt].round(4)
""")

add_md(nb, """En las 8 comparaciones, el intervalo de ΔAUC **excluye el cero** — coherente, otra vez, con
McNemar y con DeLong. La comparación más cercana a cero es `LogisticRegression` entre entornos
(ΔAUC=-0.0011, IC=[-0.0014,-0.0007]): un intervalo angosto, alejado de cero, pero de una magnitud
diminuta — el ejemplo más claro de esta sección de "estadísticamente significativo, prácticamente
irrelevante".
""")

add_md(nb, "## 6.5. Tabla resumen final: ¿coinciden DeLong, McNemar y el bootstrap?")

add_md(nb, """La consigna pide explícitamente comparar las conclusiones de las tres pruebas en una sola tabla.
Se cruzan por nombre de par (traduciendo, cuando hace falta, entre el nombre "de familia" que usa
DeLong y el nombre "de resultados" que usan McNemar/bootstrap — p.ej. `GBT_HistGB``→``GBT`).
""")

add_code(nb, """nombre_a_familia = {"LogisticRegression": "LogisticRegression", "DecisionTree": "DecisionTree",
                     "RandomForest": "RandomForest", "GBT_HistGB": "GBT", "GBT": "GBT",
                     "LinearSVC": "LinearSVC", "GaussianNB": "NaiveBayes", "NaiveBayes": "NaiveBayes"}

delong_between = pd.read_csv("../outputs/tables/delong_between_envs.csv")
delong_within_sk = pd.read_csv("../outputs/tables/delong_within_sklearn.csv")
delong_within_sp = pd.read_csv("../outputs/tables/delong_within_spark.csv")


def familia_de_resultado(etiqueta):
    return nombre_a_familia[re.sub(r'_(sklearn|spark)$', '', etiqueta)]


def delong_para(tipo, par, modelo_a_mc, modelo_b_mc):
    \"\"\"Devuelve (delta_auc, p_holm, rechaza) de DeLong, orientado igual que McNemar/bootstrap
    (modelo_a - modelo_b), buscando la fila correspondiente en la tabla de DeLong que aplique.\"\"\"
    if tipo == "entre_entornos":
        f = delong_between[delong_between["modelo_a"] == f"sk_{par}"].iloc[0]
        return f["delta_auc"], f["p_holm"], bool(f["rechaza_h0_holm_0.05"])
    tabla = delong_within_sk if tipo == "top2_dentro_sklearn" else delong_within_sp
    na, nb_ = familia_de_resultado(modelo_a_mc), familia_de_resultado(modelo_b_mc)
    pre = "sk_" if tipo == "top2_dentro_sklearn" else "sp_"
    fila = tabla[(tabla.modelo_a == pre+na) & (tabla.modelo_b == pre+nb_)]
    if len(fila):
        f = fila.iloc[0]
        return f["delta_auc"], f["p_holm"], bool(f["rechaza_h0_holm_0.05"])
    f = tabla[(tabla.modelo_a == pre+nb_) & (tabla.modelo_b == pre+na)].iloc[0]
    return -f["delta_auc"], f["p_holm"], bool(f["rechaza_h0_holm_0.05"])


filas = []
for _, m in mcnemar_resultados.iterrows():
    b = bootstrap_resultados[(bootstrap_resultados.tipo == m.tipo) & (bootstrap_resultados.par == m.par)].iloc[0]
    dd, dp, drej = delong_para(m.tipo, m.par, m.modelo_a, m.modelo_b)
    filas.append({
        "par": m.par, "tipo": m.tipo,
        "DeLong_ΔAUC": dd, "DeLong_rechaza_H0": drej, "DeLong_práctico(≥0.005)": abs(dd) >= 0.005,
        "McNemar_rechaza_H0": bool(m["rechaza_h0_holm_0.05"]),
        "Bootstrap_ΔAUC": b["delta_auc"], "Bootstrap_IC95%": f"[{b['auc_ci_low']:.4f}, {b['auc_ci_high']:.4f}]",
        "Bootstrap_excluye_0": bool(b["auc_excluye_cero"]),
    })
tabla_resumen = pd.DataFrame(filas)
tabla_resumen["las_3_coinciden_en_signo_y_significancia"] = (
    tabla_resumen["DeLong_rechaza_H0"] & tabla_resumen["McNemar_rechaza_H0"] & tabla_resumen["Bootstrap_excluye_0"]
)
tabla_resumen.round(4)
""")

add_md(nb, """**Conclusión de esta tabla:** las tres pruebas están de acuerdo en **dirección** (el signo de ΔAUC
es el mismo en DeLong y en el bootstrap en las 8 comparaciones) y en **significancia estadística**
(las 8 rechazan $H_0$ en las tres pruebas). Con n≈452.141 esto era predecible — ya se discutió en la
sección 5 — pero es una buena señal de consistencia interna: tres formas distintas de medir "¿son
diferentes?" (ranking vía covarianza analítica, aciertos a un umbral fijo, remuestreo empírico)
llegan a la misma respuesta.

Donde sí aparece un matiz es en la **significancia práctica** (|ΔAUC|≥0.005, el umbral pre-declarado
en la sección 5): `RandomForest` entre entornos (DeLong ΔAUC=0.0039, por debajo del umbral) y
`GBT_HistGB` vs. `DecisionTree` dentro de scikit-learn (DeLong ΔAUC=0.0025) son los dos casos más
ajustados — el bootstrap los sitúa un poco más alto (0.0044 y 0.0049 respectivamente) pero **igual
por debajo o justo en el límite** de 0.005. La conclusión práctica para ambos es la misma con las dos
pruebas: una diferencia real pero pequeña, del tamaño en el que "estadísticamente significativo" y
"importa para elegir un modelo en la práctica" empiezan a separarse.
""")

add_md(nb, "## 6.6. Limitaciones de McNemar y del bootstrap pareado aquí")

add_md(nb, """- **McNemar depende por completo del umbral elegido.** Un punto de corte distinto (igualmente
  válido) puede cambiar cuántos casos son discordantes y en qué dirección — a diferencia de DeLong,
  que compara el ranking completo sin fijar ningún umbral. Por eso esta prueba es complementaria, no
  un sustituto: responde "¿a *este* punto de operación, cuál se equivoca más?", no "¿cuál ordena
  mejor en general?".
- **McNemar no cuantifica magnitud, solo dirección de los aciertos discordantes** — con $n$ grande,
  cualquier asimetría, sin importar cuán chica, se vuelve significativa (la misma advertencia de
  significancia estadística vs. práctica de la sección 5, aquí sin un análogo directo de "tamaño de
  efecto" tan claro como ΔAUC).
- **El bootstrap de esta sección corrió sobre una submuestra fija de 50.000 filas**, no las 452.141
  completas, por una decisión explícita de costo computacional (sección 6.4) — un tratamiento
  distinto al del resto del proyecto, que evita cualquier reducción de escala en los datos de
  entrenamiento/prueba. Queda razonablemente justificado porque (a) es un chequeo complementario a
  DeLong, que sí corrió a escala completa, y (b) 50.000 sigue siendo una muestra grande para un
  intervalo de bootstrap — pero, con más tiempo de cómputo disponible, correr las 452.141 filas
  completas sería la versión más rigurosa.
- **Ambas pruebas, igual que DeLong, evalúan un solo split de entrenamiento/prueba.** Ninguna de las
  tres cuantifica la variabilidad que vendría de repetir todo el proceso (partición, ajuste de
  hiperparámetros, elección de umbral) con una semilla distinta — esa es una pregunta de validación
  cruzada anidada, fuera del alcance de esta comparación puntual.
""")

save(nb, "book/06_mcnemar_bootstrap.ipynb")
print("Notebook 06_mcnemar_bootstrap.ipynb creado.")
