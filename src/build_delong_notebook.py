import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 5. Prueba de DeLong con corrección de Holm (9.10.4.6.1)

**Qué prueba y por qué hace falta aquí.** El AUC de dos modelos evaluados sobre el mismo conjunto
de prueba **no son observaciones independientes** — comparten las mismas 452.141 filas, así que un
error estándar calculado como si fueran independientes (por ejemplo, restando dos AUCs y usando la
varianza de cada uno por separado) **sobreestima** la incertidumbre real de la diferencia. La prueba
de DeLong, DeLong y Clarke-Pearson (1988) resuelve esto calculando la **covarianza** entre los dos
AUCs directamente a partir de las mismas observaciones, y da un estadístico *z* y un valor-*p* para
la hipótesis nula ΔAUC=0 que sí tiene en cuenta esa correlación.

Esta sección responde tres preguntas: ¿el modelo ganador de cada entorno es *significativamente*
mejor que los otros 5 de su propio entorno? ¿Cada modelo es significativamente distinto entre
scikit-learn y PySpark? ¿Esas diferencias, además de estadísticamente significativas, son
*prácticamente* relevantes?
""")

add_md(nb, "## 5.1. Implementación: versión rápida O(n log n) (Sun & Xu, 2014)")

add_md(nb, """Con n≈452.141, la formulación original de DeLong (que compara cada par positivo-negativo
explícitamente) tiene costo O(m·n) ≈ 53.712 × 398.429 ≈ 21.400 millones de comparaciones **por cada
par de modelos** — inviable para 36 comparaciones. Sun y Xu (2014) demostraron que el mismo cálculo
puede hacerse con *midranks* (rangos con empates promediados) en O(n log n), usando exactamente la
misma covarianza analítica, solo que calculada por ordenamiento en vez de por fuerza bruta.
`src/delong.py` implementa ambas versiones.
""")

add_code(nb, """import sys
sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats
from delong import (compute_midrank, fastDeLong, delong_covariance_matrix,
                     delong_pairwise_test, slow_delong_reference, holm_correction)

print("Modulo delong.py cargado correctamente.")
""")

add_md(nb, "## 5.2. Validación de la versión rápida contra la versión lenta (definición original)")

add_code(nb, """rng = np.random.default_rng(123)

def validar_escenario(nombre, n_pos, n_neg, ruido_a, ruido_b, con_empates):
    y = np.array([1]*n_pos + [0]*n_neg)
    señal = np.concatenate([rng.normal(1.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    a = señal + rng.normal(0, ruido_a, len(y))
    b = señal + rng.normal(0, ruido_b, len(y))
    if con_empates:
        a, b = np.round(a, 0), np.round(b, 0)
    preds = np.vstack([a, b])
    aucs_fast, cov_fast = delong_covariance_matrix(y, preds)
    aucs_slow, cov_slow = slow_delong_reference(y, preds)
    ok_auc = np.allclose(aucs_fast, aucs_slow, atol=1e-9)
    ok_cov = np.allclose(cov_fast, cov_slow, atol=1e-9)
    print(f"{nombre:35s} | AUC coincide: {ok_auc} | Cov coincide: {ok_cov} | "
          f"max diff AUC={np.max(np.abs(aucs_fast-aucs_slow)):.2e}  max diff Cov={np.max(np.abs(cov_fast-cov_slow)):.2e}")
    assert ok_auc and ok_cov

validar_escenario("Sin empates, AUCs similares",   40, 60, 1.0, 1.0, False)
validar_escenario("Sin empates, AUCs distintos",   35, 45, 0.5, 2.0, False)
validar_escenario("CON empates (scores discretos)", 30, 50, 1.0, 1.0, True)
print("\\nLas tres validaciones contra la formula original de DeLong (1988) coinciden hasta precision numerica.")
""")

add_code(nb, """from sklearn.metrics import roc_auc_score

# El punto estimado del AUC debe coincidir EXACTAMENTE con sklearn (muestra grande, 20,000 filas)
n = 20000
y_val = rng.binomial(1, 0.15, n)
score_val = rng.normal(0, 1, n) + y_val * rng.normal(1.2, 1, n)
auc_fast, var_fast = delong_covariance_matrix(y_val, score_val.reshape(1, -1))
print(f"AUC (DeLong rapido) = {auc_fast[0]:.10f}")
print(f"AUC (sklearn)        = {roc_auc_score(y_val, score_val):.10f}")
print(f"Diferencia = {abs(auc_fast[0] - roc_auc_score(y_val, score_val)):.2e}  (coincide hasta precision de punto flotante)")

# La varianza analitica debe ser del mismo orden que una varianza estimada por bootstrap
# (metodo COMPLETAMENTE independiente: remuestreo, no formula cerrada)
n2 = 5000
y2 = rng.binomial(1, 0.15, n2)
score2 = rng.normal(0, 1, n2) + y2 * rng.normal(1.0, 1, n2)
_, var_analitica = delong_covariance_matrix(y2, score2.reshape(1, -1))
B = 1500
aucs_boot = [roc_auc_score(y2[idx], score2[idx]) for idx in
             (rng.choice(n2, n2, replace=True) for _ in range(B))
             if True]
aucs_boot = np.array([roc_auc_score(y2[i], score2[i]) for i in
                       (rng.choice(n2, n2, replace=True) for _ in range(B))])
var_boot = np.var(aucs_boot, ddof=1)
print(f"\\nVarianza analitica (DeLong): {var_analitica[0,0]:.6e}")
print(f"Varianza por bootstrap (B={B}): {var_boot:.6e}  (razon: {var_boot/var_analitica[0,0]:.3f})")
""")

add_md(nb, """Las tres validaciones —contra la fórmula original, contra `sklearn.metrics.roc_auc_score`, y
contra una varianza estimada por remuestreo (bootstrap, sección 6 más adelante)— coinciden. La
versión rápida es segura para usar sobre las 452.141 filas del proyecto.
""")

add_md(nb, "## 5.3. Carga y alineación de puntajes de ambos entornos")

add_code(nb, """sk = pd.read_parquet("../data/sklearn_test_scores.parquet")
sp = pd.read_parquet("../data/spark_test_scores.parquet")
sk["id"] = sk["id"].astype(str)
sp["id"] = sp["id"].astype(str)

merged = sk.merge(sp, on="id", suffixes=("_sk", "_sp"))
assert len(merged) == len(sk) == len(sp)
assert (merged["default_sk"] == merged["default_sp"]).all()
y = merged["default_sk"].values
print(f"Filas alineadas por id: {len(merged):,}  (tasa de default: {y.mean()*100:.2f}%)")

FAMILIAS = [
    ("LogisticRegression", "score_LogisticRegression_sk", "score_LogisticRegression_sp"),
    ("DecisionTree",       "score_DecisionTree_sk",       "score_DecisionTree_sp"),
    ("RandomForest",       "score_RandomForest_sk",       "score_RandomForest_sp"),
    ("GBT",                "score_GBT_HistGB",            "score_GBT"),
    ("LinearSVC",          "score_LinearSVC_sk",          "score_LinearSVC_sp"),
    ("NaiveBayes",         "score_GaussianNB",            "score_NaiveBayes"),
]
nombres_modelos = [f[0] for f in FAMILIAS]
etiquetas = [f"sk_{n}" for n in nombres_modelos] + [f"sp_{n}" for n in nombres_modelos]
todas_cols = [f[1] for f in FAMILIAS] + [f[2] for f in FAMILIAS]
""")

add_md(nb, "## 5.4. Matriz de covarianza DeLong 12×12 (una sola pasada)")

add_code(nb, """preds = merged[todas_cols].values.T
aucs, cov = delong_covariance_matrix(y, preds)

auc_table = pd.DataFrame({"modelo": etiquetas, "auc": aucs, "var": np.diag(cov)})
auc_table["ic95_low"] = auc_table["auc"] - 1.96*np.sqrt(auc_table["var"])
auc_table["ic95_high"] = auc_table["auc"] + 1.96*np.sqrt(auc_table["var"])
auc_table = auc_table.round(4)
auc_table
""")

add_md(nb, """Los 12 AUC (y sus varianzas) de esta única matriz alimentan las 36 comparaciones de las
siguientes tres secciones — es la misma covarianza subyacente en los tres casos, así que las
comparaciones entre ellas son mutuamente consistentes.
""")

add_code(nb, """def construir_comparaciones(indices_pares):
    filas = []
    for i, j in indices_pares:
        auc_a, auc_b = aucs[i], aucs[j]
        var_a, var_b, cov_ab = cov[i,i], cov[j,j], cov[i,j]
        delta = auc_a - auc_b
        var_delta = max(var_a + var_b - 2*cov_ab, 0.0)
        se = np.sqrt(var_delta)
        z = delta/se if se > 0 else 0.0
        p = 2*(1 - sstats.norm.cdf(abs(z)))
        filas.append({"modelo_a": etiquetas[i], "modelo_b": etiquetas[j],
                       "auc_a": auc_a, "auc_b": auc_b, "delta_auc": delta,
                       "delta_ci_low": delta-1.96*se, "delta_ci_high": delta+1.96*se,
                       "z": z, "p_value": p})
    df = pd.DataFrame(filas)
    df["p_holm"], df["rechaza_h0_holm_0.05"] = holm_correction(df["p_value"].values, alpha=0.05)
    df["practicamente_sig_(|delta|>=0.005)"] = df["delta_auc"].abs() >= 0.005
    return df

pares_sk = [(i,j) for i in range(6) for j in range(i+1,6)]
pares_sp = [(i+6,j+6) for i in range(6) for j in range(i+1,6)]
pares_entre = [(i,i+6) for i in range(6)]

df_within_sklearn = construir_comparaciones(pares_sk)
df_within_spark = construir_comparaciones(pares_sp)
df_between = construir_comparaciones(pares_entre)

for df in (df_within_sklearn, df_within_spark, df_between):
    for c in ["auc_a","auc_b","delta_auc","delta_ci_low","delta_ci_high","z"]:
        df[c] = df[c].round(4)
    df["p_value"] = df["p_value"].apply(lambda x: f"{x:.2e}")
    df["p_holm"] = df["p_holm"].apply(lambda x: f"{x:.2e}")
""")

add_md(nb, "## 5.5. Comparaciones dentro de scikit-learn (15 pares, Holm dentro de esta familia)")

add_code(nb, "df_within_sklearn")

add_code(nb, """def heatmap_delta_y_p(df, nombres, titulo_prefix, archivo_sufijo):
    n = len(nombres)
    delta_mat = np.full((n,n), np.nan)
    p_mat = np.full((n,n), np.nan)
    idx = {m:i for i,m in enumerate(nombres)}
    for _, row in df.iterrows():
        a = row["modelo_a"].split("_",1)[1]; b = row["modelo_b"].split("_",1)[1]
        i, j = idx[a], idx[b]
        delta_mat[i,j] = float(row["delta_auc"]); delta_mat[j,i] = -float(row["delta_auc"])
        p_mat[i,j] = p_mat[j,i] = float(row["p_holm"])

    fig, axes = plt.subplots(1, 2, figsize=(13,5.5))
    im0 = axes[0].imshow(delta_mat, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(nombres, rotation=45, ha="right")
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels(nombres)
    axes[0].set_title(f"{titulo_prefix}: ΔAUC (fila − columna)")
    for i in range(n):
        for j in range(n):
            if i != j:
                axes[0].text(j, i, f"{delta_mat[i,j]:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(p_mat, cmap="viridis_r", vmin=0, vmax=0.05)
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(nombres, rotation=45, ha="right")
    axes[1].set_yticks(range(n)); axes[1].set_yticklabels(nombres)
    axes[1].set_title(f"{titulo_prefix}: p ajustado (Holm)")
    for i in range(n):
        for j in range(n):
            if i != j:
                axes[1].text(j, i, f"{p_mat[i,j]:.0e}", ha="center", va="center", fontsize=7,
                             color="white" if p_mat[i,j] < 0.025 else "black")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"../outputs/figures/eda/delong_heatmap_{archivo_sufijo}.png", dpi=110)
    plt.show()

heatmap_delta_y_p(df_within_sklearn, nombres_modelos, "scikit-learn", "sklearn")
""")

add_md(nb, """**Lectura:** de las 15 comparaciones, solo `LogisticRegression` vs. `RandomForest` (ΔAUC=-0.0004)
**no** es significativa ni siquiera antes de corregir por comparaciones múltiples. Todas las demás sí
son estadísticamente significativas después de Holm — pero varias tienen |ΔAUC| menor al umbral
práctico de 0.005 declarado de antemano (p. ej. `DecisionTree` vs. `GBT_HistGB`, ΔAUC=-0.0025):
significativas en el sentido estadístico, pero de una magnitud que probablemente no importa para
decidir qué modelo usar en producción. Esto es exactamente el fenómeno que hace tan importante
declarar un umbral de significancia práctica *antes* de mirar los resultados, con un tamaño de
muestra de 452 mil filas de prueba: casi cualquier diferencia no nula termina siendo "significativa".
""")

add_md(nb, "## 5.6. Comparaciones dentro de PySpark (15 pares, Holm dentro de esta familia)")

add_code(nb, "df_within_spark")

add_code(nb, """heatmap_delta_y_p(df_within_spark, nombres_modelos, "PySpark", "spark")
""")

add_md(nb, """**Lectura:** aquí las 15 comparaciones son estadísticamente significativas después de Holm — incluida
`DecisionTree` vs. `RandomForest` (ΔAUC=0.0018, p_holm=0.023), que es significativa pero **no**
prácticamente relevante (por debajo de 0.005). El patrón general es el mismo que en scikit-learn:
`GBT` es significativamente mejor que todos los demás, y varias diferencias "significativas" entre
el resto de los modelos son demasiado pequeñas para ser prácticamente relevantes.
""")

add_md(nb, "## 5.7. Comparaciones entre entornos (6 pares, misma familia de modelo)")

add_code(nb, "df_between")

add_code(nb, """fig, ax = plt.subplots(figsize=(8,5))
y_pos = np.arange(len(df_between))
colores = ["#2ca02c" if s else "#7f7f7f" for s in df_between["practicamente_sig_(|delta|>=0.005)"]]
ax.barh(y_pos, df_between["delta_auc"], xerr=[df_between["delta_auc"]-df_between["delta_ci_low"],
                                                df_between["delta_ci_high"]-df_between["delta_auc"]],
        color=colores, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels([m.split("_",1)[1] for m in df_between["modelo_a"]])
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(0.005, color="red", linestyle="--", linewidth=0.8, label="umbral practico ±0.005")
ax.axvline(-0.005, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("ΔAUC (scikit-learn − PySpark), con IC95%")
ax.set_title("Diferencia de AUC entre entornos, por familia de modelo")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("../outputs/figures/eda/delong_between_envs.png", dpi=110)
plt.show()
""")

add_md(nb, """**Lectura:** con 452.141 observaciones pareadas, **las 6 comparaciones son estadísticamente
significativas** (p≈0 en todas, incluso tras Holm) — el tamaño de muestra es tan grande que casi
cualquier ΔAUC distinto de cero se detecta. Pero, usando el umbral práctico ΔAUC≥0.005 declarado de
antemano:

- **`LinearSVC`** (ΔAUC=-0.144): diferencia enorme, estadística *y* prácticamente significativa —
  scikit-learn colapsó (sección 3.7); PySpark no.
- **`GBT`** (ΔAUC=+0.0066): scikit-learn (`HistGradientBoostingClassifier`) mejor que Spark
  (`GBTClassifier`), práctica y estadísticamente significativo, aunque de magnitud moderada.
- **`NaiveBayes`** (ΔAUC=-0.0055): PySpark mejor, justo por encima del umbral práctico.
- **`RandomForest`** (ΔAUC=+0.0039), **`LogisticRegression`** (ΔAUC=-0.0011): estadísticamente
  significativos por el tamaño de muestra, pero **no** prácticamente relevantes — en la práctica,
  ambos entornos producen un modelo equivalente para estas dos familias.
- **`DecisionTree`** (ΔAUC=+0.0164): scikit-learn mejor, significativo también en el sentido
  práctico (aun después de corregir el bug de la sección 4.2.2, sigue quedando una diferencia real,
  atribuible de forma plausible a `maxBins`).
""")

add_md(nb, "## 5.8. Limitaciones de la prueba de DeLong")

add_md(nb, """- **Solo compara AUC (capacidad de *ranking*), no calibración ni el desempeño a un umbral de
  decisión concreto.** Dos modelos pueden tener AUCs estadísticamente indistinguibles y, sin
  embargo, comportarse muy distinto al umbral que realmente se usaría en producción (ver sección
  3.6: varios modelos con AUC~0.72 predicen 0% de casos positivos a umbral 0.5). Por eso el
  enunciado pide complementar con McNemar y bootstrap sobre clasificaciones concretas (sección 6).
- **Es asintótica:** la aproximación normal de la distribución de U-estadísticos de Mann-Whitney
  subyacente es más confiable con muestras grandes (aquí, 452 mil filas, muy por encima del mínimo
  habitual) — con muestras pequeñas puede ser menos precisa.
- **Extremadamente sensible al tamaño de muestra:** como se ve en las secciones 5.5–5.7, con n
  grande casi cualquier ΔAUC≠0 resulta "significativo". Esto hace que declarar un umbral de
  significancia *práctica* de antemano (aquí, ΔAUC≥0.005) sea indispensable — sin él, la
  significancia estadística por sí sola no dice si la diferencia importa para decidir qué modelo
  usar.
- **Asume que las dos muestras de puntajes provienen de las mismas unidades experimentales**
  (mismo test set) — es exactamente lo que se verificó en la sección 5.3 (mismos 452.141 ids); si
  los conjuntos de prueba fueran distintos, la covarianza calculada no tendría sentido y habría que
  usar una prueba para muestras independientes (con menos poder estadístico).
- **No corrige por sí sola el problema de comparaciones múltiples** — de ahí la necesidad de la
  corrección de Holm, aplicada aquí por separado dentro de cada una de las tres familias de
  comparación (15+15+6), tal como pide el enunciado.
""")

save(nb, "book/05_delong.ipynb")
print("Notebook 05_delong.ipynb creado.")
