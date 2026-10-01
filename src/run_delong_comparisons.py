"""Seccion 9.10.4.6.1: pruebas de DeLong (rapidas, validadas) + correccion de Holm.
Compara:
  (a) 15 pares dentro de scikit-learn (6 modelos, C(6,2)=15)
  (b) 15 pares dentro de PySpark (6 modelos, C(6,2)=15)
  (c) 6 pares entre entornos (mismo modelo/familia, sklearn vs. PySpark)
Las 36 comparaciones vienen de UNA sola matriz de covarianza 12x12 (los 12 modelos evaluados
sobre las MISMAS 452,141 filas de prueba), calculada con la version rapida O(n log n) de DeLong.
"""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from delong import delong_covariance_matrix, holm_correction

t0 = time.time()
print("Cargando y alineando puntajes de ambos entornos por 'id'...", flush=True)
sk = pd.read_parquet("data/sklearn_test_scores.parquet")
sp = pd.read_parquet("data/spark_test_scores.parquet")
sk["id"] = sk["id"].astype(str)
sp["id"] = sp["id"].astype(str)

merged = sk.merge(sp, on="id", suffixes=("_sk", "_sp"))
assert len(merged) == len(sk) == len(sp), "Las filas de sklearn y spark no coinciden 1 a 1"
assert (merged["default_sk"] == merged["default_sp"]).all(), "Las etiquetas no coinciden"
y = merged["default_sk"].values
print(f"Filas alineadas: {len(merged):,}  (tasa de default: {y.mean()*100:.2f}%)", flush=True)

# Mapeo de familias de modelos entre entornos (mismos 6 conceptos, nombres distintos en Spark)
FAMILIAS = [
    ("LogisticRegression", "score_LogisticRegression_sk", "score_LogisticRegression_sp"),
    ("DecisionTree",       "score_DecisionTree_sk",       "score_DecisionTree_sp"),
    ("RandomForest",       "score_RandomForest_sk",       "score_RandomForest_sp"),
    ("GBT",                "score_GBT_HistGB",            "score_GBT"),
    ("LinearSVC",          "score_LinearSVC_sk",          "score_LinearSVC_sp"),
    ("NaiveBayes",         "score_GaussianNB",            "score_NaiveBayes"),
]

nombres_modelos = [f[0] for f in FAMILIAS]
cols_sk = [f[1] for f in FAMILIAS]
cols_sp = [f[2] for f in FAMILIAS]
etiquetas = [f"sk_{n}" for n in nombres_modelos] + [f"sp_{n}" for n in nombres_modelos]
todas_cols = cols_sk + cols_sp

print("Calculando la matriz de covarianza DeLong 12x12 (una sola pasada, O(n log n))...", flush=True)
preds = merged[todas_cols].values.T  # (12, n)
aucs, cov = delong_covariance_matrix(y, preds)
print(f"Listo en {time.time()-t0:.2f}s", flush=True)

for etq, a in zip(etiquetas, aucs):
    print(f"  AUC {etq}: {a:.4f}", flush=True)


def construir_comparaciones(indices_pares, etiquetas_idx):
    filas = []
    for i, j in indices_pares:
        auc_a, auc_b = aucs[i], aucs[j]
        var_a, var_b, cov_ab = cov[i, i], cov[j, j], cov[i, j]
        delta = auc_a - auc_b
        var_delta = max(var_a + var_b - 2 * cov_ab, 0.0)
        se = np.sqrt(var_delta)
        z = delta / se if se > 0 else 0.0
        from scipy import stats as sstats
        p = 2 * (1 - sstats.norm.cdf(abs(z)))
        filas.append({
            "modelo_a": etiquetas_idx[i], "modelo_b": etiquetas_idx[j],
            "auc_a": auc_a, "auc_a_ci_low": auc_a - 1.96*np.sqrt(var_a), "auc_a_ci_high": auc_a + 1.96*np.sqrt(var_a),
            "auc_b": auc_b, "auc_b_ci_low": auc_b - 1.96*np.sqrt(var_b), "auc_b_ci_high": auc_b + 1.96*np.sqrt(var_b),
            "delta_auc": delta, "delta_ci_low": delta - 1.96*se, "delta_ci_high": delta + 1.96*se,
            "z": z, "p_value": p,
        })
    df = pd.DataFrame(filas)
    df["p_holm"], df["rechaza_h0_holm_0.05"] = holm_correction(df["p_value"].values, alpha=0.05)
    df["practicamente_significativo_(|delta|>=0.005)"] = df["delta_auc"].abs() >= 0.005
    return df


# etiquetas[i] para i=0..5 son sk_<modelo>, para i=6..11 son sp_<modelo> (mismo orden que "preds")

# (a) Dentro de scikit-learn: indices 0..5
pares_sk = [(i, j) for i in range(6) for j in range(i+1, 6)]
df_within_sklearn = construir_comparaciones(pares_sk, etiquetas)

# (b) Dentro de PySpark: indices 6..11
pares_sp = [(i+6, j+6) for i in range(6) for j in range(i+1, 6)]
df_within_spark = construir_comparaciones(pares_sp, etiquetas)

# (c) Entre entornos: (i, i+6) para cada uno de los 6 modelos
pares_entre = [(i, i+6) for i in range(6)]
df_between = construir_comparaciones(pares_entre, etiquetas)

# Guardar todo
df_within_sklearn.to_csv("outputs/tables/delong_within_sklearn.csv", index=False)
df_within_spark.to_csv("outputs/tables/delong_within_spark.csv", index=False)
df_between.to_csv("outputs/tables/delong_between_envs.csv", index=False)

auc_table = pd.DataFrame({"modelo": etiquetas, "auc": aucs, "var": np.diag(cov)})
auc_table["ci_low"] = auc_table["auc"] - 1.96*np.sqrt(auc_table["var"])
auc_table["ci_high"] = auc_table["auc"] + 1.96*np.sqrt(auc_table["var"])
auc_table.to_csv("outputs/tables/delong_auc_individual.csv", index=False)
np.save("outputs/tables/delong_cov_matrix.npy", cov)

print("\n=== DENTRO DE SCIKIT-LEARN (15 pares, Holm) ===")
print(df_within_sklearn[["modelo_a","modelo_b","delta_auc","p_value","p_holm","rechaza_h0_holm_0.05"]].to_string(index=False))
print("\n=== DENTRO DE PYSPARK (15 pares, Holm) ===")
print(df_within_spark[["modelo_a","modelo_b","delta_auc","p_value","p_holm","rechaza_h0_holm_0.05"]].to_string(index=False))
print("\n=== ENTRE ENTORNOS (6 pares, Holm) ===")
print(df_between[["modelo_a","modelo_b","delta_auc","p_value","p_holm","rechaza_h0_holm_0.05"]].to_string(index=False))
print(f"\nTiempo total: {time.time()-t0:.2f}s")
print("Guardado en outputs/tables/delong_*.csv")
