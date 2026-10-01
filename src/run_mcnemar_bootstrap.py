"""Seccion 9.10.4.6.2: McNemar + bootstrap pareado, con umbral elegido SOLO con datos de train.
Aplicado a: 6 pares entre entornos (mismo modelo/familia) + el par top-2-AUC dentro de cada entorno
(2 comparaciones mas) = 8 comparaciones en total, con correccion de Holm dentro de esa familia."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from mcnemar_bootstrap import umbral_youden, mcnemar_test, bootstrap_pareado
from delong import holm_correction

# Nombres de columna NATIVOS de cada dataframe de origen (sk_train/sk_test tienen sus propios
# nombres "score_<Modelo>", igual que sp_train/sp_test). Estos son los nombres correctos para
# sk_train/sp_train (Paso 1, no pasan por merge). Para el test ALINEADO (test_merged, Paso 3) hay
# que traducirlos con `col_en_merge(...)` mas abajo, porque pandas les agrega sufijo _sk/_sp SOLO
# cuando el nombre colisiona entre los dos dataframes de origen (ver esa funcion).
FAMILIAS = [
    ("LogisticRegression", "score_LogisticRegression", "score_LogisticRegression"),
    ("DecisionTree",       "score_DecisionTree",       "score_DecisionTree"),
    ("RandomForest",       "score_RandomForest",       "score_RandomForest"),
    ("GBT",                "score_GBT_HistGB",         "score_GBT"),
    ("LinearSVC",          "score_LinearSVC",          "score_LinearSVC"),
    ("NaiveBayes",         "score_GaussianNB",         "score_NaiveBayes"),
]


def col_en_merge(col_sk, col_sp):
    """Nombre real de una columna dentro de test_merged = sk_test.merge(sp_test, on='id',
    suffixes=('_sk','_sp')). Si el nombre nativo colisionaba entre sk_test y sp_test (mismo
    string en ambos), pandas le agrega el sufijo; si no colisionaba (p.ej. GBT_HistGB vs GBT),
    queda igual. ES CRITICO usar esta funcion (o test_merged directamente) para leer los
    puntajes de test de AMBOS entornos: Spark escribe su parquet en su propio orden de
    particiones, que NO coincide posicionalmente con el de sk_test, asi que indexar
    sp_test[...] por separado y emparejarlo por posicion con y_test (que viene de test_merged,
    en el orden de sk_test) mezclaria las filas en silencio."""
    if col_sk == col_sp:
        return col_sk + "_sk", col_sp + "_sp"
    return col_sk, col_sp


# Nombres nativos que colisionan entre sk_test y sp_test (y por lo tanto llevan sufijo en
# test_merged) -- se usa para traducir un nombre nativo de UN solo lado (ver top2_dentro_*).
_COLISIONA = {col_sk for _, col_sk, col_sp in FAMILIAS if col_sk == col_sp}


def col_en_merge_lado(col_native, lado):
    """Traduce un nombre nativo de columna (de un solo entorno) al nombre real que tiene en
    test_merged. `lado` es 'sk' o 'sp'."""
    return col_native + f"_{lado}" if col_native in _COLISIONA else col_native

print("Cargando puntajes de train y test de ambos entornos...", flush=True)
sk_train = pd.read_parquet("data/sklearn_train_scores.parquet")
sk_test = pd.read_parquet("data/sklearn_test_scores.parquet")
sp_train = pd.read_parquet("data/spark_train_scores_raw.parquet")
sp_test = pd.read_parquet("data/spark_test_scores.parquet")
for df in (sk_train, sk_test, sp_train, sp_test):
    df["id"] = df["id"].astype(str)

test_merged = sk_test.merge(sp_test, on="id", suffixes=("_sk", "_sp"))
assert (test_merged["default_sk"] == test_merged["default_sp"]).all()
y_test = test_merged["default_sk"].values
print(f"Test alineado: {len(test_merged):,} filas", flush=True)

# --- Paso 1: umbral de Youden por modelo, usando SOLO train ---
umbrales = {}
for nombre, col_sk, col_sp in FAMILIAS:
    u_sk = umbral_youden(sk_train["default"].values, sk_train[col_sk].values)
    u_sp = umbral_youden(sp_train["default"].values, sp_train[col_sp].values)
    umbrales[f"{nombre}_sklearn"] = u_sk
    umbrales[f"{nombre}_spark"] = u_sp
    print(f"  Umbral Youden (solo train) -- {nombre}: sklearn={u_sk:.4f}  spark={u_sp:.4f}", flush=True)

with open("outputs/tables/umbrales_youden.json", "w") as f:
    json.dump(umbrales, f, indent=2)

# --- Paso 2: identificar el par top-2-AUC dentro de cada entorno ---
res_sk = pd.read_csv("outputs/tables/sklearn_results.csv").set_index("modelo")
res_sp = pd.read_csv("outputs/tables/spark_results.csv").set_index("modelo")
top2_sk = res_sk["test_auc"].sort_values(ascending=False).index[:2].tolist()
top2_sp = res_sp["test_auc"].sort_values(ascending=False).index[:2].tolist()
print(f"\nTop-2 AUC sklearn: {top2_sk}   Top-2 AUC spark: {top2_sp}", flush=True)

# Nombre nativo (no fusionado) de columna para cada modelo, por entorno -- se traduce a la
# columna real de test_merged con col_en_merge() en el momento de usarla.
col_map_sk = {"LogisticRegression": "score_LogisticRegression", "DecisionTree": "score_DecisionTree",
              "RandomForest": "score_RandomForest", "GBT_HistGB": "score_GBT_HistGB",
              "LinearSVC": "score_LinearSVC", "GaussianNB": "score_GaussianNB"}
col_map_sp = {"LogisticRegression": "score_LogisticRegression", "DecisionTree": "score_DecisionTree",
              "RandomForest": "score_RandomForest", "GBT": "score_GBT",
              "LinearSVC": "score_LinearSVC", "NaiveBayes": "score_NaiveBayes"}

# --- Paso 3: construir las 8 comparaciones ---
# IMPORTANTE: todos los puntajes (score_a, score_b) y las etiquetas (y) se leen SIEMPRE desde
# test_merged (via col_en_merge), nunca desde sk_test/sp_test por separado, para garantizar que
# las tres columnas esten alineadas fila a fila (ver nota junto a col_en_merge mas arriba).
comparaciones = []
for nombre, col_sk, col_sp in FAMILIAS:
    col_sk_m, col_sp_m = col_en_merge(col_sk, col_sp)
    comparaciones.append({
        "tipo": "entre_entornos", "nombre_par": nombre,
        "score_a": test_merged[col_sk_m].values, "score_b": test_merged[col_sp_m].values,
        "y": y_test, "umbral_a": umbrales[f"{nombre}_sklearn"], "umbral_b": umbrales[f"{nombre}_spark"],
        "signo_a": -1 if nombre == "LinearSVC" else 1,  # (informativo; el umbral ya viene de Youden)
        "etiqueta_a": f"{nombre}_sklearn", "etiqueta_b": f"{nombre}_spark",
    })

# nombre de resultados (p.ej. "GBT_HistGB") -> nombre de familia usado en `umbrales` (p.ej. "GBT")
nombre_a_familia = {"LogisticRegression": "LogisticRegression", "DecisionTree": "DecisionTree",
                     "RandomForest": "RandomForest", "GBT_HistGB": "GBT", "GBT": "GBT",
                     "LinearSVC": "LinearSVC", "GaussianNB": "NaiveBayes", "NaiveBayes": "NaiveBayes"}

col_a = col_en_merge_lado(col_map_sk[top2_sk[0]], "sk")
col_b = col_en_merge_lado(col_map_sk[top2_sk[1]], "sk")
fam_a, fam_b = nombre_a_familia[top2_sk[0]], nombre_a_familia[top2_sk[1]]
comparaciones.append({
    "tipo": "top2_dentro_sklearn", "nombre_par": f"{top2_sk[0]}_vs_{top2_sk[1]}",
    "score_a": test_merged[col_a].values, "score_b": test_merged[col_b].values, "y": y_test,
    "umbral_a": umbrales[f"{fam_a}_sklearn"], "umbral_b": umbrales[f"{fam_b}_sklearn"],
    "etiqueta_a": f"{top2_sk[0]}_sklearn", "etiqueta_b": f"{top2_sk[1]}_sklearn",
})
col_a = col_en_merge_lado(col_map_sp[top2_sp[0]], "sp")
col_b = col_en_merge_lado(col_map_sp[top2_sp[1]], "sp")
fam_a, fam_b = nombre_a_familia[top2_sp[0]], nombre_a_familia[top2_sp[1]]
comparaciones.append({
    "tipo": "top2_dentro_spark", "nombre_par": f"{top2_sp[0]}_vs_{top2_sp[1]}",
    "score_a": test_merged[col_a].values, "score_b": test_merged[col_b].values, "y": y_test,
    "umbral_a": umbrales[f"{fam_a}_spark"], "umbral_b": umbrales[f"{fam_b}_spark"],
    "etiqueta_a": f"{top2_sp[0]}_spark", "etiqueta_b": f"{top2_sp[1]}_spark",
})

print(f"\nTotal de comparaciones: {len(comparaciones)} (6 entre entornos + 2 top-2 dentro de cada entorno)", flush=True)

# --- Submuestra estratificada FIJA para el bootstrap (no para McNemar) ---
# Con B=2000 remuestreos sobre las 452,141 filas completas, cada una de las 8 comparaciones
# tardaria ~24 minutos (perfilado empiricamente) -- ~3.2 horas en total, solo para un chequeo
# COMPLEMENTARIO a DeLong (que si corrio sobre las 452,141 filas completas, seccion 5). Se usa una
# submuestra aleatoria estratificada de 50,000 filas (semilla fija, ~11% del test set, tasa de
# default preservada), que reduce el costo a ~22 minutos totales sin comprometer la validez de un
# intervalo de confianza por bootstrap (sigue siendo un tamano de muestra grande).
rng_sub = np.random.default_rng(42)
n_sub = 50000
idx_pos = np.where(y_test == 1)[0]
idx_neg = np.where(y_test == 0)[0]
frac = n_sub / len(y_test)
sub_idx = np.concatenate([
    rng_sub.choice(idx_pos, size=int(round(len(idx_pos)*frac)), replace=False),
    rng_sub.choice(idx_neg, size=int(round(len(idx_neg)*frac)), replace=False),
])
print(f"\nSubmuestra para bootstrap: {len(sub_idx):,} filas (tasa default={y_test[sub_idx].mean()*100:.2f}%, "
      f"vs. {y_test.mean()*100:.2f}% en el test completo)", flush=True)

filas_mcnemar, filas_boot = [], []
for c in comparaciones:
    pred_a = (c["score_a"] >= c["umbral_a"]).astype(int)
    pred_b = (c["score_b"] >= c["umbral_b"]).astype(int)
    mc = mcnemar_test(c["y"], pred_a, pred_b)  # McNemar: SIEMPRE sobre las 452,141 filas completas
    filas_mcnemar.append({"tipo": c["tipo"], "par": c["nombre_par"],
                           "modelo_a": c["etiqueta_a"], "modelo_b": c["etiqueta_b"],
                           "umbral_a": c["umbral_a"], "umbral_b": c["umbral_b"],
                           **mc})
    bt = bootstrap_pareado(c["y"][sub_idx], c["score_a"][sub_idx], c["score_b"][sub_idx],
                            pred_a[sub_idx], pred_b[sub_idx], B=2000, seed=42)
    filas_boot.append({"tipo": c["tipo"], "par": c["nombre_par"],
                        "modelo_a": c["etiqueta_a"], "modelo_b": c["etiqueta_b"],
                        "delta_auc": bt["delta_auc"]["observado"], "auc_ci_low": bt["delta_auc"]["ci_low"],
                        "auc_ci_high": bt["delta_auc"]["ci_high"], "auc_excluye_cero": bt["delta_auc"]["excluye_cero"],
                        "delta_auc_pr": bt["delta_auc_pr"]["observado"], "auc_pr_ci_low": bt["delta_auc_pr"]["ci_low"],
                        "auc_pr_ci_high": bt["delta_auc_pr"]["ci_high"], "auc_pr_excluye_cero": bt["delta_auc_pr"]["excluye_cero"],
                        "delta_f1": bt["delta_f1"]["observado"], "f1_ci_low": bt["delta_f1"]["ci_low"],
                        "f1_ci_high": bt["delta_f1"]["ci_high"], "f1_excluye_cero": bt["delta_f1"]["excluye_cero"],
                        "B_efectivo": bt["B_efectivo"]})
    print(f"  {c['nombre_par']} ({c['tipo']}): McNemar p={mc['p_value']:.3e} (b+c={mc['b_mas_c']}, exacta={mc['exacta']}) "
          f"| bootstrap ΔAUC={bt['delta_auc']['observado']:.4f} [{bt['delta_auc']['ci_low']:.4f},{bt['delta_auc']['ci_high']:.4f}]", flush=True)

df_mcnemar = pd.DataFrame(filas_mcnemar)
df_mcnemar["p_holm"], df_mcnemar["rechaza_h0_holm_0.05"] = holm_correction(df_mcnemar["p_value"].values, alpha=0.05)
df_boot = pd.DataFrame(filas_boot)

df_mcnemar.to_csv("outputs/tables/mcnemar_resultados.csv", index=False)
df_boot.to_csv("outputs/tables/bootstrap_resultados.csv", index=False)

print("\n=== McNemar (con Holm) ===")
print(df_mcnemar[["par","tipo","b_mas_c","exacta","statistic","p_value","p_holm","rechaza_h0_holm_0.05"]].to_string(index=False))
print("\n=== Bootstrap pareado (B=2000) ===")
print(df_boot[["par","tipo","delta_auc","auc_ci_low","auc_ci_high","delta_f1","f1_ci_low","f1_ci_high"]].to_string(index=False))
print("\nGuardado: outputs/tables/mcnemar_resultados.csv, outputs/tables/bootstrap_resultados.csv, outputs/tables/umbrales_youden.json")
