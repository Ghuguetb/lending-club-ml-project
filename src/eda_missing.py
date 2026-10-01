import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import chi2_contingency

df = pd.read_parquet("data/accepted_raw.parquet")
df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")

profile = pd.read_csv("outputs/tables/column_profile_full.csv", index_col=0)

# Matriz de nulos (missingno) sobre una muestra representativa (2.26M filas es inviable de graficar punto por punto)
sample = df.sample(5000, random_state=42)
cols_con_nulos = profile[profile["pct_missing"] > 0].index.tolist()

fig = plt.figure(figsize=(16,8))
ax = msno.matrix(sample[cols_con_nulos[:60]], sparkline=False, fontsize=6)
plt.title("Patron de valores faltantes (muestra n=5000, primeras 60 columnas con nulos)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/missing_matrix.png", dpi=100)
plt.close()

fig = plt.figure(figsize=(10,10))
ax = msno.heatmap(sample[cols_con_nulos[:40]], fontsize=6)
plt.title("Correlacion de nulidad entre variables (muestra, 40 columnas)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/missing_heatmap.png", dpi=100)
plt.close()

# Relacion missingness vs target (chi2) para variables con 0.5% < missing < 99%
rows = []
candidatas = profile[(profile["pct_missing"]>0.5) & (profile["pct_missing"]<99)].index.tolist()
for col in candidatas:
    is_missing = df[col].isna().astype(int)
    if is_missing.nunique() < 2:
        continue
    ct = pd.crosstab(is_missing, df["default"])
    chi2, p, dof, exp = chi2_contingency(ct)
    tasa_missing = df.loc[is_missing==1, "default"].mean()*100
    tasa_no_missing = df.loc[is_missing==0, "default"].mean()*100
    rows.append({"variable": col, "pct_missing": profile.loc[col,"pct_missing"],
                 "chi2_p": p, "tasa_default_si_falta": round(tasa_missing,2),
                 "tasa_default_si_no_falta": round(tasa_no_missing,2)})
miss_target = pd.DataFrame(rows).sort_values("chi2_p")
miss_target.to_csv("outputs/tables/eda/missingness_vs_target.csv", index=False)
print("=== Relacion entre faltantes y target (top 20 mas asociadas) ===")
print(miss_target.head(20).to_string(index=False))
print(f"\nDe {len(candidatas)} variables evaluadas, {(miss_target['chi2_p']<0.05).sum()} muestran asociacion significativa (p<0.05) entre 'falta el dato' y default")
print("(con 2.26M de observaciones, incluso asociaciones muy pequenas resultan significativas)")

# Plan de tratamiento propuesto por variable (regla automatica + guardado a csv)
def plan_tratamiento(row):
    if row["pct_missing"] == 0:
        return "sin tratamiento"
    if row["pct_missing"] > 70:
        return "eliminar columna (>70% nulo)"
    if row["dtype"] in ("double[pyarrow]","int64[pyarrow]","int32[pyarrow]"):
        return "imputar con mediana (train) + indicador de faltante si asociacion con target es significativa"
    return "imputar con moda o categoria 'Desconocido' (train)"

profile["plan_tratamiento"] = profile.apply(plan_tratamiento, axis=1)
profile.to_csv("outputs/tables/eda/column_profile_con_plan.csv")
print("\nColumnas por tipo de plan de tratamiento:")
print(profile["plan_tratamiento"].value_counts())
