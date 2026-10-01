import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import json

pd.set_option("display.max_columns", 20)

df = pd.read_parquet("data/accepted_raw.parquet")

# ---- Variable objetivo ----
df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")

target_counts = df["default"].value_counts().sort_index()
target_pct = (df["default"].value_counts(normalize=True).sort_index()*100).round(2)
print("=== Distribución de la variable objetivo (regla literal del enunciado) ===")
print(pd.DataFrame({"n": target_counts, "%": target_pct}))

fig, ax = plt.subplots(figsize=(5,4))
target_counts.plot(kind="bar", ax=ax, color=["#4C72B0","#C44E52"])
ax.set_xticklabels(["0 = No Charged Off","1 = Charged Off"], rotation=0)
ax.set_ylabel("Número de préstamos")
ax.set_title("Distribución de la variable objetivo 'default'")
for i,v in enumerate(target_counts):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/eda/target_distribution.png", dpi=110)
plt.close()

# ---- Variables numericas curadas (solo info disponible al momento de originacion) ----
numeric_vars = ["loan_amnt","int_rate","installment","annual_inc","dti",
                "fico_range_low","fico_range_high","open_acc","revol_bal",
                "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc","inq_last_6mths"]

# emp_length es texto ordinal -> convertir a numerico aproximado
def parse_emp_length(x):
    if pd.isna(x): return np.nan
    x = str(x)
    if "10+" in x: return 10.0
    if "< 1" in x: return 0.0
    digits = "".join(ch for ch in x if ch.isdigit())
    return float(digits) if digits else np.nan

df["emp_length_num"] = df["emp_length"].apply(parse_emp_length)
numeric_vars.append("emp_length_num")

rows = []
for col in numeric_vars:
    s = pd.to_numeric(df[col], errors="coerce")
    desc = s.describe(percentiles=[.25,.5,.75])
    q1, q3 = desc["25%"], desc["75%"]
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((s < lo) | (s > hi)).sum()
    pct_out = n_out / s.notna().sum() * 100
    skew = s.skew()
    pct_missing = s.isna().mean()*100
    rows.append({
        "variable": col, "media": desc["mean"], "mediana": desc["50%"], "sd": desc["std"],
        "min": desc["min"], "Q1": q1, "Q3": q3, "max": desc["max"], "IQR": iqr,
        "n_outliers_IQR": n_out, "pct_outliers": round(pct_out,2),
        "skewness": round(skew,3), "pct_missing": round(pct_missing,2)
    })

numeric_summary = pd.DataFrame(rows).set_index("variable")
numeric_summary.to_csv("outputs/tables/eda/numeric_summary.csv")
print("\n=== Resumen variables numericas ===")
print(numeric_summary.round(2).to_string())

# Shapiro-Wilk en una submuestra aleatoria (n=5000) por variable, como chequeo opcional
shapiro_rows = []
rng = np.random.default_rng(42)
for col in numeric_vars:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) > 5000:
        sample = s.sample(5000, random_state=42)
    else:
        sample = s
    if len(sample) >= 3:
        stat, p = stats.shapiro(sample)
        shapiro_rows.append({"variable": col, "shapiro_stat": round(stat,4), "p_value": p, "n_muestra": len(sample)})
shapiro_df = pd.DataFrame(shapiro_rows).set_index("variable")
shapiro_df.to_csv("outputs/tables/eda/shapiro_subsample.csv")
print("\n=== Shapiro-Wilk (submuestra n=5000, solo diagnostico) ===")
print(shapiro_df.to_string())

# Histogramas + boxplots (grid)
n = len(numeric_vars)
fig, axes = plt.subplots(n, 2, figsize=(10, 3*n))
for i, col in enumerate(numeric_vars):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    axes[i,0].hist(s, bins=60, color="#4C72B0")
    axes[i,0].set_title(f"Histograma: {col}")
    axes[i,1].boxplot(s, vert=False)
    axes[i,1].set_title(f"Boxplot: {col}")
plt.tight_layout()
plt.savefig("outputs/figures/eda/numeric_hist_box.png", dpi=100)
plt.close()

print("\nFiguras guardadas en outputs/figures/eda/")
print("Tablas guardadas en outputs/tables/eda/")

# Guardar dataset intermedio con default y emp_length_num para reutilizar
df[["id","default","emp_length_num"]].to_parquet("data/target_and_derived.parquet", index=False)
