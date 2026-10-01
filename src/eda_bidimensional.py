import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

df = pd.read_parquet("data/accepted_raw.parquet")
df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")

def parse_emp_length(x):
    if pd.isna(x): return np.nan
    x = str(x)
    if "10+" in x: return 10.0
    if "< 1" in x: return 0.0
    digits = "".join(ch for ch in x if ch.isdigit())
    return float(digits) if digits else np.nan
df["emp_length_num"] = df["emp_length"].apply(parse_emp_length)

numeric_vars = ["loan_amnt","int_rate","installment","annual_inc","dti",
                "fico_range_low","fico_range_high","open_acc","revol_bal",
                "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc",
                "inq_last_6mths","emp_length_num"]
cat_vars = ["term","grade","sub_grade","home_ownership","verification_status",
            "purpose","addr_state","initial_list_status","application_type"]

# ===== 1. Numericas vs target =====
rows = []
for col in numeric_vars:
    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.notna()
    g0 = s[valid & (df["default"]==0)]
    g1 = s[valid & (df["default"]==1)]
    # Mann-Whitney (no asumimos normalidad, confirmada no-normal en analisis previo)
    stat_mw, p_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    # correlacion punto-biserial
    r_pb, p_pb = stats.pointbiserialr(df.loc[valid,"default"], s[valid])
    rows.append({
        "variable": col, "media_0": g0.mean(), "media_1": g1.mean(),
        "diff_medias": g1.mean()-g0.mean(),
        "mannwhitney_p": p_mw, "point_biserial_r": round(r_pb,4)
    })
num_vs_target = pd.DataFrame(rows).set_index("variable")
num_vs_target.to_csv("outputs/tables/eda/numeric_vs_target.csv")
print("=== Numericas vs default (Mann-Whitney + correlacion punto-biserial) ===")
print(num_vs_target.round(4).to_string())

# Boxplots comparativos por clase (grid)
n = len(numeric_vars)
fig, axes = plt.subplots((n+2)//3, 3, figsize=(14, 3.2*((n+2)//3)))
axes = axes.flatten()
for i, col in enumerate(numeric_vars):
    s = pd.to_numeric(df[col], errors="coerce")
    data0 = s[(df["default"]==0)].dropna()
    data1 = s[(df["default"]==1)].dropna()
    # recortar percentil 99 para visualizacion (outliers extremos distorsionan escala)
    cap = s.quantile(0.99)
    axes[i].boxplot([data0.clip(upper=cap), data1.clip(upper=cap)], tick_labels=["0","1"])
    axes[i].set_title(col, fontsize=10)
for j in range(i+1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("outputs/figures/eda/boxplots_by_class.png", dpi=100)
plt.close()

# ===== 2. Categoricas vs target: crosstab + chi2 =====
chi2_rows = []
fig, axes = plt.subplots(len(cat_vars), 1, figsize=(9, 3.2*len(cat_vars)))
for i, col in enumerate(cat_vars):
    ct = pd.crosstab(df[col], df["default"])
    chi2, p, dof, exp = chi2_contingency(ct)
    default_rate = (df.groupby(col, observed=True)["default"].mean()*100).sort_values(ascending=False)
    chi2_rows.append({"variable": col, "chi2": chi2, "p_value": p, "dof": dof,
                       "categoria_mayor_default_%": default_rate.index[0],
                       "tasa_mayor_default_%": round(default_rate.iloc[0],2),
                       "categoria_menor_default_%": default_rate.index[-1],
                       "tasa_menor_default_%": round(default_rate.iloc[-1],2)})
    top = default_rate.head(15)
    top.plot(kind="barh", ax=axes[i], color="#C44E52")
    axes[i].invert_yaxis()
    axes[i].set_title(f"Tasa de default por {col} (chi2 p={p:.2e})")
    axes[i].set_xlabel("% default")
plt.tight_layout()
plt.savefig("outputs/figures/eda/default_rate_by_category.png", dpi=100)
plt.close()

cat_vs_target = pd.DataFrame(chi2_rows).set_index("variable")
cat_vs_target.to_csv("outputs/tables/eda/categorical_vs_target.csv")
print("\n=== Categoricas vs default (chi-cuadrado) ===")
print(cat_vs_target.to_string())

# ===== 3. Multicolinealidad: Pearson entre numericas =====
corr = df[numeric_vars].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
corr.to_csv("outputs/tables/eda/correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(11,9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size":7})
ax.set_title("Matriz de correlacion de Pearson (variables numericas)")
plt.tight_layout()
plt.savefig("outputs/figures/eda/correlation_heatmap.png", dpi=110)
plt.close()

high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        r = corr.iloc[i,j]
        if abs(r) > 0.7:
            high_corr.append((corr.columns[i], corr.columns[j], round(r,3)))
print("\n=== Pares con |r| > 0.7 (posible multicolinealidad) ===")
for a,b,r in high_corr:
    print(f"{a} -- {b}: r={r}")

# Cramer's V entre categoricas
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    phi2 = chi2/n
    r, k = ct.shape
    return np.sqrt(phi2 / min(k-1, r-1))

cramers_rows = []
for i in range(len(cat_vars)):
    for j in range(i+1, len(cat_vars)):
        v = cramers_v(df[cat_vars[i]], df[cat_vars[j]])
        cramers_rows.append({"var1": cat_vars[i], "var2": cat_vars[j], "cramers_v": round(v,3)})
cramers_df = pd.DataFrame(cramers_rows).sort_values("cramers_v", ascending=False)
cramers_df.to_csv("outputs/tables/eda/cramers_v.csv", index=False)
print("\n=== Cramer's V mas altos entre categoricas ===")
print(cramers_df.head(10).to_string(index=False))
