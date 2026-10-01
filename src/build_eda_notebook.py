"""Construye book/01_eda.ipynb con celdas de codigo reales (no resultados pegados)."""
import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 1. Exploración de Datos (EDA)

**Proyecto Integrador — Lending Club Loan Data (2007–2020)**

Este notebook cubre la sección 9.10.4.1 del proyecto: carga inicial, análisis unidimensional,
análisis bidimensional, valores faltantes y resumen ejecutivo. El dataset completo (sin muestreo)
tiene **2.260.701 filas y 151 columnas**.
""")

# ---------- 1.1 Carga inicial ----------
add_md(nb, "## 1.1. Carga inicial y visión general")

add_code(nb, """import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import missingno as msno
import time

pd.set_option("display.max_columns", 25)
pd.set_option("display.width", 140)
""")

add_code(nb, """t0 = time.time()
df = pd.read_csv("../data/accepted_2007_to_2018Q4.csv", engine="pyarrow", dtype_backend="pyarrow")
t1 = time.time()
print(f"Tiempo de carga con pandas (motor pyarrow): {t1-t0:.2f} s")
print(f"Dimensión: {df.shape[0]:,} filas x {df.shape[1]} columnas")
""")

add_md(nb, """Como el enunciado sugiere comparar tiempos de carga, se probó también con PySpark en un
proceso aislado (una sesión de Spark y un DataFrame de pandas de este tamaño compitiendo por RAM
en una máquina de 7.8 GB no es seguro). Los tiempos obtenidos fueron:

| Motor | Tiempo | Detalle |
|---|---|---|
| pandas (engine="pyarrow") | ~15.3 s | lectura completa, tipado automático |
| Spark (arranque de sesión) | ~30.5 s | overhead fijo de la JVM, independiente del tamaño de archivo |
| Spark (lectura "lazy" del CSV) | ~4.7 s | solo registra el plan, no lee todavía |
| Spark (`count()`, ejecución real) | ~3.2 s | aquí sí se materializa la lectura |

Ambos confirman **2.260.701 filas** y **151 columnas**. A este tamaño de archivo, pandas es más
rápido en total porque Spark paga un costo fijo de arranque de JVM (~30 s) que no depende del
tamaño de los datos; ese costo se amortiza solo con datasets mucho más grandes o con múltiples
consultas sobre la misma sesión — el mismo punto que se retoma en la reflexión final (9.10.5).
""")

add_code(nb, """print(df.head(3))
""")

add_code(nb, """print(df.tail(3))
""")

add_code(nb, """buf_dtypes = df.dtypes.value_counts()
print("Resumen de tipos de datos (info):")
print(buf_dtypes)
print(f"\\nMemoria aproximada en uso: {df.memory_usage(deep=True).sum()/1e6:.0f} MB (representación pyarrow, compacta)")
""")

add_code(nb, """desc_full = df.describe(include="all").T
desc_full.to_csv("../outputs/tables/eda/describe_full.csv")
desc_full.head(20)
""")

add_md(nb, """`describe()` sobre las 151 columnas se guardó completo en
`outputs/tables/eda/describe_full.csv` (tabla muy ancha para mostrarla entera aquí); arriba se
muestran las primeras 20 columnas a modo de vista previa.
""")

# ---------- 1.2 Unidimensional ----------
add_md(nb, """## 1.2. Análisis unidimensional

### 1.2.1 Variable objetivo (`default`)

Se construye `default` exactamente como indica el enunciado:

```python
df["default"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)
```

**Hallazgo crítico antes de aplicar la regla:** `loan_status` tiene 9 categorías, no 2. Bajo la
regla literal, todo lo que no sea exactamente el texto `"Charged Off"` queda en 0 — incluyendo
**878.317 préstamos "Current"** (39% del dataset, aún vigentes y sin resolver), los "Late",
"In Grace Period", "Default", y los 33 registros con `loan_status` nulo. También incluye
**761 préstamos "Does not meet the credit policy. Status:Charged Off"**, que en la práctica sí
fueron impagos pero no calzan con el string exacto, y por lo tanto también quedan etiquetados
como 0. Se sigue la regla del enunciado tal cual (es la instrucción del proyecto), pero esto se
documenta como una limitación importante del etiquetado que se retoma en el resumen ejecutivo y
en la reflexión crítica final.
""")

add_code(nb, """print("Categorías de loan_status:")
print(df["loan_status"].value_counts(dropna=False))

df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")

target_counts = df["default"].value_counts().sort_index()
target_pct = (df["default"].value_counts(normalize=True).sort_index()*100).round(2)
print("\\nDistribución de 'default' (regla literal del enunciado):")
print(pd.DataFrame({"n": target_counts, "%": target_pct}))
""")

add_code(nb, """fig, ax = plt.subplots(figsize=(5,4))
target_counts.plot(kind="bar", ax=ax, color=["#4C72B0","#C44E52"])
ax.set_xticklabels(["0 = No 'Charged Off'","1 = Charged Off"], rotation=0)
ax.set_ylabel("Número de préstamos")
ax.set_title("Distribución de la variable objetivo 'default'")
for i,v in enumerate(target_counts):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/target_distribution.png", dpi=110)
plt.show()
""")

add_md(nb, """**Desbalance de clases:** 88.12% (0) vs 11.88% (1), razón ≈ 7.4:1. Es un desbalance
moderado-alto, típico en riesgo crediticio. Implicaciones para el modelado:

- Usar **estratificación** en la partición train/test (obligatorio, ya lo exige el enunciado).
- Usar **AUC ROC** como criterio principal (no accuracy, que con este desbalance sería engañosa:
  un modelo que siempre prediga 0 ya acertaría 88.12%).
- Reportar también **F1 y AUC-PR**, más informativas que accuracy con clases desbalanceadas.
- Considerar el umbral de decisión con cuidado (ver sección de McNemar más adelante).
""")

add_md(nb, "### 1.2.2 Variables numéricas")

add_code(nb, """def parse_emp_length(x):
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
""")

add_md(nb, """Se seleccionó un conjunto curado de variables numéricas disponibles **al momento de
originación del préstamo** (evitando fuga de información / *data leakage*): se excluyen a
propósito columnas como `total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`, etc., porque
solo existen o toman su valor final *después* de que el préstamo ya se resolvió — usarlas sería
hacer trampa (el modelo "vería el futuro"). Este es el mismo principio de fuga de datos que
aplica en cualquier problema de clasificación: solo se puede usar información que existía en el
momento en que hay que tomar la decisión.
""")

add_code(nb, """rows = []
for col in numeric_vars:
    s = pd.to_numeric(df[col], errors="coerce")
    d = s.describe(percentiles=[.25,.5,.75])
    q1, q3 = d["25%"], d["75%"]
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((s < lo) | (s > hi)).sum()
    rows.append({
        "variable": col, "media": d["mean"], "mediana": d["50%"], "sd": d["std"],
        "min": d["min"], "Q1": q1, "Q3": q3, "max": d["max"], "IQR": iqr,
        "n_outliers_IQR": n_out, "pct_outliers": round(n_out/s.notna().sum()*100,2),
        "skewness": round(s.skew(),3), "pct_missing": round(s.isna().mean()*100,2)
    })
numeric_summary = pd.DataFrame(rows).set_index("variable")
numeric_summary.to_csv("../outputs/tables/eda/numeric_summary.csv")
numeric_summary.round(2)
""")

add_md(nb, """**Lectura de la tabla:**

- `annual_inc`: asimetría de **494** y máximo de **$110.000.000** — un outlier extremo (o varios)
  que domina la media; la mediana ($65.000) es mucho más representativa. Candidata clara a
  transformación logarítmica.
- `dti`: mínimo **-1** y máximo **999** — no son valores de DTI reales (un DTI negativo no existe;
  999 es casi con certeza un código centinela para "no aplica" o dato inválido). Hay que tratarlos
  como faltantes/atípicos antes de modelar, no como observaciones válidas.
- `pub_rec` y `delinq_2yrs`: 15.8% y 18.6% "outliers" según la regla de IQR, pero son variables de
  conteo con una masa enorme en cero (la mayoría de la gente tiene 0 registros públicos negativos).
  El método de IQR no es adecuado para este tipo de distribución tan concentrada en un solo valor;
  no son atípicos en el sentido de "error de captura", sino la forma esperada de una variable de
  conteo rara.
- El resto de variables (`loan_amnt`, `int_rate`, `installment`, `fico_range_*`) tiene asimetría
  moderada (0.7–1.3), consistente con variables financieras típicas (acotadas por abajo, cola
  larga a la derecha).
""")

add_code(nb, """shapiro_rows = []
rng = np.random.default_rng(42)
for col in numeric_vars:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    sample = s.sample(5000, random_state=42) if len(s) > 5000 else s
    stat, p = stats.shapiro(sample)
    shapiro_rows.append({"variable": col, "shapiro_stat": round(stat,4), "p_value": p})
shapiro_df = pd.DataFrame(shapiro_rows).set_index("variable")
shapiro_df.to_csv("../outputs/tables/eda/shapiro_subsample.csv")
shapiro_df
""")

add_md(nb, """El test de Shapiro-Wilk no es válido/útil sobre las 2.26M de observaciones completas
(scipy lo restringe y, aunque no lo restringiera, con una muestra tan grande casi cualquier
desviación mínima de la normalidad se vuelve "significativa"). Por eso se aplica —como sugiere el
enunciado, de forma opcional— sobre una submuestra aleatoria de 5.000 registros. Todas las
variables rechazan normalidad (p < 0.001), consistente con la asimetría ya observada. Esto
confirma que, más adelante, la prueba de Mann-Whitney (no paramétrica) es la elección correcta
para comparar grupos, en vez de la prueba t de Student.
""")

add_code(nb, """n = len(numeric_vars)
fig, axes = plt.subplots(n, 2, figsize=(10, 3*n))
for i, col in enumerate(numeric_vars):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    axes[i,0].hist(s, bins=60, color="#4C72B0")
    axes[i,0].set_title(f"Histograma: {col}")
    axes[i,1].boxplot(s, orientation="horizontal")
    axes[i,1].set_title(f"Boxplot: {col}")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/numeric_hist_box.png", dpi=100)
plt.show()
""")

add_md(nb, "### 1.2.3 Variables categóricas")

add_code(nb, """cat_vars = ["term","grade","sub_grade","home_ownership","verification_status",
            "purpose","addr_state","initial_list_status","application_type","emp_length"]

summary_rows = []
for col in cat_vars:
    vc = df[col].value_counts(dropna=False)
    vc_pct = (vc/len(df)*100).round(2)
    summary_rows.append({
        "variable": col, "n_categorias": df[col].nunique(dropna=True),
        "n_categorias_raras(<1%)": (vc_pct<1).sum(),
        "pct_missing": round(df[col].isna().mean()*100,2),
        "categoria_mas_frecuente": vc.index[0], "pct_mas_frecuente": vc_pct.iloc[0]
    })
cat_summary = pd.DataFrame(summary_rows).set_index("variable")
cat_summary.to_csv("../outputs/tables/eda/categorical_summary.csv")
cat_summary
""")

add_code(nb, """fig, axes = plt.subplots(len(cat_vars), 1, figsize=(9, 3.2*len(cat_vars)))
for i, col in enumerate(cat_vars):
    top = (df[col].value_counts(normalize=True)*100).head(15)
    top.plot(kind="barh", ax=axes[i], color="#55A868")
    axes[i].invert_yaxis()
    axes[i].set_title(f"{col}")
    axes[i].set_xlabel("% del total")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/categorical_bars.png", dpi=100)
plt.show()
""")

add_md(nb, """`addr_state` tiene 51 categorías, de las cuales **23 (45%) representan menos del 1%** del
dataset cada una — candidatas naturales a agruparse en una categoría "Otros" si se usa
one-hot encoding, para no inflar la dimensionalidad. `sub_grade` (35 categorías) y `purpose`
(14, de las cuales 7 son raras) tienen el mismo patrón en menor escala.
""")

# ---------- 1.3 Bidimensional ----------
add_md(nb, """## 1.3. Análisis bidimensional

### 1.3.1 Numéricas vs. `default`
""")

add_code(nb, """rows = []
for col in numeric_vars:
    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.notna()
    g0 = s[valid & (df["default"]==0)]
    g1 = s[valid & (df["default"]==1)]
    stat_mw, p_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    r_pb, p_pb = stats.pointbiserialr(df.loc[valid,"default"], s[valid])
    rows.append({"variable": col, "media_0": g0.mean(), "media_1": g1.mean(),
                 "diff_medias": g1.mean()-g0.mean(), "mannwhitney_p": p_mw,
                 "point_biserial_r": round(r_pb,4)})
num_vs_target = pd.DataFrame(rows).set_index("variable")
num_vs_target.to_csv("../outputs/tables/eda/numeric_vs_target.csv")
num_vs_target.round(4)
""")

add_md(nb, """Con 2.26M de observaciones, **todas** las diferencias resultan significativas al test
de Mann-Whitney (p ≈ 0) — a este tamaño de muestra, incluso diferencias mínimas y sin relevancia
práctica son "estadísticamente significativas". Lo informativo aquí es la **magnitud** de la
correlación punto-biserial, no el p-valor: `int_rate` (r=0.199) es, por márgen, el predictor
individual más fuerte, seguido de `fico_range_low/high` (r≈-0.119) e `inq_last_6mths` (r=0.083).
El resto tiene correlaciones muy débiles (|r|<0.05) — esperable en riesgo crediticio, donde
ninguna variable aislada predice bien el default; se necesita combinarlas en un modelo.
""")

add_code(nb, """n = len(numeric_vars)
fig, axes = plt.subplots((n+2)//3, 3, figsize=(14, 3.2*((n+2)//3)))
axes = axes.flatten()
for i, col in enumerate(numeric_vars):
    s = pd.to_numeric(df[col], errors="coerce")
    cap = s.quantile(0.99)
    data0 = s[df["default"]==0].dropna().clip(upper=cap)
    data1 = s[df["default"]==1].dropna().clip(upper=cap)
    axes[i].boxplot([data0, data1], tick_labels=["0","1"])
    axes[i].set_title(col, fontsize=10)
for j in range(i+1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/boxplots_by_class.png", dpi=100)
plt.show()
""")

add_md(nb, "*(Nota: los boxplots recortan en el percentil 99 solo para fines de visualización; los estadísticos de la tabla anterior usan los datos completos, sin recortar.)*")

add_code(nb, """fig, axes = plt.subplots(4, 4, figsize=(16,12))
axes = axes.flatten()
for i, col in enumerate(numeric_vars):
    s = pd.to_numeric(df[col], errors="coerce")
    cap = s.quantile(0.99)
    for cls, color in [(0,"#4C72B0"), (1,"#C44E52")]:
        data = s[df["default"]==cls].dropna().clip(upper=cap)
        data.plot(kind="density", ax=axes[i], color=color, label=f"default={cls}")
    axes[i].set_title(col, fontsize=10)
    axes[i].legend(fontsize=7)
plt.tight_layout()
plt.savefig("../outputs/figures/eda/kde_by_class.png", dpi=100)
plt.show()
""")

add_md(nb, "### 1.3.2 Categóricas vs. `default`")

add_code(nb, """chi2_rows = []
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
    default_rate.head(15).plot(kind="barh", ax=axes[i], color="#C44E52")
    axes[i].invert_yaxis()
    axes[i].set_title(f"Tasa de default por {col} (chi2 p={p:.2e})")
    axes[i].set_xlabel("% default")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/default_rate_by_category.png", dpi=100)
plt.show()

cat_vs_target = pd.DataFrame(chi2_rows).set_index("variable")
cat_vs_target.to_csv("../outputs/tables/eda/categorical_vs_target.csv")
cat_vs_target
""")

add_md(nb, """`grade`/`sub_grade` muestran el gradiente más claro y monotónico: la tasa de default
sube de **3.3% en grado A a 37.5% en grado G** — coherente con que el *grade* de Lending Club ya
es, en sí mismo, un score de riesgo construido por la plataforma. `term` de 60 meses casi duplica
la tasa de default del de 36 meses (16.2% vs 10.1%). Todas las variables categóricas resultan
significativas al chi-cuadrado (de nuevo, esperable con esta n), pero las diferencias en tasa de
default sí son grandes y prácticamente relevantes para `grade`, `sub_grade` y `term`.
""")

add_md(nb, "### 1.3.3 Multicolinealidad")

add_code(nb, """corr = df[numeric_vars].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
corr.to_csv("../outputs/tables/eda/correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(11,9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size":7})
ax.set_title("Matriz de correlación de Pearson (variables numéricas)")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/correlation_heatmap.png", dpi=110)
plt.show()

high_corr = [(corr.columns[i], corr.columns[j], round(corr.iloc[i,j],3))
             for i in range(len(corr.columns)) for j in range(i+1, len(corr.columns))
             if abs(corr.iloc[i,j]) > 0.7]
print("Pares con |r| > 0.7:")
for a,b,r in high_corr:
    print(f"  {a} -- {b}: r={r}")
print(f"\\nint_rate -- fico_range_high: r={corr.loc['int_rate','fico_range_high']:.3f} (moderada, no supera 0.7)")
""")

add_md(nb, """Tres redundancias claras:

- **`fico_range_low` vs `fico_range_high`: r=1.00.** Son prácticamente el mismo dato (el rango
  FICO siempre viene en un intervalo de 4 puntos); usar solo una de las dos en el modelado.
- **`loan_amnt` vs `installment`: r=0.946.** La cuota mensual (`installment`) es una función casi
  determinística del monto, la tasa y el plazo — es información redundante con `loan_amnt` + `term`
  + `int_rate` juntas.
- **`open_acc` vs `total_acc`: r=0.718**, redundancia moderada (cuentas abiertas vs. cuentas totales
  históricas).

También se revisó, como sugiere el enunciado, `int_rate` vs `fico_range_high`: la correlación es
moderada (negativa, en torno a -0.4) pero no supera el umbral de 0.7 — tiene sentido, porque el
puntaje FICO es uno de varios factores (junto con `grade`, plazo, monto) que determinan la tasa,
no el único.
""")

add_code(nb, """def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt((chi2/n) / min(k-1, r-1))

cramers_rows = []
for i in range(len(cat_vars)):
    for j in range(i+1, len(cat_vars)):
        v = cramers_v(df[cat_vars[i]], df[cat_vars[j]])
        cramers_rows.append({"var1": cat_vars[i], "var2": cat_vars[j], "cramers_v": round(v,3)})
cramers_df = pd.DataFrame(cramers_rows).sort_values("cramers_v", ascending=False)
cramers_df.to_csv("../outputs/tables/eda/cramers_v.csv", index=False)
cramers_df.head(10)
""")

add_md(nb, """`grade` y `sub_grade` tienen **V de Cramér = 1.0**: `sub_grade` determina `grade` por
completo (A1..A5 siempre son grado A, etc.), son la misma información a distinta granularidad.
Para el modelado se usará solo `sub_grade` (más granular) **o** `grade`, no ambas.
""")

# ---------- 1.4 Missing values ----------
add_md(nb, "## 1.4. Valores faltantes (análisis detallado)")

add_code(nb, """profile = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean()*100).round(2),
    "n_unique": df.nunique(dropna=True),
}).sort_values("pct_missing", ascending=False)
profile.to_csv("../outputs/tables/column_profile_full.csv")

print(f"Columnas sin nulos: {(profile['pct_missing']==0).sum()}")
print(f"Columnas con >30% nulos: {(profile['pct_missing']>30).sum()}")
print(f"Columnas con >70% nulos: {(profile['pct_missing']>70).sum()}")
profile.head(15)
""")

add_code(nb, """sample = df.sample(5000, random_state=42)
cols_con_nulos = profile[profile["pct_missing"]>0].index.tolist()

fig = plt.figure(figsize=(16,8))
msno.matrix(sample[cols_con_nulos[:60]], sparkline=False, fontsize=6)
plt.title("Patrón de valores faltantes (muestra n=5000, primeras 60 columnas con nulos)")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/missing_matrix.png", dpi=100)
plt.show()
""")

add_code(nb, """fig = plt.figure(figsize=(10,10))
msno.heatmap(sample[cols_con_nulos[:40]], fontsize=6)
plt.title("Correlación de nulidad entre variables (muestra, 40 columnas)")
plt.tight_layout()
plt.savefig("../outputs/figures/eda/missing_heatmap.png", dpi=100)
plt.show()
""")

add_code(nb, """rows = []
candidatas = profile[(profile["pct_missing"]>0.5) & (profile["pct_missing"]<99)].index.tolist()
for col in candidatas:
    is_missing = df[col].isna().astype(int)
    if is_missing.nunique() < 2:
        continue
    ct = pd.crosstab(is_missing, df["default"])
    chi2, p, dof, exp = chi2_contingency(ct)
    rows.append({"variable": col, "pct_missing": profile.loc[col,"pct_missing"], "chi2_p": p,
                 "tasa_default_si_falta": round(df.loc[is_missing==1,"default"].mean()*100,2),
                 "tasa_default_si_no_falta": round(df.loc[is_missing==0,"default"].mean()*100,2)})
miss_target = pd.DataFrame(rows).sort_values("chi2_p")
miss_target.to_csv("../outputs/tables/eda/missingness_vs_target.csv", index=False)
print(f"{(miss_target['chi2_p']<0.05).sum()} de {len(candidatas)} variables con faltantes muestran asociación significativa entre 'falta el dato' y default")
miss_target.head(10)
""")

add_md(nb, """La ausencia de datos **no es aleatoria (no es MCAR)**: es la señal más clara de todo
el EDA. Dos patrones distintos:

- **`settlement_status` y campos relacionados** (98.5% nulos): cuando el dato *existe* (no es
  nulo), la tasa de default es **97.15%**, frente a 10.57% cuando falta. Tiene una explicación
  simple y es también una alerta de fuga de datos: un registro de "acuerdo de pago" (*settlement*)
  solo se crea *después* de que el préstamo entró en incumplimiento severo — es información
  posterior al desenlace, no debe usarse como predictor.
- **Campos `sec_app_*` / `*_joint`** (~95% nulos): faltan porque el préstamo no es una solicitud
  conjunta (`application_type != "Joint App"`), no por un error de captura. Es un missing
  **estructural (MNAR por diseño)**: la ausencia del dato es en sí misma información (préstamo
  individual), y tiene relación real con el riesgo (los conjuntos tienen menor tasa de default,
  probablemente porque combinan dos ingresos).

Ambos casos muestran por qué "imputar y ya" sin pensar en el mecanismo de faltante sería un error.
""")

add_code(nb, """def plan_tratamiento(row):
    if row["pct_missing"] == 0:
        return "sin tratamiento"
    if row["pct_missing"] > 70:
        return "eliminar columna (>70% nulo)"
    if "double" in row["dtype"] or "int" in row["dtype"]:
        return "imputar con mediana (train) + indicador de faltante si aplica"
    return "imputar con moda o categoría 'Desconocido' (train)"

profile["plan_tratamiento"] = profile.apply(plan_tratamiento, axis=1)
profile.to_csv("../outputs/tables/eda/column_profile_con_plan.csv")
profile["plan_tratamiento"].value_counts()
""")

# ---------- 1.5 Resumen ejecutivo ----------
add_md(nb, """## 1.5. Resumen ejecutivo del EDA

**Calidad de los datos**
- 2.260.701 filas, 151 columnas. 49 columnas sin nulos, 41 columnas con más de 70% de nulos
  (candidatas a eliminación directa).
- La ausencia de datos no es aleatoria: se concentra en campos de *hardship*/*settlement*
  (posteriores al desenlace del préstamo → riesgo de fuga de datos) y en campos de solicitante
  secundario (estructuralmente ausentes en solicitudes individuales).
- `dti` tiene valores centinela inválidos (mín. -1, máx. 999) que deben tratarse antes de modelar.
  `annual_inc` tiene al menos un outlier extremo ($110M) que distorsiona la media.

**Variables más prometedoras para predecir `default`**
- `int_rate` (la más fuerte, r≈0.20), `sub_grade`/`grade` (gradiente monotónico claro, 3.3%→37.5%
  de tasa de default de A a G), `fico_range_high/low` (r≈-0.12), `term` (60 vs 36 meses) e
  `inq_last_6mths`.
- Ninguna variable aislada tiene poder predictivo fuerte — es un problema donde el modelo necesita
  combinar muchas señales débiles, típico en scoring crediticio.

**Problemas detectados**
- Desbalance de clases moderado-alto (88.1% / 11.9%, ≈7.4:1) → usar AUC ROC + F1 + AUC-PR,
  estratificar la partición.
- Redundancia/multicolinealidad: `fico_range_low`≡`fico_range_high` (r=1.0), `sub_grade`≡`grade`
  (Cramér's V=1.0), `loan_amnt`↔`installment` (r=0.95).
- **Definición del target**: la regla literal del enunciado deja 878.317 préstamos "Current" (39%
  del total) etiquetados como "no default" sin que su desenlace final se conozca todavía, y
  clasifica mal 761 préstamos "Does not meet the credit policy. Status:Charged Off" como no-default.
  Se sigue la regla indicada, pero es la limitación metodológica más importante de todo el proyecto.

**Decisiones preliminares para el preprocesamiento**
- Variables a **mantener**: `loan_amnt`, `int_rate`, `sub_grade` (se prefiere sobre `grade`, ya
  contenido en `sub_grade`), `emp_length`, `annual_inc`, `home_ownership`, `verification_status`,
  `purpose`, `dti`, `addr_state`, `term`, `fico_range_high` (se descarta `fico_range_low`),
  `open_acc`, `revol_bal`, `revol_util`, `total_acc`, `pub_rec`, `delinq_2yrs`, `mort_acc`,
  `inq_last_6mths`, `initial_list_status`, `application_type`.
- Variables a **excluir**: todo lo posterior a la originación (`total_pymnt`, `recoveries`,
  `out_prncp`, `last_pymnt_*`, `settlement_*`, `hardship_*`, etc. — fuga de datos), texto libre
  de alta cardinalidad (`desc`, `emp_title`, `title`, `url`), identificadores (`id`, `member_id`),
  y columnas con >70% de nulos.
- Transformaciones a aplicar en el preprocesamiento: tratar los centinelas de `dti` (-1, 999) como
  faltantes; evaluar transformación logarítmica para `annual_inc` y `revol_bal` (colas muy
  pesadas); imputar `emp_length`/variables numéricas con mediana/moda del *train* únicamente;
  agrupar categorías raras (<1%) de `addr_state` y `purpose` en "Otros" antes del one-hot encoding.
""")

save(nb, "book/01_eda.ipynb")
print("Notebook 01_eda.ipynb creado.")
