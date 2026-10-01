import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet("data/accepted_raw.parquet")

cat_vars = ["term","grade","sub_grade","home_ownership","verification_status",
            "purpose","addr_state","initial_list_status","application_type","emp_length"]

summary_rows = []
fig, axes = plt.subplots(len(cat_vars), 1, figsize=(9, 3.2*len(cat_vars)))
for i, col in enumerate(cat_vars):
    vc = df[col].value_counts(dropna=False)
    vc_pct = (vc / len(df) * 100).round(2)
    n_categories = df[col].nunique(dropna=True)
    n_rare = (vc_pct < 1).sum()
    pct_missing = df[col].isna().mean()*100
    summary_rows.append({"variable": col, "n_categorias": n_categories,
                          "n_categorias_raras(<1%)": n_rare, "pct_missing": round(pct_missing,2),
                          "categoria_mas_frecuente": vc.index[0], "pct_mas_frecuente": vc_pct.iloc[0]})
    top = vc_pct.head(15)
    top.plot(kind="barh", ax=axes[i], color="#55A868")
    axes[i].invert_yaxis()
    axes[i].set_title(f"{col} (n_categorias={n_categories}, nulos={pct_missing:.1f}%)")
    axes[i].set_xlabel("% del total")

plt.tight_layout()
plt.savefig("outputs/figures/eda/categorical_bars.png", dpi=100)
plt.close()

cat_summary = pd.DataFrame(summary_rows).set_index("variable")
cat_summary.to_csv("outputs/tables/eda/categorical_summary.csv")
print(cat_summary.to_string())

# Detalle de addr_state (50+ categorias) y purpose por separado, guardado en csv
for col in ["addr_state","purpose","sub_grade","emp_length"]:
    vc = df[col].value_counts(dropna=False)
    vc_pct = (vc/len(df)*100).round(3)
    pd.DataFrame({"n": vc, "%": vc_pct}).to_csv(f"outputs/tables/eda/freq_{col}.csv")

print("\naddr_state - categorias con <1%:")
vc_state = df["addr_state"].value_counts(normalize=True)*100
print((vc_state < 1).sum(), "de", len(vc_state), "estados tienen <1% de los prestamos")
