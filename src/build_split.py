import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/accepted_raw.parquet", columns=["id","loan_status"])
df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")

id_train, id_test = train_test_split(
    df["id"],
    test_size=0.20,
    stratify=df["default"],
    random_state=42,
)

split_df = pd.DataFrame({
    "id": pd.concat([id_train, id_test]).values,
    "split": ["train"]*len(id_train) + ["test"]*len(id_test),
})
split_df = split_df.astype({"id": "string", "split": "string"})
split_df.to_parquet("data/split_assignment.parquet", index=False)

print(f"Train: {len(id_train):,} ({len(id_train)/len(df)*100:.2f}%)")
print(f"Test:  {len(id_test):,} ({len(id_test)/len(df)*100:.2f}%)")

# Verificar estratificacion
merged = split_df.merge(df, on="id")
print("\nTasa de default por partición (debe ser ~igual en ambas):")
print(merged.groupby("split", observed=True)["default"].mean().round(4)*100)
print("\nArchivo guardado en data/split_assignment.parquet con columnas:", split_df.columns.tolist())
