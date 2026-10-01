import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import json

NUMERIC_VARS = ["loan_amnt","int_rate","dti","fico_range_high","open_acc",
                 "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc",
                 "inq_last_6mths","emp_length_num","annual_inc_log","revol_bal_log"]
CATEGORICAL_VARS = ["term","sub_grade","home_ownership","verification_status",
                     "purpose","addr_state_grouped","initial_list_status","application_type"]
RARE_THRESHOLD = 0.01  # <1% en train -> "OTHER"

def parse_emp_length(x):
    if pd.isna(x): return np.nan
    x = str(x)
    if "10+" in x: return 10.0
    if "< 1" in x: return 0.0
    digits = "".join(ch for ch in x if ch.isdigit())
    return float(digits) if digits else np.nan

def engineer_features(df):
    df = df.copy()
    df["emp_length_num"] = df["emp_length"].apply(parse_emp_length)
    # dti: -1 y 999 son centinelas invalidos detectados en el EDA -> tratar como faltante
    df["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    df.loc[(df["dti"] == -1) | (df["dti"] >= 999), "dti"] = np.nan
    # transformaciones log para variables con cola muy pesada (EDA: skew 494 y 13.2)
    df["annual_inc_log"] = np.log1p(pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=0))
    df["revol_bal_log"] = np.log1p(pd.to_numeric(df["revol_bal"], errors="coerce").clip(lower=0))
    return df

print("Cargando datos crudos...")
cols_needed = ["id","loan_status","loan_amnt","int_rate","dti","fico_range_high","open_acc",
               "revol_util","total_acc","pub_rec","delinq_2yrs","mort_acc","inq_last_6mths",
               "emp_length","annual_inc","revol_bal","term","sub_grade","home_ownership",
               "verification_status","purpose","addr_state","initial_list_status","application_type"]
df = pd.read_parquet("data/accepted_raw.parquet", columns=cols_needed)
df["default"] = (df["loan_status"] == "Charged Off").fillna(False).astype("int64")
df = engineer_features(df)

split = pd.read_parquet("data/split_assignment.parquet")
df = df.merge(split, on="id", how="inner")
assert len(df) == len(split), "No todas las filas tienen asignacion de partición"

train_mask = df["split"] == "train"
test_mask = df["split"] == "test"
print(f"Train: {train_mask.sum():,}  Test: {test_mask.sum():,}")

# --- Agrupar categorias raras de addr_state, calculado SOLO con train ---
freq_train = df.loc[train_mask, "addr_state"].value_counts(normalize=True)
keep_states = set(freq_train[freq_train >= RARE_THRESHOLD].index)
print(f"\naddr_state: se mantienen {len(keep_states)} categorias, el resto -> 'OTHER'")
df["addr_state_grouped"] = df["addr_state"].where(df["addr_state"].isin(keep_states), "OTHER")

X_train_raw = df.loc[train_mask, NUMERIC_VARS + CATEGORICAL_VARS]
X_test_raw  = df.loc[test_mask,  NUMERIC_VARS + CATEGORICAL_VARS]
y_train = df.loc[train_mask, "default"].values
y_test  = df.loc[test_mask,  "default"].values
id_train = df.loc[train_mask, "id"].values
id_test  = df.loc[test_mask,  "id"].values

# --- Imputacion numerica: mediana, ajustada SOLO con train ---
num_imputer = SimpleImputer(strategy="median")
X_train_num = num_imputer.fit_transform(X_train_raw[NUMERIC_VARS])
X_test_num  = num_imputer.transform(X_test_raw[NUMERIC_VARS])
print("\nMedianas de imputacion (ajustadas con train):")
for v, m in zip(NUMERIC_VARS, num_imputer.statistics_):
    print(f"  {v}: {m:.3f}")

# --- Escalado: ajustado SOLO con train ---
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled  = scaler.transform(X_test_num)

# --- One-Hot categoricas: ajustado SOLO con train ---
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_cat = ohe.fit_transform(X_train_raw[CATEGORICAL_VARS].astype(str))
X_test_cat  = ohe.transform(X_test_raw[CATEGORICAL_VARS].astype(str))
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_VARS).tolist()

X_train = sparse.hstack([sparse.csr_matrix(X_train_num_scaled), X_train_cat]).tocsr()
X_test  = sparse.hstack([sparse.csr_matrix(X_test_num_scaled),  X_test_cat]).tocsr()

feature_names = NUMERIC_VARS + cat_feature_names
print(f"\nDimension final: X_train={X_train.shape}, X_test={X_test.shape}")
print(f"Total de features tras codificacion: {len(feature_names)}")

sparse.save_npz("data/sklearn_X_train.npz", X_train)
sparse.save_npz("data/sklearn_X_test.npz", X_test)
np.save("data/sklearn_y_train.npy", y_train)
np.save("data/sklearn_y_test.npy", y_test)
np.save("data/sklearn_id_train.npy", id_train)
np.save("data/sklearn_id_test.npy", id_test)
with open("data/sklearn_feature_names.json","w") as f:
    json.dump(feature_names, f)
joblib.dump({"imputer": num_imputer, "scaler": scaler, "ohe": ohe, "keep_states": keep_states},
            "data/sklearn_preprocessors.joblib")

print("\nGuardado: data/sklearn_X_train.npz, sklearn_X_test.npz, y_train/y_test, id_train/id_test, feature_names, preprocessors")
