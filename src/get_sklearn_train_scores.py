"""Puntajes de ENTRENAMIENTO de los 6 modelos de scikit-learn (necesarios para elegir el umbral de
clasificacion usando solo train, seccion 9.10.4.6.2). Los modelos ya estan entrenados y guardados;
esto solo aplica predict_proba/decision_function sobre X_train -- es rapido, no hace falta reentrenar."""
import numpy as np
import pandas as pd
from scipy import sparse
import joblib

X_train = sparse.load_npz("data/sklearn_X_train.npz")
y_train = np.load("data/sklearn_y_train.npy")
id_train = np.load("data/sklearn_id_train.npy", allow_pickle=True)
models = joblib.load("data/sklearn_fitted_models.joblib")

scores = pd.DataFrame({"id": id_train, "default": y_train})

X_train_dense = None
for nombre, modelo in models.items():
    if nombre in ("GBT_HistGB", "GaussianNB"):
        if X_train_dense is None:
            X_train_dense = X_train.toarray()
        X_in = X_train_dense
    else:
        X_in = X_train
    if nombre == "LinearSVC":
        s = modelo.decision_function(X_in)
    else:
        s = modelo.predict_proba(X_in)[:, 1]
    scores[f"score_{nombre}"] = s
    print(f"{nombre}: listo (min={s.min():.4f} max={s.max():.4f})", flush=True)

scores.to_parquet("data/sklearn_train_scores.parquet", index=False)
print("\nGuardado: data/sklearn_train_scores.parquet", scores.shape)
