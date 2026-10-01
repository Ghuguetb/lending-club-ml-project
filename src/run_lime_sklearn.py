"""Seccion 9.10.4.7: LIME sobre el mejor modelo de scikit-learn (HistGradientBoostingClassifier,
AUC=0.739, seccion 3). Explica 2 instancias del TEST mal clasificadas: un falso positivo (predicho
default, en realidad no default) y un falso negativo (predicho no default, en realidad default)."""
import json
import joblib
import numpy as np
from scipy import sparse
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORICAL_START = 14  # las primeras 14 columnas son numericas; el resto, dummies 0/1 de OHE

names = json.load(open("data/sklearn_feature_names.json"))
models = joblib.load("data/sklearn_fitted_models.joblib")
gbt = models["GBT_HistGB"]

X_test = sparse.load_npz("data/sklearn_X_test.npz").toarray()
y_test = np.load("data/sklearn_y_test.npy")
id_test = np.load("data/sklearn_id_test.npy", allow_pickle=True)

bg = np.load("data/lime_background_sklearn.npz")
X_bg = bg["X"]

pred_proba = gbt.predict_proba(X_test)[:, 1]
pred_label = (pred_proba >= 0.5).astype(int)

# Umbral natural de 0.5 para elegir los casos a explicar (no el de Youden de la seccion 6 -- aqui
# solo interesa encontrar ejemplos mal clasificados que ilustren el comportamiento del modelo).
falsos_positivos = np.where((pred_label == 1) & (y_test == 0))[0]
falsos_negativos = np.where((pred_label == 0) & (y_test == 1))[0]
print(f"Falsos positivos disponibles: {len(falsos_positivos):,}  |  Falsos negativos: {len(falsos_negativos):,}")

rng = np.random.default_rng(7)
idx_fp = falsos_positivos[rng.integers(0, len(falsos_positivos))]
idx_fn = falsos_negativos[rng.integers(0, len(falsos_negativos))]

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_bg,
    feature_names=names,
    class_names=["No default", "Default"],
    categorical_features=list(range(CATEGORICAL_START, len(names))),
    discretize_continuous=True,
    random_state=42,
)

resultados = {}
for etiqueta, idx in [("falso_positivo", idx_fp), ("falso_negativo", idx_fn)]:
    print(f"\n=== {etiqueta} (fila de test #{idx}, id={id_test[idx]}) ===")
    print(f"  y_real={y_test[idx]}  prob_predicha={pred_proba[idx]:.4f}  pred={pred_label[idx]}")
    exp = explainer.explain_instance(X_test[idx], gbt.predict_proba, num_features=10, num_samples=5000)
    lista = exp.as_list()
    for feat, peso in lista:
        print(f"    {feat:45s} {peso:+.4f}")
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    fig.savefig(f"outputs/figures/eda/lime_sklearn_{etiqueta}.png", dpi=110)
    plt.close(fig)
    resultados[etiqueta] = {
        "fila_test": int(idx), "id": str(id_test[idx]), "y_real": int(y_test[idx]),
        "prob_predicha": float(pred_proba[idx]), "pred_label": int(pred_label[idx]),
        "explicacion": [(f, float(w)) for f, w in lista],
        "intercepto_local": float(exp.intercept[1]) if hasattr(exp, "intercept") else None,
        "r2_local": float(exp.score) if hasattr(exp, "score") else None,
    }

with open("outputs/tables/lime_sklearn_resultados.json", "w") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\nGuardado: outputs/tables/lime_sklearn_resultados.json y outputs/figures/eda/lime_sklearn_*.png")
