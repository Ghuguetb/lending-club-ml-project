"""Modelado con scikit-learn: 6 modelos + GridSearchCV (sin Pipeline). Seccion 9.10.4.4."""
import numpy as np
import pandas as pd
import json, time, warnings
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

print("Cargando datos preprocesados...", flush=True)
X_train = sparse.load_npz("data/sklearn_X_train.npz")
X_test  = sparse.load_npz("data/sklearn_X_test.npz")
y_train = np.load("data/sklearn_y_train.npy")
y_test  = np.load("data/sklearn_y_test.npy")
id_train = np.load("data/sklearn_id_train.npy", allow_pickle=True)
id_test  = np.load("data/sklearn_id_test.npy", allow_pickle=True)
n_train = X_train.shape[0]
print(f"X_train={X_train.shape}, X_test={X_test.shape}", flush=True)

HARDWARE = {"cores": 2, "ram_gb": 7.8, "nota": "maquina virtual compartida, sin GPU"}

results = []
scores_test = pd.DataFrame({"id": id_test, "default": y_test})
models_fitted = {}

def evaluar_y_guardar(nombre, best_estimator, cv_auc, best_params, t_fit, y_score, t_predict):
    y_pred = (y_score >= 0.5).astype(int) if nombre != "LinearSVC" else (y_score >= 0).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)
    results.append({
        "modelo": nombre, "best_params": json.dumps(best_params), "cv_auc": cv_auc,
        "test_accuracy": acc, "test_precision": prec, "test_recall": rec,
        "test_f1": f1, "test_auc": auc, "tn": cm[0,0], "fp": cm[0,1], "fn": cm[1,0], "tp": cm[1,1],
        "tiempo_entrenamiento_s": t_fit, "tiempo_prediccion_s": t_predict,
    })
    scores_test[f"score_{nombre}"] = y_score
    print(f"  {nombre}: AUC test={auc:.4f} | CV AUC={cv_auc:.4f} | params={best_params} "
          f"| fit={t_fit:.1f}s pred={t_predict:.2f}s", flush=True)

# ============ 1. Regresion logistica ============
print("\n[1/6] LogisticRegression...", flush=True)
C_vals = [1/(rp*n_train) for rp in [1e-6, 1e-5, 1e-4]]
t0 = time.time()
gs = GridSearchCV(LogisticRegression(max_iter=500, random_state=42),
                   {"C": C_vals}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
t_fit = time.time() - t0
t0 = time.time()
y_score = gs.predict_proba(X_test)[:, 1]
t_pred = time.time() - t0
evaluar_y_guardar("LogisticRegression", gs.best_estimator_, gs.best_score_, gs.best_params_, t_fit, y_score, t_pred)
models_fitted["LogisticRegression"] = gs.best_estimator_

# ============ 2. Arbol de decision ============
print("\n[2/6] DecisionTreeClassifier...", flush=True)
t0 = time.time()
gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                   {"max_depth": [5, 10, 15]}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
t_fit = time.time() - t0
t0 = time.time()
y_score = gs.predict_proba(X_test)[:, 1]
t_pred = time.time() - t0
evaluar_y_guardar("DecisionTree", gs.best_estimator_, gs.best_score_, gs.best_params_, t_fit, y_score, t_pred)
models_fitted["DecisionTree"] = gs.best_estimator_

# ============ 3. Bosque aleatorio ============
print("\n[3/6] RandomForestClassifier (el mas pesado)...", flush=True)
t0 = time.time()
gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1),
                   {"n_estimators": [10, 50, 100], "max_depth": [5, 10, 15]},
                   cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
t_fit = time.time() - t0
t0 = time.time()
y_score = gs.predict_proba(X_test)[:, 1]
t_pred = time.time() - t0
evaluar_y_guardar("RandomForest", gs.best_estimator_, gs.best_score_, gs.best_params_, t_fit, y_score, t_pred)
models_fitted["RandomForest"] = gs.best_estimator_

# ============ 4. Gradient boosting ============
# NOTA: se usa HistGradientBoostingClassifier en vez de GradientBoostingClassifier,
# tal como permite explicitamente el enunciado ("si el costo computacional... resultara
# prohibitivo... puede sustituirse por HistGradientBoostingClassifier"). Se probo antes:
# GradientBoostingClassifier no tiene paralelismo nativo y con 1.8M filas x varios cientos
# de arboles secuenciales el tiempo esperado es de horas en esta maquina de 2 nucleos;
# HistGradientBoostingClassifier usa el mismo algoritmo de histogramas que PySpark GBT,
# lo que ademas hace la comparacion entre entornos mas justa.
print("\n[4/6] HistGradientBoostingClassifier (sustituto justificado de GBT)...", flush=True)
t0 = time.time()
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
print(f"  Conversion a denso: {time.time()-t0:.1f}s, forma={X_train_dense.shape}", flush=True)
t0 = time.time()
gs = GridSearchCV(HistGradientBoostingClassifier(learning_rate=0.1, random_state=42),
                   {"max_iter": [50, 100], "max_depth": [3, 5]}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train_dense, y_train)
t_fit = time.time() - t0
t0 = time.time()
y_score = gs.predict_proba(X_test_dense)[:, 1]
t_pred = time.time() - t0
evaluar_y_guardar("GBT_HistGB", gs.best_estimator_, gs.best_score_, gs.best_params_, t_fit, y_score, t_pred)
models_fitted["GBT_HistGB"] = gs.best_estimator_

# ============ 5. SVM lineal ============
print("\n[5/6] LinearSVC (hinge)...", flush=True)
t0 = time.time()
gs = GridSearchCV(LinearSVC(loss="hinge", dual=True, max_iter=5000, random_state=42),
                   {"C": C_vals}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
t_fit = time.time() - t0
t0 = time.time()
y_score = gs.decision_function(X_test)
t_pred = time.time() - t0
evaluar_y_guardar("LinearSVC", gs.best_estimator_, gs.best_score_, gs.best_params_, t_fit, y_score, t_pred)
models_fitted["LinearSVC"] = gs.best_estimator_

# ============ 6. Naive Bayes ============
print("\n[6/6] GaussianNB (sin busqueda de hiperparametros)...", flush=True)
t0 = time.time()
nb = GaussianNB()
nb.fit(X_train_dense, y_train)
t_fit = time.time() - t0
t0 = time.time()
y_score = nb.predict_proba(X_test_dense)[:, 1]
t_pred = time.time() - t0
# CV AUC de NB via validacion cruzada simple, para reportar de forma comparable
from sklearn.model_selection import cross_val_score
cv_auc_nb = cross_val_score(GaussianNB(), X_train_dense, y_train, cv=3, scoring="roc_auc", n_jobs=-1).mean()
evaluar_y_guardar("GaussianNB", nb, cv_auc_nb, {}, t_fit, y_score, t_pred)
models_fitted["GaussianNB"] = nb

# ============ Guardar todo ============
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/tables/sklearn_results.csv", index=False)
scores_test.to_parquet("data/sklearn_test_scores.parquet", index=False)
joblib.dump(models_fitted, "data/sklearn_fitted_models.joblib")
with open("outputs/tables/hardware_sklearn.json", "w") as f:
    json.dump(HARDWARE, f)

print("\n=== RESUMEN FINAL ===", flush=True)
print(results_df[["modelo","cv_auc","test_auc","test_accuracy","test_f1","tiempo_entrenamiento_s"]].to_string(index=False), flush=True)
print("\nGuardado: outputs/tables/sklearn_results.csv, data/sklearn_test_scores.parquet, data/sklearn_fitted_models.joblib", flush=True)
