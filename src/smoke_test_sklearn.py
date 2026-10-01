import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import time

X_train = sparse.load_npz("data/sklearn_X_train.npz")
y_train = np.load("data/sklearn_y_train.npy")

# submuestra pequena solo para probar que todo corre
rng = np.random.default_rng(0)
idx = rng.choice(X_train.shape[0], 20000, replace=False)
Xs, ys = X_train[idx], y_train[idx]
n = Xs.shape[0]

print("Probando LogisticRegression...")
t0=time.time()
C_vals = [1/(rp*n) for rp in [1e-6,1e-5,1e-4]]
gs = GridSearchCV(LogisticRegression(max_iter=200), {"C": C_vals}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(Xs, ys)
print("OK LR", time.time()-t0, gs.best_params_)

print("Probando DecisionTree...")
t0=time.time()
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), {"max_depth":[5,10,15]}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(Xs, ys)
print("OK DT", time.time()-t0, gs.best_params_)

print("Probando RandomForest...")
t0=time.time()
gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1),
                   {"n_estimators":[10,50], "max_depth":[5,10]}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(Xs, ys)
print("OK RF", time.time()-t0, gs.best_params_)

print("Probando HistGradientBoosting (requiere denso)...")
t0=time.time()
Xs_dense = Xs.toarray()
gs = GridSearchCV(HistGradientBoostingClassifier(learning_rate=0.1, random_state=42),
                   {"max_iter":[50,100], "max_depth":[3,5]}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(Xs_dense, ys)
print("OK HGB", time.time()-t0, gs.best_params_)

print("Probando LinearSVC (hinge)...")
t0=time.time()
gs = GridSearchCV(LinearSVC(loss="hinge", dual=True, max_iter=2000), {"C": C_vals}, cv=3, scoring="roc_auc", n_jobs=-1)
gs.fit(Xs, ys)
print("OK SVC", time.time()-t0, gs.best_params_)
print("decision_function ejemplo:", gs.decision_function(Xs[:3]))

print("Probando GaussianNB...")
t0=time.time()
nb = GaussianNB()
nb.fit(Xs.toarray(), ys)
print("OK NB", time.time()-t0)

print("\nTODO OK")
