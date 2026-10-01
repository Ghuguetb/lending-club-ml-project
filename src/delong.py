"""Implementacion rapida O(n log n) de la prueba de DeLong (Sun & Xu, 2014) para comparar AUCs
correlacionados (mismo conjunto de prueba, distintos clasificadores), mas una implementacion lenta
O(m*n) de referencia (formulacion original de DeLong, Delong y Clarke-Pearson, 1988) usada
UNICAMENTE para validar la version rapida en muestras pequenas -- ver validate_delong.py.
"""
import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------------
# Version rapida (Sun & Xu, 2014) -- la que se usa en todo el analisis real del proyecto
# ----------------------------------------------------------------------------------

def compute_midrank(x):
    """Rangos medios (midranks) de x, con desempate correcto para valores repetidos."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """predictions_sorted_transposed: array (k, N) con las N observaciones YA ORDENADAS de forma
    que las m positivas (label=1) ocupan las primeras columnas y las n negativas el resto, para
    k clasificadores evaluados sobre las MISMAS observaciones. Devuelve (aucs, covarianza) donde
    covarianza es la matriz de covarianza (k,k) de las AUCs -- exactamente lo que hace falta para
    comparar cualquier par de clasificadores correlacionados (mismo test set)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    if k == 1:
        delongcov = np.array([[delongcov]])
    return aucs, delongcov


def _prepare(ground_truth, predictions_2d):
    ground_truth = np.asarray(ground_truth)
    order = (-ground_truth).argsort(kind="stable")  # positivas (1) primero
    label_1_count = int(ground_truth.sum())
    predictions_sorted = np.asarray(predictions_2d)[:, order]
    return predictions_sorted, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """AUC y varianza de UN solo clasificador (conveniencia)."""
    preds_sorted, m = _prepare(ground_truth, predictions.reshape(1, -1))
    aucs, cov = fastDeLong(preds_sorted, m)
    return aucs[0], cov[0, 0]


def delong_covariance_matrix(ground_truth, predictions_2d):
    """AUCs y matriz de covarianza completa (k,k) para k clasificadores sobre el mismo test set."""
    preds_sorted, m = _prepare(ground_truth, predictions_2d)
    aucs, cov = fastDeLong(preds_sorted, m)
    return aucs, cov


def delong_pairwise_test(ground_truth, score_a, score_b):
    """Prueba de DeLong para DOS clasificadores correlacionados (mismo test set).
    Devuelve dict con AUC_a, AUC_b, IC95 de cada uno, delta_auc, IC95 del delta, z, p (2 colas)."""
    preds = np.vstack([np.asarray(score_a), np.asarray(score_b)])
    aucs, cov = delong_covariance_matrix(ground_truth, preds)
    auc_a, auc_b = aucs[0], aucs[1]
    var_a, var_b, cov_ab = cov[0, 0], cov[1, 1], cov[0, 1]
    delta = auc_a - auc_b
    var_delta = var_a + var_b - 2 * cov_ab
    var_delta = max(var_delta, 0.0)  # proteccion numerica (no debería ser negativo)
    se_delta = np.sqrt(var_delta)
    z = delta / se_delta if se_delta > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_a = (auc_a - 1.96 * np.sqrt(var_a), auc_a + 1.96 * np.sqrt(var_a))
    ci_b = (auc_b - 1.96 * np.sqrt(var_b), auc_b + 1.96 * np.sqrt(var_b))
    ci_delta = (delta - 1.96 * se_delta, delta + 1.96 * se_delta)
    return {
        "auc_a": auc_a, "auc_a_ci_low": ci_a[0], "auc_a_ci_high": ci_a[1],
        "auc_b": auc_b, "auc_b_ci_low": ci_b[0], "auc_b_ci_high": ci_b[1],
        "delta_auc": delta, "delta_ci_low": ci_delta[0], "delta_ci_high": ci_delta[1],
        "se_delta": se_delta, "z": z, "p_value": p,
    }


# ----------------------------------------------------------------------------------
# Version lenta O(m*n) de referencia -- SOLO para validar la version rapida (muestras pequenas)
# ----------------------------------------------------------------------------------

def _psi(x, y):
    if x > y:
        return 1.0
    elif x == y:
        return 0.5
    else:
        return 0.0


def slow_delong_reference(ground_truth, predictions_2d):
    """Calculo directo O(k^2 * m * n) de AUCs y covarianza siguiendo la definicion original de
    DeLong, DeLong y Clarke-Pearson (1988) via componentes estructurales V10/V01, SIN usar
    midranks ni ordenamientos -- solo para validar fastDeLong en muestras pequenas (es demasiado
    lento para las 452,141 filas del proyecto)."""
    ground_truth = np.asarray(ground_truth)
    predictions_2d = np.asarray(predictions_2d)
    k = predictions_2d.shape[0]
    pos_idx = np.where(ground_truth == 1)[0]
    neg_idx = np.where(ground_truth == 0)[0]
    m, n = len(pos_idx), len(neg_idx)

    V10 = np.zeros((k, m))  # componente estructural para cada positiva
    V01 = np.zeros((k, n))  # componente estructural para cada negativa
    for r in range(k):
        X = predictions_2d[r, pos_idx]
        Y = predictions_2d[r, neg_idx]
        for i in range(m):
            V10[r, i] = np.mean([_psi(X[i], y) for y in Y])
        for j in range(n):
            V01[r, j] = np.mean([_psi(x, Y[j]) for x in X])

    aucs = V10.mean(axis=1)
    S10 = np.cov(V10) if k > 1 else np.array([[np.var(V10[0], ddof=1)]])
    S01 = np.cov(V01) if k > 1 else np.array([[np.var(V01[0], ddof=1)]])
    if k == 1:
        S10 = S10.reshape(1, 1)
        S01 = S01.reshape(1, 1)
    cov = S10 / m + S01 / n
    return aucs, cov


def holm_correction(p_values, alpha=0.05):
    """Correccion de Holm (1979) para comparaciones multiples. Devuelve (p_ajustados, rechaza_h0)
    en el MISMO orden que p_values (no requiere que ya vengan ordenados)."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    p_sorted = p_values[order]
    p_adj_sorted = np.empty(n)
    running_max = 0.0
    for i in range(n):
        adj = (n - i) * p_sorted[i]
        running_max = max(running_max, adj)
        p_adj_sorted[i] = min(running_max, 1.0)
    p_adj = np.empty(n)
    p_adj[order] = p_adj_sorted
    reject = p_adj < alpha
    return p_adj, reject
