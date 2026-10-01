"""Utilidades para la seccion 9.10.4.6.2: umbral elegido SOLO con datos de entrenamiento (Youden),
prueba de McNemar (statsmodels, exacta si b+c<25) y bootstrap pareado para DeltaAUC/DeltaAUC-PR/DeltaF1."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar


def umbral_youden(y_train, score_train):
    """Umbral que maximiza sensibilidad+especificidad-1 (indice J de Youden), usando SOLO train."""
    fpr, tpr, thresholds = roc_curve(y_train, score_train)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def mcnemar_test(y_true, pred_a, pred_b):
    """Tabla 2x2 de concordancia de aciertos + prueba de McNemar (statsmodels)."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))   # A acierta, B falla
    n01 = int(np.sum(~correct_a & correct_b))   # A falla, B acierta
    n00 = int(np.sum(~correct_a & ~correct_b))
    tabla = np.array([[n11, n10], [n01, n00]])
    b, c = n10, n01
    es_exacta = (b + c) < 25
    resultado = mcnemar(tabla, exact=es_exacta, correction=True)
    return {
        "n11": n11, "n10_A_acierta_B_falla": n10, "n01_A_falla_B_acierta": n01, "n00": n00,
        "b_mas_c": b + c, "exacta": es_exacta,
        "statistic": float(resultado.statistic), "p_value": float(resultado.pvalue),
        "tasa_error_a": 1 - correct_a.mean(), "tasa_error_b": 1 - correct_b.mean(),
    }


def bootstrap_pareado(y_true, score_a, score_b, pred_a, pred_b, B=2000, seed=42):
    """Bootstrap pareado (mismos indices de remuestreo para A y B) para DeltaAUC, DeltaAUC-PR, DeltaF1.
    Devuelve el valor observado y el IC95% percentil de cada delta."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true); score_a = np.asarray(score_a); score_b = np.asarray(score_b)
    pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)

    deltas_auc, deltas_auc_pr, deltas_f1 = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue  # remuestreo degenerado (sin ambas clases), se descarta
        auc_a = roc_auc_score(yb, score_a[idx]); auc_b = roc_auc_score(yb, score_b[idx])
        pr_a = average_precision_score(yb, score_a[idx]); pr_b = average_precision_score(yb, score_b[idx])
        f1_a = f1_score(yb, pred_a[idx], zero_division=0); f1_b = f1_score(yb, pred_b[idx], zero_division=0)
        deltas_auc.append(auc_a - auc_b)
        deltas_auc_pr.append(pr_a - pr_b)
        deltas_f1.append(f1_a - f1_b)

    def resumen(deltas, observado):
        d = np.array(deltas)
        ci_low, ci_high = np.percentile(d, [2.5, 97.5])
        return {"observado": observado, "ci_low": ci_low, "ci_high": ci_high,
                "excluye_cero": not (ci_low <= 0 <= ci_high)}

    obs_auc = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)
    obs_pr = average_precision_score(y_true, score_a) - average_precision_score(y_true, score_b)
    obs_f1 = f1_score(y_true, pred_a, zero_division=0) - f1_score(y_true, pred_b, zero_division=0)

    return {
        "delta_auc": resumen(deltas_auc, obs_auc),
        "delta_auc_pr": resumen(deltas_auc_pr, obs_pr),
        "delta_f1": resumen(deltas_f1, obs_f1),
        "B_efectivo": len(deltas_auc),
    }
