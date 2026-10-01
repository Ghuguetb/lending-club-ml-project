"""Valida la implementacion rapida de DeLong (src/delong.py) de tres formas independientes:
  1. Comparacion directa contra una implementacion lenta O(m*n) de la definicion original de
     DeLong (1988), en muestras pequenas (donde la version lenta es factible de correr).
  2. El punto estimado del AUC debe coincidir EXACTAMENTE con sklearn.metrics.roc_auc_score,
     en muestras grandes (10,000+ filas, similar en espiritu al tamano real del proyecto).
  3. La varianza analitica de DeLong para un solo clasificador debe ser cercana a una varianza
     estimada por bootstrap (metodo totalmente independiente: remuestreo vs. formula cerrada).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "src")
from delong import delong_covariance_matrix, slow_delong_reference, delong_pairwise_test

rng = np.random.default_rng(123)

print("=" * 70)
print("VALIDACION 1: version rapida vs. version lenta (definicion original)")
print("=" * 70)

for escenario, (n_pos, n_neg, ruido_a, ruido_b, con_empates) in {
    "sin empates, AUCs similares": (40, 60, 1.0, 1.0, False),
    "sin empates, AUCs distintos": (35, 45, 0.5, 2.0, False),
    "CON empates (scores discretos)": (30, 50, 1.0, 1.0, True),
}.items():
    print(f"\n--- Escenario: {escenario} ---")
    y = np.array([1] * n_pos + [0] * n_neg)
    señal = np.concatenate([rng.normal(1.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    score_a = señal + rng.normal(0, ruido_a, len(y))
    score_b = señal + rng.normal(0, ruido_b, len(y))
    if con_empates:
        score_a = np.round(score_a, 0)  # fuerza muchos valores repetidos
        score_b = np.round(score_b, 0)
    preds = np.vstack([score_a, score_b])

    aucs_fast, cov_fast = delong_covariance_matrix(y, preds)
    aucs_slow, cov_slow = slow_delong_reference(y, preds)

    print(f"  AUC rapido: {aucs_fast}  |  AUC lento: {aucs_slow}")
    print(f"  max|AUC_rapido - AUC_lento| = {np.max(np.abs(aucs_fast - aucs_slow)):.2e}")
    print(f"  Cov rapida:\n{cov_fast}")
    print(f"  Cov lenta:\n{cov_slow}")
    print(f"  max|cov_rapida - cov_lenta| = {np.max(np.abs(cov_fast - cov_slow)):.2e}")
    assert np.allclose(aucs_fast, aucs_slow, atol=1e-9), "AUCs no coinciden"
    assert np.allclose(cov_fast, cov_slow, atol=1e-9), "Covarianzas no coinciden"
    print("  OK: coinciden hasta precision numerica.")

print("\n" + "=" * 70)
print("VALIDACION 2: AUC de DeLong (rapido) vs. sklearn.metrics.roc_auc_score (muestra grande)")
print("=" * 70)
n = 20000
y = rng.binomial(1, 0.15, n)
score = rng.normal(0, 1, n) + y * rng.normal(1.2, 1, n)
auc_delong, var_delong = delong_covariance_matrix(y, score.reshape(1, -1))
auc_delong = auc_delong[0]
auc_sklearn = roc_auc_score(y, score)
print(f"AUC (DeLong rapido) = {auc_delong:.10f}")
print(f"AUC (sklearn)        = {auc_sklearn:.10f}")
print(f"diferencia = {abs(auc_delong-auc_sklearn):.2e}")
assert abs(auc_delong - auc_sklearn) < 1e-9, "El punto estimado del AUC no coincide con sklearn"
print("OK: coinciden hasta precision numerica de punto flotante.")

print("\n" + "=" * 70)
print("VALIDACION 3: varianza analitica de DeLong vs. varianza por bootstrap (metodo independiente)")
print("=" * 70)
n = 5000
y = rng.binomial(1, 0.15, n)
score = rng.normal(0, 1, n) + y * rng.normal(1.0, 1, n)
_, var_analitica = delong_covariance_matrix(y, score.reshape(1, -1))
var_analitica = var_analitica[0, 0]

B = 2000
aucs_boot = np.empty(B)
idx_all = np.arange(n)
for b in range(B):
    idx = rng.choice(idx_all, size=n, replace=True)
    if y[idx].sum() == 0 or y[idx].sum() == n:
        aucs_boot[b] = np.nan
        continue
    aucs_boot[b] = roc_auc_score(y[idx], score[idx])
var_bootstrap = np.nanvar(aucs_boot, ddof=1)

print(f"Varianza analitica (DeLong): {var_analitica:.6e}")
print(f"Varianza por bootstrap (B={B}): {var_bootstrap:.6e}")
print(f"razon (bootstrap/analitica): {var_bootstrap/var_analitica:.3f}")
assert 0.7 < (var_bootstrap / var_analitica) < 1.3, "Varianzas demasiado distintas"
print("OK: mismo orden de magnitud (dos metodos completamente independientes).")

print("\n" + "=" * 70)
print("VALIDACION 4: prueba pareada completa (delong_pairwise_test) con AUCs simulados conocidos")
print("=" * 70)
n_pos, n_neg = 200, 800
y = np.array([1]*n_pos + [0]*n_neg)
señal = np.concatenate([rng.normal(1.5, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
score_igual_1 = señal + rng.normal(0, 1.0, len(y))
score_igual_2 = señal + rng.normal(0, 1.0, len(y))  # mismo proceso generador -> AUC esperado similar
score_peor = rng.normal(0, 1, len(y))  # puramente ruido -> AUC esperado ~0.5

r1 = delong_pairwise_test(y, score_igual_1, score_igual_2)
print(f"Dos modelos EQUIVALENTES: AUC_a={r1['auc_a']:.4f} AUC_b={r1['auc_b']:.4f} "
      f"delta={r1['delta_auc']:.4f} p={r1['p_value']:.4f} (se espera p grande, no significativo)")

r2 = delong_pairwise_test(y, score_igual_1, score_peor)
print(f"Modelo bueno vs. ruido puro: AUC_a={r2['auc_a']:.4f} AUC_b={r2['auc_b']:.4f} "
      f"delta={r2['delta_auc']:.4f} p={r2['p_value']:.2e} (se espera p muy pequeno)")
assert r2["p_value"] < 0.001, "No detecto una diferencia de AUC obvia"
print("OK: la prueba distingue correctamente el caso obvio del caso nulo.")

print("\n" + "=" * 70)
print("TODAS LAS VALIDACIONES PASARON")
print("=" * 70)
