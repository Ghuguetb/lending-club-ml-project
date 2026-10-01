# Tarea 1: Proyecto Integrador de Aprendizaje Automático

**Predicción de incumplimiento de pago - Lending Club Loan Data (2007–2020)**

Integrantes: Santiago Díaz, Gina Huguet y Leiry Mares. Maestría en Ingeniería Biomédica, Universidad del Norte.

## El problema

Lending Club fue una plataforma de préstamos entre particulares (peer-to-peer lending) que operó desde 2007 hasta 2020. A partir de los datos disponibles **en el momento de otorgar un préstamo**, surge el problema de determinar qué créditos presentan mayor riesgo de terminar en incumplimiento (default).

El objetivo de este proyecto es **desarrollar y comparar modelos de clasificación capaces de predecir**, usando únicamente la información que se tenía disponible en el momento de otorgar un préstamo, si ese **crédito terminaría en incumplimiento de pago.**

## ¿Cómo resolverlo?

Para resolver este problema entrenamos seis modelos de clasificación, regresión logística, árbol de decisión, bosque aleatorio, boosting con histogramas, SVM lineal y Naive Bayes. Todos los modelos se implementaron tanto en scikit-learn como en PySpark, utilizando la misma partición de datos para poder comparar los resultados de una forma justa.

Después evaluamos el desempeño de los modelos y comparamos los resultados entre los dos entornos utilizando diferentes métricas y pruebas estadísticas. Finalmente, utilizamos LIME para entender mejor las predicciones de los modelos y conocer qué variables tuvieron mayor influencia en cada caso. Así:

| Sección | Contenido | Numeral |
|---|---|---|
| 1. EDA | Exploración del dataset completo | 9.10.4.1 |
| 2. Preprocesamiento | Partición y preprocesamiento compartidos entre entornos | 9.10.4.2–3 |
| 3. Modelado (scikit-learn) | 6 modelos, `GridSearchCV`, cv=3 | 9.10.4.4 |
| 4. Modelado (PySpark) | Los mismos 6 modelos, `CrossValidator`, numFolds=3 | 9.10.4.5 |
| 5. Prueba de DeLong | 36 comparaciones de AUC (con corrección de Holm) | 9.10.4.6.1 |
| 6. McNemar + bootstrap | Complemento a DeLong, umbral elegido solo con train | 9.10.4.6.2 |
| 7. LIME | Interpretabilidad local sobre el mejor modelo de cada entorno | 9.10.4.7 |
| 8. Conclusiones | Comparación final | 9.10.4.8 |

## Reproducibilidad

Todo el código fuente que genera estos notebooks (no solo el que aparece en las celdas, sino los
scripts de entrenamiento, las pruebas estadísticas y la preparación de datos) vive en `src/`. Los
notebooks de las secciones con entrenamientos de varias horas (3 y 4) cargan resultados ya
calculados y guardados en `outputs/tables/` y `data/`, en vez de re-entrenar en cada ejecución; las
secciones 5 a 8, más rápidas, se ejecutan en vivo de punta a punta.
