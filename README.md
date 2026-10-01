# Tarea 1: Proyecto Integrador de Aprendizaje Automático

**Predicción de incumplimiento de pago - Lending Club Loan Data (2007-2020)**

Integrantes: Santiago Díaz, Gina Huguet y Leiry Mares. Maestría en Ingeniería Biomédica, Universidad del Norte.

## Entregables

1. **Jupyter Book (link):** https://ghuguetb.github.io/lending-club-ml-project/
2. **Notebooks compilados (.ipynb):** carpeta [`book/`](book/)

## Contenido

| Sección | Notebook | Contenido |
|---|---|---|
| 1 | [`01_eda.ipynb`](book/01_eda.ipynb) | Exploración del dataset completo |
| 2 | [`02_preprocesamiento.ipynb`](book/02_preprocesamiento.ipynb) | Partición y preprocesamiento compartidos entre entornos |
| 3 | [`03_modelado_sklearn.ipynb`](book/03_modelado_sklearn.ipynb) | 6 modelos en scikit-learn, `GridSearchCV`, cv=3 |
| 4 | [`04_modelado_pyspark.ipynb`](book/04_modelado_pyspark.ipynb) | Los mismos 6 modelos en PySpark, `CrossValidator`, numFolds=3 |
| 5 | [`05_delong.ipynb`](book/05_delong.ipynb) | 36 comparaciones de AUC (prueba de DeLong, corrección de Holm) |
| 6 | [`06_mcnemar_bootstrap.ipynb`](book/06_mcnemar_bootstrap.ipynb) | Complemento a DeLong: McNemar + bootstrap |
| 7 | [`07_lime.ipynb`](book/07_lime.ipynb) | Interpretabilidad local (LIME) sobre el mejor modelo de cada entorno |
| 8 | [`08_conclusiones.ipynb`](book/08_conclusiones.ipynb) | Comparación final |

## Estructura del repositorio

- `book/` — notebooks finales (con salidas) + configuración del Jupyter Book.
- `src/` — código fuente que genera los notebooks: scripts de entrenamiento, pruebas estadísticas y preparación de datos.
- `docs/` — sitio del Jupyter Book ya construido, publicado con GitHub Pages.

Los datos crudos y los artefactos de modelos entrenados (varios GB) no se incluyen en este repositorio por su tamaño.

## Reproducibilidad

Todo el código fuente que genera estos notebooks (no solo el que aparece en las celdas, sino los scripts de entrenamiento, las pruebas estadísticas y la preparación de datos) vive en `src/`. Los notebooks de las secciones con entrenamientos de varias horas (3 y 4) cargan resultados ya calculados y guardados en `outputs/tables/` y `data/`, en vez de re-entrenar en cada ejecución; las secciones 5 a 8, más rápidas, se ejecutan en vivo de punta a punta.
