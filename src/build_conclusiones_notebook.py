import sys
sys.path.insert(0, "src")
from nbutil import new_notebook, add_md, add_code, save

nb = new_notebook()

add_md(nb, """# 8. Comparación final y reflexión crítica (9.10.4.8)

Esta sección cierra el proyecto: una sola tabla con los 12 modelos, y luego la reflexión que pide
la consigna, pregunta por pregunta, apoyada en la evidencia ya generada en las secciones 1 a 7.
""")

add_code(nb, """import sys
sys.path.insert(0, "../src")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 160)
sk = pd.read_csv("../outputs/tables/sklearn_results.csv")
sp = pd.read_csv("../outputs/tables/spark_results.csv")
sk["entorno"] = "scikit-learn"; sp["entorno"] = "PySpark"
""")

add_md(nb, "## 8.1. Tabla comparativa final: los 12 modelos en un solo vistazo")

add_code(nb, """cols = ["entorno","modelo","best_params","cv_auc","test_auc","test_f1","tiempo_entrenamiento_s"]
tabla_final = pd.concat([sk[cols], sp[cols]], ignore_index=True).sort_values("test_auc", ascending=False)
tabla_final_fmt = tabla_final.copy()
for c in ["cv_auc","test_auc","test_f1"]:
    tabla_final_fmt[c] = tabla_final_fmt[c].round(4)
tabla_final_fmt["tiempo_entrenamiento_s"] = tabla_final_fmt["tiempo_entrenamiento_s"].round(1)
tabla_final_fmt.reset_index(drop=True)
""")

add_code(nb, """fig, ax = plt.subplots(figsize=(9,6))
colores = {"scikit-learn": "#1f77b4", "PySpark": "#ff7f0e"}
for entorno, grupo in tabla_final.groupby("entorno"):
    ax.scatter(grupo["tiempo_entrenamiento_s"]/60, grupo["test_auc"], s=90, label=entorno,
               color=colores[entorno], edgecolor="black")
    for _, fila in grupo.iterrows():
        ax.annotate(fila["modelo"], (fila["tiempo_entrenamiento_s"]/60, fila["test_auc"]),
                    fontsize=8, xytext=(4,3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Tiempo de entrenamiento (minutos, escala log)")
ax.set_ylabel("AUC de prueba")
ax.set_title("AUC vs. costo de entrenamiento, los 12 modelos")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/eda/resumen_final_auc_vs_tiempo.png", dpi=110)
plt.show()
""")

add_md(nb, """El mejor modelo en **cada** entorno es un *boosting* con histogramas
(`HistGradientBoostingClassifier`/`GBTClassifier`), y ambos quedan muy cerca uno del otro
(0.739 vs. 0.733) — ninguno de los dos entornos domina de forma aplastante en desempeño. En tiempo,
la nube de puntos deja ver el patrón real: la mayoría de los modelos de PySpark quedan a la derecha
(más lentos) de su contraparte de scikit-learn, con la notable excepción de `LinearSVC`.
""")

add_md(nb, "## 8.2. Reflexión final, pregunta por pregunta")

add_md(nb, "### 8.2.1. ¿Qué entorno fue más rápido — y por qué?")

add_code(nb, """total_sk = sk["tiempo_entrenamiento_s"].sum(); total_sp = sp["tiempo_entrenamiento_s"].sum()
print(f"Total scikit-learn: {total_sk:,.1f}s (~{total_sk/60:.1f} min)")
print(f"Total PySpark:      {total_sp:,.1f}s (~{total_sp/60:.1f} min)")
print(f"scikit-learn fue {total_sp/total_sk:.2f}x mas rapido en total.")
""")

add_md(nb, """**scikit-learn ganó el total** (~3h33min vs. ~4h5min), y gana en 4 de los 6 modelos individuales.
La razón principal es el hardware: esta máquina tiene **2 núcleos**. El paralelismo distribuido de
Spark tiene un costo fijo de coordinación (miles de *stages*, serialización entre tareas,
planificación del *scheduler*) que solo se amortiza cuando hay muchos núcleos (o muchos nodos) entre
los que repartir el trabajo — con 2, ese costo casi nunca se paga a sí mismo. La excepción es
`LinearSVC`: ahí Spark fue **8.5 veces más rápido** y con mejor AUC, pero no por ventaja de
paralelismo — fue porque scikit-learn colapsó a una solución degenerada que, irónicamente, seguía
siendo costosa de calcular (`liblinear` iteró 633 veces igual). `DecisionTree`, ya con el bug de
`rawPrediction`/`probability` corregido (sección 4.2.2), también fue ~2 veces más rápido en Spark —
esa sí es una ventaja de Spark limpia, aunque modesta y aislada a un solo modelo.
""")

add_md(nb, "### 8.2.2. ¿Qué entorno fue más preciso (AUC)?")

add_md(nb, """No hay un ganador único — depende de la familia de modelo:

| Familia | Gana | ΔAUC (sk − sp) | ¿Prácticamente significativo (≥0.005)? |
|---|---|---|---|
| LogisticRegression | (empate) | -0.0011 | No |
| DecisionTree | scikit-learn | +0.0164 | Sí |
| RandomForest | (empate) | +0.0039 | No |
| GBT | scikit-learn | +0.0066 | Sí |
| LinearSVC | PySpark | -0.1443 | Sí (colapso de sklearn) |
| NaiveBayes | PySpark | -0.0055 | Sí (por poco) |

scikit-learn gana con margen real en 2 familias (`DecisionTree`, `GBT`), PySpark gana con margen
real en 2 (`LinearSVC`, `NaiveBayes` — aunque la de `LinearSVC` es por un colapso de scikit-learn,
no por una ventaja algorítmica de Spark), y en las otras 2 (`LogisticRegression`, `RandomForest`)
son estadísticamente distintos pero prácticamente equivalentes. **El mejor modelo global es
`HistGradientBoostingClassifier` de scikit-learn (AUC=0.739)**, apenas por encima del `GBTClassifier`
de Spark (AUC=0.733) — ambos, no por casualidad, el mismo tipo de algoritmo.
""")

add_md(nb, "### 8.2.3. Significancia estadística vs. práctica")

add_md(nb, """Con 452.141 filas de prueba, la significancia estadística deja de ser informativa por sí sola: de
las **44 comparaciones de AUC** hechas en este proyecto (36 de DeLong en la sección 5 + 8 entre
McNemar/bootstrap en la sección 6), prácticamente todas rechazan $H_0$ — el tamaño de muestra por
sí solo garantiza detectar cualquier ΔAUC no nulo, sin importar cuán chico. El umbral de
significancia **práctica** declarado de antemano (|ΔAUC|≥0.005, sección 5) es lo que realmente
separa las diferencias que importan de las que no: de las 6 comparaciones entre entornos, 4 son
prácticamente relevantes (`DecisionTree`, `GBT`, `LinearSVC`, `NaiveBayes`) y 2 no lo son
(`LogisticRegression`, `RandomForest`), aunque las 6 sean "significativas". La lección metodológica
central del proyecto: **declarar el umbral de relevancia práctica antes de mirar los datos** es lo
único que evita concluir "son diferentes" cuando en realidad la diferencia es demasiado chica para
cambiar qué modelo se elegiría en producción.
""")

add_md(nb, "### 8.2.4. Diferencias de implementación que explican las discrepancias")

add_md(nb, """Dos de las discrepancias que en un primer momento parecían diferencias *algorítmicas* entre
librerías resultaron ser **bugs propios**, no diferencias reales:

- **`DecisionTree` (sección 4.2.2):** usar `rawPredictionCol="rawPrediction"` para seleccionar y
  evaluar el modelo, cuando el puntaje realmente guardado venía de `"probability"` — para un árbol
  individual esas dos columnas no son equivalentes en *ranking* (sí lo son para modelos de
  conjunto). Redujo la brecha de 14 puntos de AUC a menos de 2. La diferencia residual (~1.7 puntos)
  es plausiblemente real, atribuible a `maxBins=32` (Spark discretiza las variables continuas en
  como máximo 32 puntos de corte candidatos; scikit-learn evalúa cualquier punto real).
- **`NaiveBayes` (sección 4.6c):** el mismo bug, exactamente el mismo mecanismo — y al corregirlo,
  la "diferencia real de implementación" que se había documentado inicialmente (0.564 vs. 0.694)
  desapareció por completo (0.699 vs. 0.694, prácticamente empatados).

Una diferencia que **sí** resultó ser real y bien entendida: `LinearSVC` colapsó en scikit-learn
(`liblinear`, descenso de coordenadas dual) pero no en Spark (OWL-QN, cuasi-Newton) — dos
optimizadores distintos explorando la misma superficie de pérdida de forma distinta, y uno de ellos
quedó atrapado en una solución degenerada para este problema desbalanceado (sección 3.7/4.6b).

Y una diferencia de implementación que no afecta el AUC pero sí la interpretabilidad (sección 7.1):
el `OneHotEncoder` de scikit-learn y el de Spark, con la misma configuración conceptual, producen
espacios de variables de **distinta dimensión** (114 vs. 107) por cómo cada uno trata la categoría
"nula"/no vista.

La lección que se repite en los tres casos: **un error de implementación propio puede parecer, a
primera vista, una diferencia legítima entre librerías** — solo se distingue verificando
explícitamente, modelo por modelo, que el puntaje reportado coincide con el que realmente se usa en
el resto del pipeline.
""")

add_md(nb, "### 8.2.5. DeLong vs. McNemar/bootstrap: ¿coinciden, y qué le falta a DeLong?")

add_code(nb, """delong_between = pd.read_csv("../outputs/tables/delong_between_envs.csv")
mcnemar_res = pd.read_csv("../outputs/tables/mcnemar_resultados.csv")
boot_res = pd.read_csv("../outputs/tables/bootstrap_resultados.csv")
print(f"DeLong:            {len(delong_between)+30} comparaciones (36 en total, seccion 5), todas rechazan H0 tras Holm")
print(f"McNemar:           {len(mcnemar_res)} comparaciones (seccion 6), {mcnemar_res['rechaza_h0_holm_0.05'].sum()}/{len(mcnemar_res)} rechazan H0 tras Holm")
print(f"Bootstrap pareado: {len(boot_res)} comparaciones (seccion 6), {boot_res['auc_excluye_cero'].sum()}/{len(boot_res)} excluyen 0 en el IC95%")
""")

add_md(nb, """Las tres pruebas coinciden en dirección y en significancia estadística en las 8 comparaciones que
tienen en común (tabla resumen de la sección 6.5) — una buena señal de consistencia interna, aunque
esperable con este tamaño de muestra. DeLong compara *ranking* (AUC) con una fórmula analítica
exacta; McNemar compara *aciertos a un umbral fijo* (complementario, no un sustituto: puede cambiar
de conclusión si cambia el umbral); el bootstrap da un intervalo de confianza empírico, sin asumir
normalidad asintótica, para AUC **y** para métricas que DeLong no cubre (AUC-PR, F1). Las
limitaciones específicas de DeLong (solo AUC, asintótica, sensible al tamaño de muestra — sección
5.8) son exactamente lo que McNemar y el bootstrap vienen a compensar.
""")

add_md(nb, "### 8.2.6. ¿A qué volumen de datos empezaría PySpark a superar a scikit-learn?")

add_md(nb, """Con los datos de este proyecto (1.8M filas de entrenamiento, ~114 columnas ya con one-hot, que
caben cómodamente en los 4 GB de memoria asignados) no hay ninguna razón estructural para que Spark
gane: el cuello de botella no es memoria ni cómputo que no entre en una sola máquina, así que el
paralelismo distribuido solo añade overhead de coordinación sin nada que compensarlo. La ventaja de
Spark aparecería en dos escenarios, ninguno presente aquí: **(a)** un dataset que no entra en la
memoria de una sola máquina (varias decenas de GB o más, forzando *spilling* a disco o
imposibilitando `scikit-learn` por completo), o **(b)** un clúster real con muchos nodos/núcleos
(no 2, sino decenas o cientos), donde el trabajo sí se reparte de forma efectiva y el costo fijo de
coordinación se amortiza sobre mucho más cómputo útil. Ninguna de las dos condiciones se cumple con
2 núcleos y ~500 MB de datos ya preprocesados — así que, en este proyecto, no hay ningún punto de la
curva de tamaño de datos donde Spark hubiera empezado a ganar sin cambiar también el hardware.
""")

add_md(nb, "### 8.2.7. Contribución y limitaciones de LIME, especialmente sobre un modelo distribuido")

add_md(nb, """LIME sí aportó algo que ninguna de las pruebas globales (DeLong, McNemar, bootstrap) puede dar:
una explicación **por caso individual**, útil para entender un error concreto del modelo (sección
7.2–7.3) en vez de solo su desempeño agregado. El hallazgo más interesante no fue una variable en
particular, sino un patrón que se repitió en los dos entornos: el ajuste lineal local de LIME fue
sistemáticamente peor ($R^2$≈0.09) para los falsos negativos que para los falsos positivos
($R^2$≈0.36–0.44) — indicio de que los "incumplimientos sorpresa" caen en zonas más no lineales del
espacio de variables. Las limitaciones son las mismas de siempre (inestabilidad por muestreo
aleatorio, ruido de docenas de columnas *dummy* con peso marginal) más una específica de Spark: no
existe una implementación nativa, así que hubo que escribir a mano un adaptador que convierte cada
lote de muestras perturbadas a `DataFrame`, lo pasa por `.transform()`, y lo trae de vuelta a NumPy
(sección 7.3) — funcionó bien para explicar un puñado de casos (1–3 segundos cada uno), pero ese
costo por instancia haría impráctico generar explicaciones para miles de casos de forma rutinaria
sin un rediseño (por ejemplo, procesar muchas instancias en un solo lote en vez de una por una).
""")

add_md(nb, "### 8.2.8. Efecto de cada condición obligatoria de la consigna")

add_md(nb, """- **Split y preprocesamiento compartidos entre los dos entornos:** es la condición que hizo posible
  todo lo demás — sin las mismas 452.141 filas de prueba en el mismo orden lógico (mismos `id`), no
  existiría DeLong, McNemar ni el bootstrap pareado tal como se aplicaron (los tres dependen de
  comparar observaciones *pareadas*, no muestras independientes).
- **`cv=3` / `numFolds=3`** (en vez de 5 o 10): más barato — indispensable dado que cada
  combinación de hiperparámetros para `RandomForest`/`GBT` ya tomaba varios minutos — pero más
  ruidoso. Es una posible explicación adicional (no solo `maxBins`) de por qué `DecisionTree` eligió
  una profundidad distinta en cada entorno (10 en scikit-learn, 15 en Spark): con solo 3 particiones,
  la estimación de AUC de validación cruzada tiene más varianza, y el óptimo elegido puede
  desplazarse de una corrida a otra.
- **Sin objeto `Pipeline` en scikit-learn** (preprocesamiento explícito y separado, sección 2): más
  transparente y más fácil de auditar paso a paso — el costo es más responsabilidad manual de
  mantener sincronizados train/test y de no filtrar información (algo que sí se vigiló activamente,
  p. ej. en la elección del umbral de Youden solo con datos de entrenamiento, sección 6.1).
- **Configuración de Spark reducida y justificada para el hardware real** (2 núcleos, 4g/4g,
  `shuffle.partitions=8` en vez de los 400 "de fábrica"): documentada explícitamente en
  `src/model_pyspark.py`. Es, en el fondo, la misma razón detrás de la respuesta a 8.2.1 y 8.2.6 —
  con este hardware, cualquier configuración de Spark iba a cargar con el mismo techo estructural.
""")

add_md(nb, "## 8.3. Conclusión general")

add_md(nb, """Los dos entornos llegan a un techo predictivo muy parecido (~0.72–0.74 de AUC), con el mismo tipo
de modelo (*boosting* con histogramas) ganando en ambos — la elección de scikit-learn vs. PySpark
para este dataset en particular (1.8M filas, una sola máquina de 2 núcleos) no cambia la conclusión
de negocio de forma sustancial. Lo que sí cambia, y de forma importante, es **qué tan seguro se
puede estar** de esa conclusión: la comparación rigurosa (mismo split, tres pruebas estadísticas
complementarias, un umbral de significancia práctica declarado de antemano) fue lo que permitió
distinguir diferencias reales (el colapso de `LinearSVC`, la ventaja genuina pero modesta de
`DecisionTree` en Spark) de errores de implementación propios que, sin esa verificación cruzada,
habrían quedado documentados —incorrectamente— como hallazgos sobre las librerías. Encontrar y
corregir esos dos bugs (secciones 4.2.2 y 4.6c) es, en un sentido real, el resultado más valioso de
todo el ejercicio: más que cualquier número de AUC, es la evidencia de que la metodología de
comparación exigida por la consigna cumplió exactamente la función para la que está pensada.
""")

save(nb, "book/08_conclusiones.ipynb")
print("Notebook 08_conclusiones.ipynb creado.")
