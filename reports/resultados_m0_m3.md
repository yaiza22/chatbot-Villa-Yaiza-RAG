# Resultados de la evaluación de los motores M0–M3

**Estado:** resultados finales. El *test* (243 preguntas reales) se evaluó **una sola vez**, con los motores y su configuración congelados antes de la corrida. La partición *train* (563 preguntas) se usó para desarrollar y elegir la configuración: umbrales, modelo de embeddings, k de M3 y la combinación de motores a comparar.

**Pendiente:** la calificación manual de **corrección y alucinación de M3** (sección 11). Las demás cifras ya son definitivas.

Todas las cifras salen de archivos de `reports/`, y junto a cada una se indica el archivo y la columna de origen. Para regenerar tablas y figuras se usa `notebooks/07_resultados_m0_m3_ng.ipynb`, con `SPLIT = "train"` o `SPLIT = "test"`.

---

## 1. Fuentes

| Contenido | Archivo | Columnas clave |
|---|---|---|
| Métricas oficiales en test | `reports/eval_test/{clasicos,m3}/results_2026-11-17.csv` | `intent_accuracy`, `correct_abstention`, `unnecessary_abstention`, `hit_at_1/3/5`, `latency_median_ms`, `latency_p95_ms` y sus `*_ci_low` / `*_ci_high` |
| Métricas oficiales en train | `reports/eval_train/{clasicos,m3}/results_2026-11-17.csv` | las mismas |
| Matrices de confusión | `reports/eval_{train,test}/{clasicos,m3}/confusion_matrix_2026-11-17_<motor>_real_<split>.csv` | filas = `true_intent`, columnas = intención predicha |
| Validación fuera de pliegue de M1 y M2 (train) | `reports/eval_train/oof/` | `final_fit_choices.csv` (configuración final), `oof_folds.csv` (elección por pliegue), `oof_predictions.csv`, `threshold_tradeoff_M1.csv` y `threshold_tradeoff_M2.csv` |
| Selección del modelo de embeddings | `reports/m2_model_selection.csv` | `hit_at_1/3/5`, `auroc_answerable_vs_escalating`, `median_top1_sim_*` |
| Tablas derivadas | `reports/tables/{train,test}/*.csv` y `*.tex` | ver cada sección |
| Figuras | `reports/figures/{train,test}/*.png` y `*.pdf` | ver sección 13 |

**Verificación.** El notebook recalcula las métricas a partir de las predicciones por pregunta y las compara con `results_*.csv`. En train y en test coinciden las 18 combinaciones motor × métrica (diferencia 0,0) y las 4 matrices de confusión reconstruidas son idénticas a las oficiales.

**Hardware.** Toda la latencia se midió en el mismo equipo: AMD Ryzen 5 8600HS, NVIDIA RTX 4050 6 GB, 64 GB RAM (columna `hardware`).

## 2. Datos

`data/processed/labeled.csv` contiene 806 preguntas reales anonimizadas de chats de WhatsApp. La partición es 70/30, estratificada por intención y con semilla 42 (`tables/test/tabla_datos.csv`).

| Intención | ¿Escala? | train | test | total |
|---|---|---|---|---|
| precio | sí | 74 | 32 | 106 |
| capacidad | no | 17 | 8 | 25 |
| reglas_servicios | no | 25 | 10 | 35 |
| ubicacion | no | 36 | 16 | 52 |
| disponibilidad | sí | 188 | 81 | 269 |
| otro_negocio | sí | 178 | 76 | 254 |
| fuera_de_alcance | sí | 45 | 20 | 65 |
| **Total** | | **563** | **243** | **806** |

En test, **209 preguntas deben escalar (86 %) y 34 son respondibles (14 %)**. Todas las respondibles tienen `kb_source` (columnas `n_should_escalate`, `n_respondable`, `n_kb`).

**Consecuencia del desbalance.** Un motor que **siempre escala** acierta la decisión escalar/responder en el 86 % de las preguntas sin dar ninguna respuesta útil. Por eso no basta el acierto global: se reportan juntas la abstención correcta y la innecesaria, y su diferencia **J = abstención correcta − abstención innecesaria**. J vale 0 para "siempre escalar" y 1 para un motor perfecto. Es el mismo criterio con el que M1 y M2 eligen su umbral.

## 3. Qué se evalúa en cada motor y por qué

| Métrica | M0 | M1 | M2 | M3 | Motivo |
|---|---|---|---|---|---|
| Accuracy de intención (todas) | sí | sí | no comparable | no comparable | M2 y M3 toman la intención del fragmento top-1. La base solo tiene fragmentos de `capacidad`, `reglas_servicios` y `ubicacion`, así que no pueden predecir intenciones que escalan; su techo es n_respondibles / n_total (34/243 = 0,14 en test). Para ellos se reporta la accuracy **solo sobre respondibles**. |
| Hit@k (k = 1, 3, 5) | no aplica | no aplica | sí | sí (= M2) | M0 y M1 no recuperan fragmentos. M3 usa el recuperador de M2, así que `retrieved_str`, Hit@k y la matriz de confusión son idénticos. |
| Abstención correcta / innecesaria, J | sí | sí | sí | sí | Es la decisión escalar/responder, común a todos. |
| Corrección | automática | automática | automática | manual (pendiente) | M0 y M1 devuelven la plantilla validada de la intención y M2 el fragmento sin redactar. Su respuesta es correcta si no escaló y acertó la intención (M0/M1) o el fragmento top-1 = `kb_source` (M2). M3 redacta texto libre y se califica a mano. |
| Alucinación | 0 por construcción | 0 por construcción | 0 por construcción | manual (pendiente) | M0–M2 solo eligen entre textos ya validados de la base. |
| Latencia | sí | sí | sí | sí | Mediana y percentil 95 en el mismo equipo. |

**Diferencia de exigencia en la corrección automática.** Las plantillas de M0 y M1 reúnen todos los temas de una intención: por ejemplo, todas las reglas de la finca en un solo texto. Por eso, en M0 y M1, "correcta" significa que la información está dentro de la plantilla. M2 debe acertar el fragmento exacto, un criterio más estricto.

**M1 sobre train.** El arnés evalúa M1 con las mismas filas con las que se entrena (resustitución) y obtiene 1,00 en todo. **Ese valor no se usa.** En train se reporta la validación cruzada fuera de pliegue (sección 8).

## 4. Configuración final de los motores (elegida solo con train)

| Motor | Configuración | Cómo se eligió | Fuente |
|---|---|---|---|
| M0 | Reglas por palabras clave, construidas con la taxonomía y preguntas sintéticas, y congeladas antes de etiquetar los datos reales | No se entrena | `src/engines/m0_keywords.json` |
| M1 | TF-IDF (1–2 gramas) + regresión logística; **C = 100**, **umbral de probabilidad = 0,15** | Validación cruzada de 5 pliegues dentro de `fit()`, maximizando J | `eval_train/oof/final_fit_choices.csv` (`best_c_`, `threshold`, `cv_folds_`) |
| M2 | `intfloat/multilingual-e5-small`, un vector por párrafo, **umbral de similitud = 0,83**, top-k = 5 | Modelo: mayor Hit@1 sobre preguntas reales de train. Umbral: máximo J sobre train | `m2_model_selection.csv`; `final_fit_choices.csv` |
| M3 | Recuperador de M2 + Qwen 3.6 (35B-A3B) local en Ollama, k = 3 fragmentos en el contexto | k comparado con 12 preguntas de train (k = 3 frente a k = 5) | `src/config.py` |

**La elección es estable entre pliegues** (`oof_folds.csv`):
- M1 eligió C = 100 en 4 de 5 pliegues y umbral 0,15 en 4 de 5.
- M2 eligió umbral 0,83 en 4 pliegues y 0,84 en el quinto.

**Selección del modelo de embeddings** (`m2_model_selection.csv`, filas `source = real`, 78 respondibles y 485 que escalan):

| Modelo (un vector por párrafo) | Hit@1 | Hit@3 | Hit@5 | AUROC respondible vs. escala |
|---|---|---|---|---|
| **multilingual-e5-small** (elegido) | **0,487** | **0,731** | **0,833** | 0,687 |
| paraphrase-multilingual-mpnet-base-v2 | 0,436 | 0,679 | 0,718 | 0,763 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0,423 | 0,731 | 0,769 | 0,737 |

- **El elegido recupera mejor, pero es el que peor separa** lo respondible de lo que debe escalar (AUROC 0,687). La mediana de similitud top-1 es 0,853 en respondibles y 0,842 en las que escalan (`median_top1_sim_*`): sus similitudes quedan muy comprimidas.
- **Ningún modelo supera un AUROC de 0,77**, así que la similitud por sí sola no basta para decidir si escalar.
- **Diferencia entre preguntas sintéticas y reales:** con preguntas sintéticas (`source = synthetic`), e5-small alcanza Hit@1 = 0,846, frente a 0,487 con preguntas reales. Las preguntas escritas por el equipo son más limpias que las de WhatsApp.

## 5. Resultados en test (resultado final)

Los valores se presentan como punto [IC 95 % bootstrap, 2000 remuestreos, semilla 42]. Fuente: `tables/test/tabla_metricas.csv`.

| Motor | Accuracy intención | Accuracy intención (respondibles) | Abstención correcta | Abstención innecesaria | **J** | Correctas (respondibles) | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 0,60 [0,54–0,66] | 0,53 [0,35–0,71] | 0,92 [0,89–0,96] | 0,38 [0,21–0,56] | **0,54 [0,37–0,69]** | 0,53 [0,35–0,71] | — | — | — |
| M1 | 0,68 [0,62–0,74] | 0,32 [0,18–0,50] | 0,98 [0,96–1,00] | 0,62 [0,44–0,76] | 0,36 [0,20–0,54] | 0,32 [0,18–0,50] | — | — | — |
| M2 | 0,11 (techo 0,14) | 0,79 [0,65–0,91] | 0,32 [0,25–0,38] | 0,15 [0,03–0,29] | 0,17 [0,03–0,30] | 0,35 [0,21–0,53] | 0,38 [0,24–0,56] | 0,68 [0,50–0,82] | 0,74 [0,59–0,88] |
| M3 | 0,11 (techo 0,14) | 0,79 [0,65–0,91] | 0,62 [0,55–0,68] | 0,18 [0,06–0,32] | 0,44 [0,28–0,57] | manual | 0,38 | 0,68 | 0,74 |
| *Siempre escalar (referencia)* | — | — | 1,00 | 1,00 | 0,00 | 0,00 | — | — | — |

**Qué hace cada motor con cada pregunta en test** (`tables/test/tabla_escalar_responder.csv`; figura `escalation_outcomes`):

| Motor | Debía escalar: escaló | Debía escalar: **respondió** | Respondible: correcta | Respondible: incorrecta | Respondible: sin calificar | Respondible: escaló |
|---|---|---|---|---|---|---|
| M0 | 193 | **16 (8 %)** | 18 | 3 | 0 | 13 |
| M1 | 205 | **4 (2 %)** | 11 | 2 | 0 | 21 |
| M2 | 66 | **143 (68 %)** | 12 | 17 | 0 | 5 |
| M3 | 129 | **80 (38 %)** | — | — | 28 | 6 |

**Latencia en test.** Medianas tomadas de `latency_median_ms` de `results_*.csv`; los p95 están en `tables/test/tabla_latencia.csv`.

| Motor | M0 | M1 | M2 | M3 |
|---|---|---|---|---|
| Mediana | 0,11 ms | 0,32 ms | 15,2 ms | 4,25 s |

En train, en el mismo equipo, las medianas fueron 0,23 ms, 0,95 ms, 29,4 ms y 4,8 s (p95 de M3: 8,0 s). En ambas corridas, M3 es **de dos a tres órdenes de magnitud más lento** que M2 y unas 10 000 veces más lento que M1.

### Comparaciones pareadas en test (mismas 243 preguntas)

Fuente: `tables/test/tabla_comparaciones_pareadas.csv` (`diff_a_menos_b`, `ci_low`, `ci_high`, `ic_incluye_0`). Una diferencia es "distinguible" cuando su IC 95 % no incluye 0.

| Comparación | Métrica | a − b [IC 95 %] | ¿Distinguible? |
|---|---|---|---|
| M0 − M1 | J | +0,178 [−0,059–0,397] | No |
| M0 − M2 | J | +0,372 [0,184–0,552] | **Sí** |
| M0 − M3 | J | +0,100 [−0,094–0,296] | No |
| M1 − M2 | J | +0,194 [0,001–0,392] | Sí (al límite) |
| M1 − M3 | J | −0,078 [−0,268–0,120] | No |
| M3 − M2 | J | +0,272 [0,115–0,433] | **Sí** |
| M0 − M1 | Abstención correcta | −0,057 [−0,100 – −0,019] | Sí (M1 escala más) |
| M0 − M3 | Abstención correcta | +0,306 [0,239–0,378] | **Sí** |
| M3 − M2 | Abstención correcta | +0,301 [0,206–0,388] | **Sí** |
| M0 − M1 | Abstención innecesaria | −0,235 [−0,441 – −0,029] | Sí (M1 escala más respondibles) |
| M0 − M3 | Abstención innecesaria | +0,206 [0,029–0,382] | Sí (M0 escala más respondibles) |
| M0 − M1 | Accuracy de intención | −0,078 [−0,144 – −0,012] | Sí (M1 acierta más la intención) |
| M0 − M1 | Correctas (respondibles) | +0,206 [0,000–0,412] | No (al límite) |
| M0 − M2 | Correctas (respondibles) | +0,176 [−0,029–0,353] | No |

Con solo 34 preguntas respondibles en test, las diferencias que dependen de ellas tienen IC anchos: la abstención innecesaria, J y la corrección.

## 6. Combinaciones de motores

**Diseño.** Un clasificador (M0 o M1) decide si escalar. Si no escala, otro motor (M2 o M3) responde. El resultado se **simula** a partir de las predicciones guardadas: la pregunta escala si escala cualquiera de los dos. La comparación se fijó con train **antes** de evaluar el test, y en train se usaron las predicciones de M1 fuera de pliegue.

Test (`tables/test/tabla_combinaciones.csv`; figura `escalation_outcomes_combined`):

| | Abstención correcta | Abstención innecesaria | J | Debía escalar y respondió | Respondibles que respondió | Correctas (respondibles) |
|---|---|---|---|---|---|---|
| M0 | 0,92 | 0,38 | 0,54 | 8 % | 62 % | 0,53 |
| M3 | 0,62 | 0,18 | 0,44 | 38 % | 82 % | manual |
| **M0 + M3** | **0,95 [0,92–0,98]** | 0,44 [0,26–0,62] | 0,51 [0,34–0,67] | **5 %** | **56 %** | manual |
| M0 + M2 | 0,94 [0,91–0,97] | 0,41 [0,24–0,59] | 0,53 [0,36–0,69] | 6 % | 59 % | 0,29 [0,15–0,47] |
| M1 + M2 | 0,99 | 0,65 | 0,34 | 1 % | 35 % | 0,15 |
| M1 + M3 | 1,00 | 0,65 | 0,35 | 0 % | 35 % | manual |

En train la simulación dio el mismo orden (`tables/train/tabla_combinaciones.csv`): M0 + M3 obtuvo J = 0,51, respondió el 8 % de lo que debía escalar y el 59 % de lo respondible.

## 7. Análisis por motor

### M0: reglas por palabras clave (el método actual de la finca, equivalente a las respuestas rápidas)
- **Tiene el mejor equilibrio** en test: J = 0,54. Escala el 92 % de lo que debe escalar y responde bien 18 de 34 respondibles.
- **Escala sin necesidad el 38 % de las respondibles.** Cuando ninguna palabra clave coincide, la pregunta cae en `otro_negocio` y escala. En train, `reglas_servicios` tuvo recall de 0,28 (`tables/train/tabla_por_intencion.csv`).
- **Confunde intenciones que escalan entre sí** (por ejemplo, nunca acierta `fuera_de_alcance`), pero el cliente no lo nota: ambas escalan y el resultado es el mismo.

### M1: TF-IDF + regresión logística
- **Es el más seguro:** solo responde el 2 % de lo que debía escalar y tiene la mejor accuracy de intención (0,68).
- **Es el más conservador:** escala el **62 % de las respondibles** y responde bien apenas 11 de 34.
- **Por qué** (matriz `confusion_M1_oof` de train): hay muy pocos ejemplos respondibles para entrenar (17, 25 y 36, frente a cientos de los que escalan). El 60 % de `reglas_servicios` se clasifica como `otro_negocio` y el 29 % de `capacidad` como `precio`.
- **La estimación fuera de pliegue en train predijo bien el test** (sección 8).

### M2: embeddings + umbral de similitud
- **Recupera de forma aceptable:** Hit@3 = 0,68 en test y 0,73 en train. Entre las respondibles acierta la intención el 79 %.
- **Decide muy mal cuándo escalar:** responde con un fragmento el **68 % de las preguntas que debían escalar**. El umbral de similitud no detecta intenciones: una pregunta de precio o de fechas que menciona personas o el lugar se parece a un fragmento de capacidad o de ubicación.
- **Al devolver el top-1 sin redactar, se equivoca la mitad de las veces que responde:** 12 correctas y 17 incorrectas en test.

### M3: RAG con LLM local
- **El LLM corrige parte del problema de M2.** La abstención correcta sube de 0,32 a 0,62 (+0,30 [0,21–0,39]) sin aumentar de forma distinguible la innecesaria. J sube de 0,17 a 0,44.
- **Aun así, responde el 38 % de lo que debía escalar** (80 de 209). Como sistema único no es seguro para producción.
- **Cuánto escala en realidad es incierto** (`tables/test/tabla_m3_sensibilidad.csv`):
  - en 1 respuesta escribió "ESCALAR" sin que el motor lo registrara;
  - cerca del 31 % de sus respuestas remiten a una persona sin marcar `escalate`. Es una estimación por palabras clave, no una medida.
  - Si se contaran esas respuestas como escalamientos, la abstención correcta sería ~0,76 y la innecesaria ~0,32.
- **El fragmento correcto llegó al LLM en el 68 % de las respondibles** (Hit@3). Ese es el techo de su corrección.

## 8. Consistencia entre train y test

| Motor | J train | J test | Abst. innecesaria train | Abst. innecesaria test |
|---|---|---|---|---|
| M0 | 0,51 | 0,54 | 0,40 | 0,38 |
| M1 (train: fuera de pliegue) | 0,34 | 0,36 | 0,64 | 0,62 |
| M2 (train: fuera de pliegue) | 0,23 | 0,17 | 0,13 | 0,15 |
| M3 | 0,48 | 0,44 | 0,09 | 0,18 |

Fuentes: `tables/train/tabla_metricas_oof.csv` y `tables/test/tabla_metricas.csv`.

- **El orden de los motores se mantiene** (M0 ≥ M3 > M1 > M2) y los valores cambian poco.
- **La validación fuera de pliegue de M1 anticipó su resultado en test:**

  | M1 | Accuracy de intención | Abstención correcta | Abstención innecesaria |
  |---|---|---|---|
  | Train, fuera de pliegue | 0,69 | 0,98 | 0,64 |
  | Test | 0,68 | 0,98 | 0,62 |

  El 1,00 de la evaluación por resustitución, en cambio, no se acercaba al resultado real.
- **M3 es la única excepción notable:** su abstención innecesaria pasa de 0,09 a 0,18. En test eso son 6 de 34 preguntas, y el IC es ancho [0,06–0,32].

## 9. Conclusiones

1. **La tarea central es decidir cuándo escalar:** el 86 % de las consultas reales debe pasar a una persona. En esa tarea, **el método más simple, M0, obtiene el mejor equilibrio** (J = 0,54). No se distingue estadísticamente de M3 (0,44) ni de M1 (0,36), pero sí de M2 (0,17). Esto se reporta tal cual: el motor complejo no supera al simple.
2. **Recuperar fragmentos no sirve para decidir si escalar.** La base solo cubre temas respondibles y la similitud separa mal ambos grupos (AUROC ≤ 0,76). M2 responde dos de cada tres consultas que debían escalar.
3. **El LLM mejora la decisión de M2, pero no basta por sí solo:** M3 responde el 38 % de lo que debía escalar, y cuesta 4,3 s por respuesta con GPU.
4. **Los motores se complementan.** Con M0 decidiendo si escalar y M3 redactando, solo se responde el 5 % de lo que debía escalar (frente al 38 % de M3 solo) y se mantiene el equilibrio de M0 (J = 0,51). En la práctica, el LLM solo se llama para las preguntas que M0 no escala, así que la mayoría de las consultas no pagan su latencia. Esta combinación es la candidata a bot final, **siempre que la calificación manual confirme que las respuestas de M3 son correctas y no alucinan**. Si no lo confirma, M0 sigue siendo la opción más sólida.
5. **Respuesta a la pregunta de investigación** (¿en qué medida un agente RAG con LLM local supera a métodos más simples?):
   - **En abstención correcta, no los supera:** M3 queda por debajo de M0 y de M1.
   - **En calidad de respuesta,** la conclusión depende de la calificación manual pendiente.
6. **Metodología:** la validación cruzada fuera de pliegue estimó bien el rendimiento de M1 en test. Evaluar sobre las mismas filas de entrenamiento lo habría sobrestimado por completo.

## 10. Limitaciones

- **Pocas preguntas respondibles** (78 en train y 34 en test). Los IC de abstención innecesaria, J y corrección son anchos, y varias diferencias no son concluyentes.
- **Etiquetado:** las etiquetas las puso una sola persona, sin doble etiquetado; el error de etiqueta no está medido.
- **Solo texto:** solo se evaluaron mensajes de texto. Los audios de los chats quedaron fuera.
- **Reutilización parcial del test:** 68 preguntas de test ya se habían usado una vez en una evaluación preliminar de M0 y M1, sin ajustar nada con ellas. Parte de esas filas se reclasificó después con la base de conocimiento.
- **El k de M3 se eligió con una muestra pequeña** (12 preguntas de train).
- **Lectura de la salida de M3:** su escalamiento depende de cómo se interpreta el texto del LLM (sección 7), y la estimación de respuestas que remiten a una persona es heurística.
- **Selección del modelo de embeddings:** se eligió por recuperación (Hit@1), aunque otro modelo separa mejor las preguntas que escalan.
- **Datos sintéticos:** se evaluaron solo para la recuperación (`m2_model_selection.csv`), no para todos los motores.
- **Combinaciones simuladas:** se simularon con predicciones guardadas; su latencia no se midió directamente.

## 11. Pendiente: corrección y alucinación de M3

- **Quiénes califican:** dos personas distintas a quien construyó M3.
- **Qué califican:** en test, todas las respuestas de M3 a preguntas respondibles (28) más una muestra aleatoria de 40 respuestas (semilla 42) a preguntas que debían escalar y que M3 respondió.
- **Escala:** corrección 0, 1 o 2 frente a `reference_answer`; alucinación 0 o 1 frente a `data/kb/`.
- **Resultados que se agregan aquí:** la media con IC y el acuerdo entre calificadores.

## 12. Figuras y tablas

Las figuras están en `reports/figures/<split>/`: PNG para diapositivas y PDF para el artículo, con textos en inglés. **Para el resultado final usar `test`**; las de `train` documentan el desarrollo.

| Archivo | Uso sugerido | Pie de figura sugerido (artículo) |
|---|---|---|
| `test/escalation_outcomes` | Diapositiva principal | Resultado por pregunta en test: escalamiento de las consultas que lo requieren (izq.) y respuestas a las consultas respondibles (der.). |
| `test/escalation_outcomes_combined` | Propuesta de bot final | Motores individuales frente a combinaciones simuladas (un clasificador decide si escalar). |
| `test/metrics_panels` | Comparación general | Métricas por motor en test, con IC 95 % bootstrap. |
| `test/confusion_grid` (o `confusion_M0/M1/M2`) | Matrices | Matrices de confusión normalizadas por fila; (E) indica intenciones que escalan. |
| `test/hit_at_k` | Recuperador | Hit@k del recuperador semántico (compartido por M2 y M3). |
| `test/latency` | Costo | Latencia mediana y p95 por respuesta (escala logarítmica). |
| `test/intent_recall` | Errores por intención | Recall por intención y motor. |
| `train/metrics_panels_oof`, `train/confusion_M1_oof` | Metodología | M1 y M2 estimados fuera de pliegue en train. |

Las tablas en LaTeX están en `reports/tables/<split>/*.tex` y requieren `\usepackage{booktabs}` y `\usepackage{graphicx}`.
