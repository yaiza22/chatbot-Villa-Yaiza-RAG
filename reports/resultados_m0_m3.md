# Resultados de la evaluación de los motores M0–M3

**Estado:** resultados de **desarrollo** sobre la partición `train` (563 preguntas reales). La partición `test` (243 preguntas) todavía no se ha evaluado. Las secciones marcadas **[FALTA: test]** se completan cuando exista esa corrida, ejecutando de nuevo `notebooks/07_resultados_m0_m3_ng.ipynb` con `SPLIT = "test"`.

Todas las cifras de este documento salen de archivos de `reports/`. Junto a cada dato se indica el archivo y la columna de donde sale. Las tablas y figuras se regeneran con el notebook. Este documento no trae cifras escritas a mano.

---

## 1. Fuentes

| Contenido | Archivo | Columnas clave |
|---|---|---|
| Métricas oficiales M0, M1, M2 | `reports/eval_train/clasicos/results_2026-11-17.csv` | `intent_accuracy`, `correct_abstention`, `unnecessary_abstention`, `hit_at_1/3/5`, `latency_median_ms`, `latency_p95_ms`, y sus `*_ci_low` / `*_ci_high` |
| Métricas oficiales M3 | `reports/eval_train/m3/results_2026-11-17.csv` | las mismas |
| Predicción por pregunta (M0–M2 / M3) | `reports/eval_train/{clasicos,m3}/manual_eval_2026-11-17.csv` | `intent`, `predicted_intent`, `should_escalate`, `escalate`, `kb_source`, `retrieved_str`, `answer` |
| Matrices de confusión | `reports/eval_train/{clasicos,m3}/confusion_matrix_2026-11-17_<motor>_real_train.csv` | filas = `true_intent`, columnas = intención predicha |
| Tablas derivadas (este análisis) | `reports/tables/train/*.csv` y `*.tex` | ver cada sección |
| Figuras | `reports/figures/train/*.png` y `*.pdf` | ver sección 9 |

**Verificación.** El notebook recalcula las métricas desde `manual_eval` y las compara con `results_*.csv`. Las 18 combinaciones motor × métrica coinciden, y las 4 matrices de confusión reconstruidas son idénticas a las oficiales (sección 2 del notebook).

**Hardware de latencia:** AMD Ryzen 5 8600HS, NVIDIA RTX 4050 6 GB, 64 GB RAM (columna `hardware`). Los cuatro motores se midieron en el mismo equipo, así que su latencia es comparable entre sí, pero no con la de la Fase 2, que se midió en otro computador.

## 2. Datos

`data/processed/labeled.csv`: 806 preguntas reales anonimizadas (`source = real`), partición 70/30 estratificada por intención con semilla 42 (`tables/train/tabla_datos.csv`).

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

En `train`, 485 preguntas deben escalar (86,1 %) y 78 son respondibles (13,9 %); las 78 tienen `kb_source` (columnas `n_should_escalate`, `n_respondable` y `n_kb` de `results_*.csv`). En `test` hay 209 que escalan y 34 respondibles.

**Consecuencia del desbalance.** Un motor que **siempre** escala acierta la decisión escalar/responder en el 86,1 % de las preguntas sin responder nada. Por eso no basta con mirar el acierto global. Se reportan juntas la abstención correcta y la innecesaria, y su diferencia **J = abstención correcta − abstención innecesaria**, que vale 0 para "siempre escalar" y 1 para un motor perfecto. J es el mismo criterio con el que M1 y M2 eligen su umbral.

## 3. Qué se evalúa en cada motor y por qué

| Métrica | M0 | M1 | M2 | M3 | Motivo |
|---|---|---|---|---|---|
| Accuracy de intención (todas las preguntas) | sí | sí¹ | no aplica² | no aplica² | ² M2 y M3 toman la intención del fragmento top-1. La base solo tiene fragmentos de `capacidad`, `reglas_servicios` y `ubicacion`, así que no pueden predecir ninguna intención que escala. Su máximo posible es 78/563 = 0,139. Se reporta la accuracy **solo sobre respondibles**. |
| Hit@k (k = 1, 3, 5) | no aplica | no aplica | sí | sí (= M2) | M0 y M1 no recuperan fragmentos. M3 usa el recuperador de M2: `retrieved_str` es idéntico en las 563 filas, y por eso también lo son el Hit@k, la intención predicha y la matriz de confusión. |
| Abstención correcta e innecesaria, J | sí | sí¹ | sí | sí | Decisión escalar/responder, común a todos. |
| Corrección | automática | automática¹ | automática | manual (pendiente) | M0 y M1 devuelven la plantilla validada de la intención; M2 devuelve el fragmento sin redactarlo. Su respuesta es correcta si no escaló y acertó la intención (M0, M1) o el fragmento top-1 = `kb_source` (M2). M3 redacta texto libre, así que se califica a mano. |
| Alucinación | 0 por construcción | 0 por construcción | 0 por construcción | manual (pendiente) | M0, M1 y M2 solo eligen entre textos ya validados de la base y no pueden afirmar un dato que no esté en ella. |
| Latencia | sí | sí | sí | sí | Mediana y percentil 95 en el mismo equipo. |

¹ **M1 sobre train no es un resultado válido.** El arnés lo entrena con el `train` y lo evalúa sobre las mismas 563 filas (resustitución). TF-IDF con regresión logística las memoriza, y por eso obtiene 1,00 en todo (`results_*.csv`, fila M1). La estimación honesta en train es la validación cruzada fuera de pliegue de la sección 9 del notebook, **[FALTA: ejecutar]**. Sobre `test` M1 sí se puede medir.

## 4. Resultados principales (train)

Formato: punto [IC 95 % bootstrap, 2000 remuestreos]. Fuente: `tables/train/tabla_metricas.csv`. Los valores de accuracy, abstención y Hit@k son los de `results_*.csv`. Las demás columnas se calculan desde `manual_eval`.

| Motor | Accuracy intención | Accuracy intención (respondibles) | Abstención correcta | Abstención innecesaria | J | Correctas de punta a punta (respondibles) | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 0,62 [0,58–0,66] | 0,55 [0,44–0,65] | 0,91 [0,88–0,93] | 0,40 [0,28–0,51] | 0,51 [0,39–0,62] | 0,55 [0,44–0,65] | — | — | — |
| M1 (resustitución, no válido) | 1,00 | 1,00 | 1,00 | 0,00 | 1,00 | 1,00 | — | — | — |
| M2 | 0,11 (techo 0,14) | 0,82 [0,72–0,91] | 0,33 [0,29–0,37] | 0,08 [0,03–0,14] | 0,26 [0,18–0,32] | 0,46 [0,35–0,58] | 0,49 [0,37–0,60] | 0,73 [0,63–0,82] | 0,83 [0,74–0,91] |
| M3 | 0,11 (techo 0,14) | 0,82 [0,72–0,91] | 0,57 [0,52–0,61] | 0,09 [0,04–0,15] | 0,48 [0,40–0,55] | manual | 0,49 | 0,73 | 0,83 |
| *Siempre escalar (referencia)* | — | — | 1,00 | 1,00 | 0,00 | 0,00 | — | — | — |

**Latencia** (`latency_median_ms`, `latency_p95_ms`; `tables/train/tabla_latencia.csv`):

| Motor | Mediana | p95 |
|---|---|---|
| M0 | 0,23 ms | 0,90 ms |
| M1 | 0,95 ms | 1,30 ms |
| M2 | 29,4 ms | 41,3 ms |
| M3 | 4,8 s | 8,0 s |

M3 tarda unas 160 veces más que M2 y unas 5000 veces más que M1 en la mediana.

**Qué hace cada motor con cada pregunta** (`tables/train/tabla_escalar_responder.csv`, figura `escalation_outcomes`):

| Motor | Debía escalar: escaló | Debía escalar: **respondió** | Respondible: correcta | Respondible: incorrecta | Respondible: sin calificar | Respondible: escaló |
|---|---|---|---|---|---|---|
| M0 | 440 | 45 | 43 | 4 | 0 | 31 |
| M1 (resust.) | 485 | 0 | 78 | 0 | 0 | 0 |
| M2 | 161 | **324** | 36 | 36 | 0 | 6 |
| M3 | 275 | **210** | — | — | 71 | 7 |

## 5. Análisis por motor

### M0 (reglas por palabras clave; equivale a las respuestas rápidas actuales)
- **Escala bien:** escala el 91 % de lo que debe escalar.
- **Escala de más:** también escala el 40 % de lo respondible (31 de 78).
- **Respuestas útiles:** responde bien 43 de 78 respondibles (55 %).
- **Dónde falla** (`tables/train/tabla_por_intencion.csv`, columna `recall`; figura `confusion_M0`):
  - `reglas_servicios` tiene recall de 0,28: 14 de 25 caen en `otro_negocio`.
  - `ubicacion` tiene recall de 0,58.
  - Sin una palabra clave que coincida, la pregunta va a `otro_negocio` y escala. Eso explica la abstención innecesaria.
- **Errores que no cambian el comportamiento:** M0 nunca acierta `fuera_de_alcance` (recall 0, 38 de 45 caen en `otro_negocio`), pero como ambas intenciones escalan, el resultado para el cliente es el mismo.

### M1 (TF-IDF + regresión logística)
- **Sobre train no hay estimación válida todavía** (ver la nota ¹).
- **Resultado previo con el lote 1 (no comparable):** en la Fase 2, entrenado con el train real del lote 1 y evaluado una sola vez sobre su test (n = 68), obtuvo:
  - accuracy de intención 0,71 [0,60–0,81];
  - abstención correcta 1,00;
  - abstención innecesaria 0,78 [0,44–1,00].

  Fuente: `reports/results_fase2.csv`, fila `M1_train-real`. Ese resultado usa las etiquetas anteriores a la reclasificación del lote 1 y otro tamaño de train.
- **[FALTA: OOF y test]** Accuracy, abstención y J fuera de pliegue (`tables/train/tabla_metricas_oof.csv`) y sobre test.

### M2 (embeddings `intfloat/multilingual-e5-small` + umbral de similitud)
- **Recupera razonablemente bien:**
  - Hit@1 0,49, Hit@3 0,73, Hit@5 0,83 sobre las 78 respondibles.
  - Entre las respondibles acierta la intención el 82 %.
- **Decide mal cuándo escalar:** responde con un fragmento el **67 %** de las preguntas que debían escalar (324 de 485). La proporción que escala por intención (columna `escalo`) es:

  | Intención | Escaló |
  |---|---|
  | disponibilidad | 20 % |
  | precio | 27 % |
  | otro_negocio | 40 % |
  | fuera_de_alcance | 71 % |

- **Causa:** el umbral de similitud no es un detector de intención. Una pregunta de precio o de fechas suele mencionar personas, la finca o el lugar, y se parece a un fragmento de capacidad o ubicación (en la matriz `confusion_M2`, el 54 % de `disponibilidad` se asigna a `reglas_servicios`). El mejor umbral en train solo logra J = 0,26.
- **Respuestas:** M2 devuelve el fragmento top-1 sin redactar, así que la mitad de lo que responde en respondibles es el fragmento equivocado: 36 correctas y 36 incorrectas.

### M3 (RAG: recuperador de M2 + Qwen 3.6 local por Ollama)
- **El LLM mejora la abstención frente a M2:**
  - La abstención correcta sube de 0,33 a 0,57, una diferencia pareada de +0,235 [0,171–0,297] (`tables/train/tabla_comparaciones_pareadas.csv`).
  - Esto sin aumentar de forma distinguible la innecesaria: +0,013 [−0,064–0,090], y el IC incluye 0.
  - J pasa de 0,26 a 0,48 (+0,22 [0,12–0,32]).
- **Aun así responde el 43 % de lo que debía escalar** (210 de 485):

  | Intención | Respondidas sin escalar |
  |---|---|
  | otro_negocio | 103 |
  | disponibilidad | 63 |
  | fuera_de_alcance | 25 |
  | precio | 19 |

  Esto se cuenta con las columnas `intent` y `escalate` de `manual_eval` (M3).
- **Lectura de su salida** (`tables/train/tabla_m3_sensibilidad.csv`):
  - **"ESCALAR" sin escalar:** en 4 respuestas el LLM escribió "ESCALAR" al final y el motor no lo marcó como escalamiento. Contarlas cambia la abstención correcta de 0,567 a 0,571 y la innecesaria de 0,090 a 0,115. El efecto es pequeño.
  - **Remite a una persona sin escalar:** unas 89 respuestas mandan al cliente con una persona ("comunícate con…") con `escalate = False`. Esto se detectó con palabras clave, así que es una estimación, no una medida. Si se contaran como escalamientos, la abstención correcta sería ~0,72 y la innecesaria ~0,32.
  - **Conclusión:** la bandera `escalate` de M3 subestima cuánto deriva realmente a una persona. La calificación manual lo resolverá.
- **Precios:** en las 210 respuestas a preguntas que debían escalar no aparecen montos de dinero (búsqueda por patrones `$`, "mil", "millón"). Es una señal favorable sobre alucinación de precios, no una medida; la alucinación se califica a mano.
- **Contexto del LLM:** recibió el fragmento correcto en el 73 % de las respondibles (Hit@3). Ese es el techo de su corrección.
- **Pendiente:** la corrección y la alucinación de M3 requieren calificación manual **[FALTA: test]**.

## 6. Comparaciones pareadas (mismas 563 preguntas)

Fuente: `tables/train/tabla_comparaciones_pareadas.csv` (`diff_a_menos_b`, `ci_low`, `ci_high`, `ic_incluye_0`). Se omite M1 por la nota ¹.

| Comparación | Métrica | a − b [IC 95 %] | ¿Distinguible? |
|---|---|---|---|
| M0 − M3 | J | +0,033 [−0,081–0,147] | **No** |
| M0 − M2 | J | +0,255 [0,128–0,373] | Sí |
| M3 − M2 | J | +0,222 [0,119–0,319] | Sí |
| M0 − M3 | Abstención correcta | +0,340 [0,295–0,388] | Sí |
| M0 − M3 | Abstención innecesaria | +0,308 [0,205–0,423] | Sí (M0 escala más) |
| M0 − M2 | Correctas de punta a punta (respondibles) | +0,090 [−0,038–0,218] | **No** |
| M2 − M3 | Abstención innecesaria | −0,013 [−0,090–0,064] | **No** |

**Lectura:** M0 y M3 tienen un balance J parecido, pero por caminos opuestos. M0 escala casi todo (pocas preguntas sensibles respondidas, muchas útiles sin responder) y M3 responde casi todo (muchas útiles respondidas, muchas sensibles también).

## 7. Conclusiones preliminares (train)

1. **La tarea principal del agente es decidir si escalar:** el 86 % de las consultas reales debe pasar a una persona. En esa tarea, los métodos que clasifican la intención (M0, y presumiblemente M1, **[FALTA: OOF/test]**) son más seguros que los que solo recuperan (M2). M2 nunca puede predecir una intención que escala porque la base solo contiene temas respondibles.
2. **El LLM corrige buena parte del problema de M2** (J de 0,26 a 0,48) sin escalar de más, pero no supera a las reglas (M0) en J, y responde el 43 % de lo que debía escalar. Con estos datos, **un método complejo no supera a uno simple en abstención correcta**. Esto se reporta tal cual, como pide el protocolo.
3. **Para redactar respuestas**, la recuperación es útil (Hit@3 = 0,73), pero M2 devuelve el top-1 sin redactar (Hit@1 = 0,49). M3 puede usar los tres primeros, aunque su corrección aún no está calificada.
4. **Implicación de diseño:** las fortalezas son complementarias. El clasificador decide si escalar y el recuperador/LLM redacta las respuestas permitidas. Esa combinación se simula en la sección 11 del notebook **[FALTA: OOF/test]**.
5. **Costo:** M3 es unas 160 veces más lento que M2 (4,8 s de mediana frente a 29 ms) y necesita GPU. Es aceptable para mensajería, pero relevante frente a la mejora obtenida.

## 8. Limitaciones

- **Resultados de desarrollo:** todo lo anterior es sobre `train`. El umbral de M2 (y por tanto el de M3) se eligió con estas mismas preguntas, lo que da un leve sesgo optimista en sus números de train. M1 en train es resustitución.
- **Muestra pequeña de respondibles** (78 en train y 34 en test): los IC de abstención innecesaria y corrección son anchos (por ejemplo, M0: 0,28–0,51).
- **Etiquetas puestas por una sola persona**, sin doble etiquetado; el error de etiqueta no está medido.
- **Audios de los chats fuera de alcance:** solo se evalúan mensajes de texto.
- **Reutilización del test de la Fase 2:** las 68 preguntas de test del lote 1 ya se usaron una vez en la Fase 2, sin ajustar nada con ellas. Además, parte del lote 1 se reclasificó después con la base v0.
- **Pendientes:**
  - Evaluación sobre preguntas sintéticas (`seed.csv`), que se reporta por separado.
  - Calificación manual de corrección y alucinación de M3.
  - Medida de la abstención de M3 por patrones de texto (heurística), no por calificación humana.

## 9. Figuras y tablas para el informe y la exposición

Figuras en `reports/figures/<split>/` (PNG para diapositivas, PDF para el artículo; textos en inglés):

| Archivo | Uso sugerido | Pie de figura sugerido (artículo) |
|---|---|---|
| `escalation_outcomes` | Diapositiva principal de resultados | Resultado de cada pregunta por motor: escalamiento de consultas que lo requieren (izq.) y respuestas a consultas respondibles (der.). |
| `metrics_panels` | Comparación general | Métricas por motor con IC 95 % bootstrap. |
| `confusion_grid` | Una diapositiva con M0, M1 y M2/M3 | Matrices de confusión normalizadas por fila; (E) indica intenciones que escalan. |
| `confusion_M0`, `confusion_M1`, `confusion_M2` | Matrices individuales | ídem |
| `hit_at_k` | Recuperador | Hit@k del recuperador semántico (compartido por M2 y M3). |
| `latency` | Costo | Latencia mediana y p95 por respuesta (escala logarítmica). |
| `intent_recall` | Dónde falla cada motor | Recall por intención y motor. |
| `escalation_outcomes_oof`, `metrics_panels_oof`, `confusion_M1_oof` | M1 honesto en train **[FALTA: ejecutar]** | |
| `escalation_outcomes_combined` | Combinaciones simuladas **[FALTA]** | |

Tablas LaTeX (`reports/tables/<split>/*.tex`): requieren `\usepackage{booktabs}` y `\usepackage{graphicx}`.

## 10. [FALTA: test] Resultados finales

Se completa con `SPLIT = "test"` en el notebook, que genera los mismos archivos en `reports/figures/test/` y `reports/tables/test/`. Se reporta una sola corrida, con los mismos motores y configuraciones congelados antes de evaluar.
