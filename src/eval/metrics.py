"""Métricas de evaluación e intervalos de confianza bootstrap para los motores.

Implementa las métricas automáticas del protocolo de evaluación:
- Exactitud de intención (*intent accuracy*).
- Abstención correcta (sobre consultas que deben escalar) y abstención
  innecesaria (sobre consultas respondibles).
- Hit@k (k = 1, 3, 5) para motores con recuperación de fragmentos sobre filas
  con ``kb_source`` definido.
- Estadísticos de latencia en milisegundos (mediana y percentil 95).
- Intervalos de confianza mediante *bootstrap* con semilla fija y advertencia
  explícita cuando el tamaño de muestra es pequeño o nulo.
- Matriz de confusión sobre la taxonomía oficial de intenciones.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import INTENTS, SEED

SMALL_SAMPLE_THRESHOLD = 10
UNCONFIRMED_MARKER = "[POR CONFIRMAR]"
RETRIEVAL_ENGINES = frozenset({"M2", "M3"})


def bootstrap_ci(
    values: Sequence[float | int | bool] | np.ndarray,
    seed: int = SEED,
    n_boot: int = 1000,
    confidence: float = 0.95,
    min_n_warning: int = SMALL_SAMPLE_THRESHOLD,
) -> tuple[float, float, str]:
    """Calcula el intervalo de confianza bootstrap para la media de una muestra.

    Args:
        values: Valores numéricos o booleanos (0/1) de la métrica por fila.
        seed: Semilla del generador pseudoaleatorio para reproducibilidad.
        n_boot: Número de remuestreos bootstrap con reemplazo.
        confidence: Nivel de confianza del intervalo (por defecto 0.95).
        min_n_warning: Umbral mínimo de muestras bajo el cual se marca advertencia.

    Returns:
        Tupla ``(ci_low, ci_high, status)`` donde ``status`` es ``"n=0"`` si no
        hay datos, ``"low_n (n=<n><10)"`` si la muestra es pequeña, u ``"ok"``.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), "n=0")

    status = f"low_n (n={n}<{min_n_warning})" if n < min_n_warning else "ok"
    if n == 1:
        val = float(arr[0])
        return (val, val, status)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return (low, high, status)


def intent_accuracy(
    true_intents: Sequence[str],
    pred_intents: Sequence[str | None],
) -> tuple[float, np.ndarray]:
    """Calcula la exactitud de clasificación de intención y el vector de aciertos.

    Args:
        true_intents: Etiquetas reales de intención.
        pred_intents: Intenciones predichas por el motor.

    Returns:
        Tupla ``(accuracy, hits_array)`` con la proporción en ``[0.0, 1.0]`` (o
        ``NaN`` si ``n == 0``) y el arreglo binario de aciertos por consulta.
    """
    if len(true_intents) != len(pred_intents):
        raise ValueError("true_intents y pred_intents deben tener la misma longitud.")
    n = len(true_intents)
    if n == 0:
        return (float("nan"), np.array([], dtype=float))

    hits = np.array(
        [
            1.0 if t == p else 0.0
            for t, p in zip(true_intents, pred_intents, strict=True)
        ],
        dtype=float,
    )
    return (float(hits.mean()), hits)


def abstention_metrics(
    should_escalate: Sequence[bool],
    escalated: Sequence[bool],
) -> dict[str, Any]:
    """Calcula abstención correcta e innecesaria separando por ``should_escalate``.

    Args:
        should_escalate: Valores reales que indican si la consulta debía escalar.
        escalated: Decisiones de escalado tomadas por el motor.

    Returns:
        Diccionario con la proporción y vectores binarios de abstención correcta
        (sobre ``should_escalate=True``) y abstención innecesaria (sobre
        ``should_escalate=False``), junto con sus tamaños ``n_should_escalate`` y
        ``n_respondable``.
    """
    if len(should_escalate) != len(escalated):
        raise ValueError("should_escalate y escalated deben tener la misma longitud.")

    esc_hits = np.array(
        [
            1.0 if esc else 0.0
            for req, esc in zip(should_escalate, escalated, strict=True)
            if req
        ],
        dtype=float,
    )
    resp_unnecessary = np.array(
        [
            1.0 if esc else 0.0
            for req, esc in zip(should_escalate, escalated, strict=True)
            if not req
        ],
        dtype=float,
    )

    n_esc = int(esc_hits.size)
    n_resp = int(resp_unnecessary.size)

    return {
        "correct_abstention": float(esc_hits.mean()) if n_esc > 0 else float("nan"),
        "correct_abstention_hits": esc_hits,
        "n_should_escalate": n_esc,
        "unnecessary_abstention": (
            float(resp_unnecessary.mean()) if n_resp > 0 else float("nan")
        ),
        "unnecessary_abstention_hits": resp_unnecessary,
        "n_respondable": n_resp,
    }


def hit_at_k(
    kb_sources: Sequence[str | None],
    retrieved_lists: Sequence[Sequence[str]],
    k: int,
) -> tuple[float, int, np.ndarray]:
    """Calcula Hit@k únicamente sobre las filas con ``kb_source`` no vacío.

    Args:
        kb_sources: Identificador del fragmento esperado por consulta (o vacío).
        retrieved_lists: Lista de identificadores recuperados por el motor.
        k: Corte superior (p. ej. 1, 3 o 5).

    Returns:
        Tupla ``(hit_rate, n_kb, hits_array)``. Si ninguna fila tiene
        ``kb_source``, devuelve ``(NaN, 0, array([]))``.
    """
    if k <= 0:
        raise ValueError("El parámetro k debe ser un entero positivo.")
    if len(kb_sources) != len(retrieved_lists):
        raise ValueError("kb_sources y retrieved_lists deben tener la misma longitud.")

    hits_list: list[float] = []
    for kb, retrieved in zip(kb_sources, retrieved_lists, strict=True):
        target = "" if kb is None or pd.isna(kb) else str(kb).strip()
        if not target:
            continue
        top_k = [str(item).strip() for item in list(retrieved)[:k]]
        hits_list.append(1.0 if target in top_k else 0.0)

    hits_arr = np.asarray(hits_list, dtype=float)
    n_kb = int(hits_arr.size)
    if n_kb == 0:
        return (float("nan"), 0, hits_arr)
    return (float(hits_arr.mean()), n_kb, hits_arr)


def latency_summary(
    latencies_ms: Sequence[float | None],
) -> dict[str, float | int]:
    """Calcula la mediana y el percentil 95 de latencia en milisegundos.

    Args:
        latencies_ms: Tiempos de respuesta en milisegundos por consulta.

    Returns:
        Diccionario con ``latency_median_ms``, ``latency_p95_ms`` y ``n_latency``.
    """
    valid = np.array(
        [float(val) for val in latencies_ms if val is not None and not pd.isna(val)],
        dtype=float,
    )
    n_lat = int(valid.size)
    if n_lat == 0:
        return {
            "latency_median_ms": float("nan"),
            "latency_p95_ms": float("nan"),
            "n_latency": 0,
        }
    return {
        "latency_median_ms": float(np.median(valid)),
        "latency_p95_ms": float(np.percentile(valid, 95)),
        "n_latency": n_lat,
    }


def confusion_matrix_df(
    true_intents: Sequence[str],
    pred_intents: Sequence[str | None],
    labels: Sequence[str] = INTENTS,
) -> pd.DataFrame:
    """Construye la matriz de confusión sobre la taxonomía de intenciones.

    Args:
        true_intents: Intenciones verdaderas.
        pred_intents: Intenciones predichas.
        labels: Orden de las intenciones de la taxonomía.

    Returns:
        DataFrame con filas ``true_intent`` y columnas para cada intención
        (más columna ``unassigned`` si algún motor devuelve ``None`` o etiqueta
        fuera de la lista).
    """
    label_list = list(labels)
    matrix = pd.DataFrame(0, index=label_list, columns=label_list, dtype=int)
    matrix.index.name = "true_intent"

    for t, p in zip(true_intents, pred_intents, strict=True):
        row_key = t if t in matrix.index else "unassigned"
        if row_key not in matrix.index:
            matrix.loc[row_key] = 0
        col_key = p if (p is not None and p in matrix.columns) else "unassigned"
        if col_key not in matrix.columns:
            matrix[col_key] = 0
        matrix.loc[row_key, col_key] += 1

    return matrix


def filter_for_manual_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra las filas aptas para calificación manual de corrección y alucinación.

    Incluye filas con ``reference_answer = ESCALAR`` y respuestas confirmadas;
    excluye las filas cuyo ``reference_answer`` sea ``[POR CONFIRMAR]``.

    Args:
        df: DataFrame con la columna ``reference_answer``.

    Returns:
        Copia filtrada del DataFrame sin filas ``[POR CONFIRMAR]``.
    """
    if df.empty or "reference_answer" not in df.columns:
        return df.copy()
    mask = (
        df["reference_answer"].fillna("").astype(str).str.strip().ne(UNCONFIRMED_MARKER)
    )
    return df.loc[mask].copy()


def compute_group_metrics(
    records_df: pd.DataFrame,
    engine_name: str,
    source: str,
    split: str,
    hardware: str,
    train_source: str = "both",
    seed: int = SEED,
    n_boot: int = 1000,
    include_hit_at_k: bool | None = None,
) -> dict[str, Any]:
    """Calcula todas las métricas e intervalos de confianza de un grupo.

    Args:
        records_df: Filas evaluadas de un único ``(engine, source, split)``.
        engine_name: Nombre del motor (``"M0"``, ``"M1"``, ``"M2"`` o ``"M3"``).
        source: Origen de los datos (``"synthetic"`` o ``"real"``).
        split: Partición evaluada (``"train"`` o ``"test"``).
        hardware: Descripción textual del hardware donde se corrió la prueba.
        train_source: Origen de los datos usados en ``fit()``.
        seed: Semilla para los intervalos de confianza bootstrap.
        n_boot: Cantidad de remuestreos bootstrap.
        include_hit_at_k: Si es ``None``, se activa solo para ``M2`` y ``M3``.

    Returns:
        Diccionario plano con tamaños de muestra, métricas puntuales, límites
        inferiores/superiores de IC al 95 %, banderas de muestra pequeña y
        estadísticos de latencia.
    """
    if include_hit_at_k is None:
        include_hit_at_k = engine_name in RETRIEVAL_ENGINES

    n_total = len(records_df)
    if n_total == 0:
        true_intents: list[str] = []
        pred_intents: list[str | None] = []
        should_esc: list[bool] = []
        escalated: list[bool] = []
        kb_sources: list[str | None] = []
        retrieved_lists: list[list[str]] = []
        latencies: list[float | None] = []
        n_manual_eligible = 0
    else:
        true_intents = records_df["intent"].astype(str).tolist()
        pred_intents = records_df["predicted_intent"].tolist()
        should_esc = [
            (val if isinstance(val, bool) else str(val).strip().lower() == "true")
            for val in records_df["should_escalate"].tolist()
        ]
        escalated = [bool(val) for val in records_df["escalate"].tolist()]
        kb_sources = records_df["kb_source"].tolist()
        retrieved_lists = records_df["retrieved"].tolist()
        latencies = records_df["latency_ms"].tolist()
        n_manual_eligible = len(filter_for_manual_evaluation(records_df))

    acc, acc_hits = intent_accuracy(true_intents, pred_intents)
    acc_low, acc_high, acc_ci_status = bootstrap_ci(acc_hits, seed=seed, n_boot=n_boot)

    abst = abstention_metrics(should_esc, escalated)
    ca_low, ca_high, ca_ci_status = bootstrap_ci(
        abst["correct_abstention_hits"], seed=seed, n_boot=n_boot
    )
    ua_low, ua_high, ua_ci_status = bootstrap_ci(
        abst["unnecessary_abstention_hits"], seed=seed, n_boot=n_boot
    )

    if include_hit_at_k:
        h1, n_kb, h1_hits = hit_at_k(kb_sources, retrieved_lists, k=1)
        h3, _, h3_hits = hit_at_k(kb_sources, retrieved_lists, k=3)
        h5, _, h5_hits = hit_at_k(kb_sources, retrieved_lists, k=5)
        h1_low, h1_high, h1_status = bootstrap_ci(h1_hits, seed=seed, n_boot=n_boot)
        h3_low, h3_high, _ = bootstrap_ci(h3_hits, seed=seed, n_boot=n_boot)
        h5_low, h5_high, _ = bootstrap_ci(h5_hits, seed=seed, n_boot=n_boot)
    else:
        _, n_kb, _ = hit_at_k(kb_sources, retrieved_lists, k=1)
        h1 = h3 = h5 = float("nan")
        h1_low = h1_high = h3_low = h3_high = h5_low = h5_high = float("nan")
        h1_status = "omitted_for_engine"

    lat = latency_summary(latencies)

    return {
        "engine": engine_name,
        "train_source": train_source,
        "source": source,
        "split": split,
        "n_total": n_total,
        "n_should_escalate": abst["n_should_escalate"],
        "n_respondable": abst["n_respondable"],
        "n_kb": n_kb,
        "n_manual_eval_eligible": n_manual_eligible,
        "intent_accuracy": acc,
        "intent_accuracy_ci_low": acc_low,
        "intent_accuracy_ci_high": acc_high,
        "intent_accuracy_ci_status": acc_ci_status,
        "correct_abstention": abst["correct_abstention"],
        "correct_abstention_ci_low": ca_low,
        "correct_abstention_ci_high": ca_high,
        "correct_abstention_ci_status": ca_ci_status,
        "unnecessary_abstention": abst["unnecessary_abstention"],
        "unnecessary_abstention_ci_low": ua_low,
        "unnecessary_abstention_ci_high": ua_high,
        "unnecessary_abstention_ci_status": ua_ci_status,
        "hit_at_1": h1,
        "hit_at_1_ci_low": h1_low,
        "hit_at_1_ci_high": h1_high,
        "hit_at_3": h3,
        "hit_at_3_ci_low": h3_low,
        "hit_at_3_ci_high": h3_high,
        "hit_at_5": h5,
        "hit_at_5_ci_low": h5_low,
        "hit_at_5_ci_high": h5_high,
        "hit_at_k_status": h1_status,
        "latency_median_ms": lat["latency_median_ms"],
        "latency_p95_ms": lat["latency_p95_ms"],
        "hardware": hardware,
    }
