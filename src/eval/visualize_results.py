"""Análisis y figuras de los resultados de evaluación de los motores M0–M3.

Este módulo no ejecuta los motores sobre el test ni cambia ninguna métrica del arnés. Lee lo
que ``run_eval`` ya dejó en ``reports/`` y produce:

* Una verificación: recalcula las métricas desde las predicciones por pregunta
  (``manual_eval_<fecha>.csv``) y las compara con ``results_<fecha>.csv``.
* Métricas adicionales que el arnés no calcula: exactitud de intención solo sobre las
  preguntas respondibles, respuestas correctas de punta a punta, preguntas que debían
  escalar y se respondieron, recall por intención y diferencias pareadas entre motores.
* Figuras listas para diapositivas (PNG a 300 dpi y PDF, etiquetas en inglés) y tablas para
  el informe (CSV y LaTeX).
* Validación cruzada fuera de pliegue (out-of-fold) sobre ``train`` usando solo
  ``fit()`` y ``respond()`` de la interfaz ``Engine``: es la forma honesta de medir M1 sin
  tocar el test, porque evaluar M1 sobre las mismas filas con las que se entrenó
  (resustitución) da un resultado inflado.
* Una simulación de motores combinados (un motor decide si escalar y otro responde) a partir
  de predicciones ya guardadas.

Uso típico: desde ``notebooks/07_resultados_m0_m3_ng.ipynb``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ESCALATING_INTENTS, INTENTS, SEED

# --------------------------------------------------------------------------------------
# Constantes de presentación
# --------------------------------------------------------------------------------------
ENGINE_ORDER: tuple[str, ...] = ("M0", "M1", "M2", "M3")

# Nombres cortos en inglés para figuras y exposición.
ENGINE_LABELS: dict[str, str] = {
    "M0": "M0 Keyword rules",
    "M1": "M1 TF-IDF + LR",
    "M2": "M2 Embeddings",
    "M3": "M3 RAG (local LLM)",
}

# Paleta categórica validada (orden fijo; el color sigue al motor, nunca a su posición).
ENGINE_COLORS: dict[str, str] = {
    "M0": "#2a78d6",
    "M1": "#eb6834",
    "M2": "#1baf7a",
    "M3": "#eda100",
}

INTENT_LABELS_EN: dict[str, str] = {
    "precio": "price",
    "capacidad": "capacity",
    "reglas_servicios": "rules & services",
    "ubicacion": "location",
    "disponibilidad": "availability",
    "otro_negocio": "other business",
    "fuera_de_alcance": "out of scope",
}

# Colores de estado: siempre van con etiqueta de texto, nunca solos.
STATUS_COLORS: dict[str, str] = {
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "warning": "#fab219",
    "neutral": "#b5b3ad",
    "good_light": "#8fd18f",
}

# Rampa secuencial (un solo tono, claro -> oscuro) para las matrices de confusión.
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#fcfcfb",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

N_BOOT_DEFAULT = 2000
SMALL_N = 10
ESCALATE_TOKEN = "ESCALAR"

# Heurística (no es una métrica oficial): frases con las que M3 remite a una persona sin
# marcar escalate=True. Sirve solo para estimar cuánto se subestima su abstención.
SOFT_REDIRECT_PATTERN = re.compile(
    r"comun[ií]ca|cont[aá]ct|encargad|equipo humano|asesor|administraci",
    flags=re.IGNORECASE,
)

RESULTS_PATTERN = re.compile(r"^results_(\d{4}-\d{2}-\d{2})\.csv$")
MANUAL_PATTERN = re.compile(r"^manual_eval_(\d{4}-\d{2}-\d{2})\.csv$")
CONFUSION_PATTERN = re.compile(
    r"^confusion_matrix_(\d{4}-\d{2}-\d{2})_(M\d)_(\w+?)_(train|test)\.csv$"
)


# --------------------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------------------
@dataclass
class EvalRun:
    """Resultados de una o varias carpetas de salida de ``run_eval`` para un mismo split.

    Attributes:
        split: ``"train"`` o ``"test"``.
        summary: Filas de ``results_<fecha>.csv`` (una por motor).
        predictions: Filas de ``manual_eval_<fecha>.csv`` (una por motor y pregunta).
        confusion: Matriz de confusión oficial por motor (filas = real, columnas = predicha).
        sources: Archivo de origen de cada tabla, para citarlo en el informe.
    """

    split: str
    summary: pd.DataFrame
    predictions: pd.DataFrame
    confusion: dict[str, pd.DataFrame]
    sources: dict[str, list[str]] = field(default_factory=dict)

    @property
    def engines(self) -> list[str]:
        """Motores presentes, en el orden de ``ENGINE_ORDER``."""
        present = set(self.predictions["engine"].unique())
        return [e for e in ENGINE_ORDER if e in present]


def _latest(paths: Iterable[Path], pattern: re.Pattern[str]) -> Path | None:
    """Devuelve el archivo con la fecha más reciente en el nombre (o ``None``)."""
    dated = []
    for path in paths:
        match = pattern.match(path.name)
        if match:
            dated.append((match.group(1), path))
    return max(dated)[1] if dated else None


def _to_bool(series: pd.Series) -> pd.Series:
    """Convierte ``True``/``False`` (texto o bool) a bool; cualquier otro valor falla."""
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip()
    bad = set(text) - {"True", "False"}
    if bad:
        raise ValueError(f"Valores booleanos no válidos: {sorted(bad)}")
    return text == "True"


def load_eval_dirs(eval_dirs: Sequence[Path | str], split: str) -> EvalRun:
    """Carga y une las salidas de ``run_eval`` de varias carpetas (p. ej. clásicos y M3).

    En cada carpeta se toma el ``results_*.csv`` y el ``manual_eval_*.csv`` más recientes y
    las matrices ``confusion_matrix_*_<split>.csv``. Si un motor aparece en dos carpetas,
    gana la última carpeta de la lista.

    Args:
        eval_dirs: Carpetas de salida de ``run_eval``.
        split: Split que se quiere analizar; se comprueba contra la columna ``split``.

    Returns:
        ``EvalRun`` con las tablas unidas.

    Raises:
        FileNotFoundError: Si una carpeta no tiene ``results`` o ``manual_eval``.
        ValueError: Si los archivos traen otro split o si los motores no evaluaron las
            mismas preguntas.
    """
    summaries: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    confusion: dict[str, pd.DataFrame] = {}
    sources: dict[str, list[str]] = {"results": [], "manual_eval": [], "confusion": []}

    for raw_dir in eval_dirs:
        folder = Path(raw_dir)
        files = list(folder.glob("*.csv"))
        results_path = _latest(files, RESULTS_PATTERN)
        manual_path = _latest(files, MANUAL_PATTERN)
        if results_path is None or manual_path is None:
            raise FileNotFoundError(
                f"{folder}: faltan results_<fecha>.csv o manual_eval_<fecha>.csv"
            )
        summary = pd.read_csv(results_path)
        preds = pd.read_csv(manual_path, dtype={"question_id": str})
        summaries.append(summary)
        predictions.append(preds)
        sources["results"].append(str(results_path))
        sources["manual_eval"].append(str(manual_path))
        for path in sorted(files):
            match = CONFUSION_PATTERN.match(path.name)
            if match and match.group(4) == split:
                confusion[match.group(2)] = pd.read_csv(path, index_col=0)
                sources["confusion"].append(str(path))

    summary_df = pd.concat(summaries, ignore_index=True).drop_duplicates(
        subset=["engine"], keep="last"
    )
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df = pred_df.drop_duplicates(subset=["engine", "question_id"], keep="last")
    for column in ("should_escalate", "escalate"):
        pred_df[column] = _to_bool(pred_df[column])

    found_splits = set(summary_df["split"].astype(str)) | set(
        pred_df["split"].astype(str)
    )
    if found_splits != {split}:
        raise ValueError(
            f"Se pidió split='{split}' y los archivos traen {sorted(found_splits)}"
        )

    _check_same_questions(pred_df)
    order = {e: i for i, e in enumerate(ENGINE_ORDER)}
    summary_df = summary_df.sort_values(
        "engine", key=lambda s: s.map(order)
    ).reset_index(drop=True)
    return EvalRun(
        split, summary_df, pred_df.reset_index(drop=True), confusion, sources
    )


def _engines_in(values: Iterable[str]) -> list[str]:
    """Motores presentes: primero los de ``ENGINE_ORDER`` y luego otros (p. ej. ``M1+M3``)."""
    seen = list(dict.fromkeys(str(v) for v in values))
    return [e for e in ENGINE_ORDER if e in seen] + [
        e for e in seen if e not in ENGINE_ORDER
    ]


def _check_same_questions(predictions: pd.DataFrame) -> None:
    """Falla si los motores no evaluaron exactamente las mismas preguntas."""
    sets = {
        engine: frozenset(group["question_id"])
        for engine, group in predictions.groupby("engine")
    }
    if len(set(sets.values())) > 1:
        sizes = {engine: len(ids) for engine, ids in sets.items()}
        raise ValueError(f"Los motores no evaluaron las mismas preguntas: {sizes}")


def dataset_overview(labeled: pd.DataFrame) -> pd.DataFrame:
    """Filas por intención y split de ``labeled.csv`` (con totales y si escala)."""
    table = (
        labeled.groupby(["intent", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(INTENTS), fill_value=0)
    )
    for column in ("train", "test"):
        if column not in table.columns:
            table[column] = 0
    table = table[["train", "test"]]
    table["total"] = table.sum(axis=1)
    table.insert(0, "escala", [i in ESCALATING_INTENTS for i in table.index])
    totals = table[["train", "test", "total"]].sum()
    table = table.astype({"escala": object})
    table.loc["TOTAL"] = ["", *totals.tolist()]
    return table.astype({"train": int, "test": int, "total": int})


# --------------------------------------------------------------------------------------
# Métricas por pregunta
# --------------------------------------------------------------------------------------
def _retrieved_lists(predictions: pd.DataFrame) -> pd.Series:
    """Convierte ``retrieved_str`` (ids separados por ``|``) en listas."""
    if "retrieved_str" not in predictions.columns:
        return pd.Series([[] for _ in range(len(predictions))], index=predictions.index)
    return (
        predictions["retrieved_str"]
        .fillna("")
        .astype(str)
        .map(lambda text: [part for part in text.split("|") if part])
    )


def per_question_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas 0/1 por pregunta que alimentan todas las métricas.

    Columnas nuevas:

    * ``intent_hit``: intención predicha = etiqueta.
    * ``retrieves``: el motor devuelve fragmentos (M2/M3).
    * ``hit_at_1``, ``hit_at_3``, ``hit_at_5``: ``kb_source`` dentro del top-k
      (``NaN`` si la fila no tiene ``kb_source`` o el motor no recupera).
    * ``answered``: no escaló.
    * ``auto_correct``: solo filas respondibles. Respuesta correcta de punta a punta,
      calificada automáticamente: no escaló y (a) acertó la intención en M0/M1, cuya
      respuesta es la plantilla validada de esa intención, o (b) el fragmento top-1 es
      ``kb_source`` en M2, que devuelve ese fragmento sin redactar. En M3 queda ``NaN``:
      su texto es libre y se califica a mano.
    * ``context_has_source``: solo M3 en filas respondibles: ``kb_source`` está en el top-3
      que recibe el LLM (cota superior de lo que podría responder bien).
    * ``m3_token_leak``: M3 no escaló pero su texto contiene ``ESCALAR``.
    * ``m3_soft_redirect``: M3 no escaló pero remite a una persona (heurística).

    Args:
        predictions: Filas de ``manual_eval`` (una o varias corridas).

    Returns:
        Copia de ``predictions`` con las columnas anteriores.
    """
    out = predictions.copy()
    out["should_escalate"] = _to_bool(out["should_escalate"])
    out["escalate"] = _to_bool(out["escalate"])
    out["intent_hit"] = (
        out["intent"].astype(str).str.strip()
        == out["predicted_intent"].astype(str).str.strip()
    ).astype(float)
    retrieved = _retrieved_lists(out)
    out["retrieves"] = retrieved.map(len) > 0
    kb = (
        out["kb_source"].fillna("").astype(str).str.strip()
        if "kb_source" in out
        else ""
    )
    has_kb = kb != ""
    for k in (1, 3, 5):
        hits = [
            float(src in ids[:k]) if (flag and rec) else np.nan
            for src, ids, flag, rec in zip(
                kb, retrieved, has_kb, out["retrieves"], strict=True
            )
        ]
        out[f"hit_at_{k}"] = hits
    out["answered"] = (~out["escalate"]).astype(float)

    respondable = ~out["should_escalate"]
    # M3 redacta texto libre: no se califica automáticamente (también en combinaciones).
    is_m3 = out["engine"].astype(str).str.contains("M3", regex=False)
    top1 = retrieved.map(lambda ids: ids[0] if ids else "")
    via_intent = out["intent_hit"].astype(bool) & ~out["retrieves"]
    via_fragment = out["retrieves"] & (top1 == kb)
    correct = (~out["escalate"]) & (via_intent | via_fragment)
    out["auto_correct"] = np.where(respondable & ~is_m3, correct.astype(float), np.nan)

    in_top3 = [src in ids[:3] for src, ids in zip(kb, retrieved, strict=True)]
    out["context_has_source"] = np.where(
        respondable & is_m3, np.asarray(in_top3, dtype=float), np.nan
    )
    answer = out["answer"].fillna("").astype(str) if "answer" in out else ""
    answered_m3 = is_m3 & ~out["escalate"]
    out["m3_token_leak"] = np.where(
        answered_m3,
        answer.str.contains(ESCALATE_TOKEN, regex=False).astype(float),
        np.nan,
    )
    out["m3_soft_redirect"] = np.where(
        answered_m3, answer.str.contains(SOFT_REDIRECT_PATTERN).astype(float), np.nan
    )
    return out


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    """Media y su IC bootstrap por percentiles.

    Returns:
        ``(media, ic_bajo, ic_alto, n)``; ``NaN`` si no hay datos.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(arr.mean()),
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
        int(arr.size),
    )


# Métrica -> (columna por pregunta, filtro de filas). El filtro recibe el DataFrame de
# ``per_question_outcomes`` y devuelve una máscara booleana.
METRIC_DEFINITIONS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "intent_accuracy": ("intent_hit", lambda d: pd.Series(True, index=d.index)),
    "intent_accuracy_respondable": ("intent_hit", lambda d: ~d["should_escalate"]),
    "correct_abstention": ("escalate_f", lambda d: d["should_escalate"]),
    "unnecessary_abstention": ("escalate_f", lambda d: ~d["should_escalate"]),
    "answered_should_escalate": ("answered", lambda d: d["should_escalate"]),
    "escalation_decision_accuracy": (
        "decision_ok",
        lambda d: pd.Series(True, index=d.index),
    ),
    "auto_correct_respondable": ("auto_correct", lambda d: ~d["should_escalate"]),
    "hit_at_1": ("hit_at_1", lambda d: pd.Series(True, index=d.index)),
    "hit_at_3": ("hit_at_3", lambda d: pd.Series(True, index=d.index)),
    "hit_at_5": ("hit_at_5", lambda d: pd.Series(True, index=d.index)),
    "m3_context_has_source": ("context_has_source", lambda d: ~d["should_escalate"]),
    "m3_token_leak": ("m3_token_leak", lambda d: pd.Series(True, index=d.index)),
    "m3_soft_redirect": ("m3_soft_redirect", lambda d: pd.Series(True, index=d.index)),
}

METRIC_LABELS_ES: dict[str, str] = {
    "intent_accuracy": "Accuracy de intención (todas)",
    "intent_accuracy_respondable": "Accuracy de intención (solo respondibles)",
    "correct_abstention": "Abstención correcta",
    "unnecessary_abstention": "Abstención innecesaria",
    "answered_should_escalate": "Respondió cuando debía escalar",
    "escalation_decision_accuracy": "Acierto en la decisión escalar/responder",
    "youden_j": "J = abst. correcta − abst. innecesaria",
    "auto_correct_respondable": "Respuesta correcta de punta a punta (respondibles)",
    "hit_at_1": "Hit@1",
    "hit_at_3": "Hit@3",
    "hit_at_5": "Hit@5",
    "m3_context_has_source": "M3: el fragmento correcto llegó al LLM (top-3)",
    "m3_token_leak": "M3: respuesta con 'ESCALAR' sin escalar",
    "m3_soft_redirect": "M3: remite a una persona sin escalar (heurística)",
}

METRIC_LABELS_EN: dict[str, str] = {
    "intent_accuracy": "Intent accuracy",
    "intent_accuracy_respondable": "Intent accuracy (answerable only)",
    "correct_abstention": "Correct escalation ↑",
    "unnecessary_abstention": "Unnecessary escalation ↓",
    "answered_should_escalate": "Answered but should escalate ↓",
    "escalation_decision_accuracy": "Escalate/answer decision accuracy ↑",
    "youden_j": "Escalation balance J = CE − UE ↑",
    "auto_correct_respondable": "Correct answers (answerable) ↑",
    "hit_at_1": "Hit@1",
    "hit_at_3": "Hit@3",
    "hit_at_5": "Hit@5",
}


def _prepare(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Columnas auxiliares para las métricas de escalamiento."""
    data = outcomes.copy()
    data["escalate_f"] = data["escalate"].astype(float)
    data["decision_ok"] = (data["escalate"] == data["should_escalate"]).astype(float)
    return data


def bootstrap_youden(
    escalated_should: Sequence[float],
    escalated_respondable: Sequence[float],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """J = abstención correcta - abstención innecesaria, con IC bootstrap estratificado.

    J resume la decisión escalar/responder sin que la domine la clase mayoritaria: un motor
    que siempre escala tiene J = 0 aunque acierte la decisión en la mayoría de preguntas.
    Es el mismo criterio con el que M1 y M2 eligen su umbral. Cada grupo (debía escalar /
    respondible) se remuestrea por separado para conservar sus tamaños.
    """
    a = np.asarray(escalated_should, dtype=float)
    b = np.asarray(escalated_respondable, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ca = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    ua = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    j = ca - ua
    return (
        float(a.mean() - b.mean()),
        float(np.percentile(j, 100 * alpha)),
        float(np.percentile(j, 100 * (1 - alpha))),
    )


def compute_metrics(
    predictions: pd.DataFrame, n_boot: int = N_BOOT_DEFAULT, seed: int = SEED
) -> pd.DataFrame:
    """Tabla larga motor × métrica con punto, IC bootstrap y tamaño del subconjunto.

    Args:
        predictions: Filas de ``manual_eval`` (sin procesar o ya procesadas).
        n_boot: Remuestreos bootstrap.
        seed: Semilla del bootstrap.

    Returns:
        DataFrame con ``engine, metric, value, ci_low, ci_high, n, ci_status``. Las métricas
        que no aplican a un motor no aparecen (p. ej. Hit@k en M0).
    """
    data = (
        predictions
        if "intent_hit" in predictions
        else per_question_outcomes(predictions)
    )
    data = _prepare(data)
    rows: list[dict[str, object]] = []
    for engine in _engines_in(data["engine"]):
        group = data[data["engine"] == engine]
        for metric, (column, row_filter) in METRIC_DEFINITIONS.items():
            values = group.loc[row_filter(group), column]
            value, low, high, n = bootstrap_ci(values, n_boot=n_boot, seed=seed)
            if n == 0:
                continue
            rows.append(
                {
                    "engine": engine,
                    "metric": metric,
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                    "n": n,
                    "ci_status": "ok" if n >= SMALL_N else "small_n",
                }
            )
        should = group["should_escalate"]
        esc = group["escalate_f"]
        value, low, high = bootstrap_youden(
            esc[should].to_numpy(), esc[~should].to_numpy(), n_boot=n_boot, seed=seed
        )
        if not np.isnan(value):
            n_small = min(int(should.sum()), int((~should).sum()))
            rows.append(
                {
                    "engine": engine,
                    "metric": "youden_j",
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(group),
                    "ci_status": "ok" if n_small >= SMALL_N else "small_n",
                }
            )
    return pd.DataFrame(rows)


def check_against_summary(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Compara los puntos recalculados con los de ``results_<fecha>.csv``.

    Returns:
        Tabla con el valor del arnés, el recalculado, la diferencia y ``coincide``.
    """
    rows = []
    shared = [
        "intent_accuracy",
        "correct_abstention",
        "unnecessary_abstention",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
    ]
    for _, row in summary.iterrows():
        engine = row["engine"]
        for metric in shared:
            official = row.get(metric, np.nan)
            mine = metrics.loc[
                (metrics["engine"] == engine) & (metrics["metric"] == metric), "value"
            ]
            recomputed = float(mine.iloc[0]) if len(mine) else np.nan
            if pd.isna(official):  # el arnés no reporta esta métrica para el motor
                continue
            diff = abs(float(official) - recomputed)
            rows.append(
                {
                    "engine": engine,
                    "metric": metric,
                    "results_csv": official,
                    "recalculado": recomputed,
                    "diferencia": diff,
                    "coincide": bool(diff < 1e-9) if not np.isnan(diff) else False,
                }
            )
    return pd.DataFrame(rows)


def check_confusion(run: EvalRun) -> pd.DataFrame:
    """Compara cada matriz de confusión oficial con la reconstruida desde las predicciones."""
    rows = []
    for engine, official in run.confusion.items():
        group = run.predictions[run.predictions["engine"] == engine]
        rebuilt = pd.crosstab(group["intent"], group["predicted_intent"]).reindex(
            index=list(INTENTS), columns=list(INTENTS), fill_value=0
        )
        official = official.reindex(
            index=list(INTENTS), columns=list(INTENTS), fill_value=0
        )
        rows.append(
            {
                "engine": engine,
                "n": int(official.to_numpy().sum()),
                "coincide": bool((rebuilt.to_numpy() == official.to_numpy()).all()),
            }
        )
    return pd.DataFrame(rows)


def wide_metric_table(
    metrics: pd.DataFrame, metric_names: Sequence[str], digits: int = 2
) -> pd.DataFrame:
    """Tabla ancha motor × métrica con ``punto [bajo–alto]`` (``n/a`` si no aplica)."""
    table = pd.DataFrame(index=_engines_in(metrics["engine"]))
    for metric in metric_names:
        column = []
        for engine in table.index:
            row = metrics[(metrics["engine"] == engine) & (metrics["metric"] == metric)]
            if row.empty:
                column.append("n/a")
                continue
            r = row.iloc[0]
            flag = "*" if r["ci_status"] != "ok" else ""
            column.append(
                f"{r['value']:.{digits}f} [{r['ci_low']:.{digits}f}–{r['ci_high']:.{digits}f}]"
                f"{flag}"
            )
        table[METRIC_LABELS_ES.get(metric, metric)] = column
    table.index.name = "motor"
    return table


def per_intent_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Por motor e intención: n, recall de intención y % que escaló.

    Muestra dónde falla cada motor (p. ej. M2 nunca predice ``precio``).
    """
    data = (
        predictions
        if "intent_hit" in predictions
        else per_question_outcomes(predictions)
    )
    table = (
        data.groupby(["engine", "intent"])
        .agg(
            n=("intent_hit", "size"),
            recall=("intent_hit", "mean"),
            escalo=("escalate", "mean"),
        )
        .reset_index()
    )
    table["intent"] = pd.Categorical(
        table["intent"], categories=list(INTENTS), ordered=True
    )
    table["engine"] = pd.Categorical(
        table["engine"], categories=list(ENGINE_ORDER), ordered=True
    )
    return table.sort_values(["engine", "intent"]).reset_index(drop=True)


def escalation_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Conteos del resultado de cada pregunta por motor.

    Columnas:
        * ``esc_ok``: debía escalar y escaló.
        * ``esc_answered``: debía escalar y respondió (riesgo: dar información no validada).
        * ``resp_correct``: respondible, respondió y la respuesta es correcta (automático).
        * ``resp_wrong``: respondible, respondió con contenido incorrecto (automático).
        * ``resp_ungraded``: respondible, respondió y requiere revisión manual (M3).
        * ``resp_escalated``: respondible y escaló (abstención innecesaria).
        * ``resp_partial``: solo si hay calificación manual (``manual_partial``): respuesta
          parcialmente correcta (nota 1). No cuenta como correcta ni como incorrecta.
    """
    data = (
        predictions
        if "intent_hit" in predictions
        else per_question_outcomes(predictions)
    )
    rows = []
    for engine in _engines_in(data["engine"]):
        g = data[data["engine"] == engine]
        se, esc = g["should_escalate"], g["escalate"]
        resp_answered = g[~se & ~esc]
        graded = resp_answered["auto_correct"].notna()
        partial = (
            resp_answered["manual_partial"].fillna(0).astype(bool)
            if "manual_partial" in resp_answered
            else pd.Series(False, index=resp_answered.index)
        )
        row = {
            "engine": engine,
            "n_should_escalate": int(se.sum()),
            "esc_ok": int((se & esc).sum()),
            "esc_answered": int((se & ~esc).sum()),
            "n_respondable": int((~se).sum()),
            "resp_correct": int((resp_answered.loc[graded, "auto_correct"] == 1).sum()),
            "resp_wrong": int(
                ((resp_answered["auto_correct"] == 0) & graded & ~partial).sum()
            ),
            "resp_ungraded": int((~graded).sum()),
            "resp_escalated": int((~se & esc).sum()),
        }
        if "manual_partial" in data:
            row["resp_partial"] = int((partial & graded).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Diferencia de medias ``a - b`` con IC bootstrap pareado (mismas preguntas)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size:
        raise ValueError("Los vectores pareados deben tener la misma longitud.")
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    diffs = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(x.mean() - y.mean()),
        float(np.percentile(diffs, 100 * alpha)),
        float(np.percentile(diffs, 100 * (1 - alpha))),
    )


def paired_comparisons(
    predictions: pd.DataFrame,
    metric_names: Sequence[str] = (
        "intent_accuracy",
        "correct_abstention",
        "unnecessary_abstention",
        "escalation_decision_accuracy",
        "youden_j",
        "auto_correct_respondable",
    ),
    pairs: Sequence[tuple[str, str]] | None = None,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Diferencias pareadas entre motores sobre las mismas preguntas.

    Si el IC incluye 0, la diferencia no se distingue del azar con este tamaño de muestra.
    """
    data = (
        predictions
        if "intent_hit" in predictions
        else per_question_outcomes(predictions)
    )
    data = _prepare(data)
    engines = _engines_in(data["engine"])
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(engines) for b in engines[i + 1 :]]
    rows = []
    for metric in metric_names:
        if metric == "youden_j":
            rows.extend(_paired_youden(data, pairs, n_boot=n_boot, seed=seed))
            continue
        column, row_filter = METRIC_DEFINITIONS[metric]
        for engine_a, engine_b in pairs:
            ga = data[data["engine"] == engine_a].set_index("question_id")
            gb = (
                data[data["engine"] == engine_b]
                .set_index("question_id")
                .reindex(ga.index)
            )
            mask = row_filter(ga).to_numpy()
            va, vb = ga.loc[mask, column].to_numpy(), gb.loc[mask, column].to_numpy()
            if np.isnan(va).all() or np.isnan(vb).all():
                continue
            diff, low, high = paired_bootstrap_diff(va, vb, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "metric": metric,
                    "a": engine_a,
                    "b": engine_b,
                    "n": int((~(np.isnan(va) | np.isnan(vb))).sum()),
                    "diff_a_menos_b": diff,
                    "ci_low": low,
                    "ci_high": high,
                    "ic_incluye_0": bool(low <= 0.0 <= high),
                }
            )
    return pd.DataFrame(rows)


def _paired_youden(
    data: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED,
) -> list[dict[str, object]]:
    """Diferencia pareada de J entre motores remuestreando las mismas preguntas."""
    rows: list[dict[str, object]] = []
    for engine_a, engine_b in pairs:
        ga = data[data["engine"] == engine_a].set_index("question_id")
        gb = data[data["engine"] == engine_b].set_index("question_id").reindex(ga.index)
        should = ga["should_escalate"].to_numpy(dtype=bool)
        ea = ga["escalate"].to_numpy(dtype=float)
        eb = gb["escalate"].to_numpy(dtype=float)
        if should.all() or not should.any():
            continue

        def j(esc: np.ndarray, mask: np.ndarray) -> float:
            return float(esc[mask].mean() - esc[~mask].mean())

        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, should.size, size=should.size)
            m = should[idx]
            if m.all() or not m.any():
                continue
            diffs.append(j(ea[idx], m) - j(eb[idx], m))
        low, high = np.percentile(diffs, [2.5, 97.5])
        rows.append(
            {
                "metric": "youden_j",
                "a": engine_a,
                "b": engine_b,
                "n": int(should.size),
                "diff_a_menos_b": j(ea, should) - j(eb, should),
                "ci_low": float(low),
                "ci_high": float(high),
                "ic_incluye_0": bool(low <= 0.0 <= high),
            }
        )
    return rows


def m3_token_sensitivity(predictions: pd.DataFrame, engine: str = "M3") -> pd.DataFrame:
    """Abstención de M3 tal como la reportó el arnés y contando ``ESCALAR`` en el texto.

    Es un análisis de sensibilidad, no reemplaza la métrica oficial: muestra cuánto cambia
    la abstención si el motor hubiera detectado la palabra ``ESCALAR`` en cualquier parte
    de la respuesta (no solo como respuesta completa).
    """
    data = (
        predictions
        if "intent_hit" in predictions
        else per_question_outcomes(predictions)
    )
    g = data[data["engine"] == engine]
    if g.empty:
        return pd.DataFrame()
    leak = g["m3_token_leak"].fillna(0).astype(bool)
    soft = g["m3_soft_redirect"].fillna(0).astype(bool)
    variants = {
        "oficial (escalate del arnés)": g["escalate"],
        "+ 'ESCALAR' en el texto": g["escalate"] | leak,
        "+ remite a una persona (heurística)": g["escalate"] | leak | soft,
    }
    rows = []
    for name, esc in variants.items():
        rows.append(
            {
                "variante": name,
                "abstencion_correcta": float(esc[g["should_escalate"]].mean()),
                "abstencion_innecesaria": float(esc[~g["should_escalate"]].mean()),
                "n_escalo": int(esc.sum()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Validación cruzada fuera de pliegue (solo train) y motores combinados
# --------------------------------------------------------------------------------------
def describe_engine(engine: object) -> dict[str, object]:
    """Lee los hiperparámetros que eligió ``fit()`` si el motor los expone."""
    info: dict[str, object] = {"engine": getattr(engine, "name", type(engine).__name__)}
    for attr in ("best_c_", "threshold", "cv_folds_", "model_name", "top_k"):
        if hasattr(engine, attr):
            value = getattr(engine, attr)
            info[attr] = (
                value if np.isscalar(value) or isinstance(value, str) else str(value)
            )
    return info


def out_of_fold_predictions(
    engine_factory: Callable[[], object],
    train_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predicciones fuera de pliegue sobre ``train`` usando solo ``fit()`` y ``respond()``.

    Cada pregunta la predice un motor entrenado sin ella. Lo que el motor elija por dentro
    (por ejemplo, la validación cruzada interna de M1 para ``C`` y el umbral) se repite en
    cada pliegue, así que la estimación es honesta (validación cruzada anidada). El test no
    participa.

    Args:
        engine_factory: Función sin argumentos que crea un motor nuevo.
        train_df: Filas de ``labeled.csv`` con ``split == "train"``.
        n_splits: Número de pliegues (se reduce si alguna intención tiene menos filas).
        seed: Semilla de la partición en pliegues.

    Returns:
        ``(predicciones, elecciones)``: predicciones con el mismo esquema que
        ``manual_eval`` y una fila por pliegue con lo que eligió ``fit()``.

    Raises:
        ValueError: Si ``train_df`` trae filas de test.
    """
    from sklearn.model_selection import StratifiedKFold

    if (train_df["split"].astype(str).str.strip() != "train").any():
        raise ValueError("out_of_fold_predictions solo acepta filas de train.")
    data = train_df.reset_index(drop=True)
    min_count = int(data["intent"].value_counts().min())
    folds = max(2, min(n_splits, min_count))
    if folds < n_splits:
        warnings.warn(
            f"Alguna intención tiene {min_count} filas; se usan {folds} pliegues.",
            UserWarning,
            stacklevel=2,
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    choices: list[dict[str, object]] = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(data, data["intent"])):
        engine = engine_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            engine.fit(data.iloc[fit_idx].copy())
        info = describe_engine(engine)
        info.update({"fold": fold, "n_fit": len(fit_idx), "n_val": len(val_idx)})
        choices.append(info)
        for _, item in data.iloc[val_idx].iterrows():
            response = engine.respond(str(item["question_text"]))
            rows.append(
                {
                    "engine": getattr(engine, "name", "?"),
                    "train_source": "real",
                    "source": item.get("source", "real"),
                    "split": "train",
                    "question_id": str(item["question_id"]),
                    "intent": item["intent"],
                    "predicted_intent": response.intent,
                    "should_escalate": item["should_escalate"],
                    "escalate": bool(response.escalate),
                    "kb_source": item.get("kb_source", ""),
                    "answer": response.answer,
                    "retrieved_str": "|".join(response.retrieved or []),
                    "confidence": response.confidence,
                    "latency_ms": response.latency_ms,
                    "fold": fold,
                }
            )
    predictions = pd.DataFrame(rows)
    predictions["should_escalate"] = _to_bool(
        predictions["should_escalate"].astype(str)
    )
    return predictions, pd.DataFrame(choices)


def simulate_gate(
    predictions: pd.DataFrame, gate: str, answerer: str, name: str | None = None
) -> pd.DataFrame:
    """Simula un motor combinado: ``gate`` decide si escalar y ``answerer`` responde.

    La pregunta escala si escala cualquiera de los dos; si no, la respuesta, la intención y
    los fragmentos son los de ``answerer``. Es una simulación a partir de predicciones ya
    guardadas (no se ejecuta ningún motor), así que la latencia no se puede medir aquí.

    Args:
        predictions: Filas de ``manual_eval`` (o fuera de pliegue) de ambos motores.
        gate: Motor que decide escalar (p. ej. ``"M1"``).
        answerer: Motor que redacta o devuelve la respuesta (p. ej. ``"M2"`` o ``"M3"``).
        name: Nombre de la combinación (por defecto ``"<gate>+<answerer>"``).

    Returns:
        Filas con el mismo esquema que ``predictions`` y ``engine = name``.
    """
    name = name or f"{gate}+{answerer}"
    g = predictions[predictions["engine"] == gate].set_index("question_id")
    a = predictions[predictions["engine"] == answerer].set_index("question_id")
    if g.empty or a.empty:
        raise ValueError(f"Faltan predicciones de {gate} o {answerer}.")
    g = g.reindex(a.index)
    combined = a.copy()
    gate_escalates = _to_bool(g["escalate"])
    combined["escalate"] = _to_bool(a["escalate"]) | gate_escalates
    combined.loc[combined["escalate"], "answer"] = np.nan
    combined["engine"] = name
    return combined.reset_index()


# --------------------------------------------------------------------------------------
# Calificación manual de M3 (corrección y alucinación)
# --------------------------------------------------------------------------------------
TIPO_RESPONDABLE = "respondible"
TIPO_SHOULD_ESCALATE = "debia_escalar"
REDIRECT_NOTE = re.compile(r"remit", flags=re.IGNORECASE)


def _read_sheet(path: Path | str) -> pd.DataFrame:
    """Lee una hoja de calificación CSV con separador ``,`` o ``;`` (Excel en español)."""
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )


def _parse_grades(
    values: pd.Series, allowed: set[int], column: str, ids: pd.Series
) -> pd.Series:
    """Convierte notas de texto a números; vacío -> NaN; falla si hay valores no válidos."""
    text = values.astype(str).str.strip().str.replace(",", ".", regex=False)
    numbers = pd.to_numeric(text.where(text != ""), errors="coerce")
    bad = (text != "") & (numbers.isna() | ~numbers.isin(sorted(allowed)))
    if bad.any():
        examples = ", ".join(f"{i}={v!r}" for i, v in zip(ids[bad], values[bad]))
        raise ValueError(
            f"{column}: valores no válidos (se esperaba {sorted(allowed)}): {examples}"
        )
    return numbers


def load_manual_grades(
    path_rater_1: Path | str, path_rater_2: Path | str
) -> pd.DataFrame:
    """Une y valida las hojas de los dos calificadores.

    Cada calificador llena solo sus columnas (``*_rater_1`` o ``*_rater_2``) en su propia
    copia de la hoja. Esta función toma de cada archivo las columnas que le corresponden.

    Reglas que se comprueban:

    * Las dos hojas tienen las mismas preguntas (``question_id``) con el mismo ``tipo``.
    * Filas ``respondible``: corrección 0, 1 o 2 y alucinación 0 o 1, obligatorias.
    * Filas ``debia_escalar``: alucinación 0 o 1 obligatoria; la corrección no aplica y se
      ignora si alguien la llenó.

    Args:
        path_rater_1: Hoja del calificador 1.
        path_rater_2: Hoja del calificador 2.

    Returns:
        Una fila por pregunta con ``question_id, tipo, intent`` y, por calificador,
        ``correctness_N``, ``hallucination_N`` y ``redirect_N`` (la nota dice "remite").

    Raises:
        ValueError: Si faltan columnas, las preguntas no coinciden o hay notas no válidas
            o vacías donde son obligatorias.
    """
    frames = []
    for n, path in ((1, path_rater_1), (2, path_rater_2)):
        sheet = _read_sheet(path)
        needed = {
            "tipo",
            "question_id",
            f"correctness_rater_{n}",
            f"hallucination_rater_{n}",
            f"notes_rater_{n}",
        }
        missing = sorted(needed - set(sheet.columns))
        if missing:
            raise ValueError(f"{Path(path).name}: faltan columnas {missing}")
        sheet["question_id"] = sheet["question_id"].str.strip()
        sheet["tipo"] = sheet["tipo"].str.strip()
        if sheet["question_id"].duplicated().any():
            dup = sheet.loc[sheet["question_id"].duplicated(), "question_id"].tolist()
            raise ValueError(f"{Path(path).name}: question_id repetidos {dup}")
        bad_tipo = sorted(set(sheet["tipo"]) - {TIPO_RESPONDABLE, TIPO_SHOULD_ESCALATE})
        if bad_tipo:
            raise ValueError(f"{Path(path).name}: tipo no válido {bad_tipo}")
        ids = sheet["question_id"]
        out = sheet[["question_id", "tipo"]].copy()
        if "intent" in sheet:
            out["intent"] = sheet["intent"].str.strip()
        out[f"correctness_{n}"] = _parse_grades(
            sheet[f"correctness_rater_{n}"], {0, 1, 2}, f"correctness_rater_{n}", ids
        )
        out[f"hallucination_{n}"] = _parse_grades(
            sheet[f"hallucination_rater_{n}"], {0, 1}, f"hallucination_rater_{n}", ids
        )
        out[f"redirect_{n}"] = (
            sheet[f"notes_rater_{n}"]
            .astype(str)
            .str.contains(REDIRECT_NOTE)
            .astype(float)
        )
        is_resp = out["tipo"] == TIPO_RESPONDABLE
        ignored = ~is_resp & out[f"correctness_{n}"].notna()
        if ignored.any():
            warnings.warn(
                f"{Path(path).name}: se ignora la corrección en {int(ignored.sum())} filas "
                "'debia_escalar' (solo se califica alucinación).",
                UserWarning,
                stacklevel=2,
            )
            out.loc[ignored, f"correctness_{n}"] = np.nan
        empty_c = is_resp & out[f"correctness_{n}"].isna()
        empty_h = out[f"hallucination_{n}"].isna()
        if empty_c.any() or empty_h.any():
            raise ValueError(
                f"{Path(path).name}: faltan notas. Corrección vacía en "
                f"{ids[empty_c].tolist()}; alucinación vacía en {ids[empty_h].tolist()}"
            )
        frames.append(out)

    first, second = frames
    if set(first["question_id"]) != set(second["question_id"]):
        only_1 = sorted(set(first["question_id"]) - set(second["question_id"]))
        only_2 = sorted(set(second["question_id"]) - set(first["question_id"]))
        raise ValueError(
            f"Las hojas no tienen las mismas preguntas. Solo en la 1: {only_1}; "
            f"solo en la 2: {only_2}"
        )
    second = second.drop(columns=[c for c in ("intent",) if c in second])
    merged = first.merge(second, on="question_id", suffixes=("", "_hoja2"))
    mismatch = merged["tipo"] != merged["tipo_hoja2"]
    if mismatch.any():
        raise ValueError(
            f"El tipo no coincide entre hojas en {merged.loc[mismatch, 'question_id'].tolist()}"
        )
    return merged.drop(columns=["tipo_hoja2"]).reset_index(drop=True)


def manual_disagreements(grades: pd.DataFrame) -> pd.DataFrame:
    """Filas en que los calificadores no coinciden, con columnas vacías para el consenso.

    Se llenan en la reunión (``correctness_final`` solo en filas ``respondible``).
    """
    resp = grades["tipo"] == TIPO_RESPONDABLE
    differs = (resp & (grades["correctness_1"] != grades["correctness_2"])) | (
        grades["hallucination_1"] != grades["hallucination_2"]
    )
    out = grades.loc[
        differs,
        [
            "question_id",
            "tipo",
            "correctness_1",
            "correctness_2",
            "hallucination_1",
            "hallucination_2",
        ],
    ].copy()
    out["correctness_final"] = ""
    out["hallucination_final"] = ""
    return out.reset_index(drop=True)


def apply_resolutions(
    grades: pd.DataFrame, resolutions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Agrega ``correctness_final`` y ``hallucination_final`` (consenso).

    Donde los dos calificadores coinciden, ese es el valor final. Donde no, se toma el valor
    decidido en reunión (``resolutions``); si todavía no existe, queda ``NaN`` y la fila no
    entra en las métricas de consenso.
    """
    out = grades.copy()
    for metric in ("correctness", "hallucination"):
        a, b = out[f"{metric}_1"], out[f"{metric}_2"]
        out[f"{metric}_final"] = np.where(a == b, a, np.nan)
    if resolutions is not None and not resolutions.empty:
        res = resolutions.copy()
        res["question_id"] = res["question_id"].astype(str).str.strip()
        res = res.set_index("question_id")
        for metric, allowed in (("correctness", {0, 1, 2}), ("hallucination", {0, 1})):
            column = f"{metric}_final"
            if column not in res:
                continue
            parsed = _parse_grades(
                res[column].astype(str), allowed, column, pd.Series(res.index)
            )
            parsed.index = res.index
            idx = out["question_id"].map(parsed)
            fill = out[column].isna() & idx.notna()
            out.loc[fill, column] = idx[fill]
    out.loc[out["tipo"] != TIPO_RESPONDABLE, "correctness_final"] = np.nan
    return out


def _cohen_kappa(a: pd.Series, b: pd.Series, weights: str | None) -> float:
    """Kappa de Cohen; ``NaN`` si solo hay una categoría (kappa no está definido)."""
    from sklearn.metrics import cohen_kappa_score

    if len(set(a) | set(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a.astype(int), b.astype(int), weights=weights))


def manual_agreement(grades: pd.DataFrame) -> pd.DataFrame:
    """Acuerdo entre calificadores: % de coincidencia exacta y kappa de Cohen.

    La corrección (0/1/2) usa kappa con pesos cuadráticos: un desacuerdo 0 vs 2 pesa más
    que 1 vs 2. La alucinación y "remite" usan kappa simple. Con casi todas las notas en
    una sola categoría, kappa puede ser bajo aunque el acuerdo sea alto: por eso se
    reportan los dos.
    """
    resp = grades["tipo"] == TIPO_RESPONDABLE
    esc = grades["tipo"] == TIPO_SHOULD_ESCALATE
    specs = [
        ("correccion", "respondible", resp, "correctness", "quadratic"),
        ("alucinacion", "todas", resp | esc, "hallucination", None),
        ("alucinacion", "respondible", resp, "hallucination", None),
        ("alucinacion", "debia_escalar", esc, "hallucination", None),
        ("remite_a_persona", "debia_escalar", esc, "redirect", None),
    ]
    rows = []
    for name, group, mask, column, weights in specs:
        a = grades.loc[mask, f"{column}_1"]
        b = grades.loc[mask, f"{column}_2"]
        if a.empty:
            continue
        rows.append(
            {
                "medida": name,
                "grupo": group,
                "n": len(a),
                "acuerdo_exacto": float((a == b).mean()),
                "kappa": _cohen_kappa(a, b, weights),
                "tipo_kappa": "ponderado cuadrático" if weights else "simple",
                "n_desacuerdos": int((a != b).sum()),
            }
        )
    return pd.DataFrame(rows)


def manual_grade_summary(
    grades: pd.DataFrame, n_boot: int = N_BOOT_DEFAULT, seed: int = SEED
) -> pd.DataFrame:
    """Métricas de la calificación manual con IC bootstrap, por fuente de la nota.

    Fuentes: ``calificador_1``, ``calificador_2``, ``promedio`` (media de los dos por
    pregunta) y ``consenso`` (si ya se aplicó ``apply_resolutions``; solo filas con valor
    final).

    Métricas:

    * ``correccion_media`` (0–2), ``correcta_completa`` (% con nota 2) y
      ``correcta_al_menos_parcial`` (% con nota ≥ 1), sobre las respondibles.
    * ``alucinacion`` (% de respuestas con algún dato que no está en la base), en
      respondibles, en las que debían escalar (muestra) y en todas.
    * ``remite_a_persona`` (% de respuestas a preguntas que debían escalar que mandan al
      cliente con una persona), según las notas.
    """
    resp = grades["tipo"] == TIPO_RESPONDABLE
    esc = grades["tipo"] == TIPO_SHOULD_ESCALATE

    def full(c: pd.Series) -> pd.Series:
        return (c == 2).astype(float).where(c.notna())

    def any_credit(c: pd.Series) -> pd.Series:
        return (c >= 1).astype(float).where(c.notna())

    def per_rater(n: int) -> dict[str, pd.Series]:
        c = grades[f"correctness_{n}"]
        return {
            "c": c,
            "full": full(c),
            "any": any_credit(c),
            "h": grades[f"hallucination_{n}"],
            "r": grades[f"redirect_{n}"],
        }

    r1, r2 = per_rater(1), per_rater(2)
    # "promedio": por pregunta, la media de los dos calificadores en cada indicador.
    sources: dict[str, dict[str, pd.Series]] = {
        "calificador_1": r1,
        "calificador_2": r2,
        "promedio": {key: (r1[key] + r2[key]) / 2 for key in r1},
    }
    if "correctness_final" in grades:
        c = grades["correctness_final"]
        sources["consenso"] = {
            "c": c,
            "full": full(c),
            "any": any_credit(c),
            "h": grades["hallucination_final"],
            "r": r1["r"].where(r1["r"] == r2["r"]),
        }
    rows = []
    for source, cols in sources.items():
        metrics = [
            ("correccion_media", "respondible", cols["c"][resp]),
            ("correcta_completa", "respondible", cols["full"][resp]),
            ("correcta_al_menos_parcial", "respondible", cols["any"][resp]),
            ("alucinacion", "respondible", cols["h"][resp]),
            ("alucinacion", "debia_escalar", cols["h"][esc]),
            ("alucinacion", "todas", cols["h"][resp | esc]),
            ("remite_a_persona", "debia_escalar", cols["r"][esc]),
        ]
        for metric, group, values in metrics:
            value, low, high, n = bootstrap_ci(
                values.astype(float), n_boot=n_boot, seed=seed
            )
            rows.append(
                {
                    "fuente": source,
                    "metrica": metric,
                    "grupo": group,
                    "valor": value,
                    "ci_low": low,
                    "ci_high": high,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def apply_manual_correctness(
    outcomes: pd.DataFrame, grades: pd.DataFrame, column: str = "correctness_final"
) -> pd.DataFrame:
    """Lleva la corrección manual de M3 a las filas de M3 (y de combinaciones con M3).

    Afecta las filas respondibles de M3. Si el motor escaló, la respuesta cuenta como no
    correcta (igual que en M0–M2). Si respondió: nota 2 cuenta como
    correcta (``auto_correct = 1``); notas 0 y 1 como no correctas (``auto_correct = 0``),
    y la nota 1 se marca además en ``manual_partial`` para mostrarla aparte en las figuras.
    Las filas sin nota quedan sin calificar.

    Args:
        outcomes: Salida de ``per_question_outcomes``.
        grades: Salida de ``load_manual_grades`` (y opcionalmente ``apply_resolutions``).
        column: Nota a usar (``correctness_final``, ``correctness_1``, ...).

    Returns:
        Copia de ``outcomes`` con ``auto_correct`` y ``manual_partial`` actualizados.
    """
    out = outcomes.copy()
    if "manual_partial" not in out:
        out["manual_partial"] = np.nan
    notes = grades.set_index("question_id")[column]
    is_m3 = out["engine"].astype(str).str.contains("M3", regex=False)
    target = is_m3 & ~out["should_escalate"] & ~out["escalate"]
    # Escalar una pregunta respondible no es una respuesta correcta (igual que en M0–M2).
    escalated = is_m3 & ~out["should_escalate"] & out["escalate"]
    out.loc[escalated, "auto_correct"] = 0.0
    out.loc[escalated, "manual_partial"] = 0.0
    mapped = out.loc[target, "question_id"].astype(str).map(notes)
    graded = mapped.notna()
    idx = mapped.index[graded]
    out.loc[idx, "auto_correct"] = (mapped[graded] == 2).astype(float)
    out.loc[idx, "manual_partial"] = (mapped[graded] == 1).astype(float)
    return out


def plot_manual_grades(
    summary: pd.DataFrame,
    path: Path | None = None,
    source: str = "consenso",
    title: str | None = None,
):
    """Corrección (izq.) y alucinación (der.) de M3 según la calificación manual."""
    import matplotlib.pyplot as plt

    data = summary[summary["fuente"] == source]
    if data.empty or data["n"].sum() == 0:
        data = summary[summary["fuente"] == "promedio"]
        source = "promedio"
    source_en = {
        "consenso": "consensus",
        "promedio": "average of both raters",
        "calificador_1": "rater 1",
        "calificador_2": "rater 2",
    }.get(source, source)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    panels = [
        (
            axes[0],
            [
                ("correcta_completa", "respondible", "Fully correct (2)"),
                ("correcta_al_menos_parcial", "respondible", "At least partially (≥1)"),
            ],
            "Correctness on answerable questions",
            STATUS_COLORS["good"],
        ),
        (
            axes[1],
            [
                ("alucinacion", "respondible", "Answerable"),
                ("alucinacion", "debia_escalar", "Should have escalated"),
                ("remite_a_persona", "debia_escalar", "Refers to a person*"),
            ],
            "Hallucination and redirection",
            STATUS_COLORS["critical"],
        ),
    ]
    for ax, items, panel_title, color in panels:
        _style_axes(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        y = np.arange(len(items))[::-1]
        for k, (metric, group, label) in enumerate(items):
            row = data[(data["metrica"] == metric) & (data["grupo"] == group)]
            if row.empty or row["n"].iloc[0] == 0:
                continue
            r = row.iloc[0]
            bar_color = INK_SECONDARY if metric == "remite_a_persona" else color
            ax.barh(y[k], r["valor"], height=0.55, color=bar_color, edgecolor="white")
            ax.errorbar(
                r["valor"],
                y[k],
                xerr=[[r["valor"] - r["ci_low"]], [r["ci_high"] - r["valor"]]],
                fmt="none",
                ecolor=INK,
                capsize=3,
                elinewidth=1.2,
            )
            ax.text(
                min(r["ci_high"] + 0.02, 0.98),
                y[k],
                f"{r['valor']:.0%} (n={int(r['n'])})",
                va="center",
                fontsize=9.5,
                color=INK,
            )
        ax.set_yticks(y, [label for _, _, label in items])
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(panel_title, loc="left", fontsize=11, color=INK)
    fig.suptitle(
        title or f"M3 manual grading · test ({source_en}; *from rater notes)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    if path is not None:
        _save(fig, path)
    return fig


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------
def _style_axes(ax) -> None:
    """Ejes discretos: sin marco superior/derecho, cuadrícula tenue."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=10)
    ax.set_axisbelow(True)


def _save(fig, path: Path, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Guarda la figura en cada formato (PNG a 300 dpi) y la cierra."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
        saved.append(target)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return saved


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", list(SEQUENTIAL_BLUE))


def plot_confusion(
    matrix: pd.DataFrame,
    title: str,
    path: Path | None = None,
    ax=None,
    show_colorbar: bool = True,
):
    """Matriz de confusión: color = % de la fila, texto = conteo y %.

    El color por fila (recall) evita que las intenciones grandes (disponibilidad,
    otro_negocio) dominen la escala. Las intenciones que escalan van marcadas con (E).

    Args:
        matrix: Conteos con filas = intención real y columnas = predicha.
        title: Título (en inglés para diapositivas).
        path: Si se da, guarda PNG y PDF con ese nombre base.
        ax: Eje existente (para cuadrículas de varias matrices).
        show_colorbar: Dibuja la barra de color.

    Returns:
        La figura de matplotlib.
    """
    import matplotlib.pyplot as plt

    m = matrix.reindex(index=list(INTENTS), columns=list(INTENTS), fill_value=0)
    counts = m.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
    else:
        fig = ax.figure
    image = ax.imshow(pct, cmap=_sequential_cmap(), vmin=0, vmax=1)
    labels = [
        f"{INTENT_LABELS_EN[i]} (E)" if i in ESCALATING_INTENTS else INTENT_LABELS_EN[i]
        for i in INTENTS
    ]
    ax.set_xticks(range(len(INTENTS)), labels, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(
        range(len(INTENTS)),
        [
            f"{lab} (n={int(s)})"
            for lab, s in zip(labels, row_sums.ravel(), strict=True)
        ],
        fontsize=10,
    )
    ax.set_xlabel("Predicted intent    (E) = must be escalated", color=INK_SECONDARY)
    ax.set_ylabel("True intent", color=INK_SECONDARY)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if counts[i, j] == 0:
                continue
            color = "white" if pct[i, j] > 0.55 else INK
            ax.text(
                j,
                i,
                f"{int(counts[i, j])}\n{pct[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
                linespacing=1.1,
            )
    # Separación de 2 px entre celdas.
    ax.set_xticks(np.arange(-0.5, len(INTENTS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(INTENTS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if show_colorbar:
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label("Share of true-intent row", color=INK_SECONDARY)
        bar.outline.set_visible(False)
    if own_fig and path is not None:
        _save(fig, path)
    return fig


def plot_confusion_grid(
    matrices: dict[str, pd.DataFrame], titles: dict[str, str], path: Path | None = None
):
    """Varias matrices lado a lado (una sola diapositiva)."""
    import matplotlib.pyplot as plt

    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for k, (ax, (engine, matrix)) in enumerate(
        zip(axes, matrices.items(), strict=True)
    ):
        plot_confusion(
            matrix, titles.get(engine, engine), ax=ax, show_colorbar=(k == n - 1)
        )
        if k > 0:
            ax.set_ylabel("")
    if path is not None:
        _save(fig, path)
    return fig


def plot_metric_panels(
    metrics: pd.DataFrame,
    metric_names: Sequence[str],
    path: Path | None = None,
    engine_labels: dict[str, str] | None = None,
    na_notes: dict[tuple[str, str], str] | None = None,
    title: str | None = None,
):
    """Un panel por métrica, una barra por motor con su IC 95 % (múltiplos pequeños).

    Args:
        metrics: Salida de ``compute_metrics``.
        metric_names: Métricas a dibujar (un panel cada una).
        path: Nombre base para guardar.
        engine_labels: Nombres a mostrar por motor.
        na_notes: Texto para celdas motor×métrica que no aplican, p. ej.
            ``{("M2", "intent_accuracy"): "n/a\\nby design"}``.
        title: Título general.

    Returns:
        La figura.
    """
    import matplotlib.pyplot as plt

    labels = {**ENGINE_LABELS, **(engine_labels or {})}
    engines = _engines_in(metrics["engine"])
    na_notes = na_notes or {}
    n = len(metric_names)
    fig, axes = plt.subplots(
        1, n, figsize=(3.4 * n, 4.2), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(engines))
    for ax, metric in zip(axes, metric_names, strict=True):
        _style_axes(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        for k, engine in enumerate(engines):
            row = metrics[(metrics["engine"] == engine) & (metrics["metric"] == metric)]
            note = na_notes.get((engine, metric))
            if row.empty or note:
                ax.text(
                    k,
                    0.03,
                    note or "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=INK_SECONDARY,
                )
                continue
            r = row.iloc[0]
            color = ENGINE_COLORS.get(engine, STATUS_COLORS["neutral"])
            ax.bar(
                k, r["value"], width=0.62, color=color, edgecolor="white", linewidth=2
            )
            ax.errorbar(
                k,
                r["value"],
                yerr=[[r["value"] - r["ci_low"]], [r["ci_high"] - r["value"]]],
                fmt="none",
                ecolor=INK_SECONDARY,
                elinewidth=1.2,
                capsize=3,
            )
            ax.text(
                k,
                min(r["ci_high"] + 0.03, 1.05),
                f"{r['value']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=INK,
            )
        ax.set_xticks(
            x, [labels.get(e, e).split(" ", 1)[0] for e in engines], fontsize=10
        )
        ax.set_ylim(0, 1.12)
        ax.set_title(
            METRIC_LABELS_EN.get(metric, metric), fontsize=10.5, color=INK, loc="left"
        )
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, color=ENGINE_COLORS.get(e, STATUS_COLORS["neutral"])
        )
        for e in engines
    ]
    fig.legend(
        handles,
        [labels.get(e, e) for e in engines],
        loc="lower center",
        ncol=len(engines),
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=10,
    )
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    if path is not None:
        _save(fig, path)
    return fig


def plot_escalation_outcomes(
    outcomes: pd.DataFrame,
    path: Path | None = None,
    engine_labels: dict[str, str] | None = None,
    title: str | None = None,
):
    """Dos paneles de barras 100 %: preguntas que debían escalar y preguntas respondibles.

    Es la figura que mejor resume el comportamiento de un agente que debe escalar: muestra a
    la vez cuántas veces respondió algo que debía pasar a una persona y cuántas respuestas
    útiles dio.
    """
    import matplotlib.pyplot as plt

    labels = {**ENGINE_LABELS, **(engine_labels or {})}
    data = outcomes.set_index("engine")
    engines = list(data.index)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 0.75 * len(engines) + 2.2),
        sharey=True,
        constrained_layout=True,
    )
    panels = [
        (
            axes[0],
            "n_should_escalate",
            [
                ("esc_ok", "Escalated (correct)", STATUS_COLORS["good"]),
                (
                    "esc_answered",
                    "Answered (should have escalated)",
                    STATUS_COLORS["critical"],
                ),
            ],
            "Questions that must be escalated",
        ),
        (
            axes[1],
            "n_respondable",
            [
                ("resp_correct", "Correct answer", STATUS_COLORS["good"]),
                ("resp_partial", "Partially correct", STATUS_COLORS["good_light"]),
                ("resp_wrong", "Wrong answer", STATUS_COLORS["critical"]),
                (
                    "resp_ungraded",
                    "Answered (manual grading)",
                    STATUS_COLORS["neutral"],
                ),
                ("resp_escalated", "Escalated (unnecessary)", STATUS_COLORS["warning"]),
            ],
            "Answerable questions",
        ),
    ]
    y = np.arange(len(engines))[::-1]
    for ax, total_col, parts, panel_title in panels:
        _style_axes(ax)
        left = np.zeros(len(engines))
        for column, label, color in parts:
            if column not in data:  # p. ej. "resp_partial" sin calificación manual
                continue
            share = data[column].to_numpy() / data[total_col].to_numpy()
            ax.barh(
                y,
                share,
                left=left,
                color=color,
                edgecolor="white",
                linewidth=2,
                height=0.62,
                label=label,
            )
            for k, value in enumerate(share):
                if value >= 0.07:
                    ax.text(
                        left[k] + value / 2,
                        y[k],
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        color=(
                            INK
                            if color
                            in (STATUS_COLORS["warning"], STATUS_COLORS["good_light"])
                            else "white"
                        ),
                    )
            left += share
        n_total = int(data[total_col].iloc[0])
        ax.set_title(
            f"{panel_title} (n={n_total})", loc="left", fontsize=11.5, color=INK
        )
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=9.5,
        )
    axes[0].set_yticks(y, [labels.get(e, e) for e in engines], fontsize=10.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    if path is not None:
        _save(fig, path)
    return fig


def plot_hit_at_k(
    metrics: pd.DataFrame,
    engine: str = "M2",
    path: Path | None = None,
    title: str | None = None,
):
    """Hit@1, Hit@3 y Hit@5 del recuperador (M2 y M3 comparten el mismo)."""
    import matplotlib.pyplot as plt

    rows = metrics[
        (metrics["engine"] == engine) & metrics["metric"].str.startswith("hit_at_")
    ]
    rows = rows.set_index("metric").reindex(["hit_at_1", "hit_at_3", "hit_at_5"])
    fig, ax = plt.subplots(figsize=(5.2, 4))
    _style_axes(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    x = np.arange(3)
    ax.bar(
        x,
        rows["value"],
        width=0.55,
        color=ENGINE_COLORS["M2"],
        edgecolor="white",
        linewidth=2,
    )
    ax.errorbar(
        x,
        rows["value"],
        yerr=[rows["value"] - rows["ci_low"], rows["ci_high"] - rows["value"]],
        fmt="none",
        ecolor=INK_SECONDARY,
        capsize=3,
        elinewidth=1.2,
    )
    for k, value in enumerate(rows["value"]):
        ax.text(
            k,
            min(rows["ci_high"].iloc[k] + 0.03, 1.05),
            f"{value:.2f}",
            ha="center",
            fontsize=10,
            color=INK,
        )
    ax.set_xticks(x, ["Hit@1", "Hit@3", "Hit@5"])
    ax.set_ylim(0, 1.12)
    n = int(rows["n"].iloc[0]) if rows["n"].notna().any() else 0
    ax.set_title(
        title or f"Retriever (M2 = M3), answerable questions (n={n})",
        loc="left",
        fontsize=11.5,
        color=INK,
    )
    if path is not None:
        _save(fig, path)
    return fig


def plot_latency(
    summary: pd.DataFrame, path: Path | None = None, title: str | None = None
):
    """Latencia mediana (barra) y p95 (marca) por motor, eje logarítmico."""
    import matplotlib.pyplot as plt

    data = summary.set_index("engine").reindex(
        [e for e in ENGINE_ORDER if e in set(summary["engine"])]
    )
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(data) + 1.6))
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    y = np.arange(len(data))[::-1]
    colors = [ENGINE_COLORS[e] for e in data.index]
    ax.barh(
        y,
        data["latency_median_ms"],
        color=colors,
        height=0.55,
        edgecolor="white",
        linewidth=2,
    )
    ax.scatter(
        data["latency_p95_ms"], y, marker="|", s=320, color=INK, linewidths=2, zorder=3
    )
    for k, engine in enumerate(data.index):
        med, p95 = (
            data.loc[engine, "latency_median_ms"],
            data.loc[engine, "latency_p95_ms"],
        )
        ax.text(
            p95 * 1.25,
            y[k],
            f"median {_fmt_ms(med)} · p95 {_fmt_ms(p95)}",
            va="center",
            fontsize=9.5,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_yticks(y, [ENGINE_LABELS.get(e, e) for e in data.index])
    ax.set_xlabel(
        "Latency per answer (ms, log scale) · bar = median, tick = 95th percentile",
        color=INK_SECONDARY,
    )
    hw = str(data["hardware"].dropna().iloc[0]) if "hardware" in data else ""
    ax.set_title(title or f"Latency · {hw}", loc="left", fontsize=11, color=INK)
    ax.set_xlim(data["latency_median_ms"].min() / 3, data["latency_p95_ms"].max() * 60)
    if path is not None:
        _save(fig, path)
    return fig


def _fmt_ms(value: float) -> str:
    """Formatea milisegundos de forma legible (ms o s)."""
    return f"{value / 1000:.1f} s" if value >= 1000 else f"{value:.2f} ms"


def plot_per_intent_recall(
    table: pd.DataFrame, path: Path | None = None, title: str | None = None
):
    """Recall de intención por intención y motor (barras horizontales agrupadas)."""
    import matplotlib.pyplot as plt

    engines = [e for e in ENGINE_ORDER if e in set(table["engine"].astype(str))]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    height = 0.8 / len(engines)
    base = np.arange(len(INTENTS))[::-1]
    for k, engine in enumerate(engines):
        sub = (
            table[table["engine"].astype(str) == engine]
            .set_index("intent")
            .reindex(list(INTENTS))
        )
        offsets = base + (len(engines) / 2 - k - 0.5) * height
        recall = sub["recall"].fillna(0).to_numpy()
        ax.barh(
            offsets,
            recall,
            height=height,
            color=ENGINE_COLORS[engine],
            edgecolor="white",
            linewidth=1.5,
            label=ENGINE_LABELS[engine],
        )
        for pos, value in zip(offsets, recall, strict=True):
            if value == 0:  # una barra vacía también es un resultado: se rotula
                ax.text(
                    0.005, pos, "0%", va="center", fontsize=7.5, color=INK_SECONDARY
                )
    counts = (
        table.drop_duplicates("intent").set_index("intent")["n"].reindex(list(INTENTS))
    )
    ax.set_yticks(
        base, [f"{INTENT_LABELS_EN[i]} (n={int(counts[i])})" for i in INTENTS]
    )
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(
        "Recall (share of questions with the right intent)", color=INK_SECONDARY
    )
    ax.set_title(
        title or "Intent recall per intent", loc="left", fontsize=11.5, color=INK
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(engines), frameon=False
    )
    if path is not None:
        _save(fig, path)
    return fig


# --------------------------------------------------------------------------------------
# Tablas para el informe
# --------------------------------------------------------------------------------------
def _latex_escape(text: object) -> str:
    """Escapa los caracteres especiales de LaTeX."""
    value = (
        ""
        if text is None or (isinstance(text, float) and np.isnan(text))
        else str(text)
    )
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        value = value.replace(char, repl)
    return value.replace("–", "--")


def to_latex_table(
    df: pd.DataFrame, caption: str, label: str, index: bool = True
) -> str:
    """Tabla ``booktabs`` simple para la plantilla IEEE (sin depender de Jinja2)."""
    frame = df.reset_index() if index else df
    cols = "l" * len(frame.columns)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in frame.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(_latex_escape(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def save_table(
    df: pd.DataFrame, path: Path, caption: str, label: str, index: bool = True
) -> list[Path]:
    """Guarda una tabla en CSV (UTF-8) y en LaTeX con el mismo nombre base."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    tex_path = path.with_suffix(".tex")
    df.to_csv(csv_path, index=index, encoding="utf-8")
    tex_path.write_text(
        to_latex_table(df, caption, label, index=index), encoding="utf-8"
    )
    return [csv_path, tex_path]
