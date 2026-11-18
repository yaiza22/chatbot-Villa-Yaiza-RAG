"""Compara modelos de embeddings para elegir el de M2 (y del recuperador de M3).

Para cada modelo candidato (``EMBEDDING_CANDIDATES``) y cada forma de vectorizar los
fragmentos (un vector por fragmento o uno por párrafo), mide **solo con las filas de
``train``** (el test queda intacto):

* ``hit_at_1/3/5``: en las preguntas respondibles con ``kb_source``, qué porcentaje trae el
  fragmento correcto entre los ``k`` más parecidos. Mide si la búsqueda encuentra lo correcto.
* ``auroc_answerable_vs_escalating``: qué tan bien la similitud del mejor fragmento separa las
  preguntas que se pueden responder de las que deben escalar (1.0 = perfecta, 0.5 = azar).
  Mide si un umbral de similitud puede decidir cuándo escalar.
* ``median_ms_per_query``: latencia mediana de vectorizar una pregunta y buscar.

Si un grupo de preguntas no tiene ``kb_source`` (por ejemplo, datos reales todavía sin
etiquetar), sale con ``n_with_kb_source = 0`` y ``hit_at_k`` vacío (NaN), nunca inventado.

Uso (desde la raíz del repositorio; la primera vez descarga cada modelo de Hugging Face)::

    python -m src.eval.compare_embeddings
    python -m src.eval.compare_embeddings --models paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    EMBEDDING_CANDIDATES,
    KB_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLIT_TRAIN,
)
from src.engines.m2_embeddings import (
    EmbeddingEngine,
    Encoder,
    SentenceTransformerEncoder,
)
from src.eval.metrics import hit_at_k

EncoderFactory = Callable[[str], Encoder]
DEFAULT_OUTPUT = REPORTS_DIR / "m2_model_selection.csv"


def _as_bool(value: object) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def _auroc(answerable_scores: list[float], escalating_scores: list[float]) -> float:
    """AUROC de la similitud para separar respondibles (positivos) de las que escalan."""
    if not answerable_scores or not escalating_scores:
        return float("nan")
    labels = [1] * len(answerable_scores) + [0] * len(escalating_scores)
    return float(roc_auc_score(labels, answerable_scores + escalating_scores))


def compare_models(
    data: pd.DataFrame,
    models: Sequence[str] = EMBEDDING_CANDIDATES,
    kb_dir: Path | str = KB_DIR,
    encoder_factory: EncoderFactory = SentenceTransformerEncoder,
) -> pd.DataFrame:
    """Calcula la tabla de comparación (una fila por modelo, vectorización y ``source``).

    Args:
        data: Preguntas con ``question_text``, ``should_escalate``, ``kb_source``, ``split``
            y ``source``. Solo se usan las de ``train``.
        models: Nombres de los modelos a comparar.
        kb_dir: Carpeta con la base de conocimiento.
        encoder_factory: Crea el codificador de un modelo (se cambia en las pruebas).
    """
    train = data.loc[data["split"].astype(str).str.strip() == SPLIT_TRAIN].copy()
    rows: list[dict[str, object]] = []
    for model_name in models:
        load_start = time.perf_counter()
        encoder = encoder_factory(model_name)
        for split_paragraphs in (False, True):
            engine = EmbeddingEngine(
                kb_dir=kb_dir,
                encoder=encoder,
                model_name=model_name,
                split_paragraphs=split_paragraphs,
            )
            engine.retrieve("calentamiento")  # construye el índice antes de medir
            load_seconds = time.perf_counter() - load_start
            for source, group in train.groupby(train["source"].astype(str).str.strip()):
                answerable_scores: list[float] = []
                escalating_scores: list[float] = []
                kb_sources: list[str | None] = []
                retrieved: list[list[str]] = []
                latencies: list[float] = []
                for record in group.to_dict(orient="records"):
                    start = time.perf_counter()
                    ranked = engine.retrieve(str(record["question_text"]), k=5)
                    latencies.append((time.perf_counter() - start) * 1000.0)
                    top_score = ranked[0][1]
                    if _as_bool(record["should_escalate"]):
                        escalating_scores.append(top_score)
                        continue
                    answerable_scores.append(top_score)
                    kb_value = record.get("kb_source")
                    has_source = isinstance(kb_value, str) and kb_value.strip() != ""
                    if has_source:
                        kb_sources.append(str(kb_value).strip())
                        retrieved.append([fragment_id for fragment_id, _ in ranked])
                hits = {k: hit_at_k(kb_sources, retrieved, k)[0] for k in (1, 3, 5)}
                rows.append(
                    {
                        "model": model_name,
                        "split_paragraphs": split_paragraphs,
                        "source": source,
                        "n_answerable": len(answerable_scores),
                        "n_with_kb_source": len(kb_sources),
                        "n_escalating": len(escalating_scores),
                        "hit_at_1": hits[1],
                        "hit_at_3": hits[3],
                        "hit_at_5": hits[5],
                        "auroc_answerable_vs_escalating": _auroc(
                            answerable_scores, escalating_scores
                        ),
                        "median_top1_sim_answerable": (
                            statistics.median(answerable_scores)
                            if answerable_scores
                            else float("nan")
                        ),
                        "median_top1_sim_escalating": (
                            statistics.median(escalating_scores)
                            if escalating_scores
                            else float("nan")
                        ),
                        "median_ms_per_query": (
                            float(np.median(latencies)) if latencies else float("nan")
                        ),
                        "load_seconds": round(load_seconds, 2),
                    }
                )
    return pd.DataFrame(rows)


def _load_labeled(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [
        pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        for path in paths
        if path.exists()
    ]
    if not frames:
        raise FileNotFoundError(
            f"No se encontró ninguno de estos archivos: {list(paths)}"
        )
    return pd.concat(frames, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[PROCESSED_DIR / "seed.csv", PROCESSED_DIR / "labeled.csv"],
        help="CSV etiquetados (por defecto, la semilla y el conjunto real si existe).",
    )
    parser.add_argument("--models", nargs="+", default=list(EMBEDDING_CANDIDATES))
    parser.add_argument("--kb-dir", type=Path, default=KB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    table = compare_models(_load_labeled(args.inputs), args.models, args.kb_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))
    print(f"\nTabla guardada en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
