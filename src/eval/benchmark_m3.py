"""Mide latencia y comportamiento de M3 con unas pocas preguntas de ``train``.

Sirve para dos cosas antes de lanzar la evaluación completa (que es lenta):

1. Saber cuánto tarda de verdad el modelo en este equipo, con distintos números de
   fragmentos de contexto (``--k-values``).
2. Ver el prompt exacto y las respuestas de unas cuantas preguntas, para revisarlas a ojo.

Nunca usa filas de ``test``. Las preguntas se eligen al azar con la semilla del proyecto.

Uso (desde la raíz del repositorio; Ollama debe estar corriendo)::

    python -m src.eval.benchmark_m3 --show-prompt
    python -m src.eval.benchmark_m3 --n-answerable 10 --n-escalating 5 --k-values 3 5
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    HARDWARE_LABEL,
    LLM_MODEL,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
    SPLIT_TRAIN,
)
from src.engines.m2_embeddings import EmbeddingEngine
from src.engines.m3_rag import LlmClient, RagEngine

DEFAULT_OUTPUT = REPORTS_DIR / "m3_benchmark.csv"


def _as_bool(value: object) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def sample_questions(
    data: pd.DataFrame, n_answerable: int, n_escalating: int
) -> pd.DataFrame:
    """Elige preguntas de ``train`` al azar (semilla fija): respondibles y que escalan."""
    train = data.loc[data["split"].astype(str).str.strip() == SPLIT_TRAIN].copy()
    escalate = train["should_escalate"].map(_as_bool)
    parts = [
        train.loc[~escalate].sample(
            n=min(n_answerable, int((~escalate).sum())), random_state=SEED
        ),
        train.loc[escalate].sample(
            n=min(n_escalating, int(escalate.sum())), random_state=SEED
        ),
    ]
    return pd.concat(parts, ignore_index=True)


def run_benchmark(
    questions: pd.DataFrame,
    k_values: Sequence[int],
    retriever: EmbeddingEngine | None = None,
    show_prompt: bool = False,
    llm: LlmClient | None = None,
) -> pd.DataFrame:
    """Corre M3 sobre ``questions`` para cada valor de ``k`` y devuelve una fila por consulta."""
    retriever = retriever or EmbeddingEngine()
    rows: list[dict[str, object]] = []
    for k in k_values:
        engine = RagEngine(retriever=retriever, llm=llm, context_k=k)
        for position, record in enumerate(questions.to_dict(orient="records")):
            response = engine.respond(str(record["question_text"]))
            if show_prompt and position == 0 and engine.last_prompt is not None:
                system, user = engine.last_prompt
                print(
                    f"\n===== PROMPT EXACTO (k={k}) =====\n[SISTEMA]\n{system}\n\n[USUARIO]\n{user}"
                )
            should_escalate = _as_bool(record["should_escalate"])
            kb_source = str(record.get("kb_source", "")).strip()
            rows.append(
                {
                    "k": k,
                    "question_id": record["question_id"],
                    "should_escalate": should_escalate,
                    "escalated": response.escalate,
                    "kb_source": kb_source,
                    "top1": response.retrieved[0] if response.retrieved else "",
                    "kb_source_in_context": bool(kb_source)
                    and kb_source in response.retrieved[:k],
                    "latency_s": (response.latency_ms or 0.0) / 1000.0,
                    "answer": response.answer or "",
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Resumen por ``k``: latencia (mediana y p95) y qué tanto escala el modelo."""
    summary: list[dict[str, object]] = []
    for k, group in results.groupby("k"):
        answerable = group.loc[~group["should_escalate"]]
        escalating = group.loc[group["should_escalate"]]
        with_source = answerable.loc[answerable["kb_source"] != ""]
        summary.append(
            {
                "k": k,
                "n": len(group),
                "latency_median_s": float(group["latency_s"].median()),
                "latency_p95_s": float(np.percentile(group["latency_s"], 95)),
                "answerable_escalated": (
                    float(answerable["escalated"].mean())
                    if len(answerable)
                    else float("nan")
                ),
                "escalating_escalated": (
                    float(escalating["escalated"].mean())
                    if len(escalating)
                    else float("nan")
                ),
                "correct_fragment_in_context": (
                    float(with_source["kb_source_in_context"].mean())
                    if len(with_source)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(summary)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "labeled.csv")
    parser.add_argument("--n-answerable", type=int, default=8)
    parser.add_argument("--n-escalating", type=int, default=4)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    data = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8")
    questions = sample_questions(data, args.n_answerable, args.n_escalating)
    print(
        f"Modelo: {LLM_MODEL} | Hardware: {HARDWARE_LABEL} | Preguntas: {len(questions)}"
    )

    results = run_benchmark(questions, args.k_values, show_prompt=args.show_prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nResumen por número de fragmentos (k):")
        print(summarize(results).round(3).to_string(index=False))
    print(f"\nDetalle guardado en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
