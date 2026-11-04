"""Arnés de evaluación reproducible para los motores conversacionales (M0–M3).

Separa explícitamente los datos de entrenamiento (``--train-input`` y
``--train-source``) de los de evaluación (``--eval-input`` y ``--split``).
Garantiza que ``fit()`` solo reciba filas de ``train``, bloquea la evaluación
sobre ``test`` a menos que se autorice explícitamente (``allow_test=True`` /
``--allow-test``), desglosa siempre los resultados por ``source`` y ``split``,
y exporta la tabla resumen, las matrices de confusión y la hoja de calificación
manual para dos revisores.

Uso desde la raíz del repositorio::

    python -m src.eval.run_eval --eval-input data/processed/seed.csv --engines M0 --split train --hardware "AMD Ryzen 5 8600HS, 64 GB RAM"
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    ESCALATING_INTENTS,
    INTENTS,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
    SPLIT_TEST,
    SPLIT_TRAIN,
)
from src.engines.base import Engine
from src.engines.m0_rules import RuleEngine
from src.eval.metrics import (
    compute_group_metrics,
    confusion_matrix_df,
    filter_for_manual_evaluation,
)

REQUIRED_COLUMNS = (
    "question_id",
    "question_text",
    "intent",
    "should_escalate",
    "reference_answer",
    "kb_source",
    "split",
    "source",
)
VALID_TRAIN_SOURCES = ("synthetic", "real", "both")
VALID_SPLITS = (SPLIT_TRAIN, SPLIT_TEST)
DEFAULT_SEED_CSV = PROCESSED_DIR / "seed.csv"

# Registro extensible de motores disponibles por nombre.
ENGINE_REGISTRY: dict[str, Callable[[], Engine]] = {
    "M0": RuleEngine,
}


@dataclass
class EvaluationResult:
    """Contenedor con todos los artefactos producidos por una corrida de evaluación.

    Attributes:
        summary_df: Tabla motor × métrica desglosada por ``source`` y ``split``.
        predictions_df: Predicciones individuales fila por fila.
        manual_eval_df: Hoja filtrada para calificación manual por dos personas.
        confusion_matrices: Diccionario ``(engine, source, split) -> DataFrame``.
    """

    summary_df: pd.DataFrame
    predictions_df: pd.DataFrame
    manual_eval_df: pd.DataFrame
    confusion_matrices: dict[tuple[str, str, str], pd.DataFrame]


def register_engine(name: str, factory: Callable[[], Engine]) -> None:
    """Registra un nuevo motor en el catálogo de evaluación.

    Args:
        name: Identificador del motor (p. ej. ``"M1"``, ``"M2"``, ``"M3"``).
        factory: Función o clase sin argumentos obligatorios que instancia el motor.
    """
    ENGINE_REGISTRY[name] = factory


def load_datasets(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Carga uno o varios archivos CSV etiquetados y valida su esquema.

    Args:
        paths: Lista de rutas a archivos CSV con el esquema de evaluación.

    Returns:
        DataFrame concatenado con tipos normalizados.

    Raises:
        ValueError: Si no se indica ninguna ruta o faltan columnas requeridas.
    """
    if not paths:
        raise ValueError("Debe proporcionar al menos un archivo CSV.")

    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"El archivo {path} no contiene las columnas requeridas: {missing}"
            )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["question_id", "source"], keep="first"
        ).reset_index(drop=True)
        combined["should_escalate"] = combined["should_escalate"].map(
            lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true"
        )
    return combined


def prepare_train_data(
    train_df: pd.DataFrame,
    train_source: str = "both",
) -> pd.DataFrame:
    """Filtra los datos de entrenamiento garantizando que nunca entren filas de test.

    Args:
        train_df: DataFrame candidato para entrenamiento.
        train_source: Origen permitido (``"synthetic"``, ``"real"`` o ``"both"``).

    Returns:
        Subconjunto con ``split == "train"`` y el ``source`` solicitado.

    Raises:
        ValueError: Si ``train_source`` no es un valor válido.
    """
    if train_source not in VALID_TRAIN_SOURCES:
        raise ValueError(
            f"train_source inválido: {train_source!r}. Opciones: {VALID_TRAIN_SOURCES}"
        )
    if train_df.empty:
        return train_df.copy()

    only_train = train_df.loc[
        train_df["split"].astype(str).str.strip() == SPLIT_TRAIN
    ].copy()
    if train_source != "both":
        only_train = only_train.loc[
            only_train["source"].astype(str).str.strip() == train_source
        ].copy()
    return only_train.reset_index(drop=True)


def build_manual_eval_sheet(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Construye la hoja CSV para calificación manual por dos revisores.

    Incluye las filas con ``reference_answer = ESCALAR`` y respuestas confirmadas,
    y excluye las filas con ``reference_answer = [POR CONFIRMAR]``.

    Args:
        predictions_df: DataFrame con las predicciones de los motores.

    Returns:
        DataFrame con las columnas listas para dos calificadores humanos.
    """
    base_cols = [
        "engine",
        "train_source",
        "source",
        "split",
        "question_id",
        "question_text",
        "intent",
        "predicted_intent",
        "should_escalate",
        "escalate",
        "reference_answer",
        "answer",
        "kb_source",
        "retrieved_str",
    ]
    rater_cols = [
        "correctness_rater_1",
        "hallucination_rater_1",
        "notes_rater_1",
        "correctness_rater_2",
        "hallucination_rater_2",
        "notes_rater_2",
    ]

    filtered = filter_for_manual_evaluation(predictions_df)
    if filtered.empty:
        return pd.DataFrame(columns=base_cols + rater_cols)

    sheet = filtered.reindex(columns=base_cols).copy()
    for col in rater_cols:
        sheet[col] = ""
    return sheet.reset_index(drop=True)


def evaluate(
    eval_df: pd.DataFrame,
    engine_names: Sequence[str] = ("M0",),
    split: str = SPLIT_TRAIN,
    allow_test: bool = False,
    train_df: pd.DataFrame | None = None,
    train_source: str = "both",
    hardware: str = "CPU local",
    seed: int = SEED,
    n_boot: int = 1000,
    engine_registry: Mapping[str, Callable[[], Engine]] | None = None,
) -> EvaluationResult:
    """Ejecuta la evaluación de uno o más motores separando por ``source`` y ``split``.

    Args:
        eval_df: DataFrame con las preguntas a evaluar.
        engine_names: Nombres de los motores a ejecutar (p. ej. ``["M0"]``).
        split: Partición a evaluar (``"train"`` o ``"test"``).
        allow_test: Debe ser ``True`` para permitir evaluar sobre ``split="test"``.
        train_df: DataFrame del cual extraer el conjunto para ``fit()``. Si es
            ``None``, se extrae únicamente la partición ``train`` de ``eval_df``.
        train_source: Filtro de origen para ``fit()`` (``"synthetic"``, ``"real"``
            o ``"both"``).
        hardware: Descripción textual del equipo donde se ejecuta la prueba.
        seed: Semilla para los intervalos de confianza bootstrap.
        n_boot: Número de remuestreos bootstrap.
        engine_registry: Mapa opcional de fábricas de motores.

    Returns:
        ``EvaluationResult`` con tabla de métricas, predicciones, hoja manual y
        matrices de confusión.

    Raises:
        ValueError: Si se solicita ``split="test"`` con ``allow_test=False``, si
            el ``split`` no es válido o si un motor no está registrado.
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Partición inválida: {split!r}. Valores permitidos: {VALID_SPLITS}"
        )
    if split == SPLIT_TEST and not allow_test:
        raise ValueError(
            "No se permite evaluar sobre la partición 'test' sin autorizarlo "
            "explícitamente con allow_test=True (--allow-test)."
        )
    if split == SPLIT_TEST and allow_test:
        print("ADVERTENCIA: evaluación sobre test: un solo run")

    registry = ENGINE_REGISTRY if engine_registry is None else engine_registry
    unknown_engines = [name for name in engine_names if name not in registry]
    if unknown_engines:
        raise ValueError(
            f"Motores no registrados: {unknown_engines}. "
            f"Disponibles: {sorted(registry)}"
        )

    raw_train = eval_df if train_df is None else train_df
    fit_df = prepare_train_data(raw_train, train_source=train_source)
    assert (
        fit_df.empty or (fit_df["split"].astype(str).str.strip() == SPLIT_TRAIN).all()
    )

    if eval_df.empty:
        eval_subset = eval_df.copy()
    else:
        eval_subset = (
            eval_df.loc[eval_df["split"].astype(str).str.strip() == split]
            .copy()
            .reset_index(drop=True)
        )

    all_prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[tuple[str, str, str], pd.DataFrame] = {}

    sources_present = (
        sorted(eval_subset["source"].astype(str).str.strip().unique().tolist())
        if not eval_subset.empty
        else ["none"]
    )

    for engine_name in engine_names:
        engine = registry[engine_name]()
        engine.fit(fit_df.copy())

        engine_records: list[dict[str, Any]] = []
        for row in eval_subset.to_dict(orient="records"):
            response = engine.respond(str(row["question_text"]))
            retrieved_list = list(response.retrieved or [])
            record: dict[str, Any] = {
                "engine": engine_name,
                "train_source": train_source,
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "intent": str(row["intent"]).strip(),
                "predicted_intent": response.intent,
                "confidence": response.confidence,
                "should_escalate": bool(row["should_escalate"]),
                "escalate": bool(response.escalate),
                "reference_answer": row["reference_answer"],
                "answer": response.answer,
                "kb_source": row["kb_source"],
                "retrieved": retrieved_list,
                "retrieved_str": "|".join(str(x) for x in retrieved_list),
                "latency_ms": response.latency_ms,
                "split": str(row["split"]).strip(),
                "source": str(row["source"]).strip(),
            }
            engine_records.append(record)
            all_prediction_rows.append(record)

        engine_pred_df = pd.DataFrame(engine_records)

        for src_label in sources_present:
            if engine_pred_df.empty:
                group_df = pd.DataFrame()
            else:
                group_df = engine_pred_df.loc[
                    (engine_pred_df["source"] == src_label)
                    & (engine_pred_df["split"] == split)
                ].reset_index(drop=True)

            metrics_row = compute_group_metrics(
                records_df=group_df,
                engine_name=engine_name,
                source=src_label,
                split=split,
                hardware=hardware,
                train_source=train_source,
                seed=seed,
                n_boot=n_boot,
            )
            summary_rows.append(metrics_row)

            if group_df.empty:
                cm_df = confusion_matrix_df([], [], labels=INTENTS)
            else:
                cm_df = confusion_matrix_df(
                    group_df["intent"].tolist(),
                    group_df["predicted_intent"].tolist(),
                    labels=INTENTS,
                )
            confusion_matrices[(engine_name, src_label, split)] = cm_df

    predictions_df = pd.DataFrame(all_prediction_rows)
    summary_df = pd.DataFrame(summary_rows)
    manual_eval_df = build_manual_eval_sheet(predictions_df)
    _ = ESCALATING_INTENTS

    return EvaluationResult(
        summary_df=summary_df,
        predictions_df=predictions_df,
        manual_eval_df=manual_eval_df,
        confusion_matrices=confusion_matrices,
    )


def save_evaluation_outputs(
    result: EvaluationResult,
    output_dir: Path = REPORTS_DIR,
    date_tag: str | None = None,
    phase2: bool = False,
    output_name: str | None = None,
) -> dict[str, Path]:
    """Guarda en disco los archivos CSV de resultados, confusión y revisión manual.

    Args:
        result: Resultado devuelto por ``evaluate()``.
        output_dir: Carpeta destino de los reportes (por defecto ``reports/``).
        date_tag: Etiqueta de fecha (por defecto ``YYYY-MM-DD`` actual).
        phase2: Si es ``True``, guarda además ``reports/results_fase2.csv``.
        output_name: Nombre de archivo adicional opcional dentro de ``output_dir``.

    Returns:
        Diccionario con las rutas de los archivos generados.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tag = date_tag or date.today().isoformat()

    saved_paths: dict[str, Path] = {}

    dated_results_path = target_dir / f"results_{tag}.csv"
    result.summary_df.to_csv(dated_results_path, index=False, encoding="utf-8")
    saved_paths["results"] = dated_results_path

    if phase2:
        phase2_path = target_dir / "results_fase2.csv"
        result.summary_df.to_csv(phase2_path, index=False, encoding="utf-8")
        saved_paths["results_phase2"] = phase2_path

    if output_name:
        custom_name = (
            output_name if output_name.endswith(".csv") else f"{output_name}.csv"
        )
        custom_path = target_dir / custom_name
        result.summary_df.to_csv(custom_path, index=False, encoding="utf-8")
        saved_paths["results_custom"] = custom_path

    manual_path = target_dir / f"manual_eval_{tag}.csv"
    result.manual_eval_df.to_csv(manual_path, index=False, encoding="utf-8")
    saved_paths["manual_eval"] = manual_path

    for (
        engine_name,
        src_label,
        split_label,
    ), cm_df in result.confusion_matrices.items():
        cm_path = (
            target_dir
            / f"confusion_matrix_{tag}_{engine_name}_{src_label}_{split_label}.csv"
        )
        cm_df.to_csv(cm_path, encoding="utf-8")
        saved_paths[f"cm_{engine_name}_{src_label}_{split_label}"] = cm_path

    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Ejecuta la evaluación reproducible de los motores conversacionales."
    )
    parser.add_argument(
        "--eval-input",
        "--input",
        dest="eval_input",
        nargs="+",
        type=Path,
        default=[DEFAULT_SEED_CSV],
        help="Ruta(s) a los archivos CSV de evaluación (p. ej. seed.csv o labeled.csv).",
    )
    parser.add_argument(
        "--train-input",
        dest="train_input",
        nargs="+",
        type=Path,
        default=None,
        help="Ruta(s) a los archivos CSV para fit(). Por defecto usa --eval-input.",
    )
    parser.add_argument(
        "--train-source",
        choices=VALID_TRAIN_SOURCES,
        default="both",
        help="Filtro de source para el conjunto de entrenamiento (synthetic, real o both).",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["M0"],
        help="Nombres de los motores a evaluar (por defecto: M0).",
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default=SPLIT_TRAIN,
        help="Partición sobre la que se evalúa (train por defecto; test requiere --allow-test).",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Autoriza explícitamente evaluar sobre la partición test.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="CPU local",
        help="Descripción del hardware donde se ejecuta la evaluación.",
    )
    parser.add_argument(
        "--date",
        dest="date_tag",
        type=str,
        default=None,
        help="Fecha para los nombres de archivo en formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Guarda adicionalmente reports/results_fase2.csv.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Nombre de archivo CSV adicional para guardar la tabla de resultados.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Carpeta donde se guardan los reportes CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada principal de la interfaz de línea de comandos."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    eval_df = load_datasets(args.eval_input)
    train_paths = args.train_input if args.train_input is not None else args.eval_input
    train_df = load_datasets(train_paths)

    result = evaluate(
        eval_df=eval_df,
        engine_names=args.engines,
        split=args.split,
        allow_test=args.allow_test,
        train_df=train_df,
        train_source=args.train_source,
        hardware=args.hardware,
        seed=SEED,
    )

    saved = save_evaluation_outputs(
        result=result,
        output_dir=args.output_dir,
        date_tag=args.date_tag,
        phase2=args.phase2,
        output_name=args.output_name,
    )

    print(f"Resultados guardados en: {saved['results']}")
    if "results_phase2" in saved:
        print(f"Resultados Fase 2 guardados en: {saved['results_phase2']}")
    if "results_custom" in saved:
        print(f"Resultados adicionales guardados en: {saved['results_custom']}")
    print(f"Hoja de calificación manual en: {saved['manual_eval']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
