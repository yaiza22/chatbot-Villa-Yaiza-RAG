"""Construye el motor del bot a partir del texto de ``BOT_ENGINE``.

Permite cambiar de motor editando ``.env``, sin tocar código. Valores aceptados:

* Un motor solo: ``M0``, ``M1``, ``M2`` o ``M3``.
* Una combinación ``clasificador+redactor``: el clasificador es ``M0`` o ``M1`` y el
  redactor ``M2`` o ``M3`` (por ejemplo ``M0+M3``, el valor por defecto).

M0 y M3 no necesitan datos para arrancar: M0 lee sus reglas y M3 lee ``data/kb/``. M1 (el
clasificador) y M2 (el umbral de similitud) sí se entrenan al arrancar con el ``train`` de
``labeled.csv``, igual que en la evaluación; nunca con filas de test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.bot.combined_engine import CombinedEngine
from src.engines.base import Engine
from src.eval.run_eval import ENGINE_REGISTRY, load_datasets, prepare_train_data

if TYPE_CHECKING:
    from src.engines.m3_rag import OllamaClient

CLASSIFIER_ENGINES = ("M0", "M1")
RESPONDER_ENGINES = ("M2", "M3")
SINGLE_ENGINES = ("M0", "M1", "M2", "M3")
ENGINES_NEEDING_TRAIN = frozenset({"M1", "M2"})
# Motores lentos de cargar (modelo de embeddings y LLM): se "calientan" al arrancar.
ENGINES_TO_WARM_UP = frozenset({"M2", "M3"})
WARM_UP_QUESTION = "¿Dónde queda ubicada la finca?"


@dataclass(frozen=True)
class BuiltEngine:
    """Motor listo para usar y datos de cómo se construyó.

    Attributes:
        engine: Motor que cumple ``Engine``.
        parts: Motores que lo forman, p. ej. ``("M0", "M3")``.
        n_train_rows: Filas de train usadas para entrenar (0 si no entrenó nada).
        llm: Cliente de Ollama que usa M3 (``None`` si el motor no usa M3 o si se pasó un
            ``registry`` propio). Se expone para ajustar su tiempo máximo de espera.
    """

    engine: Engine
    parts: tuple[str, ...]
    n_train_rows: int
    llm: OllamaClient | None = None


def parse_engine_spec(spec: str) -> tuple[str, ...]:
    """Valida ``BOT_ENGINE`` y lo separa en sus motores.

    Args:
        spec: Texto como ``"M0"`` o ``"M0+M3"`` (no distingue mayúsculas ni espacios).

    Returns:
        Tupla con uno o dos nombres de motor.

    Raises:
        ValueError: Si el valor no es un motor solo ni una combinación válida.
    """
    parts = tuple(part.strip().upper() for part in str(spec).split("+"))
    if len(parts) == 1 and parts[0] in SINGLE_ENGINES:
        return parts
    if (
        len(parts) == 2
        and parts[0] in CLASSIFIER_ENGINES
        and parts[1] in RESPONDER_ENGINES
    ):
        return parts
    raise ValueError(
        f"BOT_ENGINE={spec!r} no es válido. Usa un motor solo "
        f"({', '.join(SINGLE_ENGINES)}) o 'clasificador+redactor', con clasificador en "
        f"{', '.join(CLASSIFIER_ENGINES)} y redactor en {', '.join(RESPONDER_ENGINES)} "
        "(por ejemplo, M0+M3)."
    )


def needs_training(parts: Sequence[str]) -> bool:
    """Indica si algún motor de la combinación necesita ``labeled.csv`` para arrancar."""
    return any(part in ENGINES_NEEDING_TRAIN for part in parts)


def build_engine(
    spec: str,
    train_input: Path | str,
    train_source: str = "real",
    registry: Mapping[str, Callable[[], Engine]] | None = None,
) -> BuiltEngine:
    """Crea (y entrena si hace falta) el motor indicado por ``spec``.

    Args:
        spec: Valor de ``BOT_ENGINE``.
        train_input: CSV con el esquema de ``labeled.csv`` (solo se lee si hay M1 o M2).
        train_source: ``source`` permitido al entrenar (``real``, ``synthetic`` o ``both``).
        registry: Fábricas de motores por nombre. Por defecto, las de la evaluación, con
            una diferencia: M3 recibe un cliente de Ollama propio (misma configuración de
            ``src/config.py``) para que el bot pueda ajustar su tiempo de espera.

    Returns:
        ``BuiltEngine`` con el motor listo.

    Raises:
        ValueError: Si ``spec`` no es válido o no quedan filas de train.
        FileNotFoundError: Si el motor necesita entrenar y no existe ``train_input``.
    """
    parts = parse_engine_spec(spec)
    llm: OllamaClient | None = None
    if registry is None:
        factories = dict(ENGINE_REGISTRY)
        if "M3" in parts:
            from src.engines.m3_rag import OllamaClient, RagEngine

            llm = OllamaClient()
            factories["M3"] = lambda: RagEngine(llm=llm)
        registry = factories
    engines = [registry[part]() for part in parts]

    n_train_rows = 0
    if needs_training(parts):
        path = Path(train_input)
        if not path.is_file():
            raise FileNotFoundError(
                f"BOT_ENGINE={'+'.join(parts)} entrena M1 o M2 al arrancar y necesita "
                f"{path}. Usa BOT_ENGINE=M0 o M0+M3, que no lo necesitan."
            )
        train = prepare_train_data(load_datasets([path]), train_source)
        if train.empty:
            raise ValueError(
                f"{path.name} no tiene filas de train con source={train_source!r}."
            )
        for part, engine in zip(parts, engines, strict=True):
            if part in ENGINES_NEEDING_TRAIN:
                engine.fit(train)
        n_train_rows = len(train)

    engine = engines[0] if len(engines) == 1 else CombinedEngine(*engines)
    return BuiltEngine(engine=engine, parts=parts, n_train_rows=n_train_rows, llm=llm)


def warm_up(built: BuiltEngine, question: str = WARM_UP_QUESTION) -> list[str]:
    """Carga de una vez los modelos lentos (embeddings y LLM) antes de atender clientes.

    Sin esto, el primer cliente esperaría a que se cargue el modelo de embeddings y a que
    Ollama suba el LLM a la GPU. Si Ollama no está encendido, el error sale aquí, al
    arrancar, y no con el primer cliente.

    Args:
        built: Motor construido con ``build_engine``.
        question: Pregunta de prueba (su respuesta se descarta).

    Returns:
        Nombres de los motores que se calentaron.
    """
    engine = built.engine
    components = (
        [engine.gate, engine.responder]
        if isinstance(engine, CombinedEngine)
        else [engine]
    )
    warmed: list[str] = []
    for component in components:
        if component.name in ENGINES_TO_WARM_UP:
            component.respond(question)
            warmed.append(component.name)
    return warmed
