"""Demo de consola del bot: simula una conversación con un motor, sin Telegram.

Carga un motor (M0, M1 o M2 por ahora; M3 en cuanto Nick lo registre en
``ENGINE_REGISTRY``), lo entrena con el ``train`` de los CSV indicados y abre
un bucle de preguntas y respuestas en la terminal, como si fuera la mensajería.

Uso (desde la raíz del repositorio)::

    py -m src.bot.console_demo --engine M0
    py -m src.bot.console_demo --engine M1 --train-input data/processed/labeled.csv
    py -m src.bot.console_demo --engine M2 --train-source real
"""

import argparse
import sys
from pathlib import Path

from src.config import PROCESSED_DIR
from src.eval.run_eval import ENGINE_REGISTRY, load_datasets, prepare_train_data


def _print_response(response) -> None:
    """Imprime una ``EngineResponse`` como la vería una persona en el chat."""
    if response.escalate:
        print("Bot: (esto se escala a una persona)")
    else:
        print(f"Bot: {response.answer}")
    detalles = [f"intención={response.intent}", f"confianza={response.confidence:.2f}"]
    if response.retrieved:
        detalles.append(f"recuperado={response.retrieved}")
    if response.latency_ms is not None:
        detalles.append(f"latencia={response.latency_ms:.1f} ms")
    print(f"  [{', '.join(detalles)}]")


def run_demo(engine_name: str, train_paths: list[Path], train_source: str) -> int:
    """Entrena el motor pedido y abre el bucle de preguntas en la terminal.

    Returns:
        0 si corrió bien; 2 si el motor no existe en ``ENGINE_REGISTRY``.
    """
    if engine_name not in ENGINE_REGISTRY:
        print(
            f"ERROR: «{engine_name}» no está registrado todavía. "
            f"Motores disponibles: {', '.join(ENGINE_REGISTRY)}."
        )
        return 2

    print(f"Cargando {engine_name}...")
    engine = ENGINE_REGISTRY[engine_name]()
    train_df = prepare_train_data(load_datasets(train_paths), train_source)
    engine.fit(train_df)
    print(
        f"{engine_name} listo (entrenado con {len(train_df)} filas de train, "
        f"source={train_source}). Escribe una pregunta como la escribiría un "
        "cliente. Ctrl+C o 'salir' para terminar.\n"
    )

    while True:
        try:
            message = input("Cliente: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin de la demo.")
            return 0
        if message.lower() in {"salir", "exit", "quit"}:
            print("Fin de la demo.")
            return 0
        if not message:
            continue
        _print_response(engine.respond(message))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo de consola: simula el bot respondiendo preguntas, sin Telegram."
    )
    parser.add_argument(
        "--engine",
        required=True,
        help="Motor a cargar (M0, M1, M2 o M3 cuando esté registrado).",
    )
    parser.add_argument(
        "--train-input",
        nargs="+",
        type=Path,
        default=[PROCESSED_DIR / "labeled.csv"],
        help="CSV(s) para entrenar el motor (por defecto, labeled.csv).",
    )
    parser.add_argument(
        "--train-source",
        choices=("synthetic", "real", "both"),
        default="both",
        help="Qué source usar para entrenar (por defecto, both).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_demo(args.engine.upper(), args.train_input, args.train_source)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
