"""
Partición train/test estratificada por intención.

Agrega (o completa) la columna ``split`` de un CSV etiquetado, como ``seed.csv``
o ``labeled.csv``: 70 % train y 30 % test dentro de cada intención, con la
semilla fija de ``src/config.py``. El resultado es reproducible: las mismas
preguntas con la misma semilla producen siempre la misma partición.

Reglas:

* La partición se calcula una sola vez. Si alguna fila ya tiene ``split``, el
  script se niega a continuar (a menos que se use ``--force``), porque mover
  preguntas entre train y test después de ver resultados invalida la
  evaluación: el test no se usa para ajustar nada.
* En cada intención se ordenan las preguntas por ``question_id`` antes de
  mezclarlas, así el resultado no depende del orden del archivo.
* En cada intención con 2 o más preguntas queda al menos una en test y una en
  train.

Uso (desde la raíz del repositorio)::
    python -m src.data.split --input data/processed/seed.csv
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.config import PROCESSED_DIR, SEED, SPLIT_TEST, SPLIT_TRAIN, TEST_FRACTION

REQUIRED_COLUMNS = ("question_id", "intent", "split")


def assign_splits(
    rows: list[dict[str, str]],
    seed: int = SEED,
    test_fraction: float = TEST_FRACTION,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Asigna ``train`` o ``test`` a cada fila, estratificando por ``intent``.

    Args:
        rows: Filas del CSV (cada una con ``question_id``, ``intent`` y ``split``).
        seed: Semilla del generador aleatorio.
        test_fraction: Proporción de cada intención que va a test.
        overwrite: Si es ``True``, permite reemplazar particiones ya asignadas.

    Returns:
        Copias de las filas con la columna ``split`` completa, en el mismo orden.

    Raises:
        ValueError: Si faltan columnas, hay ``question_id`` repetidos, una
            intención vacía, o ya hay particiones y ``overwrite`` es ``False``.
    """
    for column in REQUIRED_COLUMNS:
        if rows and column not in rows[0]:
            raise ValueError(f"Falta la columna '{column}'")
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Hay question_id repetidos")
    if any(not row["intent"] for row in rows):
        raise ValueError("Hay filas sin intención")
    if not overwrite and any(row["split"] for row in rows):
        raise ValueError(
            "Ya hay particiones asignadas; no se reasignan para no mover preguntas "
            "entre train y test (usa --force solo si estás seguro)"
        )

    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["intent"]].append(row["question_id"])

    rng = random.Random(seed)
    test_ids: set[str] = set()
    for intent in sorted(groups):
        members = sorted(groups[intent])
        rng.shuffle(members)
        if len(members) < 2:
            continue
        n_test = round(len(members) * test_fraction)
        n_test = min(max(1, n_test), len(members) - 1)
        test_ids.update(members[:n_test])

    return [
        {**row, "split": SPLIT_TEST if row["question_id"] in test_ids else SPLIT_TRAIN}
        for row in rows
    ]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Lee un CSV y devuelve sus columnas y sus filas."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Escribe filas en un CSV UTF-8 con las columnas dadas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Asigna la partición train/test estratificada por intención."
    )
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "seed.csv")
    parser.add_argument(
        "--output", type=Path, help="CSV de salida (por defecto, el mismo de entrada)."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reasigna aunque ya haya particiones (invalida resultados previos).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos.

    Args:
        argv: Argumentos (por defecto, los de ``sys.argv``).

    Returns:
        0 si terminó bien.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        fieldnames, rows = read_rows(args.input)
        result = assign_splits(rows, seed=args.seed, overwrite=args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    write_rows(args.output or args.input, fieldnames, result)
    counts = Counter((row["intent"], row["split"]) for row in result)
    print(f"Filas: {len(result)}")
    for intent in sorted({row["intent"] for row in result}):
        print(
            f"  {intent}: train={counts[(intent, SPLIT_TRAIN)]} "
            f"test={counts[(intent, SPLIT_TEST)]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
