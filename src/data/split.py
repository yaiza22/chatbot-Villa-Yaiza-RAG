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

Partición incremental (``--only-new``): cuando se agregan filas a un archivo que ya
tiene partición, asigna ``split`` solo a las filas con ``split`` vacío y no mueve ninguna
fila ya asignada. Para cada intención apunta a que el total acumulado (filas viejas más
nuevas) quede cerca de 70/30, y no solo el lote nuevo. Es determinista: cada intención
usa su propia semilla derivada de ``SEED`` y de su nombre, así que el resultado de una
intención no depende de las demás ni del orden del archivo. Si detecta que cambiaría una
fila existente, falla sin escribir nada.

Uso (desde la raíz del repositorio)::
    python -m src.data.split --input data/processed/seed.csv
    python -m src.data.split --input data/processed/labeled.csv --only-new
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


def _ensure_existing_unchanged(
    before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    """Verifica que las filas que ya tenían ``split`` quedaron idénticas.

    Raises:
        ValueError: Si alguna fila asignada cambió (se informa su ``question_id``).
    """
    changed = [
        old["question_id"]
        for old, new in zip(before, after, strict=True)
        if old["split"] and old != new
    ]
    if changed:
        raise ValueError(
            "La partición cambiaría filas ya asignadas: " + ", ".join(changed)
        )


def assign_new_splits(
    rows: list[dict[str, str]],
    seed: int = SEED,
    test_fraction: float = TEST_FRACTION,
) -> list[dict[str, str]]:
    """Asigna ``split`` solo a las filas nuevas, sin mover las ya asignadas.

    Las filas nuevas son las que tienen ``split`` vacío. En cada intención, el número de
    filas de test del total acumulado apunta a ``test_fraction`` (al menos una en test y
    una en train si hay 2 o más filas); a las nuevas les toca la diferencia con lo que ya
    hay en test, sin pasar del número de nuevas. Una intención con menos de 2 filas en
    total deja sus filas nuevas en train. Las nuevas se ordenan por ``question_id`` y se
    mezclan con una semilla propia de la intención (``"<seed>:<intención>"``).

    Args:
        rows: Filas del CSV (cada una con ``question_id``, ``intent`` y ``split``).
        seed: Semilla base.
        test_fraction: Proporción objetivo de test por intención en el total acumulado.

    Returns:
        Copias de las filas, en el mismo orden, con ``split`` completo. Si no hay filas
        nuevas, las devuelve sin cambios.

    Raises:
        ValueError: Si faltan columnas, hay ``question_id`` repetidos, una intención
            vacía, un ``split`` que no es ``train``/``test``/vacío, o el resultado
            cambiaría una fila ya asignada.
    """
    for column in REQUIRED_COLUMNS:
        if rows and column not in rows[0]:
            raise ValueError(f"Falta la columna '{column}'")
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Hay question_id repetidos")
    if any(not row["intent"] for row in rows):
        raise ValueError("Hay filas sin intención")
    invalid = [
        row["question_id"]
        for row in rows
        if row["split"] not in ("", SPLIT_TRAIN, SPLIT_TEST)
    ]
    if invalid:
        raise ValueError(
            f"Valores de split que no son '{SPLIT_TRAIN}' ni '{SPLIT_TEST}': "
            + ", ".join(invalid)
        )

    existing_test: Counter[str] = Counter()
    existing_total: Counter[str] = Counter()
    new_ids: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        intent = row["intent"]
        if row["split"]:
            existing_total[intent] += 1
            existing_test[intent] += row["split"] == SPLIT_TEST
        else:
            new_ids[intent].append(row["question_id"])

    test_ids: set[str] = set()
    for intent in sorted(new_ids):
        members = sorted(new_ids[intent])
        total = existing_total[intent] + len(members)
        if total < 2:
            continue
        target_test = min(max(1, round(total * test_fraction)), total - 1)
        n_test = min(max(0, target_test - existing_test[intent]), len(members))
        random.Random(f"{seed}:{intent}").shuffle(members)
        test_ids.update(members[:n_test])

    result = [
        (
            row
            if row["split"]
            else {
                **row,
                "split": SPLIT_TEST if row["question_id"] in test_ids else SPLIT_TRAIN,
            }
        )
        for row in rows
    ]
    result = [dict(row) for row in result]
    _ensure_existing_unchanged(rows, result)
    return result


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
    parser.add_argument(
        "--only-new",
        action="store_true",
        help=(
            "Asigna split solo a las filas con split vacío, sin mover las ya asignadas "
            "(apunta a 70/30 sobre el total acumulado por intención)."
        ),
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
    if args.only_new and args.force:
        parser.error("--only-new y --force no se pueden usar juntos.")
    try:
        fieldnames, rows = read_rows(args.input)
        if args.only_new:
            new_count = sum(1 for row in rows if not row.get("split"))
            if new_count == 0:
                print("No hay filas nuevas (todas tienen split); no se modificó nada.")
                return 0
            result = assign_new_splits(rows, seed=args.seed)
            print(f"Filas nuevas con split asignado: {new_count}")
        else:
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
