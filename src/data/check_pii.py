"""
Verificador de datos personales en archivos CSV.

Recorre uno o más CSV y falla (código de salida 1) si encuentra algo que parezca un dato personal:
correos, URLs, teléfonos, documentos, cuentas y, si se entregan las listas locales, nombres.
Usa exactamente los mismos detectores que ``anonymize.py``, así que un CSV recién anonimizado debe pasar.

Nunca imprime el contenido encontrado: solo el archivo, la fila, el identificador de la fila,
la columna y el tipo de dato.

Limitación: no puede detectar nombres que no estén en las listas locales, así que pasar esta
verificación no garantiza que no queden datos personales. Antes de versionar un archivo hay que
revisar además una muestra a mano.

Uso (desde la raíz del repositorio):: python -m src.data.check_pii data/processed/messages.csv
py -m src.data.check_pii data/processed/messages.csv --names-file data/interim/names.txt --index data/interim/chat_index.csv
$LASTEXITCODE
"""

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.data.anonymize import Anonymizer, add_list_arguments, build_anonymizer

# Columnas que no llevan texto libre (identificadores, fechas, etiquetas).
SKIP_COLUMNS = frozenset(
    {
        "msg_id",
        "chat_id",
        "pair_id",
        "question_id",
        "timestamp",
        "sender_role",
        "intent",
        "should_escalate",
        "kb_source",
        "split",
        "source",
    }
)
_ID_COLUMNS = ("msg_id", "pair_id", "question_id")


@dataclass(frozen=True)
class Hit:
    # Un posible dato personal encontrado (sin su contenido).
    file: str
    row: int
    row_id: str
    column: str
    tag: str
    count: int


def check_file(
    path: Path,
    anonymizer: Anonymizer,
    columns: Iterable[str] | None = None,
) -> list[Hit]:
    """
    Busca datos personales en un CSV.
    Args:
        path: CSV a revisar.
        anonymizer: Anonimizador cuyos detectores se reutilizan.
        columns: Columnas a revisar. Por defecto, todas salvo ``SKIP_COLUMNS``.
    Returns:
        Un ``Hit`` por cada fila, columna y tipo de dato encontrado.
    Raises:
        ValueError: Si alguna columna pedida no existe en el CSV.
    """
    hits: list[Hit] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if columns is None:
            targets = [name for name in fieldnames if name not in SKIP_COLUMNS]
        else:
            targets = list(columns)
            missing = [name for name in targets if name not in fieldnames]
            if missing:
                raise ValueError(f"{path.name}: falta la columna {', '.join(missing)}")
        id_column = next((name for name in _ID_COLUMNS if name in fieldnames), None)
        for number, row in enumerate(reader, start=1):
            row_id = row[id_column] if id_column else "-"
            for column in targets:
                _, counts = anonymizer.anonymize(row[column], row.get("chat_id"))
                for tag, count in counts.items():
                    hits.append(Hit(path.name, number, row_id, column, tag, count))
    return hits


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Falla si un CSV contiene datos personales."
    )
    parser.add_argument("files", nargs="+", type=Path, help="CSV a revisar.")
    parser.add_argument(
        "--columns",
        action="append",
        help="Columna a revisar (repetible; por defecto, todas las de texto).",
    )
    add_list_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Punto de entrada de la línea de comandos.
    Args:
        argv: Argumentos (por defecto, los de ``sys.argv``).
    Returns:
        0 si no se encontró nada; 1 si se encontró algún posible dato personal.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        anonymizer = build_anonymizer(args.names_file, args.keep_file, args.index)
        hits: list[Hit] = []
        for path in args.files:
            hits.extend(check_file(path, anonymizer, args.columns))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    for hit in hits:
        print(
            f"{hit.file}: fila {hit.row} ({hit.row_id}), columna {hit.column}: "
            f"{hit.tag} x{hit.count}"
        )
    if hits:
        totals = Counter()
        for hit in hits:
            totals[hit.tag] += hit.count
        summary = ", ".join(f"{tag}: {count}" for tag, count in sorted(totals.items()))
        print(f"FALLA: posibles datos personales ({summary}).")
        return 1
    print("OK: no se detectaron datos personales con las reglas actuales.")
    if not anonymizer.has_names:
        print("AVISO: sin lista de nombres, los nombres no se verificaron.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
