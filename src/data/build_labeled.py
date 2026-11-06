"""
Arma ``labeled.csv`` a partir de la hoja de etiquetado completada a mano.

La hoja (exportada de Excel) solo necesita las columnas ``question_id``, ``question_text``,
``intent`` y ``dudosa``. Las demás (``should_escalate``, ``responsable``, ``pair_id``,
``answer_text``...) son opcionales. El script:

1. Lee el CSV detectando el separador (coma, punto y coma o tabulador) y la codificación
   (UTF-8 con o sin BOM). Si alguna fila no tiene tantas columnas como el encabezado,
   informa su número y su ``question_id`` (nunca el texto) y no escribe nada.
2. Separa las filas descartadas: las que tienen ``intent`` vacío. El motivo sale de
   ``dudosa``: ``descartada`` es un mensaje sin sentido (``sin_sentido``) y
   ``un comentario`` es un mensaje que podría tener sentido con su contexto
   (``depende_contexto``). Una fila sin intención y sin motivo se considera olvidada y
   detiene el proceso.
3. Valida las intenciones contra la taxonomía y recalcula ``should_escalate`` a partir de
   la intención (``True``/``False``), avisando si el valor de la hoja no coincidía.
4. Escribe ``labeled.csv`` (UTF-8 sin BOM, separado por comas, con comillas donde haga
   falta) y un archivo aparte con las filas descartadas, sin el texto de la pregunta.
   ``check_pii`` se ejecuta antes de escribir: si encuentra algo, no se escribe nada.

``split`` queda vacío: lo asigna ``split.py`` una sola vez. Por eso el script no
sobrescribe un ``labeled.csv`` existente sin ``--force``.

Uso (desde la raíz del repositorio)::

    py -m src.data.build_labeled --input data/processed/hoja_etiquetada.csv
"""

import argparse
import csv
import io
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ESCALATING_INTENTS, INTENTS, PROCESSED_DIR
from src.data.anonymize import add_list_arguments, build_anonymizer
from src.data.check_pii import check_file
from src.data.prepare_labeling import SHEET_FILENAME, write_csv

LABELED_FILENAME = "labeled.csv"
LABELED_COLUMNS = (
    "question_id",
    "question_text",
    "intent",
    "should_escalate",
    "reference_answer",
    "kb_source",
    "split",
    "source",
)
DISCARDED_COLUMNS = ("question_id", "motivo", "dudosa", "pair_id")
REQUIRED_COLUMNS = frozenset({"question_id", "question_text", "intent"})
COMPARE_COLUMNS = frozenset(
    {"question_id", "question_text", "intent", "should_escalate"}
)

DELIMITERS = (",", ";", "\t")
ID_PATTERN = re.compile(r"R\d{3,}")
ESCALATE_ANSWER = "ESCALAR"
PENDING_ANSWER = "[POR CONFIRMAR]"
SOURCE_REAL = "real"
MIN_ROWS_WARNING = 5
MIN_ROWS_STRATIFY = 2

REASON_NO_SENSE = "sin_sentido"
REASON_CONTEXT = "depende_contexto"
REASON_UNLABELED = "sin_clasificar"


@dataclass
class Table:
    # Contenido de un CSV leído, con lo detectado del archivo.

    header: list[str]
    rows: list[dict[str, str]]
    delimiter: str
    had_bom: bool
    misaligned: list[tuple[int, str]]


@dataclass
class Built:
    # Resultado de procesar la hoja (solo ids y conteos en los avisos, nunca texto).

    labeled: list[dict[str, str]] = field(default_factory=list)
    discarded: list[dict[str, str]] = field(default_factory=list)
    bad_ids: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    empty_text: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    unlabeled: list[str] = field(default_factory=list)
    invalid_intents: list[tuple[str, str]] = field(default_factory=list)
    escalation_mismatch: list[str] = field(default_factory=list)
    with_doubt: list[str] = field(default_factory=list)
    intent_counts: Counter = field(default_factory=Counter)
    reason_counts: Counter = field(default_factory=Counter)


def _fold(text: str) -> str:
    # Minúsculas, sin tildes y sin espacios en los bordes.
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(char for char in text if not unicodedata.combining(char)).strip()


def _squash(text: str) -> str:
    return " ".join((text or "").split())


def parse_bool(value: str) -> bool | None:
    # Interpreta True/False en inglés o español (Excel); ``None`` si está vacío o es raro.
    folded = _fold(value)
    if folded in {"true", "verdadero", "1"}:
        return True
    if folded in {"false", "falso", "0"}:
        return False
    return None


def discard_reason(note: str) -> str:
    """
    Motivo de descarte según la nota de ``dudosa``.

    Returns:
        ``sin_sentido`` si empieza por ``#descartada`` (o ``descartar``),
        ``depende_contexto`` si empieza por ``un comentario``; si no, ``sin_clasificar``.
    """
    folded = _fold(note)
    if folded.startswith(("#descartad", "descartar")):
        return REASON_NO_SENSE
    # if folded.startswith(("un comentario", "comentario")):
    if folded:
        return REASON_CONTEXT
    return REASON_UNLABELED

def read_table(path: Path) -> Table:
    """
    Lee un CSV detectando separador y BOM.

    El separador se toma del encabezado (el carácter más frecuente entre coma, punto y
    coma y tabulador), así que las comas dentro de un texto entre comillas no molestan.
    Las filas con distinto número de columnas que el encabezado no se leen y quedan en
    ``misaligned`` (número de registro e id).

    Raises:
        ValueError: Si el archivo no es UTF-8, está vacío, no tiene separador o le faltan
            columnas obligatorias.
    """
    raw = path.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path.name}: no está en UTF-8; expórtalo desde Excel como «CSV UTF-8»"
        ) from exc
    if not text.strip():
        raise ValueError(f"{path.name}: el archivo está vacío")
    first_line = text.split("\n", 1)[0]
    delimiter = max(DELIMITERS, key=first_line.count)
    if first_line.count(delimiter) == 0:
        raise ValueError(f"{path.name}: no se encontró un separador en el encabezado")
    records = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    header = [name.strip() for name in records[0]]
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {', '.join(sorted(missing))}")
    id_index = header.index("question_id")
    rows: list[dict[str, str]] = []
    misaligned: list[tuple[int, str]] = []
    for number, record in enumerate(records[1:], start=1):
        if not any(cell.strip() for cell in record):
            continue  # Excel suele dejar filas finales vacías
        if len(record) != len(header):
            row_id = record[id_index].strip() if id_index < len(record) else "-"
            misaligned.append((number, row_id or "-"))
            continue
        rows.append(dict(zip(header, record, strict=True)))
    return Table(header, rows, delimiter, had_bom, misaligned)


def build_labeled(rows: Sequence[dict[str, str]]) -> Built:
    """
    Separa las filas etiquetadas de las descartadas y valida la hoja.

    Args:
        rows: Filas de la hoja (diccionarios por nombre de columna).
    Returns:
        ``Built`` con las filas listas para ``labeled.csv``, las descartadas y las listas
        de problemas encontrados (vacías si todo está bien).
    """
    built = Built()
    seen: set[str] = set()
    for row in rows:
        question_id = row["question_id"].strip()
        if not ID_PATTERN.fullmatch(question_id):
            built.bad_ids.append(question_id or "-")
            continue
        if question_id in seen:
            built.duplicates.append(question_id)
            continue
        seen.add(question_id)
        intent = row["intent"].strip().lower()
        note = _squash(row.get("dudosa", ""))
        reason = discard_reason(note)
        if not intent:
            built.reason_counts[reason] += 1
            if reason == REASON_UNLABELED:
                built.unlabeled.append(question_id)
            built.discarded.append(
                {
                    "question_id": question_id,
                    "motivo": reason,
                    "dudosa": note,
                    "pair_id": row.get("pair_id", "").strip(),
                }
            )
            continue
        if reason != REASON_UNLABELED:
            built.conflicts.append(question_id)
            continue
        if intent not in INTENTS:
            built.invalid_intents.append((question_id, intent[:30]))
            continue
        text = _squash(row["question_text"])
        if not text:
            built.empty_text.append(question_id)
            continue
        escalate = intent in ESCALATING_INTENTS
        declared = parse_bool(row.get("should_escalate", ""))
        if declared is not None and declared != escalate:
            built.escalation_mismatch.append(question_id)
        if note:
            built.with_doubt.append(question_id)
        if escalate:
            reference, kb_source = ESCALATE_ANSWER, ""
        else:
            reference = _squash(row.get("reference_answer", "")) or PENDING_ANSWER
            kb_source = row.get("kb_source", "").strip()
        built.intent_counts[intent] += 1
        built.labeled.append(
            {
                "question_id": question_id,
                "question_text": text,
                "intent": intent,
                "should_escalate": str(escalate),
                "reference_answer": reference,
                "kb_source": kb_source,
                "split": "",
                "source": SOURCE_REAL,
            }
        )
    return built


def _print_ids(label: str, ids: Sequence[str]) -> None:
    print(f"{label}: {len(ids)} ({', '.join(ids)})")


def _compare(existing: Table, labeled: Sequence[dict[str, str]], name: str) -> None:
    """Compara lo generado con un ``labeled.csv`` hecho a mano; imprime solo ids."""
    missing_columns = COMPARE_COLUMNS - set(existing.header)
    if missing_columns:
        raise ValueError(
            f"{name}: faltan columnas {', '.join(sorted(missing_columns))}"
        )
    current = {row["question_id"].strip(): row for row in existing.rows}
    generated = {row["question_id"]: row for row in labeled}
    common = sorted(set(generated) & set(current))
    text_diff, intent_diff, escalate_diff = [], [], []
    for question_id in common:
        old, new = current[question_id], generated[question_id]
        if _squash(old["question_text"]) != new["question_text"]:
            text_diff.append(question_id)
        if old["intent"].strip().lower() != new["intent"]:
            intent_diff.append(question_id)
        if parse_bool(old["should_escalate"]) != (new["should_escalate"] == "True"):
            escalate_diff.append(question_id)
    different = set(text_diff) | set(intent_diff) | set(escalate_diff)
    print(
        f"Comparación con {name}: {len(common) - len(different)} de {len(common)} "
        "filas comunes son idénticas."
    )
    for label, ids in (
        ("  Solo en lo generado", sorted(set(generated) - set(current))),
        ("  Solo en el archivo comparado", sorted(set(current) - set(generated))),
        ("  Texto distinto", text_diff),
        ("  Intención distinta", intent_diff),
        ("  should_escalate distinto", escalate_diff),
    ):
        if ids:
            _print_ids(label, ids)
    if existing.misaligned:
        _print_ids(
            "  Filas ilegibles en el archivo comparado",
            [row_id for _, row_id in existing.misaligned],
        )


def _print_summary(table: Table, built: Built) -> None:
    separator = {",": "coma", ";": "punto y coma", "\t": "tabulador"}[table.delimiter]
    print(
        f"Hoja leída: {len(table.rows)} filas (separador: {separator}; "
        f"BOM: {'sí' if table.had_bom else 'no'})."
    )
    print(f"Etiquetadas: {len(built.labeled)}; descartadas: {len(built.discarded)}.")
    if built.reason_counts:
        reasons = ", ".join(
            f"{name}: {count}" for name, count in sorted(built.reason_counts.items())
        )
        print(f"  Descartadas por motivo: {reasons}.")
    print("Filas por intención:")
    for intent in INTENTS:
        print(f"  {intent}: {built.intent_counts.get(intent, 0)}")
    if built.with_doubt:
        _print_ids("Etiquetadas con alguna duda anotada", built.with_doubt)
    if built.escalation_mismatch:
        _print_ids(
            "AVISO: should_escalate de la hoja no coincide con la intención "
            "(se usó el de la intención)",
            built.escalation_mismatch,
        )
    few = [
        f"{intent} ({built.intent_counts.get(intent, 0)})"
        for intent in INTENTS
        if built.intent_counts.get(intent, 0) < MIN_ROWS_WARNING
    ]
    if few:
        print(
            f"AVISO: intenciones con menos de {MIN_ROWS_WARNING} filas: {', '.join(few)}."
        )
    if any(
        built.intent_counts.get(intent, 0) < MIN_ROWS_STRATIFY for intent in INTENTS
    ):
        print(
            f"AVISO: hay intenciones con menos de {MIN_ROWS_STRATIFY} filas: "
            "no se puede estratificar la partición."
        )


def _blocking_problems(built: Built) -> list[tuple[str, list[str]]]:
    invalid = [
        f"{question_id} ({value!r})" for question_id, value in built.invalid_intents
    ]
    candidates = [
        ("question_id con formato inválido", built.bad_ids),
        ("question_id repetidos", built.duplicates),
        ("intención que no está en la taxonomía", invalid),
        ("texto de pregunta vacío", built.empty_text),
        ("intención y motivo de descarte a la vez (deja solo uno)", built.conflicts),
        ("sin intención y sin motivo en dudosa (¿se olvidó?)", built.unlabeled),
    ]
    return [(label, ids) for label, ids in candidates if ids]


def run(
    input_path: Path,
    output_path: Path,
    *,
    anonymizer,
    discarded_path: Path | None = None,
    compare_path: Path | None = None,
    force: bool = False,
) -> int:
    """
    Genera ``labeled.csv`` y el archivo de descartadas desde la hoja completada.

    Returns:
        0 si todo salió bien; 1 si ``check_pii`` encontró algo (no se escribe nada);
        2 si no se pudo continuar (archivo existente sin ``force`` o problemas en la hoja).
    """
    discarded_path = discarded_path or output_path.with_name(
        f"{output_path.stem}_discarded.csv"
    )
    if output_path.exists() and not force:
        print(
            f"ERROR: ya existe {output_path.name}. Si ya le asignaste la partición, "
            "generarlo de nuevo la borraría. Usa --force solo antes de asignarla."
        )
        return 2

    table = read_table(input_path)
    if table.misaligned:
        print(
            f"ERROR: {len(table.misaligned)} filas con distinto número de columnas que el "
            "encabezado (¿comas sin comillas dentro del texto?). Nada se escribió."
        )
        for number, row_id in table.misaligned:
            print(f"  registro {number} ({row_id})")
        return 2
    built = build_labeled(table.rows)
    problems = _blocking_problems(built)
    if problems:
        for label, ids in problems:
            _print_ids(f"ERROR: {label}", ids)
        print("Corrige la hoja y vuelve a ejecutar. No se escribió ningún archivo.")
        return 2
    if not built.labeled:
        print("ERROR: no quedó ninguna fila etiquetada.")
        return 2

    if compare_path is not None:
        _compare(read_table(compare_path), built.labeled, compare_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".tmp_") as tmp:
        tmp_labeled = Path(tmp) / output_path.name
        tmp_discarded = Path(tmp) / discarded_path.name
        write_csv(tmp_labeled, LABELED_COLUMNS, built.labeled)
        write_csv(tmp_discarded, DISCARDED_COLUMNS, built.discarded)
        hits = check_file(tmp_labeled, anonymizer)
        hits += check_file(tmp_discarded, anonymizer, ["dudosa"])
        if hits:
            totals: Counter = Counter()
            for hit in hits:
                print(
                    f"{hit.file}: fila {hit.row} ({hit.row_id}), columna {hit.column}: "
                    f"{hit.tag} x{hit.count}"
                )
                totals[hit.tag] += hit.count
            summary = ", ".join(
                f"{tag}: {count}" for tag, count in sorted(totals.items())
            )
            print(
                f"FALLA: posibles datos personales ({summary}). No se escribió ningún archivo."
            )
            return 1
        tmp_labeled.replace(output_path)
        tmp_discarded.replace(discarded_path)

    _print_summary(table, built)
    print(
        f"OK: check_pii sin hallazgos. Escrito {output_path} y {discarded_path.name}."
    )
    if not anonymizer.has_names:
        print("AVISO: sin lista de nombres, los nombres no se verificaron.")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Arma labeled.csv desde la hoja de etiquetado completada."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / SHEET_FILENAME,
        help="Hoja de etiquetado completada y exportada de Excel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / LABELED_FILENAME,
        help="CSV etiquetado de salida.",
    )
    parser.add_argument(
        "--discarded",
        type=Path,
        help="CSV de filas descartadas (por defecto, junto a la salida).",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="Otro labeled.csv (por ejemplo, el hecho a mano) para comparar por id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe la salida aunque exista (se pierde la partición asignada).",
    )
    add_list_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Punto de entrada de la línea de comandos.

    Returns:
        Código de salida de ``run`` (0 bien, 1 datos personales, 2 no se pudo continuar).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        anonymizer = build_anonymizer(args.names_file, args.keep_file, args.index)
        return run(
            args.input,
            args.output,
            anonymizer=anonymizer,
            discarded_path=args.discarded,
            compare_path=args.compare_with,
            force=args.force,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
