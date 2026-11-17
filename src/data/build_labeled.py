"""
Arma ``labeled.csv`` a partir de las hojas de etiquetado completadas a mano.

La hoja (exportada de Excel) solo necesita las columnas ``question_id``, ``question_text``,
``intent`` y ``dudosa``. Las demás (``kb_source``, ``should_escalate``, ``responsable``,
``pair_id``, ``answer_text``...) son opcionales. Cada fila se etiqueta una sola vez, de
una de estas tres maneras:

* **Escala o se abstiene:** se elige solo la ``intent``.
* **Descarte:** se escribe una nota en ``dudosa`` y se deja ``intent`` vacío.
* **Respondible:** se elige solo el ``kb_source`` (id del fragmento de ``data/kb/``). La
  intención se toma del encabezado del fragmento y ``reference_answer`` es el texto del
  fragmento. Si además se escribió una intención distinta, se avisa y se usa la del
  fragmento.

El script:

1. Lee el CSV detectando el separador (coma, punto y coma o tabulador) y la codificación
   (UTF-8 con o sin BOM). Si alguna fila no tiene tantas columnas como el encabezado,
   informa su número y su ``question_id`` (nunca el texto) y no escribe nada.
2. Separa las filas descartadas: las que no tienen ``intent`` ni ``kb_source``. El motivo
   sale de ``dudosa``: si empieza por ``#descartada`` (o ``descartar``) es un mensaje sin
   sentido (``sin_sentido``). Cualquier otra nota es un mensaje que necesita contexto
   (``depende_contexto``). Una fila con nota en ``dudosa`` no puede tener intención ni
   ``kb_source``.
3. Valida las intenciones contra la taxonomía y los ``kb_source`` contra ``data/kb/``, y
   calcula ``should_escalate`` (``True``/``False``). Un fragmento cuya intención escala se
   rechaza, y una intención que escala junto con un ``kb_source`` también.
4. Escribe ``labeled.csv`` (UTF-8 sin BOM, separado por comas) y un archivo aparte con las
   filas descartadas, sin el texto de la pregunta. ``check_pii`` se ejecuta antes de
   escribir: si encuentra algo, no se escribe nada.

Filas sin etiquetar (``--allow-pending``): por defecto una fila vacía (sin intención, sin
``kb_source`` y sin nota) o una respondible sin ``kb_source`` detiene el proceso. Con
``--allow-pending`` esas filas quedan fuera sin detenerlo y el script informa cuántas son y
sus ids (en intervalos); se completan en la misma hoja y se vuelve a ejecutar.

Fusión (``--merge``): agrega a un ``labeled.csv`` existente solo las filas nuevas de la
hoja. De las filas que ya existen nunca se tocan ``question_text``, ``split`` ni
``source``. Si la hoja cambia ``intent``, ``should_escalate``, ``reference_answer`` o
``kb_source`` de una fila existente, el script se detiene y lista las filas, salvo que se
use ``--allow-relabel``; entonces actualiza solo esos cuatro campos e imprime cada cambio.

``split`` queda vacío en las filas nuevas: lo asigna ``split.py``. Por eso, sin ``--merge``,
el script no sobrescribe un ``labeled.csv`` existente sin ``--force``.

Uso (desde la raíz del repositorio)::

    # Primera vez
    py -m src.data.build_labeled --input data/processed/hoja_etiquetada.csv
    # Agregar un lote nuevo (o corregir etiquetas) sin tocar las particiones
    py -m src.data.build_labeled --input data/processed/labeling_sheet_batch2.csv --merge --allow-pending
    py -m src.data.build_labeled --input data/processed/labeling_sheet_batch1_etiquetadas.csv --merge --allow-relabel
"""

import argparse
import csv
import io
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ESCALATING_INTENTS, INTENTS, KB_DIR, PROCESSED_DIR
from src.data.anonymize import add_list_arguments, build_anonymizer
from src.data.check_pii import check_file
from src.data.prepare_labeling import write_csv #, SHEET_FILENAME

#KB_DIR = PROCESSED_DIR.parent / "kb"
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

# Campos de una fila ya existente que ``--merge`` solo cambia con ``--allow-relabel``.
MERGE_FIELDS = ("intent", "should_escalate", "reference_answer", "kb_source")

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

KB_FRONT_MATTER_FENCE = "---"


@dataclass
class Table:
    # Contenido de un CSV leído, con lo detectado del archivo.

    header: list[str]
    rows: list[dict[str, str]]
    delimiter: str
    had_bom: bool
    misaligned: list[tuple[int, str]]


@dataclass(frozen=True)
class KbFragment:
    # Fragmento de la base de conocimiento (``data/kb/<id>.md``).

    id: str
    intent: str
    since: str
    body: str


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
    # Respondibles sin kb_source (solo cuando se usa la base de conocimiento).
    incomplete: list[str] = field(default_factory=list)
    kb_unknown: list[tuple[str, str]] = field(default_factory=list)
    kb_escalating: list[str] = field(default_factory=list)
    intent_kb_conflict: list[str] = field(default_factory=list)
    intent_kb_mismatch: list[str] = field(default_factory=list)
    kb_counts: Counter = field(default_factory=Counter)
    kb_pending_text: set[str] = field(default_factory=set)
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
        ``sin_sentido`` si empieza por ``#descartad`` (o ``descartar``);
        ``depende_contexto`` si hay cualquier otra nota (necesita contexto);
        ``sin_clasificar`` si la nota está vacía.
    """
    folded = _fold(note)
    if folded.startswith(("#descartad", "descartar")):
        return REASON_NO_SENSE
    if folded:
        return REASON_CONTEXT
    return REASON_UNLABELED


def normalize_kb_id(value: str) -> str:
    # Acepta el id con o sin extensión (``reglas_licor`` o ``reglas_licor.md``).
    cleaned = (value or "").strip()
    if cleaned.lower().endswith(".md"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _parse_kb_file(path: Path) -> KbFragment:
    # Lee un fragmento: encabezado entre ``---`` (id, intent, vigente_desde) y cuerpo.
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or lines[0].strip() != KB_FRONT_MATTER_FENCE:
        raise ValueError("no empieza con el encabezado '---'")
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == KB_FRONT_MATTER_FENCE
        )
    except StopIteration:
        raise ValueError("el encabezado no se cierra con '---'") from None
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")
    for key in ("id", "intent"):
        if not meta.get(key):
            raise ValueError(f"falta '{key}' en el encabezado")
    body_lines = lines[end + 1 :]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    if body_lines and body_lines[0].lstrip().startswith("# "):
        body_lines.pop(0)  # el título no forma parte de la respuesta
    body = _squash(" ".join(body_lines))
    if not body:
        raise ValueError("el fragmento no tiene contenido")
    return KbFragment(meta["id"], meta["intent"].lower(), meta.get("vigente_desde", ""), body)


def load_kb(kb_dir: Path) -> dict[str, KbFragment]:
    """
    Carga los fragmentos de ``kb_dir`` (un ``.md`` por tema con encabezado YAML simple).

    El encabezado solo necesita ``id``, ``intent`` y ``vigente_desde``; se lee con un
    analizador propio para no depender de una librería de YAML.

    Returns:
        Diccionario ``id -> KbFragment``.

    Raises:
        ValueError: Si la carpeta no existe, no tiene fragmentos, o algún fragmento está
            mal formado (sin encabezado, ``id`` distinto del nombre del archivo,
            intención fuera de la taxonomía o sin contenido). Se listan todos los errores.
    """
    if not kb_dir.is_dir():
        raise ValueError(f"no se encontró la carpeta de la base de conocimiento: {kb_dir}")
    fragments: dict[str, KbFragment] = {}
    errors: list[str] = []
    for path in sorted(kb_dir.glob("*.md")):
        try:
            fragment = _parse_kb_file(path)
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if fragment.id != path.stem:
            errors.append(f"{path.name}: el id '{fragment.id}' no coincide con el nombre del archivo")
        elif fragment.intent not in INTENTS:
            errors.append(f"{path.name}: intención '{fragment.intent}' fuera de la taxonomía")
        else:
            fragments[fragment.id] = fragment
    if errors:
        raise ValueError("base de conocimiento inválida: " + "; ".join(errors))
    if not fragments:
        raise ValueError(f"no hay fragmentos .md en {kb_dir}")
    return fragments


def compact_ids(ids: Sequence[str]) -> str:
    """
    Resume una lista de ids en intervalos (``R251-R260, R300``).

    Los ids con formato ``R<número>`` consecutivos se agrupan; los demás se dejan tal cual.
    """
    numbered: list[tuple[int, str, str]] = []
    others: list[str] = []
    for question_id in ids:
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", question_id)
        if match:
            numbered.append((int(match.group(2)), match.group(1), question_id))
        else:
            others.append(question_id)
    parts: list[str] = []
    group: list[tuple[int, str, str]] = []

    def flush() -> None:
        if not group:
            return
        first, last = group[0][2], group[-1][2]
        parts.append(first if len(group) == 1 else f"{first}-{last}")

    for item in sorted(numbered):
        if group and item[1] == group[-1][1] and item[0] == group[-1][0] + 1:
            group.append(item)
        else:
            flush()
            group = [item]
    flush()
    return ", ".join(parts + others)


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


def build_labeled(
    rows: Sequence[dict[str, str]],
    kb: Mapping[str, KbFragment] | None = None,
    allow_pending: bool = False,
) -> Built:
    """
    Separa las filas etiquetadas de las descartadas y valida la hoja.

    Args:
        rows: Filas de la hoja (diccionarios por nombre de columna).
        kb: Base de conocimiento (``id -> fragmento``). Si se da, una fila con
            ``kb_source`` toma su intención y su ``reference_answer`` del fragmento, y una
            fila respondible sin ``kb_source`` cuenta como incompleta. Si es ``None`` se
            usa el comportamiento sin base: ``reference_answer`` de la hoja o
            ``[POR CONFIRMAR]``.
        allow_pending: Si es ``True``, las filas totalmente vacías no se tratan como
            descartadas ni como error: quedan fuera y se listan en ``Built.unlabeled``.
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
        kb_id = normalize_kb_id(row.get("kb_source", "")) if kb is not None else ""
        note = _squash(row.get("dudosa", ""))
        reason = discard_reason(note)
        if not intent and not kb_id:
            if reason == REASON_UNLABELED and allow_pending:
                built.unlabeled.append(question_id)  # queda pendiente, no es descarte
                continue
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
        if intent and intent not in INTENTS:
            built.invalid_intents.append((question_id, intent[:30]))
            continue
        text = _squash(row["question_text"])
        if not text:
            built.empty_text.append(question_id)
            continue

        if kb is not None and kb_id:
            fragment = kb.get(kb_id)
            if fragment is None:
                built.kb_unknown.append((question_id, kb_id[:40]))
                continue
            if fragment.intent in ESCALATING_INTENTS:
                built.kb_escalating.append(question_id)
                continue
            if intent in ESCALATING_INTENTS:
                built.intent_kb_conflict.append(question_id)
                continue
            if intent and intent != fragment.intent:
                built.intent_kb_mismatch.append(question_id)
            intent = fragment.intent
            escalate = False
            reference, kb_source = fragment.body, fragment.id
            built.kb_counts[fragment.id] += 1
            if PENDING_ANSWER in fragment.body:
                built.kb_pending_text.add(fragment.id)
        else:
            escalate = intent in ESCALATING_INTENTS
            if kb is not None and not escalate:
                built.incomplete.append(question_id)  # respondible sin fragmento
                continue
            if escalate:
                reference, kb_source = ESCALATE_ANSWER, ""
            else:
                reference = _squash(row.get("reference_answer", "")) or PENDING_ANSWER
                kb_source = row.get("kb_source", "").strip()
        declared = parse_bool(row.get("should_escalate", ""))
        if declared is not None and declared != escalate:
            built.escalation_mismatch.append(question_id)
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


def read_plain_csv(path: Path) -> list[dict[str, str]]:
    # Lee un CSV sencillo escrito por este mismo script (UTF-8, con o sin BOM).
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _norm_escalate(value: str) -> str:
    # ``True``/``False`` aunque Excel lo haya cambiado a VERDADERO/FALSO.
    parsed = parse_bool(value)
    return value if parsed is None else str(parsed)


@dataclass
class MergeResult:
    # Resultado de fusionar filas nuevas en un ``labeled.csv`` existente.

    rows: list[dict[str, str]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    changed: list[tuple[str, dict[str, tuple[str, str]]]] = field(default_factory=list)
    unchanged: int = 0
    text_changed: list[str] = field(default_factory=list)
    relabel_blocked: list[str] = field(default_factory=list)
    now_discarded: list[str] = field(default_factory=list)


def merge_labeled(
    existing: Sequence[dict[str, str]],
    new: Sequence[dict[str, str]],
    discarded_ids: Sequence[str] = (),
    allow_relabel: bool = False,
) -> MergeResult:
    """
    Agrega a ``existing`` las filas de ``new`` que todavía no están.

    De las filas que ya existen no se tocan ``question_text``, ``split`` ni ``source``
    (el texto distinto se rechaza siempre). Si ``new`` trae otro valor en ``intent``,
    ``should_escalate``, ``reference_answer`` o ``kb_source``, el cambio solo se aplica con
    ``allow_relabel``; si no, queda en ``relabel_blocked``. Las filas de ``existing`` que no
    aparecen en ``new`` se conservan igual.

    Args:
        existing: Filas actuales de ``labeled.csv``.
        new: Filas etiquetadas que salen de la hoja.
        discarded_ids: ids que la hoja marca ahora como descartados; si alguno ya estaba
            etiquetado se informa en ``now_discarded`` (no se borra nada solo).
        allow_relabel: Permite actualizar los campos de ``MERGE_FIELDS``.
    """
    result = MergeResult()
    #current = {row["question_id"]: {**{c: "" for c in LABELED_COLUMNS}, **row} for row in existing}
    current = {row["question_id"]: {c: row.get(c, "") for c in LABELED_COLUMNS} for row in existing}
    result.now_discarded = [qid for qid in discarded_ids if qid in current]
    for row in new:
        question_id = row["question_id"]
        old = current.get(question_id)
        if old is None:
            current[question_id] = dict(row)
            result.added.append(question_id)
            continue
        if _squash(old["question_text"]) != row["question_text"]:
            result.text_changed.append(question_id)
            continue
        diffs = {
            name: (old[name], row[name])
            for name in MERGE_FIELDS
            if (
                _norm_escalate(old[name]) != row[name]
                if name == "should_escalate"
                else old[name] != row[name]
            )
        }
        if not diffs:
            result.unchanged += 1
        elif not allow_relabel:
            result.relabel_blocked.append(question_id)
        else:
            result.changed.append((question_id, diffs))
            for name, (_, value) in diffs.items():
                old[name] = value
    result.rows = [current[row["question_id"]] for row in existing] + [
        current[question_id] for question_id in result.added
    ]
    return result


def merge_discarded(
    existing: Sequence[dict[str, str]],
    new: Sequence[dict[str, str]],
    labeled_ids: set[str],
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Une las descartadas de la hoja con las ya guardadas.

    Una fila vuelve a descartarse con el motivo nuevo; las que ahora tienen etiqueta salen
    de las descartadas.

    Returns:
        Filas resultantes y los ids que dejaron de estar descartados.
    """
    merged = {row["question_id"]: row for row in existing}
    for row in new:
        merged[row["question_id"]] = row
    recovered = sorted(qid for qid in merged if qid in labeled_ids)
    rows = [row for qid, row in merged.items() if qid not in labeled_ids]
    return rows, recovered


def _print_pending(label: str, ids: Sequence[str]) -> None:
    print(f"{label}: {len(ids)} ({compact_ids(ids)})")


def _print_summary(
    table: Table, built: Built, counts: Mapping[str, int] | None = None
) -> None:
    counts = built.intent_counts if counts is None else counts
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
    if built.unlabeled or built.incomplete:
        print("Pendientes (quedaron fuera; completa la hoja y vuelve a ejecutar):")
        if built.unlabeled:
            _print_pending(
                "  Vacías (sin intención, sin kb_source y sin nota)", built.unlabeled
            )
        if built.incomplete:
            _print_pending("  Incompletas (respondibles sin kb_source)", built.incomplete)
    if built.kb_counts:
        print("Filas por fragmento de la base:")
        for kb_id, count in sorted(built.kb_counts.items()):
            print(f"  {kb_id}: {count}")
    print("Filas por intención:")
    for intent in INTENTS:
        print(f"  {intent}: {counts.get(intent, 0)}")
    # if built.with_doubt:
    #     _print_ids("Etiquetadas con alguna duda anotada", built.with_doubt)
    if built.escalation_mismatch:
        _print_ids(
            "AVISO: should_escalate de la hoja no coincide con la intención "
            "(se usó el de la intención)",
            built.escalation_mismatch,
        )
    if built.intent_kb_mismatch:
        _print_ids(
            "AVISO: la intención de la hoja no coincide con la del fragmento "
            "(se usó la del fragmento)",
            built.intent_kb_mismatch,
        )
    if built.kb_pending_text:
        print(
            "AVISO: fragmentos usados que contienen [POR CONFIRMAR]: "
            f"{', '.join(sorted(built.kb_pending_text))}."
        )
    few = [
        f"{intent} ({counts.get(intent, 0)})"
        for intent in INTENTS
        if counts.get(intent, 0) < MIN_ROWS_WARNING
    ]
    if few:
        print(
            f"AVISO: intenciones con menos de {MIN_ROWS_WARNING} filas: {', '.join(few)}."
        )
    if any(counts.get(intent, 0) < MIN_ROWS_STRATIFY for intent in INTENTS):
        print(
            f"AVISO: hay intenciones con menos de {MIN_ROWS_STRATIFY} filas: "
            "no se puede estratificar la partición."
        )


def _print_merge_report(result: MergeResult, recovered: Sequence[str]) -> None:
    print(
        f"Fusión: nuevas {len(result.added)}; actualizadas {len(result.changed)}; "
        f"sin cambios {result.unchanged}."
    )
    for question_id, diffs in result.changed:
        parts = []
        for name, (old, new) in diffs.items():
            if name == "reference_answer":
                parts.append("reference_answer actualizada")
            else:
                parts.append(f"{name} {old or '(vacío)'} -> {new or '(vacío)'}")
        print(f"  {question_id}: {'; '.join(parts)}")
    if recovered:
        _print_ids("Salieron de las descartadas por tener etiqueta ahora", recovered)
    open_rows = [
        row["question_id"]
        for row in result.rows
        if row["should_escalate"] == "False" and not row["kb_source"]
    ]
    if open_rows:
        _print_pending(
            "AVISO: filas respondibles de labeled.csv sin kb_source (aún [POR CONFIRMAR])",
            open_rows,
        )
    waiting = [row["question_id"] for row in result.rows if not row["split"]]
    if waiting:
        _print_pending("Filas sin split (falta ejecutar split.py --only-new)", waiting)


def _blocking_problems(
    built: Built, allow_pending: bool = False
) -> list[tuple[str, list[str]]]:
    invalid = [
        f"{question_id} ({value!r})" for question_id, value in built.invalid_intents
    ]
    unknown_kb = [f"{question_id} ({value!r})" for question_id, value in built.kb_unknown]
    candidates = [
        ("question_id con formato inválido", built.bad_ids),
        ("question_id repetidos", built.duplicates),
        ("intención que no está en la taxonomía", invalid),
        ("kb_source que no existe en la base de conocimiento", unknown_kb),
        ("kb_source de un fragmento que escala (no es respondible)", built.kb_escalating),
        (
            "intención que escala junto con kb_source (deja solo una de las dos)",
            built.intent_kb_conflict,
        ),
        ("texto de pregunta vacío", built.empty_text),
        ("intención o kb_source y nota en dudosa a la vez (si hay nota, déjalos vacíos)", built.conflicts),
    ]
    if not allow_pending:
        candidates += [
            ("sin intención y sin motivo en dudosa (¿se olvidó?)", built.unlabeled),
            ("respondible sin kb_source (elige el fragmento o cambia la intención)", built.incomplete),
        ]
    return [(label, ids) for label, ids in candidates if ids]


def _report_hits(hits) -> None:
    totals: Counter = Counter()
    for hit in hits:
        print(
            f"{hit.file}: fila {hit.row} ({hit.row_id}), columna {hit.column}: "
            f"{hit.tag} x{hit.count}"
        )
        totals[hit.tag] += hit.count
    summary = ", ".join(f"{tag}: {count}" for tag, count in sorted(totals.items()))
    print(f"FALLA: posibles datos personales ({summary}). No se escribió ningún archivo.")


def run(
    input_path: Path,
    output_path: Path,
    *,
    anonymizer,
    discarded_path: Path | None = None,
    compare_path: Path | None = None,
    force: bool = False,
    kb_dir: Path | None = None,
    allow_pending: bool = False,
    merge: bool = False,
    allow_relabel: bool = False,
) -> int:
    """
    Genera ``labeled.csv`` y el archivo de descartadas desde la hoja completada.

    Args:
        input_path: Hoja completada y exportada de Excel.
        output_path: ``labeled.csv`` de salida.
        anonymizer: Anonimizador usado por ``check_pii``.
        discarded_path: Archivo de descartadas (por defecto, junto a la salida).
        compare_path: Otro ``labeled.csv`` para comparar por id.
        force: Sobrescribe una salida existente (se pierde la partición asignada).
        kb_dir: Carpeta de la base de conocimiento. Si es ``None`` no se usa la base.
        allow_pending: Deja fuera, sin detener el proceso, las filas vacías y las
            respondibles sin ``kb_source``, e informa cuántas son.
        merge: Agrega las filas nuevas a un ``labeled.csv`` existente sin tocar su
            texto, ``split`` ni ``source``.
        allow_relabel: Con ``merge``, permite actualizar ``intent``,
            ``should_escalate``, ``reference_answer`` y ``kb_source`` de filas existentes.

    Returns:
        0 si todo salió bien; 1 si ``check_pii`` encontró algo (no se escribe nada);
        2 si no se pudo continuar (archivo existente sin ``force``/``merge``, opciones
        incompatibles o problemas en la hoja).
    """
    discarded_path = discarded_path or output_path.with_name(
        f"{output_path.stem}_discarded.csv"
    )
    if merge and force:
        print("ERROR: --merge y --force no se pueden usar juntos.")
        return 2
    if allow_relabel and not merge:
        print("ERROR: --allow-relabel solo tiene sentido junto con --merge.")
        return 2
    if output_path.exists() and not force and not merge:
        print(
            f"ERROR: ya existe {output_path.name}. Si ya le asignaste la partición, "
            "generarlo de nuevo la borraría. Usa --merge para agregar filas nuevas "
            "(o --force solo antes de asignarla)."
        )
        return 2

    kb = load_kb(kb_dir) if kb_dir is not None else None
    table = read_table(input_path)
    if table.misaligned:
        print(
            f"ERROR: {len(table.misaligned)} filas con distinto número de columnas que el "
            "encabezado (¿comas sin comillas dentro del texto?). Nada se escribió."
        )
        for number, row_id in table.misaligned:
            print(f"  registro {number} ({row_id})")
        return 2
    built = build_labeled(table.rows, kb, allow_pending)
    problems = _blocking_problems(built, allow_pending)
    if problems:
        for label, ids in problems:
            _print_ids(f"ERROR: {label}", ids)
        print("Corrige la hoja y vuelve a ejecutar. No se escribió ningún archivo.")
        return 2
    if not built.labeled and not (merge and built.discarded):
        print("ERROR: no quedó ninguna fila etiquetada.")
        return 2

    if compare_path is not None:
        _compare(read_table(compare_path), built.labeled, compare_path.name)

    final_labeled = built.labeled
    final_discarded = built.discarded
    merge_result: MergeResult | None = None
    recovered: list[str] = []
    if merge:
        existing_rows: list[dict[str, str]] = []
        if output_path.exists():
            existing = read_table(output_path)
            if existing.misaligned:
                print(f"ERROR: {output_path.name} tiene filas ilegibles. Nada se escribió.")
                return 2
            existing_rows = existing.rows
        merge_result = merge_labeled(
            existing_rows,
            built.labeled,
            [row["question_id"] for row in built.discarded],
            allow_relabel,
        )
        merge_problems = [
            ("el texto de la pregunta no coincide con labeled.csv (el texto no se modifica)", merge_result.text_changed),
            ("la etiqueta cambia y falta --allow-relabel", merge_result.relabel_blocked),
            ("ya estaba etiquetada en labeled.csv y la hoja ahora la descarta (corrígelo a mano)", merge_result.now_discarded),
        ]
        merge_problems = [(label, ids) for label, ids in merge_problems if ids]
        if merge_problems:
            for label, ids in merge_problems:
                _print_ids(f"ERROR: {label}", ids)
            print("No se escribió ningún archivo.")
            return 2
        final_labeled = merge_result.rows
        old_discarded = read_plain_csv(discarded_path) if discarded_path.exists() else []
        final_discarded, recovered = merge_discarded(
            old_discarded, built.discarded, {row["question_id"] for row in final_labeled}
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".tmp_") as tmp:
        tmp_labeled = Path(tmp) / output_path.name
        tmp_discarded = Path(tmp) / discarded_path.name
        write_csv(tmp_labeled, LABELED_COLUMNS, final_labeled)
        write_csv(tmp_discarded, DISCARDED_COLUMNS, final_discarded)
        hits = check_file(tmp_labeled, anonymizer, ["question_text"]) #hits = check_file(tmp_labeled, anonymizer)
        hits += check_file(tmp_discarded, anonymizer, ["dudosa"])
        if hits:
            _report_hits(hits)
            return 1
        tmp_labeled.replace(output_path)
        tmp_discarded.replace(discarded_path)

    counts = Counter(row["intent"] for row in final_labeled)
    _print_summary(table, built, counts)
    if merge_result is not None:
        _print_merge_report(merge_result, recovered)
        print(f"Total en {output_path.name}: {len(final_labeled)} filas.")
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
        #default=PROCESSED_DIR / SHEET_FILENAME,
        required=True,
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
        "--kb-dir",
        type=Path,
        default=KB_DIR,
        help="Carpeta con los fragmentos de la base de conocimiento.",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="Otro labeled.csv (por ejemplo, el hecho a mano) para comparar por id.",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help=(
            "No se detiene por filas vacías ni por respondibles sin kb_source: las deja "
            "fuera e informa cuántas son y sus ids."
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Agrega las filas nuevas a un labeled.csv existente sin tocar su split.",
    )
    parser.add_argument(
        "--allow-relabel",
        action="store_true",
        help=(
            "Con --merge, permite cambiar intent, should_escalate, reference_answer y "
            "kb_source de filas que ya existen (imprime cada cambio)."
        ),
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
            kb_dir=args.kb_dir,
            allow_pending=args.allow_pending,
            merge=args.merge,
            allow_relabel=args.allow_relabel,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
