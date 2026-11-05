"""
Prepara los pares pregunta-respuesta y la hoja de etiquetado del primer lote.

A partir de ``messages.csv`` (ya anonimizado en data/processed) hace tres cosas:
1. Descarta los mensajes de cliente que no son una consulta (saludos, agradecimientos,
   confirmaciones, multimedia y mensajes vacíos) y une los mensajes seguidos del mismo
   cliente en una sola consulta.
2. Empareja cada consulta con lo que la administración respondió a continuación y guarda
   todos los pares en ``qa_pairs.csv``.
3. Toma una muestra al azar (con la semilla global) y genera la hoja de etiquetado, con
   ``intent``, ``should_escalate`` y ``dudosa`` vacías para llenarlas a mano.

El filtro de mensajes es deliberadamente conservador: solo descarta lo que es claramente
cortesía o ruido y no usa palabras clave de temas (precio, fecha, mascotas...). Así la
muestra no se inclina hacia lo que un motor de reglas ya sabe detectar, y los casos dudosos
llegan a la hoja para que quien etiqueta los marque en la columna ``dudosa``.

Antes de escribir algo en la carpeta de salida se ejecuta la verificación de datos
personales sobre lo generado. Si encuentra algo, no se escribe ningún archivo y solo se
informa fila, columna y tipo (nunca el contenido).

Dentro de cada chat, ``messages.csv`` está en orden cronológico.

Uso (desde la raíz del repositorio, con las mismas listas locales que usó el anonimizador)::
    py -m src.data.prepare_labeling --names-file data/interim/names.txt --index data/interim/chat_index.csv
"""

import argparse
import csv
import random
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import PROCESSED_DIR, SEED
from src.data.anonymize import add_list_arguments, build_anonymizer
from src.data.check_pii import check_file

QA_FILENAME = "qa_pairs.csv"
SHEET_FILENAME = "labeling_sheet_batch1.csv"

QA_COLUMNS = ("pair_id", "chat_id", "question_text", "answer_text", "timestamp")
# ``pair_id`` va al final solo para poder volver a ``qa_pairs.csv``; se quita al armar el archivo etiquetado definitivo.
SHEET_COLUMNS = (
    "question_id",
    "question_text",
    "answer_text",
    "intent",
    "should_escalate",
    "dudosa",
    "responsable",
    "pair_id",
)
REQUIRED_MESSAGE_COLUMNS = frozenset(
    {"msg_id", "chat_id", "timestamp", "sender_role", "text"}
)

DEFAULT_SAMPLE_SIZE = 250
DEFAULT_MAX_GAP_MIN = 10.0
DEFAULT_RESPONSABLES = ("YS",)

CLIENT_ROLE = "cliente"
ADMIN_ROLE = "admin"

# Marcadores que deja el anonimizador; no cuentan como contenido.
_MARKER_RE = re.compile(r"\[(?:TEL|EMAIL|URL|NOMBRE|DOC|CUENTA)\]")
_EDITED_RE = re.compile(r"<se edit[óo] este mensaje\.?>", re.IGNORECASE)
# Textos que deja WhatsApp en lugar de multimedia o eventos del sistema.
_MEDIA_RE = re.compile(
    r"^<?\s*(?:multimedia|imagen|audio|video|sticker|gif|documento|contacto)"
    r"\s+omitid[oa]\s*>?$"
    r"|^<?\s*(?:image|audio|video|sticker|gif|document|contact)\s+omitted\s*>?$"
    r"|^<adjunto:.*>$"
    r"|se elimin[oó] este mensaje|eliminaste este mensaje"
    r"|(?:video)?llamada perdida",
    re.IGNORECASE,
)

# Palabras de cortesía y confirmación. Un mensaje formado solo por estas palabras (sin signo de pregunta) no es una consulta.
_SOCIAL_TOKENS = frozenset(
    {
        "hola",
        "holi",
        "buenas",
        "buenos",
        "buen",
        "buena",
        "dia",
        "dias",
        "tarde",
        "tardes",
        "noche",
        "noches",
        "saludos",
        "saludo",
        "gracias",
        "grac",
        "muchas",
        "mil",
        "mucho",
        "ok",
        "okay",
        "oki",
        "okey",
        "listo",
        "vale",
        "perfecto",
        "perfecta",
        "bien",
        "excelente",
        "genial",
        "entendido",
        "entiendo",
        "dale",
        "claro",
        "si",
        "no",
        "por",
        "favor",
        "porfa",
        "amable",
        "amables",
        "atencion",
        "atento",
        "atenta",
        "pendiente",
        "pendientes",
        "quedo",
        "quedamos",
        "luego",
        "hablamos",
        "adios",
        "chao",
        "hasta",
        "pronto",
        "bendiciones",
        "bendecido",
        "bendecida",
        "dios",
        "feliz",
        "un",
        "una",
        "abrazo",
        "de",
        "nada",
        "igualmente",
        "con",
        "gusto",
        "muy",
        "amabilidad",
        "gentil",
        "tan",
        "que",
        "te",
        "le",
        "lo",
        "la",
        "les",
        "se",
        "y",
        "a",
        "el",
        "para",
        "su",
        "ud",
        "usted",
        "ustedes",
        "super",
        "chevere",
        "ya",
        "esta",
        "bueno",
    }
)
_LAUGH_RE = re.compile(r"(?:ja|je|ji|jo)+h?")
_OK_RE = re.compile(r"o+k+y?")

VERDICT_KEEP = "keep"
VERDICT_EMPTY = "empty"
VERDICT_MEDIA = "media"
VERDICT_SOCIAL = "social"


@dataclass
class Stats:
    # Conteos del proceso (solo números, nunca texto de los chats).
    messages: int = 0
    client_messages: int = 0
    dropped: Counter = field(default_factory=Counter)
    unparsed_timestamps: int = 0
    pairs: int = 0
    answered_pairs: int = 0


@dataclass
class _Block:
    # Consulta en construcción: mensajes del cliente y respuesta de la administración.

    chat_id: str
    timestamp: str
    last_time: datetime | None
    question: list[str]
    answer: list[str] = field(default_factory=list)


def _squash(text: str) -> str:
    # Quita la marca de edición y pasa el texto a una sola línea.
    return " ".join(_EDITED_RE.sub(" ", text or "").split())


def _tokens(text: str) -> list[str]:
    # Palabras en minúscula, sin tildes, sin marcadores y con letras repetidas reducidas.
    text = _MARKER_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # "holaaa" -> "hola"
    return re.findall(r"[a-z0-9]+", text)


def _is_social_token(token: str) -> bool:
    return (
        token in _SOCIAL_TOKENS
        or _LAUGH_RE.fullmatch(token) is not None
        or _OK_RE.fullmatch(token) is not None
    )


def classify_message(text: str) -> str:
    """
    Decide si un mensaje puede ser (parte de) una consulta.

    Args:
        text: Texto del mensaje, ya anonimizado.
    Returns:
        ``"keep"`` si se conserva; ``"empty"`` si no tiene texto (vacío, solo emojis,
        signos o marcadores); ``"media"`` si es un marcador de multimedia o de sistema;
        ``"social"`` si es solo saludo, agradecimiento o confirmación. Un mensaje con
        signo de pregunta siempre se conserva.
    """
    cleaned = _squash(text)
    if not cleaned:
        return VERDICT_EMPTY
    if _MEDIA_RE.search(cleaned):
        return VERDICT_MEDIA
    tokens = _tokens(cleaned)
    if not tokens:
        return VERDICT_EMPTY
    if "?" in cleaned or "¿" in cleaned:
        return VERDICT_KEEP
    if all(_is_social_token(token) for token in tokens):
        return VERDICT_SOCIAL
    return VERDICT_KEEP


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat((value or "").strip())
    except ValueError:
        return None


def _within_gap(
    previous: datetime | None, current: datetime | None, max_gap_min: float
) -> bool:
    # True si dos mensajes están lo bastante cerca como para ser la misma consulta.
    if previous is None or current is None:
        return True
    try:
        return (current - previous).total_seconds() <= max_gap_min * 60
    except TypeError:  # fechas con y sin zona horaria mezcladas
        return True


def read_messages(path: Path) -> list[dict[str, str]]:
    """
    Lee ``messages.csv``.

    Se usa el módulo ``csv`` y no pandas para que textos como "NA" o "null" no se
    conviertan en valores vacíos.

    Raises:
        ValueError: Si faltan columnas del esquema.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MESSAGE_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name}: faltan columnas {', '.join(sorted(missing))}"
            )
        return list(reader)


def build_pairs(
    messages: Sequence[dict[str, str]],
    max_gap_min: float = DEFAULT_MAX_GAP_MIN,
    stats: Stats | None = None,
) -> list[dict[str, str]]:
    """
    Empareja cada consulta de cliente con la respuesta siguiente de la administración.

    Reglas:
      - Los mensajes del cliente que no son consulta se ignoran. Si llegan después de que
        la administración ya respondió (por ejemplo, un "gracias"), cierran el intercambio.
      - Los mensajes seguidos del cliente, sin respuesta de por medio y separados por
        menos de ``max_gap_min`` minutos, forman una sola consulta.
      - La respuesta reúne los mensajes de la administración hasta el siguiente mensaje
        del cliente. Si no hubo respuesta, ``answer_text`` queda vacío.
      - Los mensajes de la administración sin consulta previa se ignoran.

    Args:
        messages: Filas de ``messages.csv``, en orden cronológico dentro de cada chat.
        max_gap_min: Minutos máximos entre mensajes seguidos para unirlos.
        stats: Contadores opcionales que se actualizan.
    Returns:
        Filas con las columnas de ``QA_COLUMNS``; ``pair_id`` es secuencial (P00001...).
    """
    stats = stats if stats is not None else Stats()
    chats: dict[str, list[dict[str, str]]] = {}
    for row in messages:
        stats.messages += 1
        chats.setdefault(row["chat_id"], []).append(row)

    blocks: list[_Block] = []

    #for chat_id, rows in chats.items():
    for chat_id in sorted(chats):
        rows = chats[chat_id]
        current: _Block | None = None
        for row in rows:
            text = _squash(row["text"])
            verdict = classify_message(row["text"])
            role = row["sender_role"].strip().lower()
            if role == CLIENT_ROLE:
                stats.client_messages += 1
                if verdict != VERDICT_KEEP:
                    stats.dropped[verdict] += 1
                    if current is not None and current.answer:
                        blocks.append(current)
                        current = None
                    continue
                moment = _parse_timestamp(row["timestamp"])
                if moment is None:
                    stats.unparsed_timestamps += 1
                if (
                    current is not None
                    and not current.answer
                    and _within_gap(current.last_time, moment, max_gap_min)
                ):
                    current.question.append(text)
                    current.last_time = moment
                else:
                    if current is not None:
                        blocks.append(current)
                    current = _Block(chat_id, row["timestamp"], moment, [text])
            elif role == ADMIN_ROLE:
                if current is not None and verdict == VERDICT_KEEP:
                    current.answer.append(text)
        if current is not None:
            blocks.append(current)

    pairs = [
        {
            "pair_id": f"P{number:05d}",
            "chat_id": block.chat_id,
            "question_text": " ".join(block.question),
            "answer_text": " ".join(block.answer),
            "timestamp": block.timestamp,
        }
        for number, block in enumerate(blocks, start=1)
    ]
    stats.pairs = len(pairs)
    stats.answered_pairs = sum(1 for pair in pairs if pair["answer_text"])
    return pairs


def sample_pairs(
    pairs: Sequence[dict[str, str]], size: int, seed: int = SEED
) -> list[dict[str, str]]:
    """
    Muestra al azar reproducible de pares.

    Se ordena antes por ``pair_id`` para que el resultado no dependa del orden de entrada.
    Si hay menos pares que ``size``, se devuelven todos (mezclados).
    """
    ordered = sorted(pairs, key=lambda pair: pair["pair_id"])
    return random.Random(seed).sample(ordered, k=min(size, len(ordered)))


def build_sheet(
    sample: Sequence[dict[str, str]],
    responsables: Sequence[str] = DEFAULT_RESPONSABLES,
) -> list[dict[str, str]]:
    """
    Arma las filas de la hoja de etiquetado (R001, R002...).

    Las columnas a etiquetar quedan vacías. ``responsable`` se reparte por turnos, así que
    con varias personas las filas quedan equilibradas (diferencia máxima de una).
    """
    if not responsables:
        raise ValueError("Se necesita al menos un responsable.")
    return [
        {
            "question_id": f"R{number:03d}",
            "question_text": pair["question_text"],
            "answer_text": pair["answer_text"],
            "intent": "",
            "should_escalate": "",
            "dudosa": "",
            "responsable": responsables[(number - 1) % len(responsables)],
            "pair_id": pair["pair_id"],
        }
        for number, pair in enumerate(sample, start=1)
    ]


def write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]
) -> None:
    # Escribe un CSV UTF-8 sin BOM y con fin de línea ``\\n`` (igual en cualquier sistema).
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(stats: Stats, sample_size: int, responsables: Sequence[str]) -> None:
    dropped = ", ".join(
        f"{name}: {count}" for name, count in sorted(stats.dropped.items())
    )
    print(f"Mensajes leídos: {stats.messages} (de cliente: {stats.client_messages}).")
    print(
        f"Mensajes de cliente descartados: {sum(stats.dropped.values())} ({dropped or '-'})."
    )
    print(f"Pares generados: {stats.pairs} (con respuesta: {stats.answered_pairs}).")
    print(
        f"Filas en la hoja de etiquetado: {sample_size}; responsables: {', '.join(responsables)}."
    )
    if stats.unparsed_timestamps:
        print(
            f"AVISO: {stats.unparsed_timestamps} mensajes con fecha ilegible; "
            "no se pudo aplicar el límite de minutos para unirlos."
        )


def run(
    messages_path: Path,
    out_dir: Path,
    *,
    anonymizer,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_gap_min: float = DEFAULT_MAX_GAP_MIN,
    responsables: Sequence[str] = DEFAULT_RESPONSABLES,
    force: bool = False,
    seed: int = SEED,
) -> int:
    """
    Genera ``qa_pairs.csv`` y la hoja de etiquetado en ``out_dir``.

    Los archivos se escriben primero en una carpeta temporal, se verifican con
    ``check_pii`` y solo si no hay hallazgos pasan a ``out_dir``.

    Returns:
        0 si todo salió bien; 1 si ``check_pii`` encontró algo (no se escribe nada);
        2 si no se pudo continuar (hoja ya existente sin ``force``, o sin pares).
    """
    qa_path = out_dir / QA_FILENAME
    sheet_path = out_dir / SHEET_FILENAME
    if sheet_path.exists() and not force:
        print(
            f"ERROR: ya existe {sheet_path.name}. Generarlo de nuevo borraría las etiquetas. "
            "Usa --force solo si quieres empezar de cero."
        )
        return 2

    stats = Stats()
    pairs = build_pairs(read_messages(messages_path), max_gap_min, stats)
    if not pairs:
        print("ERROR: no se encontró ninguna consulta de cliente en el archivo.")
        return 2
    sheet = build_sheet(sample_pairs(pairs, sample_size, seed), responsables)

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".tmp_") as tmp:
        tmp_qa = Path(tmp) / QA_FILENAME
        tmp_sheet = Path(tmp) / SHEET_FILENAME
        write_csv(tmp_qa, QA_COLUMNS, pairs)
        write_csv(tmp_sheet, SHEET_COLUMNS, sheet)
        hits = check_file(tmp_qa, anonymizer)
        hits += check_file(tmp_sheet, anonymizer, ["question_text", "answer_text"])
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
        tmp_qa.replace(qa_path)
        tmp_sheet.replace(sheet_path)

    _print_summary(stats, len(sheet), responsables)
    print("OK: check_pii sin hallazgos. Archivos escritos en", out_dir)
    if not anonymizer.has_names:
        print("AVISO: sin lista de nombres, los nombres no se verificaron.")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera qa_pairs.csv y la hoja de etiquetado del primer lote."
    )
    parser.add_argument(
        "--messages",
        type=Path,
        default=PROCESSED_DIR / "messages.csv",
        help="CSV de mensajes anonimizados.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROCESSED_DIR, help="Carpeta de salida."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Número de consultas candidatas en la hoja.",
    )
    parser.add_argument(
        "--max-gap-min",
        type=float,
        default=DEFAULT_MAX_GAP_MIN,
        help="Minutos máximos entre mensajes seguidos del cliente para unirlos.",
    )
    parser.add_argument(
        "--responsables",
        nargs="+",
        default=list(DEFAULT_RESPONSABLES),
        help="Iniciales de quienes etiquetan; las filas se reparten por turnos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe la hoja aunque ya exista (borra las etiquetas hechas).",
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
    if args.sample_size < 1 or args.max_gap_min < 0:
        parser.error(
            "--sample-size debe ser al menos 1 y --max-gap-min no puede ser negativo."
        )
    try:
        anonymizer = build_anonymizer(args.names_file, args.keep_file, args.index)
        return run(
            args.messages,
            args.out_dir,
            anonymizer=anonymizer,
            sample_size=args.sample_size,
            max_gap_min=args.max_gap_min,
            responsables=args.responsables,
            force=args.force,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
