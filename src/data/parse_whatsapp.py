"""
Parser de exportaciones de chats de WhatsApp

Lee archivos .txt o .zip que contengan uno, exportados desde WhatsApp y
genera [messages.csv] con el esquema: msg_id, chat_id, timestamp, sender_role, text.
Solo se conservan los mensajes de texto; las llamadas, los mensajes del sistema, los mensajes eliminados
y los archivos multimedia (imágenes, videos, audios, documentos) se descartan,
pero se cuentan para poder revisar el resultado sin mirar el contenido de los chats.

1. El nombre del remitente de la finca se recibe como parámetro (``--admin``), nunca queda escrito en el código, porque es un dato personal.
2. El CSV no incluye nombres de remitentes: solo el rol (``cliente``/``admin``).
3. ``chat_id`` es secuencial (``chat_0001``...) y no se deriva del nombre del archivo, que suele contener el nombre del contacto.
4. Si el archivo de entrada es un ``.zip``, solo se lee el ``.txt``: las fotos, audios y videos del zip no se extraen al disco.
5. La salida va a ``data/interim/`` (sin anonimizar, no se versiona).

Uso (desde la raíz del repositorio): python -m src.data.parse_whatsapp --admin "Nombre en el export"
"""

import argparse
import csv
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "interim" / "messages.csv"

CSV_COLUMNS = ("msg_id", "chat_id", "timestamp", "sender_role", "text")
ROLE_CLIENT = "cliente"
ROLE_ADMIN = "admin"

FORMAT_ANDROID = "android"
FORMAT_IOS = "ios"

KIND_TEXT = "text"
KIND_SYSTEM = "system"
KIND_CALL = "call"
KIND_DELETED = "deleted"
KIND_AUDIO = "audio"
KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_DOCUMENT = "document"
KIND_MEDIA = "media"  # multimedia de tipo desconocido ("<Multimedia omitido>")

# Un remitente real tiene un nombre corto; si el texto antes de ": " es muy largo,
# se trata de un mensaje del sistema que casualmente contiene ": ".
_MAX_SENDER_LENGTH = 60

# --- Limpieza de caracteres -------------------------------------------------

# Marcas invisibles que WhatsApp agrega al inicio de líneas. (Para iOS)
_INVISIBLE = re.compile("[\u200e\u200f\u202a-\u202e\ufeff]")
# Espacios "duros" (p. ej. antes de "p. m." en exports recientes de Android).
_HARD_SPACES = re.compile("[\u00a0\u202f]")

# --- Encabezados de mensaje -------------------------------------------------
_DATE = r"(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{2,4})"
_TIME = (
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
    r"(?:\s*(?P<ampm>[ap])\.?\s*m\.?)?"
)
# Android: "12/03/2024, 3:45 p. m. - Nombre: texto"
_ANDROID_HEADER = re.compile(
    rf"^{_DATE},?\s+{_TIME}\s+-\s+(?P<rest>.*)$", re.IGNORECASE
)
# iOS: "[12/03/2024, 3:45:12 p. m.] Nombre: texto"
_IOS_HEADER = re.compile(rf"^\[{_DATE},?\s+{_TIME}\]\s*(?P<rest>.*)$", re.IGNORECASE)

# --- Clasificación del contenido -------------------------------------------
_SYSTEM_BODY = re.compile(
    r"^(los mensajes y las llamadas est[aá]n cifrados"
    r"|messages and calls are end-to-end encrypted"
    r"|este chat es con una cuenta de empresa"
    r"|esta empresa usa un servicio seguro"
    r"|los mensajes temporales"
    r"|cambiaste el c[oó]digo de seguridad"
    r"|el c[oó]digo de seguridad de .* cambi[oó])",
    re.IGNORECASE,
)

_CALL_PART = (
    r"(?:de\s+(?:voz|video|v[ií]deo)|perdida|rechazada"
    r"|sin\s+respuesta|no\s+contestada)"
)
_CALL = re.compile(
    rf"^(?:video)?llamada(?:\s+{_CALL_PART})+(?:,?\s+\d+\s*(?:s|seg|min|h)\b.*)?\.?$",
    re.IGNORECASE,
)

_DELETED = re.compile(
    r"^(se elimin[oó] este mensaje|eliminaste este mensaje"
    r"|este mensaje fue eliminado|you deleted this message"
    r"|this message was deleted)\.?$",
    re.IGNORECASE,
)

_EDITED = re.compile(
    r"\s*<(?:se edit[oó] este mensaje|este mensaje fue editado"
    r"|this message was edited)\.?>\s*$",
    re.IGNORECASE,
)

# Marcadores de multimedia: deben ocupar toda la primera línea del mensaje,
# para no confundir un texto que mencione ".pdf" con un adjunto.
_MEDIA_GENERIC = re.compile(r"^<?multimedia omitido>?$", re.IGNORECASE)
_MEDIA_WORD = re.compile(
    r"^(?P<what>imagen|video|v[ií]deo|audio|sticker|gif|documento|contacto"
    r"|ubicaci[oó]n) omitid[oa]$",
    re.IGNORECASE,
)
_MEDIA_IOS_ATTACHED = re.compile(
    r"^<(?:adjunto|attached):\s*(?P<name>.+)>$", re.IGNORECASE
)
_MEDIA_ANDROID_ATTACHED = re.compile(
    r"^(?P<name>.+?)\s*\((?:archivo adjunto|file attached)\)$", re.IGNORECASE
)

_WORD_KINDS = {
    "imagen": KIND_IMAGE,
    "sticker": KIND_IMAGE,
    "gif": KIND_IMAGE,
    "video": KIND_VIDEO,
    "vídeo": KIND_VIDEO,
    "audio": KIND_AUDIO,
    "documento": KIND_DOCUMENT,
}
_EXTENSION_KINDS = {
    **dict.fromkeys(("jpg", "jpeg", "png", "webp", "gif"), KIND_IMAGE),
    **dict.fromkeys(("mp4", "mov", "3gp", "avi"), KIND_VIDEO),
    **dict.fromkeys(("opus", "m4a", "aac", "mp3", "ogg", "amr"), KIND_AUDIO),
    **dict.fromkeys(
        ("pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"), KIND_DOCUMENT
    ),
}


# Mensaje del chat ya clasificado
@dataclass(frozen=True)
class ParsedMessage:
    # Fecha y hora del mensaje
    timestamp: datetime
    # Nombre del remitente como aparece en el export o None si es mensaje del sistema
    sender: str | None
    # Texto del mensaje, da vacío si no es texto
    text: str
    # Tipo de mensaje (``text``, ``system``, ``call``, ``deleted``, ``audio``, ``image``, ``video``, ``document`` o ``media``).
    kind: str


# Resultado de parsear un chat.
@dataclass
class ParsedChat:
    # Mensajes clasificados, en el orden del archivo.
    messages: list[ParsedMessage]
    # Líneas no vacías antes del primer encabezado reconocible.
    orphan_lines: int = 0


# Conteos del procesamiento (sin contenido de los chats).
@dataclass
class ParseStats:
    chats: int = 0
    failed_chats: list[str] = field(default_factory=list)
    kinds: Counter[str] = field(default_factory=Counter)
    roles: Counter[str] = field(default_factory=Counter)
    orphan_lines: int = 0


# Filas de ``messages.csv`` más el índice ``chat_id`` -> archivo de origen.
@dataclass
class Dataset:
    rows: list[dict[str, str]] = field(default_factory=list)
    chat_index: list[tuple[str, str]] = field(default_factory=list)
    stats: ParseStats = field(default_factory=ParseStats)


# Encabezado de un mensaje más las líneas de continuación.
@dataclass
class _RawEntry:
    timestamp: datetime
    first: str
    extra: list[str] = field(default_factory=list)


# --- Lectura ---------------------------------------------------------------
def _clean(line: str) -> str:
    # Quita saltos de línea, marcas invisibles y espacios duros de una línea.
    line = _INVISIBLE.sub("", line.rstrip("\r\n"))
    return _HARD_SPACES.sub(" ", line)


def read_chat_lines(path: Path) -> list[str]:
    """
    En un ``.zip`` solo se lee el primer ``.txt`` (por orden alfabético); los
    archivos multimedia del zip no se extraen.
    Args:
        path: Ruta al ``.txt`` o ``.zip`` exportado.
    Returns:
        Lista de líneas del chat.
    Raises:
        ValueError: Si el ``.zip`` no contiene ningún ``.txt``.
    """
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".txt") and not name.startswith("__MACOSX")
            )
            if not names:
                raise ValueError("El .zip no contiene ningún archivo .txt")
            text = archive.read(names[0]).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    # Se separa solo por "\n": splitlines() también cortaría en otros separadores Unicode que pueden aparecer dentro de un mensaje.
    return text.split("\n")


def list_chat_files(input_path: Path) -> list[Path]:
    """
    Devuelve los archivos de chat a procesar, en orden alfabético.

    Args:
        input_path: Un ``.txt``/``.zip`` o una carpeta que los contenga (sin buscar en subcarpetas).
    Returns:
        Lista ordenada de rutas.
    Raises:
        ValueError: Si la ruta no existe o no hay archivos ``.txt``/``.zip``.
    """
    if not input_path.exists():
        raise ValueError("La ruta de entrada no existe")
    if input_path.is_file():
        return [input_path]
    files = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".zip"}
    )
    if not files:
        raise ValueError("No se encontraron archivos .txt ni .zip en la carpeta")
    return files


# --- Parseo ----------------------------------------------------------------
def detect_format(lines: Iterable[str]) -> str:
    """
    Detecta si el export es de Android o de iOS.
    Args:
        lines: Líneas del chat.
    Returns:
        ``"android"`` o ``"ios"``.
    Raises:
        ValueError: Si ninguna línea tiene un encabezado reconocible.
    """
    for raw in lines:
        line = _clean(raw)
        if _ANDROID_HEADER.match(line):
            return FORMAT_ANDROID
        if _IOS_HEADER.match(line):
            return FORMAT_IOS
    raise ValueError("No se reconoció el formato del export (Android o iOS)")


def _build_timestamp(match: re.Match[str], line_number: int) -> datetime:
    # Construye el ``datetime`` de un encabezado (día/mes/año).
    year = int(match["year"])
    if year < 100:
        year += 2000
    hour = int(match["hour"])
    ampm = (match["ampm"] or "").lower()
    if ampm == "p" and hour < 12:
        hour += 12
    elif ampm == "a" and hour == 12:
        hour = 0
    try:
        return datetime(
            year,
            int(match["month"]),
            int(match["day"]),
            hour,
            int(match["minute"]),
            int(match["second"] or 0),
        )
    except ValueError as exc:
        raise ValueError(
            f"Fecha u hora inválida en la línea {line_number} "
            "(se asume formato día/mes/año)"
        ) from exc


def _split_entries(lines: list[str], fmt: str) -> tuple[list[_RawEntry], int]:
    # Agrupa las líneas en mensajes: encabezado + líneas de continuación.
    header = _ANDROID_HEADER if fmt == FORMAT_ANDROID else _IOS_HEADER
    entries: list[_RawEntry] = []
    orphans = 0
    for number, raw in enumerate(lines, start=1):
        line = _clean(raw)
        match = header.match(line)
        if match:
            entries.append(_RawEntry(_build_timestamp(match, number), match["rest"]))
        elif entries:
            entries[-1].extra.append(line)
        elif line.strip():
            orphans += 1
    return entries, orphans


def _media_kind(first_line: str) -> str | None:
    # Devuelve el tipo de multimedia si la línea es un marcador; si no, None.
    if _MEDIA_GENERIC.match(first_line):
        return KIND_MEDIA
    word = _MEDIA_WORD.match(first_line)
    if word:
        return _WORD_KINDS.get(word["what"].lower(), KIND_MEDIA)
    attached = _MEDIA_IOS_ATTACHED.match(first_line) or _MEDIA_ANDROID_ATTACHED.match(
        first_line
    )
    if attached:
        name = attached["name"].strip()
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        return _EXTENSION_KINDS.get(extension, KIND_MEDIA)
    return None


def _classify(entry: _RawEntry) -> ParsedMessage:
    # Clasifica un mensaje y extrae su texto cuando corresponde.
    rest = entry.first.strip()
    sender: str | None = None
    body = rest
    if ": " in rest:
        candidate, body = rest.split(": ", 1)
        if len(candidate.strip()) <= _MAX_SENDER_LENGTH:
            sender = candidate.strip()
        else:
            body = rest

    def make(kind: str, text: str = "") -> ParsedMessage:
        return ParsedMessage(entry.timestamp, sender, text, kind)

    if sender is None:
        return make(KIND_CALL if _CALL.match(rest) else KIND_SYSTEM)

    first = body.strip()
    if _SYSTEM_BODY.match(first):
        return make(KIND_SYSTEM)
    if _DELETED.match(first) and not entry.extra:
        return make(KIND_DELETED)
    if _CALL.match(first) and not entry.extra:
        return make(KIND_CALL)

    media_kind = _media_kind(first)
    if media_kind is not None:
        # Un adjunto con pie de foto: el pie de foto sí es texto del cliente.
        caption = "\n".join(entry.extra).strip()
        return make(KIND_TEXT, caption) if caption else make(media_kind)

    text = _EDITED.sub("", "\n".join([body, *entry.extra])).strip()
    return make(KIND_TEXT, text)


def parse_chat_lines(lines: Iterable[str]) -> ParsedChat:
    """
    Parsea las líneas de un chat exportado.
    Args:
        lines: Líneas del archivo de chat (Android o iOS).
    Returns:
        Los mensajes clasificados y el número de líneas huérfanas.
    Raises:
        ValueError: Si no se reconoce el formato o hay una fecha inválida.
    """
    lines = list(lines)
    fmt = detect_format(lines)
    entries, orphans = _split_entries(lines, fmt)
    return ParsedChat([_classify(entry) for entry in entries], orphans)


# --- Salida ----------------------------------------------------------------
def _normalize_name(name: str) -> str:
    # Normaliza un nombre para compararlo (mayúsculas y espacios)
    return " ".join(name.split()).casefold()


def messages_to_rows(
    chat_id: str, messages: Iterable[ParsedMessage], admin_names: Iterable[str]
) -> list[dict[str, str]]:
    """
    Convierte los mensajes de texto en filas con el esquema de ``messages.csv``.
    Args:
        chat_id: Identificador anónimo del chat (p. ej. ``chat_0001``).
        messages: Mensajes ya clasificados.
        admin_names: Nombres de remitente que corresponden a la finca. Cualquier otro remitente se marca como ``cliente``.
    Returns:
        Filas con las columnas de ``CSV_COLUMNS``; solo mensajes de texto.
    """
    admins = {_normalize_name(name) for name in admin_names}
    rows: list[dict[str, str]] = []
    for message in messages:
        if message.kind != KIND_TEXT or not message.text:
            continue
        is_admin = message.sender is not None and (
            _normalize_name(message.sender) in admins
        )
        rows.append(
            {
                "msg_id": f"{chat_id}_{len(rows) + 1:05d}",
                "chat_id": chat_id,
                "timestamp": message.timestamp.isoformat(sep=" "),
                "sender_role": ROLE_ADMIN if is_admin else ROLE_CLIENT,
                "text": message.text,
            }
        )
    return rows


def build_dataset(input_path: Path, admin_names: Iterable[str]) -> Dataset:
    """
    Procesa todos los chats de ``input_path`` y arma las filas de salida.
    Un chat que falla (formato no reconocido, fecha inválida, zip dañado) no
    detiene el proceso: se anota en ``stats.failed_chats`` con su ``chat_id``.
    Args:
        input_path: Un ``.txt``/``.zip`` o una carpeta con varios.
        admin_names: Nombres de remitente que corresponden a la finca.
    Returns:
        El conjunto de filas, el índice de chats y los conteos.
    """
    admin_names = list(admin_names)
    dataset = Dataset()
    for number, source in enumerate(list_chat_files(input_path), start=1):
        chat_id = f"chat_{number:04d}"
        dataset.chat_index.append((chat_id, source.name))
        dataset.stats.chats += 1
        try:
            parsed = parse_chat_lines(read_chat_lines(source))
        except UnicodeDecodeError:
            dataset.stats.failed_chats.append(f"{chat_id}: codificación no UTF-8")
            continue
        except (ValueError, zipfile.BadZipFile, OSError) as exc:
            dataset.stats.failed_chats.append(f"{chat_id}: {exc}")
            continue
        rows = messages_to_rows(chat_id, parsed.messages, admin_names)
        dataset.rows.extend(rows)
        dataset.stats.kinds.update(message.kind for message in parsed.messages)
        dataset.stats.roles.update(row["sender_role"] for row in rows)
        dataset.stats.orphan_lines += parsed.orphan_lines
    return dataset


def write_messages_csv(rows: Iterable[dict[str, str]], path: Path) -> None:
    # Escribe ``messages.csv`` (UTF-8) con las columnas del esquema acordado.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_chat_index(index: Iterable[tuple[str, str]], path: Path) -> None:
    """
    Escribe la correspondencia ``chat_id`` -> archivo de origen.
    Contiene nombres de archivo (que pueden incluir nombres de contactos), por
    eso debe quedarse en ``data/interim/`` y no versionarse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("chat_id", "source_file"))
        writer.writerows(index)


# --- Línea de comandos -----------------------------------------------------
def _read_admin_file(path: Path) -> list[str]:
    # Lee nombres de remitente (uno por línea; ``#`` inicia un comentario).
    names = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convierte exports de WhatsApp (.txt o .zip) en messages.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Archivo .txt/.zip o carpeta con varios (por defecto data/raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV de salida (por defecto data/interim/messages.csv).",
    )
    parser.add_argument(
        "--admin",
        action="append",
        default=[],
        help="Nombre del remitente de la finca tal como sale en el export "
        "(se puede repetir si el nombre ha cambiado).",
    )
    parser.add_argument(
        "--admin-file",
        type=Path,
        help="Archivo local con un nombre de remitente de la finca por línea.",
    )
    return parser


def _print_summary(dataset: Dataset) -> None:
    # Imprime solo conteos: nunca el contenido de los chats.
    stats = dataset.stats
    print(f"Chats procesados: {stats.chats} (con error: {len(stats.failed_chats)})")
    for failure in stats.failed_chats:
        print(f"  - {failure}")
    print("Mensajes por tipo (antes de filtrar):")
    for kind, count in sorted(stats.kinds.items()):
        print(f"  {kind}: {count}")
    print(
        f"Filas escritas: {len(dataset.rows)} "
        f"(cliente: {stats.roles[ROLE_CLIENT]}, admin: {stats.roles[ROLE_ADMIN]})"
    )
    print(f"Líneas sin encabezado antes del primer mensaje: {stats.orphan_lines}")
    if dataset.rows and stats.roles[ROLE_ADMIN] == 0:
        print(
            "AVISO: no se encontró ningún mensaje del admin; revisa que el "
            "nombre de --admin coincida con el del export."
        )


def main(argv: list[str] | None = None) -> int:
    """
    Punto de entrada de la línea de comandos.
    Args:
        argv: Argumentos (por defecto, los de ``sys.argv``).
    Returns:
        0 si todos los chats se procesaron; 1 si alguno falló.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    admin_names = list(args.admin)
    if args.admin_file:
        admin_names.extend(_read_admin_file(args.admin_file))
    if not admin_names:
        parser.error(
            "Indica al menos un remitente de la finca con --admin o --admin-file."
        )
    try:
        dataset = build_dataset(args.input, admin_names)
    except ValueError as exc:
        parser.error(str(exc))
    write_messages_csv(dataset.rows, args.output)
    write_chat_index(dataset.chat_index, args.output.with_name("chat_index.csv"))
    _print_summary(dataset)
    return 1 if dataset.stats.failed_chats else 0


if __name__ == "__main__":
    sys.exit(main())
