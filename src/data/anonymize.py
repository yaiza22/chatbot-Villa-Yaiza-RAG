"""
Anonimizador de datos personales para los chats de la finca.

Lee un CSV sin anonimizar (como: ``data/interim/messages.csv``) y escribe otro con el mismo esquema
en el que los datos personales de las columnas de texto se reemplazan por marcadores fijos:

* ``[EMAIL]``: correos electrónicos.
* ``[URL]``: enlaces, dominios y usuarios de redes (@usuario).
* ``[TEL]``: teléfonos colombianos (móviles, fijos con indicativo) e internacionales con +.
* ``[DOC]``: cédulas, NIT y similares cuando van precedidos de una palabra clave ("cc", "cédula", "nit"...).
* ``[CUENTA]``: números de cuenta (con palabra clave bancaria) y otros números largos de 9 o más dígitos.
* ``[NOMBRE]``: nombres de personas de una lista local.

Reglas:
* Los reemplazos son marcadores fijos: no se guarda ningún mapeo entre el valor original
y el marcador, así que la operación no se puede revertir.
* El reporte contiene solo conteos por tipo, nunca contenido.
* Las reglas se aplican en un orden que evita que unas pisen a otras: correos, URLs, usuarios, teléfonos,
documentos, cuentas y, al final, nombres.
* Los precios, fechas y cantidades ("45 personas", "$1.500.000") no se tocan.
* Los nombres solo se detectan si están en la lista (archivo local, nunca en el repositorio) o, con
``--index``, si aparecen en el nombre del archivo del chat (se aplican únicamente a ese chat).
Un nombre que no esté en esas listas pasa sin enmascarar, por eso hace falta una revisión manual de
una muestra.
* Los términos de una lista de excepciones (p. ej. el nombre de la finca) nunca se enmascaran como nombre.

Uso (desde la raíz del repositorio):: python -m src.data.anonymize --names-file data/interim/names.txt
Se puede usar este comando más completo que tomas los nombres de los contactos (clientes) de chat_index.csv para aplicar esos nombres también.
py -m src.data.anonymize --names-file data/interim/names.txt --index data/interim/chat_index.csv
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "interim" / "messages.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "messages.csv"
REPORT_NAME = "anonymization_report.csv"

TAG_PHONE = "[TEL]"
TAG_EMAIL = "[EMAIL]"
TAG_URL = "[URL]"
TAG_NAME = "[NOMBRE]"
TAG_DOC = "[DOC]"
TAG_ACCOUNT = "[CUENTA]"
TAGS = (TAG_PHONE, TAG_EMAIL, TAG_URL, TAG_NAME, TAG_DOC, TAG_ACCOUNT)

# Nombre de la finca: no es un dato personal y no debe enmascararse aunque alguna de sus palabras figure en la lista de nombres.
DEFAULT_KEEP_TERMS = ("Alojamientos Villa Yaiza", "Villa Yaiza", "Finca Villa Yaiza")

# Palabras que no se consideran nombres cuando se sacan del nombre del archivo (un contacto guardado como "Finca Rosa" no debe enmascarar la palabra "finca").
_NEVER_NAME = frozenset(
    {
        "finca",
        "fincas",
        "villa",
        "alojamientos",
        "cliente",
        "clientes",
        "evento",
        "eventos",
        "casa",
        "grupo",
        "familia",
        "chat",
        "whatsapp",
        "reserva",
        "reservas",
        "señor",
        "señora",
        "don",
        "doña",
        "admin",
        "empresa",
        "hotel",
        "cabaña",
        "cabañas",
    }
)
# Partículas de nombres compuestos que no se enmascaran por sí solas.
_NAME_STOPWORDS = frozenset(
    {"del", "las", "los", "san", "van", "von", "dos", "das", "con", "por", "que"}
)
_MIN_NAME_LENGTH = 3

# --- Patrones ---------------------------------------------------------------
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

_URL = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+"
    r"|(?<![\w@.-])(?:wa\.me|goo\.gl|bit\.ly|youtu\.be|instagr\.am)(?:/[^\s<>\"']*)?"
    r"|(?<![\w@.-])(?:[a-z0-9-]+\.)+(?:com|net|org|app|info|co)"
    r"(?:/[^\s<>\"']*)?(?![\w@-])",
    re.IGNORECASE,
)
_URL_TRAILING = ".,;:!?)]}»\"'"

_HANDLE = re.compile(r"(?<![\w@.])@[A-Za-z0-9_][A-Za-z0-9_.]{2,}")

# Teléfono Móvil (colombiano): 3XX + 7 dígitos, con prefijo (+57, 57, 0057) y
# separadores (espacio, punto o guion) entre cualquier par de dígitos.
_PHONE_MOBILE = re.compile(
    r"(?<![\w.])(?:(?:\+|00)?57[\s.-]*)?\(?3\d{2}\)?(?:[\s.-]?\d){7}(?!\d)"
)
# Teléfono Fijo (colombiano): solo si trae indicativo, (60X) o (X), seguido de 7 dígitos.
_PHONE_LANDLINE = re.compile(
    r"(?<![\w.])(?:\(?60\d\)?|\([1-8]\))[\s.-]*[2-8](?:[\s.-]?\d){6}(?!\d)"
)
_PHONE_INTERNATIONAL = re.compile(r"(?<![\w.])\+\d{1,3}(?:[\s.-]?\d){7,12}(?!\d)")

# Documentos: el número debe ir precedido de una palabra clave.
_DOC = re.compile(
    r"(?P<kw>\b(?:cc|c\.c\.?|c[eé]dula|nit|identificaci[oó]n|documento|pasaporte)\b"
    r"[^\d\n]{0,15})(?P<num>\d(?:[.\s-]?\d){5,11})(?!\d)",
    re.IGNORECASE,
)

# Cuentas: palabra clave bancaria y un número de 8 a 19 dígitos.
_ACCOUNT_KEYWORD = re.compile(
    r"(?P<kw>\b(?:cuenta|cta|ahorros|corriente|nequi|daviplata|bancolombia"
    r"|davivienda|bbva|llave)\b[^\d\n]{0,30})(?P<num>\d(?:[\s.-]?\d){7,18})(?!\d)",
    re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"(?<![\d.,/-])\d{9,20}(?!\d)")

# Un número con puntos de mil ("1.500.000") es un precio, no una cuenta.
_PRICE = re.compile(r"\d{1,3}(?:\.\d{3})+")

_TAG_IN_TEXT = r"\[(?:tel|email|url|nombre|doc|cuenta)\]"


# --- Utilidades de texto ----------------------------------------------------
def _fold_char(char: str) -> str:
    """Pasa un carácter a minúscula sin tilde, conservando la longitud (1 a 1)."""
    lower = char.lower()
    if len(lower) != 1:
        lower = char
    if lower == "ñ":  # se conserva: "peña" y "pena" son palabras distintas
        return lower
    decomposed = unicodedata.normalize("NFD", lower)
    if len(decomposed) > 1 and all(unicodedata.combining(c) for c in decomposed[1:]):
        return decomposed[0]
    return lower


def fold(text: str) -> str:
    """
    Normaliza a minúsculas y sin tildes, con la misma longitud que ``text``.
    Al conservar la longitud, las posiciones encontradas en el texto normalizado
    sirven directamente sobre el texto original.
    """
    return "".join(_fold_char(char) for char in text)


def _name_tokens(names: Iterable[str], skip: frozenset[str] = frozenset()) -> set[str]:
    # Separa nombres completos en palabras sueltas normalizadas.
    tokens: set[str] = set()
    for name in names:
        for token in re.findall(r"[^\W\d_]+", fold(name)):
            if (
                len(token) >= _MIN_NAME_LENGTH
                and token not in _NAME_STOPWORDS
                and token not in skip
            ):
                tokens.add(token)
    return tokens


def _replace_spans(text: str, spans: list[tuple[int, int]], tag: str) -> str:
    # Reemplaza los tramos ``spans`` (ordenados y sin solapar) por ``tag``
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        pieces.append(tag)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _merge_adjacent(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # Une tramos separados solo por espacios ("Juan Pérez" cuenta como uno).
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and not text[merged[-1][1] : start].strip():
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _overlaps(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in others)


# --- Carga de listas --------------------------------------------------------
def load_lines(path: Path) -> list[str]:
    # Lee un archivo de listas (una entrada por línea; ``#`` inicia comentario).
    lines = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


_CONTACT_PREFIX = re.compile(
    r"^(?:chat de whatsapp con|whatsapp chat with|whatsapp chat\s*-)\s*", re.IGNORECASE
)


def contact_tokens(source_file: str) -> set[str]:
    """
    Saca las palabras del nombre del contacto a partir del nombre del archivo.

    Args:
        source_file: Nombre del archivo exportado, p. ej. "Chat de WhatsApp con <contacto>.zip".

    Returns:
        Palabras normalizadas del nombre (vacío si el contacto es un número).
    """
    stem = _CONTACT_PREFIX.sub("", Path(source_file).stem)
    return _name_tokens([stem], skip=_NEVER_NAME)


def load_chat_names(index_path: Path) -> dict[str, set[str]]:
    # Lee ``chat_index.csv`` y devuelve las palabras de nombre de cada chat.
    with index_path.open(encoding="utf-8", newline="") as handle:
        return {
            row["chat_id"]: contact_tokens(row["source_file"])
            for row in csv.DictReader(handle)
        }


# --- Anonimizador -----------------------------------------------------------
class Anonymizer:
    """
    Reemplaza datos personales de un texto por marcadores fijos.

    Args:
        names: Nombres de personas a enmascarar en todos los chats. Se separan en palabras sueltas, sin distinguir mayúsculas ni tildes.
        keep_terms: Frases que nunca se enmascaran como nombre.
        chat_names: Palabras de nombre adicionales por ``chat_id``, que solo se aplican a ese chat.
    """

    def __init__(
        self,
        names: Iterable[str] = (),
        keep_terms: Iterable[str] = DEFAULT_KEEP_TERMS,
        chat_names: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self._global_tokens = _name_tokens(names)
        self._chat_tokens = {
            chat_id: set(tokens) for chat_id, tokens in (chat_names or {}).items()
        }
        protected = [re.escape(fold(term)) for term in keep_terms if term.strip()]
        protected.append(_TAG_IN_TEXT)
        self._protected = re.compile("|".join(protected))
        self._patterns: dict[str | None, re.Pattern[str] | None] = {}

    @property
    def has_names(self) -> bool:
        # Indica si hay alguna lista de nombres cargada.
        return bool(self._global_tokens or self._chat_tokens)

    @property
    def name_token_count(self) -> int:
        # Número de palabras de nombre globales cargadas (solo para el reporte).
        return len(self._global_tokens)

    def anonymize(
        self, text: str, chat_id: str | None = None
    ) -> tuple[str, Counter[str]]:
        """
        Anonimiza un texto.
        Args:
            text: Texto original.
            chat_id: Chat al que pertenece, para aplicar sus nombres propios.
        Returns:
            El texto con marcadores y un conteo de reemplazos por marcador.
        """
        counts: Counter[str] = Counter()
        text = self._sub(_EMAIL, text, TAG_EMAIL, counts)
        text = self._sub(_URL, text, TAG_URL, counts, strip_trailing=True)
        text = self._sub(_HANDLE, text, TAG_URL, counts)
        # Primero los números con "+": si no, el móvil colombiano se quedaría
        # con la parte final de un número extranjero y dejaría el prefijo suelto.
        text = self._sub(_PHONE_INTERNATIONAL, text, TAG_PHONE, counts)
        text = self._sub(_PHONE_MOBILE, text, TAG_PHONE, counts)
        text = self._sub(_PHONE_LANDLINE, text, TAG_PHONE, counts)
        text = self._sub_keyword(_DOC, text, TAG_DOC, counts, min_digits=6)
        text = self._sub_keyword(
            _ACCOUNT_KEYWORD, text, TAG_ACCOUNT, counts, min_digits=8, skip_price=True
        )
        text = self._sub(_LONG_NUMBER, text, TAG_ACCOUNT, counts)
        text = self._mask_names(text, chat_id, counts)
        return text, counts

    @staticmethod
    def _sub(
        pattern: re.Pattern[str],
        text: str,
        tag: str,
        counts: Counter[str],
        strip_trailing: bool = False,
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            tail = ""
            if strip_trailing:
                stripped = value.rstrip(_URL_TRAILING)
                tail = value[len(stripped) :]
            counts[tag] += 1
            return tag + tail

        return pattern.sub(replace, text)

    @staticmethod
    def _sub_keyword(
        pattern: re.Pattern[str],
        text: str,
        tag: str,
        counts: Counter[str],
        min_digits: int,
        skip_price: bool = False,
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            number = match.group("num")
            digits = sum(char.isdigit() for char in number)
            if digits < min_digits or (skip_price and _PRICE.fullmatch(number)):
                return match.group(0)
            counts[tag] += 1
            return match.group("kw") + tag

        return pattern.sub(replace, text)

    def _name_pattern(self, chat_id: str | None) -> re.Pattern[str] | None:
        # Patrón con los nombres globales más los propios del chat (con caché).
        if chat_id not in self._patterns:
            tokens = self._global_tokens | self._chat_tokens.get(chat_id or "", set())
            if tokens:
                ordered = sorted(tokens, key=lambda token: (-len(token), token))
                body = "|".join(re.escape(token) for token in ordered)
                self._patterns[chat_id] = re.compile(rf"(?<!\w)(?:{body})(?!\w)")
            else:
                self._patterns[chat_id] = None
        return self._patterns[chat_id]

    def _mask_names(self, text: str, chat_id: str | None, counts: Counter[str]) -> str:
        pattern = self._name_pattern(chat_id)
        if pattern is None:
            return text
        folded = fold(text)
        protected = [match.span() for match in self._protected.finditer(folded)]
        spans = [
            match.span()
            for match in pattern.finditer(folded)
            if not _overlaps(match.span(), protected)
        ]
        spans = _merge_adjacent(text, spans)
        if not spans:
            return text
        counts[TAG_NAME] += len(spans)
        return _replace_spans(text, spans, TAG_NAME)


def build_anonymizer(
    names_file: Path | None = None,
    keep_file: Path | None = None,
    index_path: Path | None = None,
) -> Anonymizer:
    """
    Arma un ``Anonymizer`` a partir de archivos locales opcionales.
    Args:
        names_file: Lista de nombres a enmascarar en todos los chats.
        keep_file: Términos adicionales que nunca se enmascaran.
        index_path: ``chat_index.csv`` para sacar los nombres de cada contacto.
    Returns:
        El anonimizador configurado.
    """
    names = load_lines(names_file) if names_file else []
    keep_terms = list(DEFAULT_KEEP_TERMS)
    if keep_file:
        keep_terms.extend(load_lines(keep_file))
    chat_names = load_chat_names(index_path) if index_path else None
    return Anonymizer(names, keep_terms, chat_names)


# --- Procesamiento de CSV ---------------------------------------------------
@dataclass
class Report:
    # Conteos de una ejecución (sin contenido de los chats).
    rows_total: int = 0
    rows_changed: int = 0
    counts: Counter[str] = field(default_factory=Counter)


def anonymize_csv(
    input_path: Path,
    output_path: Path,
    anonymizer: Anonymizer,
    columns: Iterable[str] = ("text",),
) -> Report:
    """
    Anonimiza las columnas indicadas de un CSV y escribe el resultado.
    Args:
        input_path: CSV sin anonimizar.
        output_path: CSV de salida (mismas columnas y mismo orden).
        anonymizer: Anonimizador a usar.
        columns: Columnas de texto a anonimizar.
    Returns:
        Los conteos de la corrida.
    Raises:
        ValueError: Si alguna columna pedida no existe en el CSV.
    """
    columns = list(columns)
    report = Report()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open(encoding="utf-8", newline="") as source,
        output_path.open("w", encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in columns if column not in fieldnames]
        if missing:
            raise ValueError(f"El CSV no tiene la(s) columna(s): {', '.join(missing)}")
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            changed = False
            for column in columns:
                new_text, counts = anonymizer.anonymize(row[column], row.get("chat_id"))
                if counts:
                    changed = True
                    report.counts.update(counts)
                row[column] = new_text
            report.rows_total += 1
            report.rows_changed += int(changed)
            writer.writerow(row)
    return report


def write_report(report: Report, path: Path) -> None:
    # Escribe el reporte (solo conteos) en un CSV ``metric,count``.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "count"))
        writer.writerow(("rows_total", report.rows_total))
        writer.writerow(("rows_changed", report.rows_changed))
        for tag in TAGS:
            writer.writerow((tag, report.counts[tag]))


# --- Línea de comandos ------------------------------------------------------
def add_list_arguments(parser: argparse.ArgumentParser) -> None:
    # Agrega las opciones de listas locales (compartidas con ``check_pii``).
    parser.add_argument(
        "--names-file",
        type=Path,
        help="Archivo local con un nombre de persona por línea (no se versiona).",
    )
    parser.add_argument(
        "--keep-file",
        type=Path,
        help="Archivo local con términos que nunca se enmascaran como nombre.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="chat_index.csv: usa el nombre del contacto de cada archivo de chat.",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reemplaza datos personales de un CSV por marcadores fijos."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--columns",
        action="append",
        help="Columna de texto a anonimizar (repetible; por defecto 'text').",
    )
    add_list_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Punto de entrada de la línea de comandos.
    Args:
        argv: Argumentos (por defecto, los de ``sys.argv``).
    Returns:
        0 si terminó bien.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("La salida no puede ser el mismo archivo que la entrada.")
    try:
        anonymizer = build_anonymizer(args.names_file, args.keep_file, args.index)
        report = anonymize_csv(
            args.input, args.output, anonymizer, args.columns or ["text"]
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    write_report(report, args.output.with_name(REPORT_NAME))
    print(f"Filas: {report.rows_total} (con cambios: {report.rows_changed})")
    for tag in TAGS:
        print(f"  {tag}: {report.counts[tag]}")
    if not anonymizer.has_names:
        print(
            "AVISO: no se cargó ninguna lista de nombres; los nombres de personas NO se enmascararon."
        )
    print(f"Salida: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
