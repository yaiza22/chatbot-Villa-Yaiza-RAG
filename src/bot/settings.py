"""Configuración del bot de Telegram, leída de variables de entorno o del archivo ``.env``.

El archivo ``.env`` vive en la raíz del repositorio y no se versiona (está en ``.gitignore``)
porque guarda el token del bot, que funciona como una contraseña. Variables que se leen:

* ``TELEGRAM_BOT_TOKEN`` (obligatoria): token que entrega @BotFather.
* ``BOT_ENGINE`` (opcional, por defecto ``M0+M3``): motor que responde. Puede ser un motor
  solo (``M0``, ``M1``, ``M2``, ``M3``) o ``clasificador+redactor`` (``M0+M3``, ``M1+M2``...).
* ``ADMIN_CHAT_ID`` (opcional): chat que recibe las consultas escaladas. Vacío = sin avisos.
* ``BOT_TRAIN_INPUT`` y ``BOT_TRAIN_SOURCE`` (opcionales): datos con los que se entrenan M1 y
  M2 si el motor elegido los usa. Por defecto ``data/processed/labeled.csv`` y ``real``, igual
  que en la evaluación final.
* ``BOT_LLM_TIMEOUT_S`` (opcional, por defecto 60): segundos máximos que el bot espera a
  Ollama por respuesta. Si se pasa, la consulta escala. En la evaluación el p95 de M3 fue de
  6 a 8 s; el límite de ``src/config.py`` (300 s) es para las corridas de evaluación y solo
  se usa al arrancar, cuando Ollama puede tardar en cargar el modelo.

Se lee el ``.env`` con unas pocas líneas propias para no agregar otra dependencia.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from src.config import PROCESSED_DIR, REPO_ROOT
from src.eval.run_eval import VALID_TRAIN_SOURCES

ENV_FILE = REPO_ROOT / ".env"
DEFAULT_ENGINE_SPEC = "M0+M3"
DEFAULT_TRAIN_INPUT = PROCESSED_DIR / "labeled.csv"
DEFAULT_TRAIN_SOURCE = "real"
DEFAULT_LLM_TIMEOUT_S = 60.0

# Formato de los tokens de @BotFather: "<número>:<texto>". Solo se usa para avisar de un
# token mal copiado; el token nunca se imprime.
_TOKEN_PATTERN = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")


class SettingsError(ValueError):
    """La configuración del bot está incompleta o tiene un valor inválido."""


@dataclass(frozen=True)
class BotSettings:
    """Configuración ya validada del bot.

    Attributes:
        token: Token del bot. ``repr=False`` evita que aparezca si se imprime el objeto.
        engine_spec: Motor pedido, normalizado (p. ej. ``"M0+M3"``).
        admin_chat_id: Chat que recibe las consultas escaladas, o ``None``.
        train_input: CSV con el que se entrenan M1 y M2 (si el motor los usa).
        train_source: ``source`` permitido al entrenar (``real``, ``synthetic`` o ``both``).
        llm_timeout_s: Segundos máximos de espera a Ollama mientras el bot atiende.
    """

    token: str = field(repr=False)
    engine_spec: str
    admin_chat_id: int | None
    train_input: Path
    train_source: str
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S


def load_env_file(
    path: Path | str = ENV_FILE,
    environ: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Carga las líneas ``CLAVE=valor`` de un archivo ``.env`` en las variables de entorno.

    No sobrescribe variables que ya existan (así se puede probar otro valor desde la
    terminal sin editar el archivo). Ignora líneas vacías y comentarios (``#``), acepta el
    prefijo ``export`` y quita comillas alrededor del valor. Lee con ``utf-8-sig`` porque el
    Bloc de notas puede guardar el archivo con BOM.

    Args:
        path: Ruta del archivo. Si no existe, no hace nada.
        environ: Diccionario donde se cargan los valores (por defecto ``os.environ``).

    Returns:
        Nombres de las variables cargadas (nunca sus valores).

    Raises:
        SettingsError: Si una línea no tiene el formato ``CLAVE=valor``.
    """
    environ = os.environ if environ is None else environ
    path = Path(path)
    if not path.is_file():
        return []

    loaded: list[str] = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            raise SettingsError(
                f"La línea {line_number} de {path.name} no tiene el formato CLAVE=valor."
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key not in environ:
            environ[key] = value
            loaded.append(key)
    return loaded


def _parse_admin_chat_id(raw_value: str) -> int | None:
    """Convierte ``ADMIN_CHAT_ID`` en entero (los grupos tienen id negativo) o ``None``."""
    value = raw_value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise SettingsError(
            "ADMIN_CHAT_ID debe ser un número entero (puede empezar con '-'). Obtenlo "
            "enviando /miid al bot desde el chat que recibirá los avisos, o déjalo vacío."
        ) from exc


def read_settings(environ: Mapping[str, str] | None = None) -> BotSettings:
    """Lee y valida la configuración del bot.

    Args:
        environ: Variables de entorno (por defecto ``os.environ``).

    Returns:
        ``BotSettings`` validado.

    Raises:
        SettingsError: Si falta el token o algún valor no es válido.
    """
    environ = os.environ if environ is None else environ

    token = environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SettingsError(
            "Falta TELEGRAM_BOT_TOKEN. Agrégalo al archivo .env de la raíz del repositorio "
            "con el token que entregó @BotFather."
        )
    if not _TOKEN_PATTERN.match(token):
        raise SettingsError(
            "TELEGRAM_BOT_TOKEN no tiene el formato de un token de @BotFather "
            "(número, dos puntos y texto). Revisa que se haya copiado completo."
        )

    engine_spec = environ.get("BOT_ENGINE", "").replace(" ", "").upper()
    engine_spec = engine_spec or DEFAULT_ENGINE_SPEC

    train_input = Path(
        environ.get("BOT_TRAIN_INPUT", "").strip() or DEFAULT_TRAIN_INPUT
    )
    if not train_input.is_absolute():
        train_input = REPO_ROOT / train_input

    train_source = environ.get("BOT_TRAIN_SOURCE", "").strip() or DEFAULT_TRAIN_SOURCE
    if train_source not in VALID_TRAIN_SOURCES:
        raise SettingsError(
            f"BOT_TRAIN_SOURCE={train_source!r} no es válido. "
            f"Opciones: {', '.join(VALID_TRAIN_SOURCES)}."
        )

    raw_timeout = environ.get("BOT_LLM_TIMEOUT_S", "").strip()
    try:
        llm_timeout_s = float(raw_timeout) if raw_timeout else DEFAULT_LLM_TIMEOUT_S
    except ValueError:
        llm_timeout_s = -1.0
    if not llm_timeout_s > 0:
        raise SettingsError(
            "BOT_LLM_TIMEOUT_S debe ser un número de segundos mayor que 0 (p. ej. 60)."
        )

    return BotSettings(
        token=token,
        engine_spec=engine_spec,
        admin_chat_id=_parse_admin_chat_id(environ.get("ADMIN_CHAT_ID", "")),
        train_input=train_input,
        train_source=train_source,
        llm_timeout_s=llm_timeout_s,
    )
