"""Textos del bot y reglas de qué contestar, sin depender de Telegram.

Se separa de ``telegram_bot.py`` para poder probar la lógica sin conexión ni token: aquí se
decide qué texto recibe el cliente, cuándo se avisa al chat de administración y cómo se
reenvían las respuestas del equipo al cliente.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

from src.engines.base import Engine, EngineResponse
from src.engines.m0_rules import normalize_text

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
OUT_OF_SCOPE_INTENT = "fuera_de_alcance"

WELCOME_TEXT = (
    "¡Hola! Soy el asistente automático de Alojamientos Villa Yaiza, la finca para "
    "eventos en Ibagué.\n\n"
    "Puedo responderte sobre la capacidad de la finca, sus reglas y servicios, y cómo "
    "llegar. Las preguntas sobre precios, fechas disponibles y reservas las atiende una "
    "persona del equipo.\n\n"
    "Escribe tu pregunta. Por favor no envíes datos personales (teléfonos, documentos o "
    "números de cuenta) por este chat."
)
HELP_TEXT = (
    "Puedes preguntarme, por ejemplo:\n"
    "• ¿Para cuántas personas es la finca?\n"
    "• ¿Se pueden llevar mascotas?\n"
    "• ¿Hasta qué hora se puede poner música?\n"
    "• ¿A qué hora es el check-in?\n"
    "• ¿Dónde queda la finca?\n\n"
    "Para precios, fechas disponibles o reservas te comunico con una persona del equipo."
)
ESCALATION_WITH_ADMIN_TEXT = (
    "Esa consulta la atiende directamente una persona del equipo de Villa Yaiza. Ya le "
    "pasé tu mensaje y te responderá por este chat lo antes posible."
)
ESCALATION_WITHOUT_ADMIN_TEXT = (
    "Esa consulta la debe atender una persona del equipo de Villa Yaiza, así que no puedo "
    "responderla por aquí. Por favor comunícate con la administración de la finca."
)
OUT_OF_SCOPE_TEXT = (
    "Solo puedo ayudarte con preguntas sobre Alojamientos Villa Yaiza: capacidad, reglas, "
    "servicios y ubicación. ¿Tienes alguna pregunta sobre la finca?"
)
NON_TEXT_TEXT = "Por ahora solo puedo leer mensajes de texto. ¿Me escribes tu pregunta?"
ADMIN_REPLY_HEADER = "Respuesta del equipo de Villa Yaiza:"
ADMIN_REPLY_PREFIX = ADMIN_REPLY_HEADER + "\n\n"
ADMIN_REPLY_SENT_TEXT = "Respuesta enviada al cliente."
ADMIN_REPLY_FAILED_TEXT = (
    "No se pudo enviar la respuesta al cliente (pudo haber bloqueado el bot)."
)
UNKNOWN_NOTICE_TEXT = (
    "Ese mensaje no es un aviso de consulta, o es de antes de reiniciar el bot, así que "
    "no sé a qué cliente enviarlo. Para contestarle a un cliente, responde directamente "
    "al aviso de su consulta."
)

# --------------------------------------------------------------------------------------
# Saludos, agradecimientos y confirmaciones
# --------------------------------------------------------------------------------------
# Un mensaje que solo saluda, agradece, se despide o confirma no es una consulta: no se le
# pasa al motor (que lo escalaría como ``otro_negocio``) y se contesta con un texto fijo.
# Son las mismas categorías que se descartan al construir las preguntas de evaluación
# (saludos, agradecimientos, confirmaciones y mensajes vacíos), así que esta regla no
# cambia nada de lo evaluado. Basta una palabra fuera de estas listas (p. ej. "piscina" o
# "información") para que el mensaje sí vaya al motor.
SMALL_TALK_MAX_WORDS = 8
SMALL_TALK_GREETING = "saludo"
SMALL_TALK_THANKS = "agradecimiento"
SMALL_TALK_FAREWELL = "despedida"
SMALL_TALK_ACK = "confirmacion"
SMALL_TALK_EMPTY = "vacio"

_THANKS_WORDS = {"gracias", "agradezco", "agradecida", "agradecido", "bendiga"}
_FAREWELL_WORDS = {"chao", "chau", "adios", "luego", "vemos", "pronto"}
_GREETING_WORDS = {"hola", "holi", "buenas", "buenos", "buen", "saludos", "hey", "tal"}
_ACK_WORDS = {
    "ok",
    "okey",
    "okay",
    "oki",
    "vale",
    "listo",
    "perfecto",
    "bueno",
    "bien",
    "entendido",
    "entiendo",
    "claro",
    "si",
    "dale",
    "genial",
    "excelente",
    "super",
    "ah",
    "ya",
    "ja",
    "jaja",
    "jajaja",
}
# Palabras que acompañan a las anteriores pero no deciden la categoría.
_FILLER_WORDS = {
    "dia",
    "dias",
    "tardes",
    "noches",
    "que",
    "como",
    "esta",
    "estas",
    "estan",
    "muy",
    "muchas",
    "muchisimas",
    "mil",
    "por",
    "todo",
    "a",
    "y",
    "de",
    "la",
    "el",
    "le",
    "te",
    "lo",
    "los",
    "las",
    "usted",
    "ustedes",
    "nada",
    "no",
    "senor",
    "senora",
    "don",
    "dona",
    "amigo",
    "amiga",
    "dios",
    "hasta",
    "nos",
    "feliz",
    "igualmente",
}
# Solo cuentan como charla si el mensaje agradece ("gracias por la información"); solas
# ("información") son una consulta.
_THANKS_ONLY_WORDS = {"info", "informacion", "atencion", "ayuda", "respuesta"}
_WORD_RE = re.compile(r"[a-z]+")

SMALL_TALK_REPLIES = {
    SMALL_TALK_GREETING: (
        "¡Hola! ¿En qué te puedo ayudar? Puedo responderte sobre la capacidad de la "
        "finca, sus reglas y servicios, y cómo llegar."
    ),
    SMALL_TALK_THANKS: "¡Con gusto! Si tienes otra pregunta sobre la finca, aquí estoy.",
    SMALL_TALK_FAREWELL: (
        "¡Hasta pronto! Cuando quieras, escríbeme tu pregunta sobre la finca."
    ),
    SMALL_TALK_ACK: "Listo. Si tienes otra pregunta sobre la finca, escríbeme.",
    SMALL_TALK_EMPTY: "¿En qué te puedo ayudar? Escríbeme tu pregunta sobre la finca.",
}


def classify_small_talk(message: str) -> str | None:
    """Indica si el mensaje es solo un saludo, agradecimiento, despedida o confirmación.

    Usa la misma normalización que M0 (minúsculas, sin tildes ni letras repetidas, así
    "Holaaa" cuenta como "hola"). Un mensaje sin palabras (solo emojis o signos) cuenta
    como ``vacio``.

    Returns:
        La categoría (``saludo``, ``agradecimiento``, ``despedida``, ``confirmacion`` o
        ``vacio``), o ``None`` si el mensaje debe ir al motor.
    """
    words = _WORD_RE.findall(normalize_text(message))
    if not words:
        return SMALL_TALK_EMPTY
    if len(words) > SMALL_TALK_MAX_WORDS:
        return None

    vocabulary = (
        _THANKS_WORDS | _FAREWELL_WORDS | _GREETING_WORDS | _ACK_WORDS | _FILLER_WORDS
    )
    present = set(words)
    thanks = bool(present & _THANKS_WORDS)
    if thanks:
        vocabulary = vocabulary | _THANKS_ONLY_WORDS
    if not present <= vocabulary:
        return None

    if thanks:
        return SMALL_TALK_THANKS
    if present & _FAREWELL_WORDS:
        return SMALL_TALK_FAREWELL
    if present & _GREETING_WORDS:
        return SMALL_TALK_GREETING
    if present & _ACK_WORDS:
        return SMALL_TALK_ACK
    return SMALL_TALK_EMPTY  # solo palabras de relleno ("¿qué?", "no")


@dataclass(frozen=True)
class BotReply:
    """Qué se le contesta al cliente.

    Attributes:
        text: Texto para el cliente.
        notify_admin: ``True`` si además se avisa al chat de administración.
    """

    text: str
    notify_admin: bool


def safe_respond(engine: Engine, message: str) -> tuple[EngineResponse, str | None]:
    """Llama al motor y, si falla por cualquier motivo, escala en lugar de caerse.

    Se captura cualquier excepción a propósito: un cliente nunca debe quedarse sin
    respuesta porque, por ejemplo, Ollama se apagó a mitad de la demo. Escalar es la opción
    segura, la misma que toma M3 cuando una respuesta puntual falla.

    Returns:
        La respuesta del motor y ``None``, o una respuesta que escala y el nombre del error.
    """
    try:
        return engine.respond(message), None
    except Exception as exc:  # noqa: BLE001 (ver docstring)
        fallback = EngineResponse(
            answer=None, intent=None, confidence=0.0, escalate=True
        )
        return fallback, type(exc).__name__


def build_reply(response: EngineResponse, admin_enabled: bool) -> BotReply:
    """Decide el texto para el cliente y si se avisa a administración.

    * Si el motor responde, se envía su texto tal cual.
    * Si escala por ``fuera_de_alcance``, se aclara qué temas atiende el bot y no se molesta
      al equipo.
    * Si escala por otro motivo, se avisa al equipo (si hay chat de administración).
    """
    answer = (response.answer or "").strip()
    if not response.escalate and answer:
        return BotReply(text=answer, notify_admin=False)
    if response.intent == OUT_OF_SCOPE_INTENT:
        return BotReply(text=OUT_OF_SCOPE_TEXT, notify_admin=False)
    if admin_enabled:
        return BotReply(text=ESCALATION_WITH_ADMIN_TEXT, notify_admin=True)
    return BotReply(text=ESCALATION_WITHOUT_ADMIN_TEXT, notify_admin=False)


def admin_notification_text(
    message: str, response: EngineResponse, engine_name: str
) -> str:
    """Texto del aviso que recibe el chat de administración por cada consulta escalada."""
    intent = response.intent or "sin intención"
    return (
        f"Consulta escalada por el bot ({engine_name}).\n"
        f"Intención estimada: {intent}\n\n"
        f"Mensaje del cliente:\n{message}\n\n"
        "Responde a ESTE mensaje (Responder / Reply) y el bot le enviará tu respuesta al "
        "cliente."
    )


def split_message(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Parte un texto largo en mensajes que Telegram acepte (máximo 4096 caracteres).

    Corta preferiblemente en un salto de línea y, si no hay, en un espacio.
    """
    if limit < 1:
        raise ValueError("limit debe ser un entero positivo.")
    remaining = text.strip()
    parts: list[str] = []
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = remaining.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


class RelayRegistry:
    """Recuerda a qué cliente corresponde cada aviso enviado al chat de administración.

    Cuando alguien del equipo responde a un aviso, el bot busca aquí el chat del cliente y
    le reenvía la respuesta. Solo vive en memoria (no se guarda en disco) y conserva los
    ``max_items`` avisos más recientes; al reiniciar el bot se pierde, y para responder a
    un aviso viejo hay que contactar al cliente por otro medio.
    """

    def __init__(self, max_items: int = 1000) -> None:
        if max_items < 1:
            raise ValueError("max_items debe ser un entero positivo.")
        self.max_items = max_items
        self._client_by_notice: OrderedDict[int, int] = OrderedDict()

    def remember(self, notice_message_id: int, client_chat_id: int) -> None:
        """Asocia el id del aviso con el chat del cliente."""
        self._client_by_notice[notice_message_id] = client_chat_id
        self._client_by_notice.move_to_end(notice_message_id)
        while len(self._client_by_notice) > self.max_items:
            self._client_by_notice.popitem(last=False)

    def lookup(self, notice_message_id: int) -> int | None:
        """Devuelve el chat del cliente de ese aviso, o ``None`` si no se conoce."""
        return self._client_by_notice.get(notice_message_id)

    def __len__(self) -> int:
        return len(self._client_by_notice)
