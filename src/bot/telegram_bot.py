"""Bot de Telegram de Alojamientos Villa Yaiza.

Recibe los mensajes de los clientes, se los pasa al motor elegido en ``BOT_ENGINE`` con
``engine.respond(texto)`` (la misma interfaz que se evaluó) y contesta. Si el motor escala,
avisa al chat de administración (si ``ADMIN_CHAT_ID`` está configurado), y cuando alguien
del equipo responde a ese aviso, el bot le reenvía la respuesta al cliente.

Funciona con *long polling*: el bot le pregunta a Telegram si hay mensajes nuevos, así que
no hace falta servidor, dominio ni abrir puertos. Basta con que este PC esté encendido, con
internet y (si el motor usa M3) con Ollama abierto.

Privacidad: el bot no guarda el texto de las conversaciones. En la terminal solo registra
el motor, la intención, si escaló y la latencia.

Uso (desde la raíz del repositorio, con el entorno activo)::

    python -m src.bot.telegram_bot

Comandos para el cliente: ``/start`` y ``/ayuda``. ``/miid`` muestra el id del chat, que se
copia en ``ADMIN_CHAT_ID`` para activar los avisos.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import suppress

from telegram import Bot, Update
from telegram.constants import ChatAction, ChatType
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.bot.conversation import (
    ADMIN_REPLY_FAILED_TEXT,
    ADMIN_REPLY_HEADER,
    ADMIN_REPLY_PREFIX,
    ADMIN_REPLY_SENT_TEXT,
    ESCALATION_WITHOUT_ADMIN_TEXT,
    HELP_TEXT,
    NON_TEXT_TEXT,
    SMALL_TALK_REPLIES,
    UNKNOWN_NOTICE_TEXT,
    WELCOME_TEXT,
    BotReply,
    RelayRegistry,
    admin_notification_text,
    build_reply,
    classify_small_talk,
    safe_respond,
    split_message,
)
from src.bot.engine_factory import build_engine, warm_up
from src.bot.settings import BotSettings, SettingsError, load_env_file, read_settings
from src.engines.base import Engine, EngineResponse

logger = logging.getLogger("villa_yaiza_bot")

# "escribiendo..." dura unos 5 s en Telegram; se renueva mientras el motor trabaja.
TYPING_REFRESH_S = 4.0

# Claves de ``application.bot_data``.
ENGINE_KEY = "engine"
SETTINGS_KEY = "settings"
RELAY_KEY = "relay"
LOCK_KEY = "engine_lock"


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------
async def _keep_typing(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    """Muestra "escribiendo..." en el chat hasta que se active ``stop``."""
    while not stop.is_set():
        with suppress(TelegramError):
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=TYPING_REFRESH_S)


async def _ask_engine(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str
) -> tuple[EngineResponse, str | None]:
    """Consulta al motor en un hilo aparte, con "escribiendo..." mientras tanto.

    El motor es código bloqueante (M3 tarda segundos): correrlo en un hilo deja al bot
    libre para recibir otros mensajes. El candado hace que el motor atienda una consulta a
    la vez, porque los modelos locales no están pensados para usarse en paralelo.
    """
    data = context.application.bot_data
    stop = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(context.bot, chat_id, stop))
    try:
        async with data[LOCK_KEY]:
            return await asyncio.to_thread(safe_respond, data[ENGINE_KEY], text)
    finally:
        stop.set()
        await typing_task


async def _notify_admin(
    context: ContextTypes.DEFAULT_TYPE,
    settings: BotSettings,
    text: str,
    response: EngineResponse,
    client_chat_id: int,
) -> bool:
    """Envía el aviso de consulta escalada y recuerda a qué cliente pertenece.

    Returns:
        ``True`` si el aviso se envió.
    """
    data = context.application.bot_data
    engine: Engine = data[ENGINE_KEY]
    try:
        notice = await context.bot.send_message(
            chat_id=settings.admin_chat_id,
            text=admin_notification_text(text, response, engine.name),
        )
    except TelegramError as exc:
        logger.warning(
            "No se pudo avisar al chat de administración (%s). Revisa ADMIN_CHAT_ID y "
            "que ese chat haya iniciado conversación con el bot.",
            type(exc).__name__,
        )
        return False
    data[RELAY_KEY].remember(notice.message_id, client_chat_id)
    return True


# --------------------------------------------------------------------------------------
# Manejadores
# --------------------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/start``: presentación del bot y aviso de privacidad."""
    if update.effective_message is not None:
        await update.effective_message.reply_text(WELCOME_TEXT)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/ayuda``: ejemplos de preguntas que el bot sabe responder."""
    if update.effective_message is not None:
        await update.effective_message.reply_text(HELP_TEXT)


async def chat_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/miid``: muestra el id del chat para configurar ``ADMIN_CHAT_ID``."""
    if update.effective_message is None or update.effective_chat is None:
        return
    await update.effective_message.reply_text(
        f"El id de este chat es: {update.effective_chat.id}\n"
        "Para recibir aquí las consultas escaladas, cópialo en ADMIN_CHAT_ID del archivo "
        ".env y reinicia el bot."
    )


async def _relay_admin_reply(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """Si el equipo respondió a un aviso, le reenvía la respuesta al cliente.

    Acepta texto y también fotos, ubicaciones, documentos o audios (se copian tal cual),
    por ejemplo para enviarle al cliente el pin de la finca.

    Returns:
        ``True`` si el mensaje era del chat de administración y ya se atendió aquí;
        ``False`` si debe seguir el flujo normal.
    """
    message = update.effective_message
    chat = update.effective_chat
    settings: BotSettings = context.application.bot_data[SETTINGS_KEY]
    if (
        message is None
        or chat is None
        or settings.admin_chat_id is None
        or chat.id != settings.admin_chat_id
        or message.reply_to_message is None
    ):
        return False

    relay: RelayRegistry = context.application.bot_data[RELAY_KEY]
    client_chat_id = relay.lookup(message.reply_to_message.message_id)
    if client_chat_id is None:
        # En un chat privado de administración, responder a otro mensaje se trata como
        # una pregunta normal (sirve para probar). En el grupo se explica qué pasó.
        if chat.type == ChatType.PRIVATE:
            return False
        replied_author = message.reply_to_message.from_user
        if replied_author is not None and replied_author.id == context.bot.id:
            await message.reply_text(UNKNOWN_NOTICE_TEXT)
        return True

    try:
        if message.text:
            await context.bot.send_message(
                chat_id=client_chat_id, text=ADMIN_REPLY_PREFIX + message.text
            )
        else:
            await context.bot.send_message(
                chat_id=client_chat_id, text=ADMIN_REPLY_HEADER
            )
            await context.bot.copy_message(
                chat_id=client_chat_id,
                from_chat_id=chat.id,
                message_id=message.message_id,
            )
    except TelegramError as exc:
        logger.warning(
            "No se pudo reenviar la respuesta al cliente (%s).", type(exc).__name__
        )
        await message.reply_text(ADMIN_REPLY_FAILED_TEXT)
        return True
    await message.reply_text(ADMIN_REPLY_SENT_TEXT)
    logger.info("respuesta_del_equipo_reenviada=True")
    return True


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mensaje de texto: respuesta del equipo a un aviso, o pregunta de un cliente."""
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None or not message.text:
        return
    settings: BotSettings = context.application.bot_data[SETTINGS_KEY]

    # 1) Alguien del equipo responde a un aviso: se le reenvía al cliente.
    if await _relay_admin_reply(update, context):
        return

    # 2) En grupos el bot no conversa: solo atiende las respuestas del equipo.
    if chat.type != ChatType.PRIVATE:
        return

    # 3) Saludo, agradecimiento o confirmación: texto fijo, sin motor ni aviso.
    small_talk = classify_small_talk(message.text)
    if small_talk is not None:
        await message.reply_text(SMALL_TALK_REPLIES[small_talk])
        logger.info("charla=%s escalo=False aviso_admin=False", small_talk)
        return

    # 4) Pregunta de un cliente.
    response, error = await _ask_engine(context, chat.id, message.text)
    reply = build_reply(response, admin_enabled=settings.admin_chat_id is not None)
    if reply.notify_admin:
        sent = await _notify_admin(context, settings, message.text, response, chat.id)
        if not sent:
            reply = BotReply(text=ESCALATION_WITHOUT_ADMIN_TEXT, notify_admin=False)

    for part in split_message(reply.text):
        await message.reply_text(part)

    engine: Engine = context.application.bot_data[ENGINE_KEY]
    latency = response.latency_ms if response.latency_ms is not None else float("nan")
    logger.info(
        "motor=%s intencion=%s escalo=%s latencia_ms=%.0f aviso_admin=%s",
        engine.name,
        response.intent,
        response.escalate,
        latency,
        reply.notify_admin,
    )
    if error is not None:
        logger.warning("El motor falló (%s); la consulta se escaló.", error)


async def non_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fotos, audios, ubicaciones, stickers, etc.

    En el chat de administración pueden ser la respuesta del equipo a un aviso. En un chat
    privado con un cliente, el bot explica que solo lee texto. En grupos no hace nada.
    """
    if await _relay_admin_reply(update, context):
        return
    chat = update.effective_chat
    if (
        update.effective_message is not None
        and chat is not None
        and chat.type == ChatType.PRIVATE
    ):
        await update.effective_message.reply_text(NON_TEXT_TEXT)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Registra errores no controlados sin imprimir el contenido de los mensajes."""
    logger.error("Error no controlado en el bot: %s", type(context.error).__name__)


# --------------------------------------------------------------------------------------
# Arranque
# --------------------------------------------------------------------------------------
def build_application(settings: BotSettings, engine: Engine) -> Application:
    """Crea la aplicación de Telegram con sus manejadores y el motor ya cargado."""
    application = ApplicationBuilder().token(settings.token).concurrent_updates(True)
    application = application.build()
    application.bot_data.update(
        {
            ENGINE_KEY: engine,
            SETTINGS_KEY: settings,
            RELAY_KEY: RelayRegistry(),
            LOCK_KEY: asyncio.Lock(),
        }
    )
    new_messages = filters.UpdateType.MESSAGE  # ignora mensajes editados
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler(["ayuda", "help"], help_command))
    application.add_handler(CommandHandler("miid", chat_id_command))
    application.add_handler(
        MessageHandler(new_messages & filters.TEXT & ~filters.COMMAND, text_message)
    )
    application.add_handler(
        MessageHandler(
            new_messages & ~filters.TEXT & ~filters.COMMAND, non_text_message
        )
    )
    application.add_error_handler(on_error)
    return application


def _configure_logging() -> None:
    """Muestra los registros del bot y, de las demás librerías, solo avisos y errores.

    Las librerías de red (``httpx`` y similares) escriben en nivel INFO la URL de cada
    petición, y en Telegram esa URL contiene el token del bot. Por eso todo queda en
    WARNING y solo el registro del bot baja a INFO.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.WARNING,
    )
    logger.setLevel(logging.INFO)


def main() -> int:
    """Lee la configuración, prepara el motor y deja el bot escuchando.

    Returns:
        Código de salida: 0 si terminó normalmente (Ctrl+C), 2 si hubo un error al arrancar.
    """
    _configure_logging()
    try:
        load_env_file()
        settings = read_settings()
    except SettingsError as exc:
        print(f"ERROR de configuración: {exc}")
        return 2

    print(f"Cargando el motor {settings.engine_spec}...")
    try:
        built = build_engine(
            settings.engine_spec, settings.train_input, settings.train_source
        )
        if built.n_train_rows:
            print(f"  Entrenado con {built.n_train_rows} filas de train.")
        if "M2" in built.parts or "M3" in built.parts:
            print("  Cargando modelos (la primera vez puede tardar hasta un minuto)...")
        warm_up(built)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR al preparar el motor: {exc}")
        return 2
    except RuntimeError as exc:  # p. ej. Ollama apagado (LlmUnavailableError)
        print(
            f"ERROR al preparar el motor: {exc}\n"
            "Abre Ollama y vuelve a intentar, o cambia a BOT_ENGINE=M0 en el archivo .env."
        )
        return 2

    # Al arrancar se usa el límite largo de config.py (Ollama puede tardar en cargar el
    # modelo); atendiendo clientes, uno corto: si se pasa, la consulta escala.
    if built.llm is not None:
        built.llm.timeout_s = settings.llm_timeout_s
        print(
            f"  Espera máxima a Ollama por respuesta: {settings.llm_timeout_s:.0f} s."
        )

    application = build_application(settings, built.engine)
    avisos = (
        "activados"
        if settings.admin_chat_id is not None
        else "desactivados (ADMIN_CHAT_ID vacío)"
    )
    print(
        f"Bot listo con {built.engine.name}. Avisos a administración: {avisos}.\n"
        "Escríbele al bot en Telegram. Ctrl+C para detenerlo."
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    print("Bot detenido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
