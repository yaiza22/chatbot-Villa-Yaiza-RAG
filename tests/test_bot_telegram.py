"""Pruebas de los manejadores del bot de Telegram, sin conexión y sin token.

Usan objetos falsos en lugar de los de Telegram: un ``bot`` que guarda lo que enviaría y
mensajes con solo los campos que lee el bot. Necesitan ``python-telegram-bot`` instalado
(está en ``requirements.txt``); si no está, se omiten.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("telegram")

from src.bot import telegram_bot as tb
from src.bot.combined_engine import CombinedEngine
from src.bot.conversation import (
    ADMIN_REPLY_HEADER,
    ADMIN_REPLY_PREFIX,
    ADMIN_REPLY_SENT_TEXT,
    ESCALATION_WITH_ADMIN_TEXT,
    ESCALATION_WITHOUT_ADMIN_TEXT,
    NON_TEXT_TEXT,
    SMALL_TALK_GREETING,
    SMALL_TALK_REPLIES,
    UNKNOWN_NOTICE_TEXT,
    RelayRegistry,
)
from src.bot.settings import BotSettings
from src.engines.base import EngineResponse

ADMIN_GROUP = -100123
CLIENT = 555
BOT_ID = 42


class FakeEngine:
    """Responde capacidad y escala todo lo que mencione precio; cuenta las llamadas."""

    name = "M0+M3"

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def fit(self, train: object) -> None:
        """No entrena."""

    def respond(self, message: str) -> EngineResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("Ollama apagado")
        if "vale" in message.lower():
            return EngineResponse(None, "precio", 1.0, True, latency_ms=1.0)
        return EngineResponse("Caben 65 personas.", "capacidad", 0.9, False)


class FakeBot:
    """Guarda los mensajes que el bot enviaría."""

    id = BOT_ID

    def __init__(self, fail_send_to: int | None = None) -> None:
        self.sent: list[tuple[int, str]] = []
        self.copied: list[tuple[int, int, int]] = []
        self.fail_send_to = fail_send_to
        self._next_id = 1000

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        """No hace nada."""

    async def send_message(self, chat_id: int, text: str) -> SimpleNamespace:
        if chat_id == self.fail_send_to:
            raise tb.TelegramError("Forbidden: bot was blocked by the user")
        self._next_id += 1
        self.sent.append((chat_id, text))
        return SimpleNamespace(message_id=self._next_id)

    async def copy_message(
        self, chat_id: int, from_chat_id: int, message_id: int
    ) -> None:
        self.copied.append((chat_id, from_chat_id, message_id))


class FakeMessage:
    """Mensaje recibido con los campos que usa el bot."""

    def __init__(
        self,
        text: str | None,
        message_id: int = 1,
        reply_to: SimpleNamespace | None = None,
    ) -> None:
        self.text = text
        self.message_id = message_id
        self.reply_to_message = reply_to
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


def _context(
    engine: FakeEngine,
    admin_chat_id: int | None = ADMIN_GROUP,
    bot: FakeBot | None = None,
) -> SimpleNamespace:
    settings = BotSettings(
        token="x",
        engine_spec="M0+M3",
        admin_chat_id=admin_chat_id,
        train_input=None,  # type: ignore[arg-type]
        train_source="real",
    )
    bot_data = {
        tb.ENGINE_KEY: engine,
        tb.SETTINGS_KEY: settings,
        tb.RELAY_KEY: RelayRegistry(),
        tb.LOCK_KEY: asyncio.Lock(),
    }
    return SimpleNamespace(
        application=SimpleNamespace(bot_data=bot_data), bot=bot or FakeBot()
    )


def _update(message: FakeMessage, chat_id: int, private: bool) -> SimpleNamespace:
    chat_type = tb.ChatType.PRIVATE if private else tb.ChatType.SUPERGROUP
    return SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
    )


def _reply_to(message_id: int, author_id: int = BOT_ID) -> SimpleNamespace:
    return SimpleNamespace(
        message_id=message_id, from_user=SimpleNamespace(id=author_id)
    )


def _run(coroutine: object) -> None:
    asyncio.run(coroutine)  # type: ignore[arg-type]


def test_cliente_recibe_respuesta_del_motor_sin_aviso() -> None:
    context = _context(FakeEngine())
    message = FakeMessage("para cuántas personas es?")
    _run(tb.text_message(_update(message, CLIENT, private=True), context))
    assert message.replies == ["Caben 65 personas."]
    assert context.bot.sent == []


def test_consulta_que_escala_avisa_y_la_respuesta_del_equipo_llega_al_cliente() -> None:
    context = _context(FakeEngine())
    question = FakeMessage("cuánto vale el 24?")
    _run(tb.text_message(_update(question, CLIENT, private=True), context))
    assert question.replies == [ESCALATION_WITH_ADMIN_TEXT]
    assert context.bot.sent[-1][0] == ADMIN_GROUP
    assert "cuánto vale el 24?" in context.bot.sent[-1][1]
    notice_id = context.bot._next_id

    answer = FakeMessage("Está libre, te paso la cotización", 7, _reply_to(notice_id))
    _run(tb.text_message(_update(answer, ADMIN_GROUP, private=False), context))
    assert context.bot.sent[-1] == (
        CLIENT,
        ADMIN_REPLY_PREFIX + "Está libre, te paso la cotización",
    )
    assert answer.replies == [ADMIN_REPLY_SENT_TEXT]


def test_el_equipo_puede_responder_con_foto_o_ubicacion() -> None:
    context = _context(FakeEngine())
    question = FakeMessage("cuánto vale?")
    _run(tb.text_message(_update(question, CLIENT, private=True), context))
    notice_id = context.bot._next_id

    pin = FakeMessage(None, 9, _reply_to(notice_id))
    _run(tb.non_text_message(_update(pin, ADMIN_GROUP, private=False), context))
    assert context.bot.sent[-1] == (CLIENT, ADMIN_REPLY_HEADER)
    assert context.bot.copied == [(CLIENT, ADMIN_GROUP, 9)]
    assert pin.replies == [ADMIN_REPLY_SENT_TEXT]


def test_responder_a_un_mensaje_que_no_es_aviso_explica_que_pasa() -> None:
    context = _context(FakeEngine())
    answer = FakeMessage("hola", 3, _reply_to(999))
    _run(tb.text_message(_update(answer, ADMIN_GROUP, private=False), context))
    assert answer.replies == [UNKNOWN_NOTICE_TEXT]
    assert context.bot.sent == []


def test_en_grupos_el_bot_no_conversa() -> None:
    engine = FakeEngine()
    context = _context(engine)
    message = FakeMessage("cuánto vale?")
    _run(tb.text_message(_update(message, ADMIN_GROUP, private=False), context))
    assert message.replies == [] and engine.calls == 0


def test_saludo_no_llama_al_motor_ni_avisa() -> None:
    engine = FakeEngine()
    context = _context(engine)
    message = FakeMessage("Holaaa buenas noches")
    _run(tb.text_message(_update(message, CLIENT, private=True), context))
    assert message.replies == [SMALL_TALK_REPLIES[SMALL_TALK_GREETING]]
    assert engine.calls == 0
    assert context.bot.sent == []


def test_si_el_motor_falla_escala_en_vez_de_caerse() -> None:
    context = _context(FakeEngine(fail=True))
    message = FakeMessage("dónde queda la finca?")
    _run(tb.text_message(_update(message, CLIENT, private=True), context))
    assert message.replies == [ESCALATION_WITH_ADMIN_TEXT]
    assert context.bot.sent[-1][0] == ADMIN_GROUP


def test_si_no_se_puede_avisar_al_equipo_se_le_dice_al_cliente() -> None:
    context = _context(FakeEngine(), bot=FakeBot(fail_send_to=ADMIN_GROUP))
    message = FakeMessage("cuánto vale?")
    _run(tb.text_message(_update(message, CLIENT, private=True), context))
    assert message.replies == [ESCALATION_WITHOUT_ADMIN_TEXT]


def test_sin_chat_de_administracion_no_se_avisa() -> None:
    context = _context(FakeEngine(), admin_chat_id=None)
    message = FakeMessage("cuánto vale?")
    _run(tb.text_message(_update(message, CLIENT, private=True), context))
    assert message.replies == [ESCALATION_WITHOUT_ADMIN_TEXT]
    assert context.bot.sent == []


def test_cliente_que_envia_audio_recibe_aviso_de_solo_texto() -> None:
    context = _context(FakeEngine())
    audio = FakeMessage(None)
    _run(tb.non_text_message(_update(audio, CLIENT, private=True), context))
    assert audio.replies == [NON_TEXT_TEXT]


def test_build_application_registra_los_manejadores() -> None:
    settings = BotSettings(
        token="123456789:" + "A" * 35,
        engine_spec="M0+M3",
        admin_chat_id=None,
        train_input=None,  # type: ignore[arg-type]
        train_source="real",
    )
    application = tb.build_application(
        settings, CombinedEngine(FakeEngine(), FakeEngine())
    )
    n_handlers = sum(len(group) for group in application.handlers.values())
    assert n_handlers == 5
    assert isinstance(application.bot_data[tb.RELAY_KEY], RelayRegistry)
