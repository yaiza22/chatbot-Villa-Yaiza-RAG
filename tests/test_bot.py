"""Pruebas de la lógica del bot sin Telegram, sin token y sin modelos.

Se prueban la combinación de motores, el selector ``BOT_ENGINE``, la lectura del ``.env`` y
las reglas de qué se contesta. ``telegram_bot.py`` solo conecta estas piezas con Telegram, así
que estas pruebas no necesitan ``python-telegram-bot``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.bot.combined_engine import CombinedEngine
from src.bot.conversation import (
    ESCALATION_WITH_ADMIN_TEXT,
    ESCALATION_WITHOUT_ADMIN_TEXT,
    OUT_OF_SCOPE_TEXT,
    SMALL_TALK_ACK,
    SMALL_TALK_EMPTY,
    SMALL_TALK_FAREWELL,
    SMALL_TALK_GREETING,
    SMALL_TALK_REPLIES,
    SMALL_TALK_THANKS,
    RelayRegistry,
    admin_notification_text,
    build_reply,
    classify_small_talk,
    safe_respond,
    split_message,
)
from src.bot.engine_factory import (
    build_engine,
    needs_training,
    parse_engine_spec,
    warm_up,
)
from src.bot.settings import (
    DEFAULT_ENGINE_SPEC,
    SettingsError,
    load_env_file,
    read_settings,
)
from src.engines.base import Engine, EngineResponse

VALID_TOKEN = "123456789:" + "A" * 35


class FakeEngine:
    """Motor falso: devuelve una respuesta fija y cuenta cuántas veces lo llaman."""

    def __init__(
        self,
        name: str,
        escalate: bool = False,
        answer: str | None = "respuesta",
        intent: str | None = "capacidad",
        fail: bool = False,
    ) -> None:
        self.name = name
        self.escalate = escalate
        self.answer = answer
        self.intent = intent
        self.fail = fail
        self.calls = 0
        self.fit_rows: int | None = None

    def fit(self, train: pd.DataFrame) -> None:
        self.fit_rows = len(train)

    def respond(self, message: str) -> EngineResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("motor caído")
        return EngineResponse(
            answer=None if self.escalate else self.answer,
            intent=self.intent,
            confidence=0.9,
            escalate=self.escalate,
            retrieved=["frag_1"] if self.name in {"M2", "M3"} else [],
            latency_ms=1.0,
        )


def _registry(engines: dict[str, FakeEngine]) -> dict[str, object]:
    return {name: (lambda engine=engine: engine) for name, engine in engines.items()}


def _write_labeled(path: Path) -> Path:
    rows = [
        ("R1", "cuantas personas caben", "capacidad", False, "train", "real"),
        ("R2", "cuanto vale", "precio", True, "train", "real"),
        ("R3", "donde queda", "ubicacion", False, "test", "real"),
        ("S1", "hay wifi", "reglas_servicios", False, "train", "synthetic"),
    ]
    pd.DataFrame(
        [
            {
                "question_id": qid,
                "question_text": text,
                "intent": intent,
                "should_escalate": escalate,
                "reference_answer": "",
                "kb_source": "",
                "split": split,
                "source": source,
            }
            for qid, text, intent, escalate, split, source in rows
        ]
    ).to_csv(path, index=False, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# CombinedEngine
# --------------------------------------------------------------------------------------
def test_combined_cumple_protocolo_y_nombre() -> None:
    engine = CombinedEngine(FakeEngine("M0"), FakeEngine("M3"))
    assert isinstance(engine, Engine)
    assert engine.name == "M0+M3"


def test_combined_si_el_clasificador_escala_no_llama_al_redactor() -> None:
    gate = FakeEngine("M0", escalate=True, intent="precio")
    responder = FakeEngine("M3")
    response = CombinedEngine(gate, responder).respond("cuanto vale")
    assert response.escalate is True
    assert response.answer is None
    assert response.intent == "precio"
    assert responder.calls == 0


def test_combined_responde_con_texto_del_redactor_e_intencion_del_clasificador() -> (
    None
):
    gate = FakeEngine("M0", intent="reglas_servicios", answer="plantilla")
    responder = FakeEngine("M3", intent="capacidad", answer="  texto redactado ")
    response = CombinedEngine(gate, responder).respond("se puede llevar perro")
    assert response.escalate is False
    assert response.answer == "texto redactado"
    assert response.intent == "reglas_servicios"
    assert response.retrieved == ["frag_1"]
    assert response.latency_ms is not None


@pytest.mark.parametrize(
    "escalate, answer", [(True, None), (False, "   "), (False, None)]
)
def test_combined_escala_si_el_redactor_escala_o_no_da_texto(
    escalate: bool, answer: str | None
) -> None:
    responder = FakeEngine("M3", escalate=escalate, answer=answer)
    response = CombinedEngine(FakeEngine("M0"), responder).respond("hola")
    assert response.escalate is True
    assert response.answer is None


def test_combined_fit_entrena_ambos() -> None:
    gate, responder = FakeEngine("M1"), FakeEngine("M2")
    CombinedEngine(gate, responder).fit(pd.DataFrame({"a": [1, 2]}))
    assert gate.fit_rows == 2
    assert responder.fit_rows == 2


# --------------------------------------------------------------------------------------
# Selector de motor
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec, expected",
    [
        ("M0", ("M0",)),
        ("m3", ("M3",)),
        ("M0+M3", ("M0", "M3")),
        (" m1 + m2 ", ("M1", "M2")),
    ],
)
def test_parse_engine_spec_validos(spec: str, expected: tuple[str, ...]) -> None:
    assert parse_engine_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "M4", "M3+M0", "M0+M1", "M2+M3", "M0+M3+M2"])
def test_parse_engine_spec_invalidos(spec: str) -> None:
    with pytest.raises(ValueError, match="BOT_ENGINE"):
        parse_engine_spec(spec)


def test_needs_training_solo_con_m1_o_m2() -> None:
    assert not needs_training(("M0",))
    assert not needs_training(("M0", "M3"))
    assert needs_training(("M0", "M2"))
    assert needs_training(("M1", "M3"))


def test_build_engine_m0_m3_no_lee_datos(tmp_path: Path) -> None:
    engines = {"M0": FakeEngine("M0"), "M3": FakeEngine("M3")}
    built = build_engine(
        "M0+M3", tmp_path / "no_existe.csv", registry=_registry(engines)
    )
    assert isinstance(built.engine, CombinedEngine)
    assert built.parts == ("M0", "M3")
    assert built.n_train_rows == 0
    assert engines["M0"].fit_rows is None and engines["M3"].fit_rows is None


def test_build_engine_entrena_m1_solo_con_train_real(tmp_path: Path) -> None:
    labeled = _write_labeled(tmp_path / "labeled.csv")
    engines = {"M1": FakeEngine("M1"), "M3": FakeEngine("M3")}
    built = build_engine("M1+M3", labeled, "real", registry=_registry(engines))
    # Solo R1 y R2: R3 es test y S1 es sintética.
    assert built.n_train_rows == 2
    assert engines["M1"].fit_rows == 2
    assert engines["M3"].fit_rows is None


def test_build_engine_sin_datos_para_entrenar_falla_con_mensaje(tmp_path: Path) -> None:
    engines = {"M0": FakeEngine("M0"), "M2": FakeEngine("M2")}
    with pytest.raises(FileNotFoundError, match="BOT_ENGINE=M0"):
        build_engine("M0+M2", tmp_path / "no_existe.csv", registry=_registry(engines))


def test_warm_up_solo_calienta_m2_y_m3() -> None:
    engines = {"M0": FakeEngine("M0"), "M3": FakeEngine("M3")}
    built = build_engine("M0+M3", "x.csv", registry=_registry(engines))
    assert warm_up(built) == ["M3"]
    assert engines["M0"].calls == 0
    assert engines["M3"].calls == 1

    solo_m0 = build_engine("M0", "x.csv", registry=_registry({"M0": FakeEngine("M0")}))
    assert warm_up(solo_m0) == []


# --------------------------------------------------------------------------------------
# Configuración (.env)
# --------------------------------------------------------------------------------------
def test_load_env_file_lee_formato_y_no_sobrescribe(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "﻿# comentario\n\n"
        f"TELEGRAM_BOT_TOKEN={VALID_TOKEN}\n"
        'export BOT_ENGINE="M0"\n'
        "ADMIN_CHAT_ID=\n",
        encoding="utf-8",
    )
    environ = {"BOT_ENGINE": "M0+M3"}
    loaded = load_env_file(env_file, environ)
    assert loaded == ["TELEGRAM_BOT_TOKEN", "ADMIN_CHAT_ID"]
    assert environ["TELEGRAM_BOT_TOKEN"] == VALID_TOKEN
    assert environ["BOT_ENGINE"] == "M0+M3"  # ya existía: no se pisa
    assert environ["ADMIN_CHAT_ID"] == ""


def test_load_env_file_inexistente_no_hace_nada(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    assert load_env_file(tmp_path / ".env", environ) == []
    assert environ == {}


def test_load_env_file_linea_invalida(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ESTO NO ES VALIDO\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="línea 1"):
        load_env_file(env_file, {})


def test_read_settings_valores_por_defecto() -> None:
    settings = read_settings({"TELEGRAM_BOT_TOKEN": VALID_TOKEN})
    assert settings.engine_spec == DEFAULT_ENGINE_SPEC == "M0+M3"
    assert settings.admin_chat_id is None
    assert settings.train_source == "real"
    assert settings.train_input.name == "labeled.csv"
    assert VALID_TOKEN not in repr(settings)


def test_read_settings_normaliza_motor_y_admin() -> None:
    settings = read_settings(
        {
            "TELEGRAM_BOT_TOKEN": VALID_TOKEN,
            "BOT_ENGINE": " m0 + m3 ",
            "ADMIN_CHAT_ID": "-1001234",
        }
    )
    assert settings.engine_spec == "M0+M3"
    assert settings.admin_chat_id == -1001234


@pytest.mark.parametrize(
    "environ, match",
    [
        ({}, "Falta TELEGRAM_BOT_TOKEN"),
        ({"TELEGRAM_BOT_TOKEN": "pega_aqui_el_token"}, "formato"),
        ({"TELEGRAM_BOT_TOKEN": VALID_TOKEN, "ADMIN_CHAT_ID": "abc"}, "ADMIN_CHAT_ID"),
        ({"TELEGRAM_BOT_TOKEN": VALID_TOKEN, "BOT_TRAIN_SOURCE": "x"}, "BOT_TRAIN"),
    ],
)
def test_read_settings_errores(environ: dict[str, str], match: str) -> None:
    with pytest.raises(SettingsError, match=match):
        read_settings(environ)


# --------------------------------------------------------------------------------------
# Qué se contesta
# --------------------------------------------------------------------------------------
def _response(
    escalate: bool, answer: str | None = None, intent: str | None = None
) -> EngineResponse:
    return EngineResponse(
        answer=answer, intent=intent, confidence=0.5, escalate=escalate
    )


def test_build_reply_responde_con_texto_del_motor() -> None:
    reply = build_reply(_response(False, " La finca es para 65 personas. "), True)
    assert reply.text == "La finca es para 65 personas."
    assert reply.notify_admin is False


def test_build_reply_escala_con_y_sin_admin() -> None:
    con_admin = build_reply(_response(True, intent="precio"), admin_enabled=True)
    sin_admin = build_reply(_response(True, intent="precio"), admin_enabled=False)
    assert con_admin.text == ESCALATION_WITH_ADMIN_TEXT and con_admin.notify_admin
    assert sin_admin.text == ESCALATION_WITHOUT_ADMIN_TEXT
    assert not sin_admin.notify_admin


def test_build_reply_fuera_de_alcance_no_avisa() -> None:
    reply = build_reply(_response(True, intent="fuera_de_alcance"), True)
    assert reply.text == OUT_OF_SCOPE_TEXT
    assert reply.notify_admin is False


def test_build_reply_sin_texto_cuenta_como_escalar() -> None:
    reply = build_reply(_response(False, answer="  "), admin_enabled=True)
    assert reply.notify_admin is True


def test_safe_respond_escala_si_el_motor_falla() -> None:
    response, error = safe_respond(FakeEngine("M3", fail=True), "hola")
    assert response.escalate is True and response.answer is None
    assert error == "RuntimeError"

    ok, no_error = safe_respond(FakeEngine("M0"), "hola")
    assert ok.answer == "respuesta" and no_error is None


def test_admin_notification_text_incluye_mensaje_e_intencion() -> None:
    text = admin_notification_text(
        "cuanto vale", _response(True, intent="precio"), "M0+M3"
    )
    assert "cuanto vale" in text
    assert "precio" in text
    assert "M0+M3" in text


def test_split_message() -> None:
    assert split_message("corto") == ["corto"]
    parts = split_message("uno dos\ntres cuatro", limit=9)
    assert parts == ["uno dos", "tres", "cuatro"]
    assert all(len(part) <= 9 for part in parts)
    assert split_message("a" * 10, limit=4) == ["aaaa", "aaaa", "aa"]


def test_relay_registry_recuerda_y_olvida_los_mas_viejos() -> None:
    relay = RelayRegistry(max_items=2)
    relay.remember(1, 100)
    relay.remember(2, 200)
    relay.remember(3, 300)
    assert len(relay) == 2
    assert relay.lookup(1) is None
    assert relay.lookup(2) == 200
    assert relay.lookup(3) == 300


# --------------------------------------------------------------------------------------
# Saludos, agradecimientos y confirmaciones
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message, expected",
    [
        ("Hola", SMALL_TALK_GREETING),
        ("Holaaaa!!", SMALL_TALK_GREETING),
        ("Buenas noches", SMALL_TALK_GREETING),
        ("Buen día señora", SMALL_TALK_GREETING),
        ("Hola, ¿qué tal?", SMALL_TALK_GREETING),
        ("Muchas gracias", SMALL_TALK_THANKS),
        ("ok gracias por la información", SMALL_TALK_THANKS),
        ("Dios le bendiga", SMALL_TALK_THANKS),
        ("Chao, hasta luego", SMALL_TALK_FAREWELL),
        ("Nos vemos", SMALL_TALK_FAREWELL),
        ("Ok", SMALL_TALK_ACK),
        ("Listo, perfecto", SMALL_TALK_ACK),
        ("👍", SMALL_TALK_EMPTY),
        ("???", SMALL_TALK_EMPTY),
        ("¿Qué?", SMALL_TALK_EMPTY),
    ],
)
def test_classify_small_talk_reconoce_charla(message: str, expected: str) -> None:
    assert classify_small_talk(message) == expected
    assert SMALL_TALK_REPLIES[expected]


@pytest.mark.parametrize(
    "message",
    [
        "Información",
        "Información por favor",
        "Hola, para cuántas personas es la finca?",
        "Buenas, tienen wifi?",
        "Qué tal la piscina",
        "Gracias, y cuánto vale?",
        "Hola buenas noches quisiera saber si tienen disponible el 24",
        "Naguaraaaa",
        "Cuánto vale para el 24 de diciembre?",
        "hola hola hola hola hola hola hola hola hola",
    ],
)
def test_classify_small_talk_deja_pasar_las_consultas(message: str) -> None:
    assert classify_small_talk(message) is None


# --------------------------------------------------------------------------------------
# Tiempo de espera a Ollama
# --------------------------------------------------------------------------------------
def test_read_settings_timeout_por_defecto_y_personalizado() -> None:
    assert read_settings({"TELEGRAM_BOT_TOKEN": VALID_TOKEN}).llm_timeout_s == 60.0
    custom = read_settings(
        {"TELEGRAM_BOT_TOKEN": VALID_TOKEN, "BOT_LLM_TIMEOUT_S": "45"}
    )
    assert custom.llm_timeout_s == 45.0


@pytest.mark.parametrize("value", ["0", "-5", "abc", "nan"])
def test_read_settings_timeout_invalido(value: str) -> None:
    with pytest.raises(SettingsError, match="BOT_LLM_TIMEOUT_S"):
        read_settings({"TELEGRAM_BOT_TOKEN": VALID_TOKEN, "BOT_LLM_TIMEOUT_S": value})


def test_build_engine_m3_por_defecto_expone_su_cliente_de_ollama() -> None:
    from src.engines.m3_rag import OllamaClient, RagEngine

    built = build_engine("M0+M3", "x.csv")
    assert isinstance(built.engine, CombinedEngine)
    assert isinstance(built.engine.responder, RagEngine)
    assert isinstance(built.llm, OllamaClient)
    # Cambiar el tiempo de espera del cliente expuesto cambia el que usa M3.
    built.llm.timeout_s = 12.0
    assert built.llm.timeout_s == 12.0


def test_build_engine_sin_m3_no_crea_cliente() -> None:
    assert build_engine("M0", "x.csv").llm is None
