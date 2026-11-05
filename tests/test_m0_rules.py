"""Pruebas unitarias para el contrato Engine y el motor de reglas M0."""

import pandas as pd
import pytest

from src.config import ESCALATING_INTENTS, INTENTS
from src.engines.base import Engine, EngineResponse
from src.engines.m0_rules import RuleEngine, normalize_text

# 15 mensajes sintéticos inventados que cubren todos los escenarios del motor M0.
SYNTHETIC_CASES: list[tuple[str, str, bool, float]] = [
    # 1. Precio sin fecha concreta (con letras repetidas y tildes).
    (
        "Holaaa buenas tardes, cuánto vale alquilar la finca?",
        "precio",
        True,
        1.0,
    ),
    # 2. Precio sin fecha concreta (consulta de tarifas generales).
    (
        "Me regala las tarifas y si hacen descuento por favor",
        "precio",
        True,
        1.0,
    ),
    # 3. Precio con fecha concreta día + mes -> disponibilidad.
    (
        "Cuánto cuesta el alquiler para el 24 de diciembre?",
        "disponibilidad",
        True,
        1.0,
    ),
    # 4. Precio con puente festivo -> disponibilidad.
    (
        "Qué precio tiene la finca para el puente de reyes?",
        "disponibilidad",
        True,
        1.0,
    ),
    # 5. Precio con nombre de mes -> disponibilidad.
    (
        "Me puede cotizar para un evento en noviembre?",
        "disponibilidad",
        True,
        1.0,
    ),
    # 6. Disponibilidad pura sin mencionar precio.
    (
        "Tienen disponible del 10 al 12 de enero?",
        "disponibilidad",
        True,
        1.0,
    ),
    # 7. Capacidad (respondible con plantilla [POR CONFIRMAR]).
    (
        "Cuántas habitaciones y camas tiene la finca para dormir?",
        "capacidad",
        False,
        1.0,
    ),
    # 8. Reglas y servicios (respondible con plantilla [POR CONFIRMAR]).
    (
        "Se permiten mascotas y tienen piscina y wifi?",
        "reglas_servicios",
        False,
        1.0,
    ),
    # 9. Ubicación (respondible con plantilla [POR CONFIRMAR]).
    (
        "En qué vereda queda y cómo llego desde Ibagué en carro?",
        "ubicacion",
        False,
        1.0,
    ),
    # 10. Otro negocio (escala).
    (
        "Ya hice la consignación, me envían el contrato y la factura?",
        "otro_negocio",
        True,
        1.0,
    ),
    # 11. Fuera de alcance (escala/abstiene).
    (
        "Tienen domicilio de pizza o venden repuestos para moto?",
        "fuera_de_alcance",
        True,
        1.0,
    ),
    # 12. Varias intenciones: no escalable (reglas_servicios) + escalable (precio) -> gana la que escala.
    (
        "Tienen piscina y cuánto vale el alquiler?",
        "precio",
        True,
        1.0,
    ),
    # 13. Varias intenciones: ambas escalan -> gana la que aparece primero en el texto.
    (
        "Necesito factura para mi empresa y saber si está libre el sábado",
        "otro_negocio",
        True,
        1.0,
    ),
    # 14. Varias intenciones: ninguna escala -> gana la que aparece primero en el texto.
    (
        "Cuántas personas caben y me envía la ubicación en google maps?",
        "capacidad",
        False,
        1.0,
    ),
    # 15. Mensaje sin coincidencias (con subcadena engañosa "estuvo" != "tv") -> otro_negocio, confidence=0.0.
    (
        "Hola buenas noches, estuvo muy bonito el saludo, muchas gracias",
        "otro_negocio",
        True,
        0.0,
    ),
]


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    """Instancia compartida del motor M0 para las pruebas."""
    return RuleEngine()


def test_engine_smoke_protocol(engine: RuleEngine) -> None:
    """Smoke test que verifica el cumplimiento del contrato Engine y EngineResponse."""
    assert isinstance(engine, Engine)
    assert engine.name == "M0"

    empty_df = pd.DataFrame(columns=["question_text", "intent"])
    engine.fit(empty_df)

    response = engine.respond("¿Dónde queda la finca?")
    assert isinstance(response, EngineResponse)
    assert response.intent in INTENTS
    assert response.confidence == 1.0
    assert response.escalate is False
    assert response.answer is not None
    assert "[POR CONFIRMAR]" in response.answer
    assert response.retrieved == []
    assert response.latency_ms is not None
    assert response.latency_ms >= 0.0


def test_synthetic_cases_count() -> None:
    """Garantiza que la batería principal tenga exactamente los 15 mensajes requeridos."""
    assert len(SYNTHETIC_CASES) == 15


@pytest.mark.parametrize(
    ("message", "expected_intent", "expected_escalate", "expected_confidence"),
    SYNTHETIC_CASES,
)
def test_m0_synthetic_messages(
    engine: RuleEngine,
    message: str,
    expected_intent: str,
    expected_escalate: bool,
    expected_confidence: float,
) -> None:
    """Evalúa los 15 mensajes sintéticos sobre el motor de reglas M0."""
    response = engine.respond(message)

    assert response.intent == expected_intent
    assert response.escalate is expected_escalate
    assert response.confidence == expected_confidence
    assert response.latency_ms is not None
    assert response.latency_ms >= 0.0

    if expected_escalate:
        assert expected_intent in ESCALATING_INTENTS
        assert response.answer is None
    else:
        assert expected_intent not in ESCALATING_INTENTS
        assert response.answer is not None
        assert "[POR CONFIRMAR]" in response.answer


def test_normalize_text_accents_and_repeated_letters() -> None:
    """Verifica normalización de minúsculas, tildes y letras repetidas."""
    assert normalize_text("¡¡Holaaa, cuántooo cuesta!!") == "¡¡hola, cuanto cuesta!!"
    assert normalize_text("Ubicación en Ibagué") == "ubicacion en ibague"
    # Debe conservar dígrafos válidos en español ('rr', 'll') y 'bb'.
    assert (
        normalize_text("perro carro llegar toallas bbq")
        == "perro carro llegar toallas bbq"
    )


def test_specific_date_detection(engine: RuleEngine) -> None:
    """Comprueba la detección de fechas concretas (día+mes, mes, festivos y puentes)."""
    assert engine.has_specific_date("para el 24 de diciembre") is True
    assert engine.has_specific_date("del 26 al 28 de dic") is True
    assert engine.has_specific_date("en octubre tienen cupo") is True
    assert engine.has_specific_date("para el puente de reyes") is True
    assert engine.has_specific_date("en semana santa") is True
    assert engine.has_specific_date("fin de semana del 8") is True
    assert engine.has_specific_date("cuánto cuesta alquilar la finca") is False
