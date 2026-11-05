"""Pruebas unitarias para el motor M1 (TF-IDF + Regresión Logística)."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import ESCALATING_INTENTS, INTENTS, SPLIT_TEST, SPLIT_TRAIN
from src.engines.base import Engine, EngineResponse
from src.engines.m1_tfidf_rl import TfidfRlEngine


@pytest.fixture()
def synthetic_train_df() -> pd.DataFrame:
    """Construye un conjunto sintético inventado con al menos 5 ejemplos por intención."""
    samples_by_intent: dict[str, list[str]] = {
        "precio": [
            "cuánto vale el alquiler de la finca",
            "qué precio tiene la tarifa por noche",
            "me regala una cotización y el valor del alquiler",
            "tienen descuento en el precio por tres noches",
            "cuánto cuesta alquilar el lugar para un evento",
        ],
        "capacidad": [
            "cuántas personas caben en las habitaciones",
            "cuál es la capacidad máxima de camas para dormir",
            "cuántos invitados caben cómodamente en el salón",
            "cuántas habitaciones y camas disponibles tienen",
            "alcanza el espacio de alojamiento para cincuenta personas",
        ],
        "reglas_servicios": [
            "se permiten mascotas y perros en la zona verde",
            "a qué hora es el horario de check in y check out",
            "el alquiler incluye piscina wifi y cocina equipada",
            "hay que llevar sábanas toallas y utensilios de cocina",
            "hasta qué hora dejan poner música en la piscina",
        ],
        "ubicacion": [
            "en qué vereda de ibagué queda ubicada la finca",
            "me comparte la dirección exacta y el link de google maps",
            "cómo llego en carro desde el centro de ibagué",
            "se puede entrar en automóvil normal o toca camioneta",
            "pasa transporte público cerca de la ubicación",
        ],
        "disponibilidad": [
            "tienen fecha disponible para el 24 de diciembre",
            "está libre el calendario para reservar en enero",
            "hay cupo disponible este fin de semana",
            "quisiera reservar fecha para el puente de noviembre",
            "me confirma disponibilidad del 10 al 12 de diciembre",
        ],
        "otro_negocio": [
            "ya envié el comprobante de la consignación bancaria",
            "necesito factura electrónica a nombre de mi empresa",
            "me pueden enviar el contrato firmado de la reserva",
            "olvidé una maleta en la casa y quiero hablar con el administrador",
            "si cancelo el evento me devuelven el anticipo abonado",
        ],
        "fuera_de_alcance": [
            "cómo preparo una receta de arroz con pollo casero",
            "cuánto está el cambio del dólar hoy en el banco",
            "me ayudas a resolver una tarea de matemáticas",
            "venden repuestos mecánicos para motocicleta",
            "quién ganó el partido de fútbol de anoche",
        ],
    }

    rows: list[dict[str, object]] = []
    counter = 1
    for intent, phrases in samples_by_intent.items():
        for text in phrases:
            rows.append(
                {
                    "question_id": f"SYN_{counter:03d}",
                    "question_text": text,
                    "intent": intent,
                    "should_escalate": intent in ESCALATING_INTENTS,
                    "reference_answer": (
                        "ESCALAR" if intent in ESCALATING_INTENTS else "[POR CONFIRMAR]"
                    ),
                    "kb_source": "",
                    "split": SPLIT_TRAIN,
                    "source": "synthetic",
                }
            )
            counter += 1
    return pd.DataFrame(rows)


def test_m1_smoke_protocol_and_persistence(
    synthetic_train_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Smoke test que verifica contrato Engine, tabla de umbral y guardado joblib."""
    engine = TfidfRlEngine()
    assert isinstance(engine, Engine)
    assert engine.name == "M1"

    engine.fit(synthetic_train_df)
    assert engine.cv_folds_ == 5
    assert engine.best_c_ in (1.0, 10.0, 100.0)
    assert set(engine.c_losses_.keys()) == {1.0, 10.0, 100.0}
    assert engine.candidate_thresholds[0] == pytest.approx(0.15)
    assert 0.15 <= engine.threshold <= 0.90

    table = engine.get_threshold_tradeoff_table()
    assert list(table.columns) == [
        "threshold",
        "correct_abstention",
        "unnecessary_abstention",
        "youden_j",
    ]
    assert len(table) == len(engine.candidate_thresholds)

    response = engine.respond("cuántas habitaciones y camas tienen para dormir")
    assert isinstance(response, EngineResponse)
    assert response.intent in INTENTS
    assert 0.0 <= response.confidence <= 1.0
    assert response.latency_ms is not None
    assert response.latency_ms >= 0.0

    model_file = tmp_path / "m1_test.joblib"
    saved_path = engine.save(model_file)
    assert saved_path.exists()

    loaded = TfidfRlEngine.load(saved_path)
    loaded_resp = loaded.respond("cuántas habitaciones y camas tienen para dormir")
    assert loaded_resp.intent == response.intent
    assert loaded_resp.confidence == pytest.approx(response.confidence)
    assert loaded.best_c_ == pytest.approx(engine.best_c_)
    assert loaded.threshold == pytest.approx(engine.threshold)


def test_m1_threshold_triggers_escalation_when_confidence_is_low(
    synthetic_train_df: pd.DataFrame,
) -> None:
    """Comprueba que por debajo del umbral se escale y por encima se responda."""
    engine = TfidfRlEngine()
    engine.fit(synthetic_train_df)

    # Con un umbral bajo, una consulta clara de capacidad debe responderse con plantilla.
    engine.threshold = 0.10
    resp_high = engine.respond("cuántas habitaciones y camas disponibles tienen")
    assert resp_high.intent == "capacidad"
    assert resp_high.escalate is False
    assert resp_high.answer is not None
    assert "[POR CONFIRMAR]" in resp_high.answer

    # Si el umbral supera la probabilidad máxima predicha, debe escalar con answer=None.
    engine.threshold = min(0.99, resp_high.confidence + 0.05)
    resp_low = engine.respond("cuántas habitaciones y camas disponibles tienen")
    assert resp_low.intent == "capacidad"
    assert resp_low.escalate is True
    assert resp_low.answer is None


def test_m1_escalating_intents_always_escalate_regardless_of_confidence(
    synthetic_train_df: pd.DataFrame,
) -> None:
    """Un mensaje cuya intención está en ESCALATING_INTENTS siempre devuelve escalate=True."""
    engine = TfidfRlEngine()
    engine.fit(synthetic_train_df)
    # Forzamos umbral mínimo para asegurar que la confianza supere ampliamente el umbral.
    engine.threshold = 0.01

    queries = [
        ("cuánto vale el alquiler de la finca y qué precio tiene", "precio"),
        ("tienen fecha disponible para reservar el 24 de diciembre", "disponibilidad"),
        (
            "necesito factura electrónica y el contrato de la consignación",
            "otro_negocio",
        ),
        (
            "cómo preparo una receta de arroz con pollo y cuánto está el dólar",
            "fuera_de_alcance",
        ),
    ]
    for query, expected_intent in queries:
        response = engine.respond(query)
        assert response.intent == expected_intent
        assert response.intent in ESCALATING_INTENTS
        assert response.confidence > engine.threshold
        assert response.escalate is True
        assert response.answer is None


def test_m1_fit_rejects_test_split_rows(synthetic_train_df: pd.DataFrame) -> None:
    """Verifica que fit() lance ValueError si recibe cualquier fila con split == 'test'."""
    contaminated = synthetic_train_df.copy()
    contaminated.loc[0, "split"] = SPLIT_TEST

    engine = TfidfRlEngine()
    with pytest.raises(ValueError, match="split == 'test'"):
        engine.fit(contaminated)


def test_m1_reduces_folds_with_warning_and_fails_clearly_under_2_examples(
    synthetic_train_df: pd.DataFrame,
) -> None:
    """Reduce los pliegues con aviso si una clase tiene < 5 ejemplos y falla si tiene < 2."""
    # Dejamos solo 3 ejemplos por clase (< 5 pliegues).
    small_df = (
        synthetic_train_df.groupby("intent", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    engine = TfidfRlEngine()
    with pytest.warns(UserWarning, match="se reduce el número de pliegues"):
        engine.fit(small_df)
    assert engine.cv_folds_ == 3

    # Si una clase tiene solo 1 ejemplo (< 2), debe lanzar ValueError explicativo.
    tiny_df = (
        synthetic_train_df.groupby("intent", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="menos de 2 ejemplos"):
        engine.fit(tiny_df)
