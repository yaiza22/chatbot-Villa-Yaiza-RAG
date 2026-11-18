"""Combinación de dos motores: uno decide si escalar y otro redacta la respuesta.

Por qué existe: en la evaluación, el motor que mejor decide cuándo escalar (M0) no es el que
mejor redacta, y el que redacta (M3) responde demasiadas preguntas que debían escalar. Al
combinarlos, el clasificador filtra primero y el redactor solo trabaja con lo que pasa el
filtro. Es la misma regla con la que se simularon las combinaciones en la evaluación: la
consulta escala si escala cualquiera de los dos motores.

La clase cumple la interfaz ``Engine`` sin cambiarla, así que el bot la usa igual que a un
motor solo.
"""

from __future__ import annotations

import time

import pandas as pd

from src.engines.base import Engine, EngineResponse


class CombinedEngine:
    """Motor compuesto ``clasificador + redactor``.

    Args:
        gate: Motor que decide si la consulta escala (M0 o M1).
        responder: Motor que redacta la respuesta cuando ``gate`` no escala (M2 o M3).

    Attributes:
        name: Nombre compuesto, p. ej. ``"M0+M3"``.
    """

    def __init__(self, gate: Engine, responder: Engine) -> None:
        self.gate = gate
        self.responder = responder
        self.name = f"{gate.name}+{responder.name}"

    def fit(self, train: pd.DataFrame) -> None:
        """Entrena los dos motores con el mismo ``train`` (los que no entrenan lo ignoran)."""
        self.gate.fit(train)
        self.responder.fit(train)

    def respond(self, message: str) -> EngineResponse:
        """Pasa el mensaje por el clasificador y, si no escala, por el redactor.

        Returns:
            Si el clasificador escala, su respuesta (sin llamar al redactor, así la mayoría de
            consultas no paga la latencia del LLM). Si no, la respuesta del redactor con la
            intención del clasificador; escala si el redactor escala o no devuelve texto.
            ``latency_ms`` es el tiempo total de la combinación.
        """
        start = time.perf_counter()
        gate_response = self.gate.respond(message)
        if gate_response.escalate:
            return EngineResponse(
                answer=None,
                intent=gate_response.intent,
                confidence=gate_response.confidence,
                escalate=True,
                retrieved=list(gate_response.retrieved),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        responder_response = self.responder.respond(message)
        answer = (responder_response.answer or "").strip()
        escalate = responder_response.escalate or not answer
        return EngineResponse(
            answer=None if escalate else answer,
            intent=gate_response.intent,
            confidence=responder_response.confidence,
            escalate=escalate,
            retrieved=list(responder_response.retrieved),
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


def _ensure_protocol_compliance(gate: Engine, responder: Engine) -> Engine:
    """Falla en el análisis de tipos si ``CombinedEngine`` deja de cumplir ``Engine``."""
    return CombinedEngine(gate, responder)
