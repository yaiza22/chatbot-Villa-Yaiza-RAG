"""Contrato común para todos los motores conversacionales del proyecto.

Define la estructura de respuesta estándar (``EngineResponse``) y el protocolo
(``Engine``) que deben implementar todos los motores (M0 a M3) para ser
evaluados bajo las mismas condiciones e integrados en el canal de mensajería.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class EngineResponse:
    """Respuesta estandarizada devuelta por cualquier motor conversacional.

    Attributes:
        answer: Texto de respuesta para el cliente (``None`` si escala o se abstiene).
        intent: Intención predicha dentro de la taxonomía definida, o ``None``.
        confidence: Nivel de confianza o similitud en el rango ``[0.0, 1.0]``.
        escalate: ``True`` si la consulta debe pasar a una persona administradora.
        retrieved: Lista de identificadores de fragmentos recuperados de la base.
        latency_ms: Tiempo de ejecución de la respuesta en milisegundos.
    """

    answer: str | None  # texto para el cliente (None si escala/abstiene)
    intent: str | None  # intención predicha
    confidence: float  # 0–1 (probabilidad o similitud)
    escalate: bool  # True si debe pasar a una persona
    retrieved: list[str] = field(default_factory=list)  # ids de fragmentos (M2/M3)
    latency_ms: float | None = None


@runtime_checkable
class Engine(Protocol):
    """Protocolo común que deben cumplir los motores M0, M1, M2 y M3."""

    name: str  # "M0" | "M1" | "M2" | "M3"

    def fit(self, train: pd.DataFrame) -> None:
        """Ajusta el motor con el conjunto de entrenamiento si aplica.

        Args:
            train: DataFrame con las preguntas etiquetadas de entrenamiento.
        """
        ...

    def respond(self, message: str) -> EngineResponse:
        """Procesa un mensaje del cliente y genera una respuesta estandarizada.

        Args:
            message: Texto de la consulta enviada por el cliente.

        Returns:
            Instancia de ``EngineResponse`` con la predicción y metadatos.
        """
        ...

