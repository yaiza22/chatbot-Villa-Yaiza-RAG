"""Motor conversacional basado en reglas de palabras clave (M0).

Representa el método base de respuestas rápidas por palabras clave (equivalente
al esquema manual de WhatsApp Business). Normaliza el texto del cliente, busca
coincidencias por palabra completa a partir de un archivo de configuración JSON
y aplica reglas de negocio para priorizar el escalado a una persona cuando
corresponde.
"""

import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import ESCALATING_INTENTS, INTENTS, SEED
from src.engines.base import Engine, EngineResponse

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "m0_keywords.json"
FALLBACK_INTENT = "otro_negocio"

# Expresiones regulares para colapsar letras repetidas ("holaaa" -> "hola").
_TRIPLE_OR_MORE_RE = re.compile(r"([a-z])\1{2,}")
_DOUBLE_NON_DIGRAPH_RE = re.compile(r"([aeioudfghjkmpqstvwxyz])\1+")
_WHITESPACE_RE = re.compile(r"\s+")


def _collapse_repeated_char(match: re.Match[str]) -> str:
    """Conserva dígrafos válidos del español ('ll', 'rr', 'cc') o 'bb' al reducir repeticiones."""
    char = match.group(1)
    if char in {"l", "r", "c", "b"}:
        return char * 2
    return char


def normalize_text(text: str) -> str:
    """Normaliza un texto pasando a minúsculas, quitando tildes y letras repetidas.

    Args:
        text: Cadena de entrada en español informal.

    Returns:
        Texto en minúsculas, sin marcas diacríticas, sin alargamientos de letras
        y con espacios colapsados.
    """
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    without_accents = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    collapsed_triples = _TRIPLE_OR_MORE_RE.sub(_collapse_repeated_char, without_accents)
    collapsed_doubles = _DOUBLE_NON_DIGRAPH_RE.sub(r"\1", collapsed_triples)
    return _WHITESPACE_RE.sub(" ", collapsed_doubles).strip()


def _build_word_boundary_pattern(phrase: str) -> re.Pattern[str]:
    """Compila un patrón con límites de palabra completa para una palabra o frase."""
    normalized = normalize_text(phrase)
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\b")


class RuleEngine:
    """Motor M0 basado en coincidencia de palabras clave y reglas de negocio.

    Attributes:
        name: Identificador corto del motor (``"M0"``).
        seed: Semilla global de reproducibilidad del proyecto.
    """

    name: str = "M0"

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Inicializa el motor cargando el archivo de configuración de palabras clave.

        Args:
            config_path: Ruta al archivo JSON con las palabras clave, indicadores
                de fecha y respuestas fijas.

        Raises:
            ValueError: Si el archivo de configuración contiene intenciones fuera
                de la taxonomía oficial.
        """
        self.seed = SEED
        self.config_path = Path(config_path)
        with self.config_path.open(encoding="utf-8") as handle:
            raw_config = json.load(handle)

        raw_keywords: dict[str, list[str]] = raw_config.get("keywords", {})
        unknown_intents = set(raw_keywords) - set(INTENTS)
        if unknown_intents:
            raise ValueError(
                f"Intenciones desconocidas en la configuración: {sorted(unknown_intents)}"
            )

        self._intent_patterns: dict[str, list[re.Pattern[str]]] = {
            intent: [
                _build_word_boundary_pattern(kw) for kw in raw_keywords.get(intent, [])
            ]
            for intent in INTENTS
        }

        date_cfg: dict[str, list[str]] = raw_config.get("date_indicators", {})
        months = [normalize_text(m) for m in date_cfg.get("months", [])]
        festivals = [
            normalize_text(f) for f in date_cfg.get("festivals_and_periods", [])
        ]

        self._date_patterns: list[re.Pattern[str]] = []
        if months:
            months_alt = "|".join(
                re.escape(m) for m in sorted(months, key=len, reverse=True)
            )
            # Día + mes (p. ej. "24 de diciembre", "31 dic").
            self._date_patterns.append(
                re.compile(rf"\b(?:[1-9]|[12]\d|3[01])\s+(?:de\s+)?(?:{months_alt})\b")
            )
            # Rango de días (p. ej. "del 26 al 28 de dic").
            self._date_patterns.append(
                re.compile(
                    rf"\bdel\s+(?:[1-9]|[12]\d|3[01])\s+al\s+(?:[1-9]|[12]\d|3[01])"
                    rf"(?:\s+de\s+(?:{months_alt}))?\b"
                )
            )
            # Nombre de mes como palabra completa.
            self._date_patterns.extend(_build_word_boundary_pattern(m) for m in months)

        # Fin de semana con número de día (p. ej. "fin de semana del 8").
        self._date_patterns.append(
            re.compile(r"\bfin\s+de\s+semana\s+del\s+(?:[1-9]|[12]\d|3[01])\b")
        )
        # Festivos y puentes concretos.
        self._date_patterns.extend(_build_word_boundary_pattern(f) for f in festivals)

        self._canned_answers: dict[str, str] = dict(
            raw_config.get("canned_answers", {})
        )

    def fit(self, train: pd.DataFrame) -> None:
        """Cumple el contrato ``Engine``; M0 no requiere ajuste estadístico.

        Args:
            train: DataFrame de entrenamiento (no utilizado por M0).
        """
        _ = train

    def has_specific_date(self, message: str) -> bool:
        """Indica si el mensaje contiene una fecha, mes, festivo o puente concreto.

        Args:
            message: Texto original o normalizado del cliente.

        Returns:
            ``True`` si se detecta una referencia temporal concreta.
        """
        normalized = normalize_text(message)
        return self._find_specific_date_position(normalized) is not None

    def _find_specific_date_position(self, normalized_text: str) -> int | None:
        """Devuelve el índice de inicio de la primera fecha o periodo concreto."""
        positions: list[int] = []
        for pattern in self._date_patterns:
            match = pattern.search(normalized_text)
            if match is not None:
                positions.append(match.start())
        return min(positions) if positions else None

    def respond(self, message: str) -> EngineResponse:
        """Clasifica el mensaje por reglas y genera la respuesta o escalado.

        Reglas de resolución:
        1. Si el mensaje pregunta por precio e incluye una fecha o periodo
           concreto, la intención resultante es ``disponibilidad``.
        2. Si coinciden varias intenciones, prevalece la que obliga a escalar
           sobre la que no escala.
        3. Si hay empate entre intenciones del mismo tipo de escalado, gana la
           que aparece primero en el texto.
        4. Si ninguna regla coincide, se asigna ``otro_negocio`` con confianza
           ``0.0`` y se escala a una persona.

        Args:
            message: Consulta escrita por el cliente.

        Returns:
            ``EngineResponse`` con intención, confianza, bandera de escalado,
            respuesta fija (si aplica) y latencia en milisegundos.
        """
        start_time = time.perf_counter()
        normalized = normalize_text(message)

        matched_positions: dict[str, int] = {}
        for intent in INTENTS:
            earliest: int | None = None
            for pattern in self._intent_patterns.get(intent, []):
                match = pattern.search(normalized)
                if match is not None:
                    pos = match.start()
                    if earliest is None or pos < earliest:
                        earliest = pos
            if earliest is not None:
                matched_positions[intent] = earliest

        date_pos = self._find_specific_date_position(normalized)
        if date_pos is not None:
            if "disponibilidad" in matched_positions:
                matched_positions["disponibilidad"] = min(
                    matched_positions["disponibilidad"], date_pos
                )
            else:
                matched_positions["disponibilidad"] = date_pos

            # Si pregunta por precio para una fecha o periodo concreto, es disponibilidad.
            if "precio" in matched_positions:
                price_pos = matched_positions.pop("precio")
                matched_positions["disponibilidad"] = min(
                    matched_positions["disponibilidad"], price_pos
                )

        if not matched_positions:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return EngineResponse(
                answer=None,
                intent=FALLBACK_INTENT,
                confidence=0.0,
                escalate=True,
                latency_ms=latency_ms,
            )

        has_escalating = any(
            intent in ESCALATING_INTENTS for intent in matched_positions
        )
        if has_escalating:
            candidates = {
                intent: pos
                for intent, pos in matched_positions.items()
                if intent in ESCALATING_INTENTS
            }
        else:
            candidates = matched_positions

        best_intent = min(candidates, key=lambda intent: candidates[intent])
        should_escalate = best_intent in ESCALATING_INTENTS
        answer = None if should_escalate else self._canned_answers.get(best_intent)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return EngineResponse(
            answer=answer,
            intent=best_intent,
            confidence=1.0,
            escalate=should_escalate,
            latency_ms=latency_ms,
        )


def _ensure_protocol_compliance() -> Engine:
    """Verificación estática de que RuleEngine cumple el protocolo Engine."""
    return RuleEngine()
