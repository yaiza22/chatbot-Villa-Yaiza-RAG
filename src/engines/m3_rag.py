"""Motor conversacional M3: RAG (recuperar y generar) con un modelo de lenguaje local.

Idea general (para quien lo ve por primera vez):

1. **Recuperar**: se busca en la base de conocimiento (``data/kb``) los fragmentos más
   parecidos a la pregunta, con el mismo recuperador de M2 (embeddings multilingües).
2. **Generar**: se le entregan esos fragmentos a un modelo de lenguaje local (Ollama) junto con
   la pregunta, y un prompt que le exige responder **solo** con ese contexto. Si el contexto no
   contiene la respuesta, debe contestar exactamente ``ESCALAR``.
3. **Decidir**: si el modelo contesta ``ESCALAR`` (o no contesta), el motor escala a una
   persona; si no, devuelve el texto redactado por el modelo.

Diferencia con M2: M2 devuelve el fragmento tal cual; M3 lo redacta con el modelo. Eso lo
hace más natural, pero abre la puerta a la alucinación (datos que no están en la base), que
se mide aparte con revisión manual.

Qué se aprende de los datos: nada. M3 no entrena; ``fit()`` solo comprueba que no le lleguen
filas de test.

Decisiones de diseño que conviene recordar en el informe:

* La intención devuelta es la del fragmento más parecido (como en M2). Como los fragmentos de
  la base no tienen intenciones que escalan, la decisión de escalar precios, fechas y temas
  ajenos recae en el prompt, no en la intención.
* ``confidence`` es la similitud del mejor fragmento: el modelo de lenguaje no entrega una
  probabilidad.
* ``retrieved`` siempre trae los ``M2_TOP_K`` mejores ids (aunque el modelo reciba menos),
  para poder medir Hit@1, Hit@3 y Hit@5.
* Si Ollama no está disponible se lanza un error claro: no se finge una abstención. Si una
  respuesta puntual falla (tiempo agotado, respuesta ilegible), se escala y se cuenta en
  ``n_llm_failures``.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from src.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    M2_TOP_K,
    M3_CONTEXT_K,
    OLLAMA_HOST,
    SEED,
    SPLIT_TEST,
)
from src.engines.base import Engine, EngineResponse
from src.engines.m2_embeddings import EmbeddingEngine, KbFragment

ESCALATE_TOKEN = "ESCALAR"

SYSTEM_PROMPT = f"""Eres el asistente de mensajería de Alojamientos Villa Yaiza, una finca de \
alquiler para eventos en Ibagué (Colombia). Respondes por mensaje a posibles clientes, en \
español, con tono amable y breve (máximo 4 frases).

Reglas estrictas:
1. Usa SOLO la información de los FRAGMENTOS que se te entregan. No uses conocimiento propio \
ni supongas datos.
2. Nunca des precios, tarifas ni descuentos, y nunca confirmes si una fecha está disponible: \
eso lo atiende una persona.
3. Si los FRAGMENTOS no contienen la respuesta, o la pregunta trata de precios, fechas \
disponibles, reservas concretas, otros servicios de la finca que no aparecen en los \
FRAGMENTOS o temas que no tienen que ver con la finca, responde ÚNICAMENTE la palabra \
{ESCALATE_TOKEN}.
4. Si los FRAGMENTOS cubren solo una parte de la pregunta, responde esa parte y no agregues \
nada que no esté en ellos.
5. No menciones los FRAGMENTOS ni sus identificadores, y no escribas tu razonamiento."""

_ESCALATE_PATTERN = re.compile(rf"^\W*{ESCALATE_TOKEN}\b", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------
def build_user_prompt(message: str, fragments: Sequence[KbFragment]) -> str:
    """Arma el mensaje del usuario: fragmentos recuperados y pregunta del cliente."""
    context = "\n\n---\n\n".join(
        f"[{fragment.id}] {fragment.title}\n{fragment.text}" for fragment in fragments
    )
    return f"FRAGMENTOS:\n\n{context}\n\nPREGUNTA DEL CLIENTE:\n{message}"


def is_escalation(raw_answer: str) -> bool:
    """Indica si la salida del modelo significa "escalar".

    Escala si la salida está vacía o empieza con ``ESCALAR`` (ignorando comillas, asteriscos
    o puntuación inicial y mayúsculas).
    """
    text = _THINK_BLOCK.sub("", raw_answer or "").strip()
    return not text or _ESCALATE_PATTERN.match(text) is not None


# --------------------------------------------------------------------------------------
# Cliente del modelo de lenguaje
# --------------------------------------------------------------------------------------
class LlmUnavailableError(RuntimeError):
    """Ollama no responde (no está corriendo o la dirección es incorrecta)."""


class LlmFailedError(RuntimeError):
    """Una respuesta puntual falló (tiempo agotado, error del servidor, salida ilegible)."""


class LlmClient(Protocol):
    """Interfaz mínima de un modelo de lenguaje (permite usar uno falso en las pruebas)."""

    def chat(self, system: str, user: str) -> str:
        """Devuelve el texto generado para un mensaje de sistema y uno de usuario."""
        ...


class OllamaClient:
    """Cliente de la API local de Ollama (``/api/chat``), solo con la librería estándar.

    Args:
        model: Nombre exacto del modelo, como aparece en ``ollama list``.
        host: Dirección del servidor de Ollama.
        temperature: Aleatoriedad de la generación (0 = reproducible).
        seed: Semilla de la generación.
        num_ctx: Tamaño de la ventana de contexto que se le pide al modelo.
        max_tokens: Máximo de tokens generados.
        timeout_s: Segundos máximos de espera por respuesta.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        host: str = OLLAMA_HOST,
        temperature: float = LLM_TEMPERATURE,
        seed: int = SEED,
        num_ctx: int = LLM_NUM_CTX,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout_s: float = LLM_TIMEOUT_S,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def _payload(self, system: str, user: str) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,  # sin "modo pensar": responde directo y mucho más rápido
            "keep_alive": "30m",  # mantiene el modelo cargado entre preguntas
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }

    def chat(self, system: str, user: str) -> str:
        """Envía la consulta a Ollama y devuelve el texto de la respuesta.

        Raises:
            LlmUnavailableError: Si no se puede conectar con Ollama.
            LlmFailedError: Si la respuesta tarda demasiado, falla o no se puede leer.
        """
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(self._payload(system, user)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["message"]["content"]
        except urllib.error.HTTPError as exc:
            raise LlmFailedError(
                f"Ollama respondió con error HTTP {exc.code}."
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LlmFailedError("Ollama no respondió a tiempo.") from exc
            raise LlmUnavailableError(
                f"No se pudo conectar con Ollama en {self.host}. ¿Está corriendo "
                f"(`ollama serve`) y existe el modelo '{self.model}' (`ollama list`)?"
            ) from exc
        except TimeoutError as exc:
            raise LlmFailedError("Ollama no respondió a tiempo.") from exc
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            raise LlmFailedError(
                "La respuesta de Ollama no tiene el formato esperado."
            ) from exc
        return _THINK_BLOCK.sub("", str(content)).strip()


# --------------------------------------------------------------------------------------
# Motor M3
# --------------------------------------------------------------------------------------
class RagEngine:
    """Motor M3: recupera fragmentos con M2 y redacta la respuesta con un LLM local.

    Attributes:
        name: Identificador del motor (``"M3"``).
        context_k: Cuántos fragmentos se le entregan al modelo.
        n_llm_failures: Respuestas puntuales que fallaron y se escalaron (se reporta).
        last_prompt: ``(system, user)`` de la última consulta, para inspeccionarlo.
    """

    name: str = "M3"

    def __init__(
        self,
        retriever: EmbeddingEngine | None = None,
        llm: LlmClient | None = None,
        context_k: int = M3_CONTEXT_K,
    ) -> None:
        if context_k < 1:
            raise ValueError("context_k debe ser un entero positivo.")
        self.context_k = context_k
        self._retriever = retriever
        self._llm = llm
        self.n_llm_failures = 0
        self.last_prompt: tuple[str, str] | None = None

    def fit(self, train: pd.DataFrame) -> None:
        """M3 no entrena nada; solo verifica que no se le entreguen filas de test.

        Raises:
            ValueError: Si ``train`` trae filas con ``split == "test"``.
        """
        if (
            train is not None
            and "split" in train.columns
            and (train["split"].astype(str).str.strip() == SPLIT_TEST).any()
        ):
            raise ValueError("fit() recibió filas con split == 'test'.")

    def respond(self, message: str) -> EngineResponse:
        """Recupera el contexto, consulta al modelo y decide si responde o escala.

        Returns:
            ``EngineResponse`` con la intención del fragmento más parecido, su similitud como
            ``confidence``, ``retrieved`` con los mejores ids, y ``answer`` con el texto del
            modelo (``None`` si escala).

        Raises:
            LlmUnavailableError: Si Ollama no está disponible.
        """
        start = time.perf_counter()
        if self._retriever is None:
            self._retriever = EmbeddingEngine()
        if self._llm is None:
            self._llm = OllamaClient()

        ranked = self._retriever.retrieve(message, k=max(M2_TOP_K, self.context_k))
        fragments = [self._retriever.fragment(fragment_id) for fragment_id, _ in ranked]
        best_score = ranked[0][1]
        intent = fragments[0].intent

        system, user = (
            SYSTEM_PROMPT,
            build_user_prompt(message, fragments[: self.context_k]),
        )
        self.last_prompt = (system, user)
        try:
            raw_answer = self._llm.chat(system, user)
        except LlmFailedError as exc:
            self.n_llm_failures += 1
            warnings.warn(
                f"AVISO M3: falló una respuesta ({exc}); se escala.", stacklevel=2
            )
            raw_answer = ESCALATE_TOKEN

        escalate = is_escalation(raw_answer)
        return EngineResponse(
            answer=None if escalate else raw_answer.strip(),
            intent=intent,
            confidence=float(np.clip(best_score, 0.0, 1.0)),
            escalate=escalate,
            retrieved=[fragment_id for fragment_id, _ in ranked[:M2_TOP_K]],
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


def _ensure_protocol_compliance() -> Engine:
    """Falla en tiempo de análisis si ``RagEngine`` deja de cumplir ``Engine``."""
    return RagEngine()
