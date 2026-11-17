"""Pruebas de M3 (RAG) con un modelo de lenguaje y un codificador falsos.

No necesitan Ollama, GPU ni descargar modelos: ``FakeLlm`` devuelve respuestas fijas y
``FakeEncoder`` vectoriza con una bolsa de palabras determinista.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest

from src.config import SEED
from src.engines.base import Engine, EngineResponse
from src.engines.m2_embeddings import EmbeddingEngine, KbFragment
from src.engines.m3_rag import (
    ESCALATE_TOKEN,
    SYSTEM_PROMPT,
    LlmFailedError,
    LlmUnavailableError,
    OllamaClient,
    RagEngine,
    build_user_prompt,
    is_escalation,
)

DIM = 256


class FakeEncoder:
    """Bolsa de palabras con hashing: textos con palabras en común quedan cercanos."""

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIM)
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode("utf-8")) % DIM] += 1.0
        return vector

    def encode_queries(self, texts):
        return np.array([self._vector(t) for t in texts])

    def encode_documents(self, texts):
        return np.array([self._vector(t) for t in texts])


class FakeLlm:
    """Modelo de lenguaje falso: devuelve una respuesta fija o lanza un error."""

    def __init__(
        self, answer: str = "Se admiten mascotas.", error: Exception | None = None
    ):
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.answer


def _write_fragment(
    directory: Path, name: str, intent: str, title: str, body: str
) -> None:
    (directory / f"{name}.md").write_text(
        f"---\nid: {name}\nintent: {intent}\n---\n# {title}\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture()
def retriever(tmp_path: Path) -> EmbeddingEngine:
    _write_fragment(
        tmp_path,
        "reglas_mascotas",
        "reglas_servicios",
        "Mascotas",
        "Se admiten perros y gatos con correa dentro de la finca.",
    )
    _write_fragment(
        tmp_path,
        "capacidad_personas",
        "capacidad",
        "Capacidad",
        "La finca recibe hasta 65 personas para dormir.",
    )
    _write_fragment(
        tmp_path,
        "ubicacion_direccion",
        "ubicacion",
        "Ubicación",
        "La finca queda en la vereda Cural, a 20 minutos de Ibagué.",
    )
    return EmbeddingEngine(kb_dir=tmp_path, encoder=FakeEncoder())


# ------------------------------------------------------------------- utilidades del prompt
@pytest.mark.parametrize(
    "text",
    ["ESCALAR", "escalar", "**ESCALAR**", ' "ESCALAR" ', "ESCALAR.", "", "   ", "\n"],
)
def test_is_escalation_true(text: str) -> None:
    assert is_escalation(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Se admiten mascotas con correa.",
        "No necesitas escalar nada: la finca queda en Cural.",
        "Hay escalares de piedra en el jardín.",
    ],
)
def test_is_escalation_false(text: str) -> None:
    assert is_escalation(text) is False


def test_is_escalation_ignores_think_blocks() -> None:
    assert is_escalation("<think>no sé si está</think>ESCALAR") is True
    assert is_escalation("<think>ok</think>Sí se admiten mascotas.") is False


def test_build_user_prompt_contains_fragments_in_order_and_question() -> None:
    fragments = [
        KbFragment("a_uno", "capacidad", "Uno", "Texto uno."),
        KbFragment("b_dos", "ubicacion", "Dos", "Texto dos."),
    ]
    prompt = build_user_prompt("¿Cuántos caben?", fragments)
    assert prompt.index("[a_uno] Uno") < prompt.index("[b_dos] Dos")
    assert "Texto uno." in prompt and "Texto dos." in prompt
    assert prompt.endswith("PREGUNTA DEL CLIENTE:\n¿Cuántos caben?")


def test_system_prompt_forces_grounding_and_escalation_token() -> None:
    assert ESCALATE_TOKEN in SYSTEM_PROMPT
    assert "SOLO la información de los FRAGMENTOS" in SYSTEM_PROMPT
    assert "precios" in SYSTEM_PROMPT and "disponible" in SYSTEM_PROMPT


# -------------------------------------------------------------------------------- motor
def test_respond_returns_model_answer_and_retrieval_metadata(
    retriever: EmbeddingEngine,
) -> None:
    llm = FakeLlm("Sí, se admiten perros y gatos con correa.")
    engine = RagEngine(retriever=retriever, llm=llm)
    response = engine.respond("se admiten perros y gatos?")
    assert isinstance(response, EngineResponse)
    assert response.escalate is False
    assert response.answer == "Sí, se admiten perros y gatos con correa."
    assert response.intent == "reglas_servicios"
    assert response.retrieved[0] == "reglas_mascotas"
    assert 0.0 <= response.confidence <= 1.0
    assert response.latency_ms is not None and response.latency_ms >= 0.0
    assert len(llm.calls) == 1  # siempre se consulta al modelo


def test_respond_escalates_when_model_says_escalar(retriever: EmbeddingEngine) -> None:
    engine = RagEngine(retriever=retriever, llm=FakeLlm("ESCALAR"))
    response = engine.respond("cuanto cuesta el alquiler")
    assert response.escalate is True
    assert response.answer is None
    assert response.retrieved  # se conserva para medir Hit@k


def test_context_k_limits_fragments_in_prompt_but_not_retrieved(
    retriever: EmbeddingEngine,
) -> None:
    llm = FakeLlm()
    engine = RagEngine(retriever=retriever, llm=llm, context_k=1)
    response = engine.respond("se admiten perros y gatos?")
    user_prompt = llm.calls[0][1]
    assert user_prompt.count("[") == 1  # solo un fragmento en el contexto
    assert "[reglas_mascotas]" in user_prompt
    assert len(response.retrieved) == 3  # la base de prueba tiene 3 fragmentos
    assert engine.last_prompt == (SYSTEM_PROMPT, user_prompt)


def test_failed_answer_escalates_counts_and_warns(retriever: EmbeddingEngine) -> None:
    engine = RagEngine(
        retriever=retriever, llm=FakeLlm(error=LlmFailedError("tiempo agotado"))
    )
    with pytest.warns(UserWarning, match="AVISO M3"):
        response = engine.respond("se admiten perros?")
    assert response.escalate is True and response.answer is None
    assert engine.n_llm_failures == 1


def test_unavailable_llm_is_not_hidden(retriever: EmbeddingEngine) -> None:
    engine = RagEngine(
        retriever=retriever, llm=FakeLlm(error=LlmUnavailableError("sin Ollama"))
    )
    with pytest.raises(LlmUnavailableError):
        engine.respond("se admiten perros?")


def test_context_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="context_k"):
        RagEngine(context_k=0)


def test_fit_rejects_test_rows_and_accepts_train(retriever: EmbeddingEngine) -> None:
    engine = RagEngine(retriever=retriever, llm=FakeLlm())
    engine.fit(pd.DataFrame({"question_text": ["hola"], "split": ["train"]}))
    with pytest.raises(ValueError, match="test"):
        engine.fit(pd.DataFrame({"question_text": ["hola"], "split": ["test"]}))


def test_engine_follows_protocol(retriever: EmbeddingEngine) -> None:
    engine: Engine = RagEngine(retriever=retriever, llm=FakeLlm())
    assert engine.name == "M3"


def test_m3_is_registered_in_the_evaluation_harness() -> None:
    from src.eval.run_eval import ENGINE_REGISTRY

    assert "M3" in ENGINE_REGISTRY


# ---------------------------------------------------------------------- cliente de Ollama
class _FakeResponse(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result if isinstance(result, bytes) else b"")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def test_ollama_client_sends_expected_payload_and_parses_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = json.dumps(
        {"message": {"content": "<think>x</think> Hola, soy el asistente. "}}
    )
    captured = _patch_urlopen(monkeypatch, reply.encode("utf-8"))
    client = OllamaClient(
        model="modelo:prueba", host="http://localhost:1234/", timeout_s=7.0
    )
    assert client.chat("sistema", "usuario") == "Hola, soy el asistente."
    body = captured["body"]
    assert captured["url"] == "http://localhost:1234/api/chat"
    assert captured["timeout"] == 7.0
    assert body["model"] == "modelo:prueba"
    assert body["stream"] is False and body["think"] is False
    assert body["options"]["seed"] == SEED
    assert body["options"]["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_ollama_client_connection_refused_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_urlopen(monkeypatch, urllib.error.URLError(ConnectionRefusedError()))
    with pytest.raises(LlmUnavailableError, match="ollama list"):
        OllamaClient().chat("s", "u")


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(TimeoutError()),
        TimeoutError(),
        urllib.error.HTTPError("http://x", 500, "boom", None, None),
    ],
)
def test_ollama_client_timeouts_and_http_errors_are_failed(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    _patch_urlopen(monkeypatch, error)
    with pytest.raises(LlmFailedError):
        OllamaClient().chat("s", "u")


@pytest.mark.parametrize("payload", [b"no es json", b'{"otra_cosa": 1}'])
def test_ollama_client_unreadable_body_is_failed(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    _patch_urlopen(monkeypatch, payload)
    with pytest.raises(LlmFailedError, match="formato"):
        OllamaClient().chat("s", "u")


# ------------------------------------------------------------------------------- benchmark
def _questions() -> pd.DataFrame:
    rows = [
        ("Q1", "se admiten perros y gatos?", "False", "reglas_mascotas", "train"),
        (
            "Q2",
            "cuantas personas pueden dormir",
            "False",
            "capacidad_personas",
            "train",
        ),
        ("Q3", "cuanto cuesta el alquiler", "True", "", "train"),
        ("Q4", "hay fecha libre en diciembre", "True", "", "train"),
        ("Q5", "se admiten perros", "False", "reglas_mascotas", "test"),  # no se usa
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "question_id",
            "question_text",
            "should_escalate",
            "kb_source",
            "split",
        ],
    )


def test_benchmark_samples_only_train_rows_and_is_reproducible() -> None:
    from src.eval.benchmark_m3 import sample_questions

    first = sample_questions(_questions(), n_answerable=5, n_escalating=5)
    second = sample_questions(_questions(), n_answerable=5, n_escalating=5)
    assert "Q5" not in set(first["question_id"])
    assert len(first) == 4
    assert first["question_id"].tolist() == second["question_id"].tolist()


def test_benchmark_runs_for_each_k_and_summarizes(retriever: EmbeddingEngine) -> None:
    from src.eval.benchmark_m3 import run_benchmark, sample_questions, summarize

    questions = sample_questions(_questions(), n_answerable=2, n_escalating=2)
    results = run_benchmark(
        questions, [1, 3], retriever=retriever, llm=FakeLlm("ESCALAR")
    )
    assert set(results["k"]) == {1, 3}
    assert len(results) == 8
    assert results["escalated"].all()
    summary = summarize(results)
    assert summary["escalating_escalated"].tolist() == [1.0, 1.0]
    assert summary["answerable_escalated"].tolist() == [1.0, 1.0]
    assert (summary["latency_median_s"] >= 0).all()
