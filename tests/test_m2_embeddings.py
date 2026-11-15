"""Pruebas de M2 (embeddings + búsqueda vectorial) con un codificador falso.

No descargan ningún modelo: ``FakeEncoder`` convierte cada texto en una bolsa de palabras
con hashing determinista. Así se prueba la lógica de M2 (carga de la base, búsqueda, umbral,
escalamiento) de forma rápida y reproducible. La calidad del modelo real se mide aparte con
``python -m src.eval.compare_embeddings``.
"""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ESCALATING_INTENTS, KB_DIR
from src.engines.base import Engine, EngineResponse
from src.engines.m2_embeddings import (
    EmbeddingEngine,
    KbFragment,
    SentenceTransformerEncoder,
    load_kb,
)
from src.eval.compare_embeddings import compare_models

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


def _write_fragment(
    directory: Path, name: str, intent: str, title: str, body: str
) -> None:
    (directory / f"{name}.md").write_text(
        f"---\nid: {name}\nintent: {intent}\n---\n# {title}\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture()
def kb_dir(tmp_path: Path) -> Path:
    _write_fragment(
        tmp_path,
        "reglas_mascotas",
        "reglas_servicios",
        "Mascotas",
        "Se admiten perros y gatos con correa dentro de la finca.\n\n"
        "Los dueños recogen los desechos de su mascota.",
    )
    _write_fragment(
        tmp_path,
        "capacidad_personas",
        "capacidad",
        "Capacidad",
        "La finca recibe hasta 65 personas para dormir y más para eventos.",
    )
    _write_fragment(
        tmp_path,
        "ubicacion_direccion",
        "ubicacion",
        "Ubicación",
        "La finca queda en la vereda Cural, a 20 minutos de Ibagué.",
    )
    return tmp_path


@pytest.fixture()
def train() -> pd.DataFrame:
    rows = [
        ("T1", "Se admiten perros en la finca?", "reglas_servicios", False, "train"),
        ("T2", "Cuantas personas pueden dormir?", "capacidad", False, "train"),
        ("T3", "Donde queda la finca en la vereda?", "ubicacion", False, "train"),
        ("T4", "cuanto cuesta el alquiler", "precio", True, "train"),
        ("T5", "hay fecha libre en diciembre", "disponibilidad", True, "train"),
        ("T6", "venden empanadas congeladas", "fuera_de_alcance", True, "train"),
    ]
    return pd.DataFrame(
        rows,
        columns=["question_id", "question_text", "intent", "should_escalate", "split"],
    )


@pytest.fixture()
def engine(kb_dir: Path, train: pd.DataFrame) -> EmbeddingEngine:
    model = EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder())
    model.fit(train)
    return model


# ---------------------------------------------------------------- base de conocimiento
def test_load_kb_reads_fragments_without_header_or_title(kb_dir: Path) -> None:
    fragments = load_kb(kb_dir)
    assert [f.id for f in fragments] == [
        "capacidad_personas",
        "reglas_mascotas",
        "ubicacion_direccion",
    ]
    pets = next(f for f in fragments if f.id == "reglas_mascotas")
    assert pets.title == "Mascotas"
    assert pets.text.startswith("Se admiten perros")
    assert "---" not in pets.text and "# " not in pets.text


def test_load_kb_drops_pending_section(tmp_path: Path) -> None:
    (tmp_path / "reglas_musica.md").write_text(
        "---\nid: reglas_musica\nintent: reglas_servicios\n---\n# Música\n\n"
        "Se permite música.\n\n## Pendiente de confirmar\n\n- ¿Hasta qué hora?\n",
        encoding="utf-8",
    )
    (fragment,) = load_kb(tmp_path)
    assert fragment.text == "Se permite música."


def test_load_kb_accepts_windows_line_endings(tmp_path: Path) -> None:
    raw = "---\r\nid: reglas_musica\r\nintent: reglas_servicios\r\n---\r\n# Música\r\n\r\nHola.\r\n"
    (tmp_path / "reglas_musica.md").write_bytes(raw.encode("utf-8"))
    (fragment,) = load_kb(tmp_path)
    assert fragment == KbFragment(
        "reglas_musica", "reglas_servicios", "Música", "Hola."
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# sin encabezado\n\ntexto", "encabezado YAML"),
        ("---\nintent: capacidad\n---\n# T\n\ntexto\n", "no tiene 'id'"),
        ("---\nid: otro\nintent: capacidad\n---\n# T\n\ntexto\n", "no coincide"),
        ("---\nid: tema\nintent: inventada\n---\n# T\n\ntexto\n", "desconocida"),
        ("---\nid: tema\nintent: precio\n---\n# T\n\ntexto\n", "escala"),
        ("---\nid: tema\nintent: capacidad\n---\n# T\n", "no tiene contenido"),
    ],
)
def test_load_kb_rejects_malformed_files(
    tmp_path: Path, content: str, message: str
) -> None:
    (tmp_path / "tema.md").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_kb(tmp_path)


def test_load_kb_rejects_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No se encontraron"):
        load_kb(tmp_path)


def test_real_kb_loads_and_has_no_escalating_intents() -> None:
    fragments = load_kb(KB_DIR)
    assert len(fragments) >= 10
    assert all(f.intent not in ESCALATING_INTENTS for f in fragments)
    assert all("pendiente de confirmar" not in f.text.lower() for f in fragments)
    assert len({f.id for f in fragments}) == len(fragments)


# ---------------------------------------------------------------------- búsqueda
def test_retrieve_ranks_the_matching_fragment_first(engine: EmbeddingEngine) -> None:
    ranked = engine.retrieve("se admiten perros y gatos?", k=3)
    assert ranked[0][0] == "reglas_mascotas"
    assert len(ranked) == 3
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_does_not_require_fit(kb_dir: Path) -> None:
    fresh = EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder())
    assert fresh.retrieve("perros", k=1)[0][0] == "reglas_mascotas"


def test_retrieve_is_deterministic(engine: EmbeddingEngine) -> None:
    assert engine.retrieve("cuantas personas caben") == engine.retrieve(
        "cuantas personas caben"
    )


def test_split_paragraphs_scores_with_best_paragraph(kb_dir: Path) -> None:
    engine = EmbeddingEngine(
        kb_dir=kb_dir, encoder=FakeEncoder(), split_paragraphs=True
    )
    assert engine.retrieve("recogen los desechos", k=1)[0][0] == "reglas_mascotas"
    assert engine._chunk_vectors is not None
    assert engine._chunk_vectors.shape[0] == 4  # 1 + 2 + 1 párrafos


def test_top_k_must_be_positive(kb_dir: Path) -> None:
    with pytest.raises(ValueError, match="top_k"):
        EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder(), top_k=0)


# --------------------------------------------------------------------- respond
def test_respond_before_fit_raises(kb_dir: Path) -> None:
    fresh = EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder())
    with pytest.raises(RuntimeError, match="fit"):
        fresh.respond("hola")


def test_respond_returns_fragment_text_unmodified(engine: EmbeddingEngine) -> None:
    response = engine.respond("Se admiten perros y gatos con correa en la finca?")
    assert isinstance(response, EngineResponse)
    assert response.escalate is False
    assert response.intent == "reglas_servicios"
    fragment = next(f for f in load_kb(engine.kb_dir) if f.id == "reglas_mascotas")
    assert response.answer == fragment.text
    assert response.retrieved[0] == "reglas_mascotas"
    assert len(response.retrieved) == engine.top_k or len(response.retrieved) == 3
    assert 0.0 <= response.confidence <= 1.0
    assert response.latency_ms >= 0.0


def test_respond_escalates_below_threshold_and_keeps_retrieved(
    engine: EmbeddingEngine,
) -> None:
    engine.threshold = 0.99
    response = engine.respond("zzz qqq xxx")
    assert response.escalate is True
    assert response.answer is None
    assert response.retrieved  # se conserva para medir Hit@k


def test_respond_does_not_escalate_when_similarity_is_above_threshold(
    engine: EmbeddingEngine,
) -> None:
    engine.threshold = 0.0
    assert engine.respond("perros").escalate is False


# -------------------------------------------------------------------------- fit
def test_fit_rejects_test_rows(kb_dir: Path, train: pd.DataFrame) -> None:
    leaked = train.copy()
    leaked.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="test"):
        EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder()).fit(leaked)


def test_fit_rejects_empty_and_missing_columns(
    kb_dir: Path, train: pd.DataFrame
) -> None:
    model = EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder())
    with pytest.raises(ValueError, match="vacío"):
        model.fit(train.iloc[0:0])
    with pytest.raises(ValueError, match="Faltan columnas"):
        model.fit(train.drop(columns=["question_text"]))


def test_fit_builds_tradeoff_table(engine: EmbeddingEngine) -> None:
    table = engine.get_threshold_tradeoff_table()
    assert list(table.columns) == [
        "threshold",
        "correct_abstention",
        "unnecessary_abstention",
        "youden_j",
    ]
    assert len(table) == 101
    assert table["threshold"].is_monotonic_increasing
    # Con umbral 0 solo escalan las preguntas cuyo fragmento más cercano es de intención
    # que escala; como la base no tiene ninguna, nada escala por similitud.
    assert table.loc[0, "unnecessary_abstention"] == 0.0
    assert table.loc[0, "correct_abstention"] == 0.0
    assert table.iloc[-1]["correct_abstention"] == 1.0
    assert 0.0 <= engine.threshold <= 1.0


def test_fit_threshold_is_separating_when_data_is_separable(
    engine: EmbeddingEngine,
) -> None:
    table = engine.get_threshold_tradeoff_table()
    best = table.loc[table["threshold"] == engine.threshold].iloc[0]
    assert best["youden_j"] == table["youden_j"].max()
    assert best["youden_j"] > 0.5


def test_select_best_threshold_tie_break() -> None:
    table = pd.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3, 0.4],
            "correct_abstention": [0.5, 0.9, 0.9, 1.0],
            "unnecessary_abstention": [0.0, 0.4, 0.4, 0.5],
            "youden_j": [0.5, 0.5, 0.5, 0.5],
        }
    )
    # Mismo J: gana la mayor abstención correcta (0.4), no el umbral más bajo.
    assert EmbeddingEngine._select_best_threshold(table) == 0.4
    table["correct_abstention"] = [0.9, 0.9, 0.9, 0.9]
    # Todo empatado: gana el umbral más bajo.
    assert EmbeddingEngine._select_best_threshold(table) == 0.1


def test_select_best_threshold_falls_back_when_no_valid_rows() -> None:
    table = pd.DataFrame(
        {
            "threshold": [0.1],
            "correct_abstention": [float("nan")],
            "unnecessary_abstention": [0.0],
            "youden_j": [float("nan")],
        }
    )
    assert EmbeddingEngine._select_best_threshold(table) == 0.5


def test_fit_warns_when_train_has_one_class(kb_dir: Path, train: pd.DataFrame) -> None:
    only_answerable = train.loc[~train["should_escalate"]]
    model = EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder())
    with pytest.warns(UserWarning, match="AVISO M2"):
        model.fit(only_answerable)
    assert model.threshold == 0.5


def test_fit_is_reproducible(kb_dir: Path, train: pd.DataFrame) -> None:
    thresholds = []
    for _ in range(2):
        model = EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder())
        model.fit(train)
        thresholds.append(model.threshold)
    assert thresholds[0] == thresholds[1]


# ----------------------------------------------------------- contrato e integración
def test_engine_follows_protocol(kb_dir: Path) -> None:
    model: Engine = EmbeddingEngine(kb_dir=kb_dir, encoder=FakeEncoder())
    assert model.name == "M2"


def test_missing_sentence_transformers_gives_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        SentenceTransformerEncoder("paraphrase-multilingual-MiniLM-L12-v2")


def test_m2_is_registered_in_the_evaluation_harness() -> None:
    from src.eval.run_eval import ENGINE_REGISTRY

    assert "M2" in ENGINE_REGISTRY


# --------------------------------------------------------------- comparación de modelos
def test_compare_models_reports_hit_at_k_and_uses_only_train(kb_dir: Path) -> None:
    data = pd.DataFrame(
        [
            ("A1", "se admiten perros y gatos", "reglas_servicios", "False",
             "reglas_mascotas", "train", "synthetic"),
            ("A2", "cuantas personas pueden dormir", "capacidad", "False",
             "capacidad_personas", "train", "synthetic"),
            ("A3", "donde queda la finca vereda", "ubicacion", "False",
             "", "train", "synthetic"),
            ("A4", "cuanto cuesta el alquiler", "precio", "True", "", "train", "synthetic"),
            ("A5", "se admiten perros", "reglas_servicios", "False",
             "capacidad_personas", "test", "synthetic"),  # no debe contarse
        ],
        columns=[
            "question_id", "question_text", "intent", "should_escalate",
            "kb_source", "split", "source",
        ],
    )  # fmt: skip
    table = compare_models(
        data,
        models=["fake-model"],
        kb_dir=kb_dir,
        encoder_factory=lambda name: FakeEncoder(),
    )
    assert set(table["split_paragraphs"]) == {False, True}
    row = table.loc[~table["split_paragraphs"]].iloc[0]
    assert row["source"] == "synthetic"
    assert row["n_answerable"] == 3
    assert row["n_with_kb_source"] == 2
    assert row["n_escalating"] == 1
    assert row["hit_at_1"] == 1.0
    assert row["hit_at_5"] == 1.0
    assert 0.0 <= row["auroc_answerable_vs_escalating"] <= 1.0
    assert row["median_ms_per_query"] >= 0.0


def test_compare_models_without_kb_source_reports_nan_not_invented(
    kb_dir: Path,
) -> None:
    data = pd.DataFrame(
        [("R1", "se admiten perros", "reglas_servicios", "False", "", "train", "real")],
        columns=[
            "question_id", "question_text", "intent", "should_escalate",
            "kb_source", "split", "source",
        ],
    )  # fmt: skip
    table = compare_models(
        data,
        models=["fake-model"],
        kb_dir=kb_dir,
        encoder_factory=lambda name: FakeEncoder(),
    )
    assert table["hit_at_1"].isna().all()
    assert (table["n_with_kb_source"] == 0).all()
