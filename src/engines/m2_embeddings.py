"""Motor conversacional M2: búsqueda semántica con embeddings multilingües.

Idea general (para quien lo ve por primera vez):

1. La base de conocimiento (``data/kb/*.md``) se divide en fragmentos: un archivo es un
   fragmento con un ``id`` y una ``intent``.
2. Un modelo de embeddings convierte cada fragmento y cada pregunta del cliente en un vector
   de números. Dos textos con significado parecido quedan con vectores cercanos aunque no
   compartan palabras ("¿puedo llevar mi perro?" ~ "Se admiten mascotas").
3. ``respond()`` busca el fragmento más parecido a la pregunta (similitud coseno). Si esa
   similitud supera un umbral, responde con el texto del fragmento **sin modificarlo**; si
   no, escala a una persona.

Qué se aprende de los datos y qué no: M2 **no entrena** ningún modelo. Lo único que sale de
``train`` es un número, el umbral de similitud, elegido para maximizar la abstención correcta
menos la abstención innecesaria (el mismo criterio que usa M1). Como no hay nada más que
ajustar, el test queda intacto, y ``fit()`` rechaza cualquier fila con ``split == "test"``.

Limitación conocida: los fragmentos de la base solo cubren ``capacidad``,
``reglas_servicios`` y ``ubicacion``. La intención predicha es la del fragmento más cercano
(según el protocolo de evaluación), así que M2 nunca predice ``precio``, ``disponibilidad``,
``otro_negocio`` ni ``fuera_de_alcance``: esas preguntas solo se detectan si su similitud con
la base es baja. Es justo lo que la comparación con M1 debe medir.
"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.config import (
    EMBEDDING_MODEL,
    ESCALATING_INTENTS,
    INTENTS,
    KB_DIR,
    M2_SPLIT_PARAGRAPHS,
    M2_TOP_K,
    SPLIT_TEST,
)
from src.engines.base import Engine, EngineResponse

# Umbrales candidatos de similitud coseno: de 0.00 a 1.00 en pasos de 0.01. La rejilla es
# fina a propósito: algunos modelos (p. ej. los E5) comprimen sus similitudes entre 0.7 y 1.0.
CANDIDATE_THRESHOLDS: tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(101))
DEFAULT_THRESHOLD = 0.5
PENDING_HEADING = re.compile(r"^##\s+pendiente de confirmar\b", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Base de conocimiento
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KbFragment:
    """Un tema de la base de conocimiento.

    Attributes:
        id: Identificador del fragmento (coincide con el nombre del archivo).
        intent: Intención del tema (``capacidad``, ``reglas_servicios`` o ``ubicacion``).
        title: Encabezado ``# ...`` del archivo.
        text: Cuerpo del fragmento, sin encabezado YAML, sin título y sin la sección
            ``## Pendiente de confirmar``. Es lo que M2 devuelve al cliente.
    """

    id: str
    intent: str
    title: str
    text: str


def _parse_fragment(path: Path) -> KbFragment:
    """Lee un archivo ``.md`` de la base y lo convierte en ``KbFragment``."""
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    match = re.match(r"---\n(.*?)\n---\n(.*)", raw, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"{path.name}: falta el encabezado YAML (--- ... ---).")

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    for key in ("id", "intent"):
        if not meta.get(key):
            raise ValueError(f"{path.name}: el encabezado YAML no tiene '{key}'.")
    if meta["id"] != path.stem:
        raise ValueError(
            f"{path.name}: el id '{meta['id']}' no coincide con el nombre del archivo."
        )
    if meta["intent"] not in INTENTS:
        raise ValueError(f"{path.name}: intención desconocida '{meta['intent']}'.")
    if meta["intent"] in ESCALATING_INTENTS:
        raise ValueError(
            f"{path.name}: un fragmento de la base no puede tener la intención "
            f"'{meta['intent']}' porque esa intención escala."
        )

    title = ""
    body_lines: list[str] = []
    for line in match.group(2).splitlines():
        if PENDING_HEADING.match(line.strip()):
            break  # lo que sigue es una nota para el equipo, no contenido para el cliente
        if not title and line.startswith("# "):
            title = line[2:].strip()
            continue
        body_lines.append(line)
    text = "\n".join(body_lines).strip()
    if not text:
        raise ValueError(f"{path.name}: el fragmento no tiene contenido.")
    return KbFragment(id=meta["id"], intent=meta["intent"], title=title, text=text)


def load_kb(kb_dir: Path | str = KB_DIR) -> list[KbFragment]:
    """Carga todos los fragmentos de ``kb_dir`` en orden alfabético de archivo.

    Raises:
        ValueError: Si un archivo está mal formado o si la carpeta no tiene fragmentos.
    """
    files = sorted(Path(kb_dir).glob("*.md"))
    fragments = [_parse_fragment(path) for path in files]
    if not fragments:
        raise ValueError(f"No se encontraron fragmentos .md en {kb_dir}.")
    return fragments


# --------------------------------------------------------------------------------------
# Codificadores de texto
# --------------------------------------------------------------------------------------
class Encoder(Protocol):
    """Convierte textos en vectores. Permite cambiar de modelo (o usar uno falso en pruebas)."""

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Vectores de las preguntas del cliente, forma ``(n, dimensión)``."""
        ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Vectores de los fragmentos de la base, forma ``(n, dimensión)``."""
        ...


class SentenceTransformerEncoder:
    """Codificador con un modelo de ``sentence-transformers`` (gratuito y local).

    Los modelos de la familia E5 exigen los prefijos ``query: `` y ``passage: ``; se agregan
    solos cuando el nombre del modelo contiene ``e5``. El resto no usa prefijos.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "M2 necesita sentence-transformers. Instálalo con: "
                "pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = "query: " if is_e5 else ""
        self._document_prefix = "passage: " if is_e5 else ""

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        vectors = self._model.encode(
            [prefix + str(text) for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Vectores normalizados de las preguntas."""
        return self._encode(texts, self._query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Vectores normalizados de los fragmentos."""
        return self._encode(texts, self._document_prefix)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normaliza cada fila a longitud 1 para que el producto punto sea la similitud coseno."""
    matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


# --------------------------------------------------------------------------------------
# Motor M2
# --------------------------------------------------------------------------------------
class EmbeddingEngine:
    """Motor M2: recupera el fragmento más parecido y lo devuelve sin redactarlo.

    Attributes:
        name: Identificador del motor (``"M2"``).
        top_k: Cuántos fragmentos se devuelven en ``EngineResponse.retrieved``.
        split_paragraphs: Si es ``True``, cada párrafo de un fragmento se vectoriza por
            separado y el fragmento puntúa con su mejor párrafo. Ayuda con fragmentos largos
            (algunos modelos truncan a 128 tokens). El valor por defecto sale de
            ``M2_SPLIT_PARAGRAPHS`` en ``src/config.py``.
        threshold: Umbral de similitud elegido en ``fit()``.
        threshold_table_: Tabla de compromiso de cada umbral candidato (última ``fit``).
    """

    name: str = "M2"

    def __init__(
        self,
        kb_dir: Path | str = KB_DIR,
        encoder: Encoder | None = None,
        model_name: str = EMBEDDING_MODEL,
        top_k: int = M2_TOP_K,
        split_paragraphs: bool = M2_SPLIT_PARAGRAPHS,
        candidate_thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k debe ser un entero positivo.")
        self.kb_dir = Path(kb_dir)
        self.model_name = model_name
        self.top_k = top_k
        self.split_paragraphs = split_paragraphs
        self.candidate_thresholds = tuple(float(t) for t in candidate_thresholds)
        self.threshold: float = DEFAULT_THRESHOLD
        self.threshold_table_: pd.DataFrame = pd.DataFrame()

        self._encoder = encoder
        self._fitted = False
        self._fragments: list[KbFragment] = []
        self._by_id: dict[str, KbFragment] = {}
        self._chunk_vectors: np.ndarray | None = None
        self._chunk_owner: np.ndarray | None = None

    # ---- índice -----------------------------------------------------------------
    def _chunks_of(self, fragment: KbFragment) -> list[str]:
        """Textos que se vectorizan por cada fragmento (todos llevan el título)."""
        prefix = f"{fragment.title}. " if fragment.title else ""
        if not self.split_paragraphs:
            return [prefix + fragment.text]
        paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", fragment.text) if p.strip()
        ]
        return [prefix + paragraph for paragraph in paragraphs]

    def _ensure_index(self) -> None:
        """Carga la base y calcula los vectores la primera vez que se necesitan."""
        if self._chunk_vectors is not None:
            return
        if self._encoder is None:
            self._encoder = SentenceTransformerEncoder(self.model_name)

        self._fragments = load_kb(self.kb_dir)
        self._by_id = {fragment.id: fragment for fragment in self._fragments}
        texts: list[str] = []
        owners: list[int] = []
        for index, fragment in enumerate(self._fragments):
            for chunk in self._chunks_of(fragment):
                texts.append(chunk)
                owners.append(index)
        self._chunk_vectors = _l2_normalize(self._encoder.encode_documents(texts))
        self._chunk_owner = np.asarray(owners, dtype=int)

    def _fragment_scores(self, query_vectors: np.ndarray) -> np.ndarray:
        """Similitud de cada pregunta con cada fragmento, forma ``(n_preguntas, n_fragmentos)``.

        Un fragmento puntúa con su mejor párrafo (con un solo párrafo, es su único vector).
        """
        assert self._chunk_vectors is not None and self._chunk_owner is not None
        sims = _l2_normalize(query_vectors) @ self._chunk_vectors.T
        scores = np.full((sims.shape[0], len(self._fragments)), -np.inf)
        for owner in range(len(self._fragments)):
            scores[:, owner] = sims[:, self._chunk_owner == owner].max(axis=1)
        return scores

    # ---- API pública --------------------------------------------------------------
    def retrieve(self, message: str, k: int | None = None) -> list[tuple[str, float]]:
        """Devuelve los ``k`` fragmentos más parecidos como ``(id, similitud)``.

        No usa el umbral ni requiere ``fit()``: sirve para medir la calidad de la búsqueda
        (Hit@k) y para comparar modelos de embeddings.
        """
        self._ensure_index()
        assert self._encoder is not None
        scores = self._fragment_scores(self._encoder.encode_queries([str(message)]))[0]
        order = np.argsort(
            -scores, kind="stable"
        )  # empates: orden de la base, reproducible
        limit = self.top_k if k is None else k
        return [(self._fragments[i].id, float(scores[i])) for i in order[:limit]]

    def fragment(self, fragment_id: str) -> KbFragment:
        """Devuelve el fragmento de la base con ese ``id`` (carga la base si hace falta).

        Raises:
            KeyError: Si no existe un fragmento con ese ``id``.
        """
        self._ensure_index()
        return self._by_id[fragment_id]

    def fit(self, train: pd.DataFrame) -> None:
        """Elige el umbral de similitud con las preguntas de ``train``.

        Args:
            train: DataFrame con ``question_text`` e ``intent`` (y opcionalmente
                ``should_escalate``), con el esquema de ``seed.csv`` / ``labeled.csv``.

        Raises:
            ValueError: Si ``train`` está vacío, si faltan columnas o si trae filas de test.
        """
        if train is None or train.empty:
            raise ValueError(
                "El conjunto de entrenamiento para M2 no puede estar vacío."
            )
        missing = sorted({"question_text", "intent"} - set(train.columns))
        if missing:
            raise ValueError(f"Faltan columnas requeridas en train para M2: {missing}")
        if (
            "split" in train.columns
            and (train["split"].astype(str).str.strip() == SPLIT_TEST).any()
        ):
            raise ValueError(
                "fit() recibió filas con split == 'test'. La selección del umbral debe "
                "usar únicamente datos de 'train'."
            )

        self._ensure_index()
        assert self._encoder is not None
        labels = train["intent"].astype(str).str.strip().tolist()
        if "should_escalate" in train.columns:
            should_escalate = np.array(
                [
                    v if isinstance(v, bool) else str(v).strip().lower() == "true"
                    for v in train["should_escalate"].tolist()
                ],
                dtype=bool,
            )
        else:
            should_escalate = np.array(
                [label in ESCALATING_INTENTS for label in labels], dtype=bool
            )

        scores = self._fragment_scores(
            self._encoder.encode_queries(train["question_text"].astype(str).tolist())
        )
        top_index = scores.argmax(axis=1)
        top_scores = scores.max(axis=1)
        top_intents = np.array([self._fragments[i].intent for i in top_index])

        self.threshold_table_ = self._compute_threshold_table(
            top_intents, top_scores, should_escalate
        )
        if not should_escalate.any() or should_escalate.all():
            warnings.warn(
                "AVISO M2: train no tiene a la vez preguntas que escalan y que se pueden "
                f"responder; se usa el umbral por defecto ({DEFAULT_THRESHOLD}).",
                UserWarning,
                stacklevel=2,
            )
        self.threshold = self._select_best_threshold(self.threshold_table_)
        self._fitted = True

    def _compute_threshold_table(
        self,
        top_intents: np.ndarray,
        top_scores: np.ndarray,
        should_escalate: np.ndarray,
    ) -> pd.DataFrame:
        """Abstención correcta e innecesaria de cada umbral candidato, sobre ``train``."""
        predicts_escalation = np.array(
            [intent in ESCALATING_INTENTS for intent in top_intents], dtype=bool
        )
        rows: list[dict[str, float]] = []
        for threshold in self.candidate_thresholds:
            escalated = predicts_escalation | (top_scores < threshold)
            correct = (
                float(escalated[should_escalate].mean())
                if should_escalate.any()
                else float("nan")
            )
            unnecessary = (
                float(escalated[~should_escalate].mean())
                if (~should_escalate).any()
                else float("nan")
            )
            youden = (
                float("nan")
                if np.isnan(correct) or np.isnan(unnecessary)
                else correct - unnecessary
            )
            rows.append(
                {
                    "threshold": threshold,
                    "correct_abstention": correct,
                    "unnecessary_abstention": unnecessary,
                    "youden_j": youden,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _select_best_threshold(table: pd.DataFrame) -> float:
        """Elige el umbral con mayor J = abstención correcta - abstención innecesaria.

        Desempata con mayor abstención correcta y, después, con el umbral más bajo (el
        mismo criterio que M1).
        """
        valid = table.dropna(subset=["youden_j"])
        if valid.empty:
            return DEFAULT_THRESHOLD
        best = valid.sort_values(
            by=["youden_j", "correct_abstention", "threshold"],
            ascending=[False, False, True],
        )
        return float(best.iloc[0]["threshold"])

    def get_threshold_tradeoff_table(self) -> pd.DataFrame:
        """Copia de la tabla de compromiso calculada en el último ``fit()``."""
        return self.threshold_table_.copy()

    def respond(self, message: str) -> EngineResponse:
        """Busca el fragmento más parecido y decide si responde o escala.

        Returns:
            ``EngineResponse`` con la intención del fragmento más parecido, su similitud
            como ``confidence``, ``retrieved`` con los ``top_k`` ids (siempre, aunque
            escale, para poder medir Hit@k) y ``answer`` con el texto del fragmento sin
            modificar (``None`` si escala).

        Raises:
            RuntimeError: Si se llama antes de ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError(
                "El motor M2 no ha elegido su umbral. Llame a fit(train) antes de respond()."
            )
        start = time.perf_counter()
        ranked = self.retrieve(message)
        best_id, best_score = ranked[0]
        fragment = self._by_id[best_id]
        escalate = fragment.intent in ESCALATING_INTENTS or best_score < self.threshold
        latency_ms = (time.perf_counter() - start) * 1000.0
        return EngineResponse(
            answer=None if escalate else fragment.text,
            intent=fragment.intent,
            confidence=float(np.clip(best_score, 0.0, 1.0)),
            escalate=escalate,
            retrieved=[fragment_id for fragment_id, _ in ranked],
            latency_ms=latency_ms,
        )


def _ensure_protocol_compliance() -> Engine:
    """Falla en tiempo de análisis si ``EmbeddingEngine`` deja de cumplir ``Engine``."""
    return EmbeddingEngine(encoder=None)
