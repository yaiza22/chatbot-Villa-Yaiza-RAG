"""Motor conversacional M1 basado en TF-IDF y Regresión Logística.

Entrena un clasificador multiclase sobre las preguntas de entrenamiento usando
n-gramas de 1 a 2 palabras (sin tildes) y regresión logística con pesos
balanceados por clase. El umbral de probabilidad para decidir la abstención por
baja confianza se selecciona automáticamente en cada llamada a ``fit()``
mediante validación cruzada estratificada de 5 pliegues exclusivamente sobre el
conjunto de entrenamiento recibido.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import ESCALATING_INTENTS, INTENTS, REPO_ROOT, SEED, SPLIT_TEST
from src.engines.base import Engine, EngineResponse

DEFAULT_TEMPLATES_PATH = Path(__file__).resolve().parent / "m1_templates.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "m1_tfidf_rl.joblib"
CANDIDATE_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.10 + 0.05 * idx, 2) for idx in range(17)
)


class TfidfRlEngine:
    """Motor M1 con vectorización TF-IDF, regresión logística y umbral por CV.

    Attributes:
        name: Identificador del motor (``"M1"``).
        seed: Semilla fija de reproducibilidad.
        threshold: Umbral de probabilidad máxima elegido en ``fit()``.
        cv_folds_: Número de pliegues utilizado en la última validación cruzada.
        threshold_table_: Tabla de compromiso entre abstención correcta e
            innecesaria para cada umbral candidato.
    """

    name: str = "M1"

    def __init__(
        self,
        templates_path: Path = DEFAULT_TEMPLATES_PATH,
        candidate_thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
        default_cv_folds: int = 5,
    ) -> None:
        """Inicializa la configuración del motor y carga las plantillas de respuesta.

        Args:
            templates_path: Ruta al archivo JSON con las respuestas fijas.
            candidate_thresholds: Secuencia de umbrales en ``(0, 1)`` a evaluar en CV.
            default_cv_folds: Número estándar de pliegues para validación cruzada.

        Raises:
            ValueError: Si las plantillas contienen intenciones fuera de la taxonomía.
        """
        self.seed = SEED
        self.templates_path = Path(templates_path)
        self.candidate_thresholds = tuple(float(t) for t in candidate_thresholds)
        self.default_cv_folds = int(default_cv_folds)

        with self.templates_path.open(encoding="utf-8") as handle:
            raw_cfg = json.load(handle)

        canned: dict[str, str] = dict(raw_cfg.get("canned_answers", {}))
        unknown_intents = set(canned) - set(INTENTS)
        if unknown_intents:
            raise ValueError(
                f"Intenciones desconocidas en plantillas de M1: {sorted(unknown_intents)}"
            )

        self._canned_answers = canned
        self.threshold: float = 0.5
        self.cv_folds_: int | None = None
        self.threshold_table_: pd.DataFrame = pd.DataFrame(
            columns=[
                "threshold",
                "correct_abstention",
                "unnecessary_abstention",
                "youden_j",
            ]
        )
        self._pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """Construye un pipeline nuevo de TF-IDF (1–2 n-gramas) y Regresión Logística."""
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        strip_accents="unicode",
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=self.seed,
                        max_iter=1000,
                    ),
                ),
            ]
        )

    def fit(self, train: pd.DataFrame) -> None:
        """Ajusta el pipeline y selecciona el umbral de confianza con CV sobre ``train``.

        Args:
            train: DataFrame con el esquema de ``seed.csv``/``labeled.csv`` que
                contiene al menos ``question_text`` e ``intent``.

        Raises:
            ValueError: Si ``train`` está vacío, si faltan columnas requeridas,
                si contiene filas con ``split == "test"``, o si alguna clase tiene
                menos de 2 ejemplos (impidiendo la validación cruzada estratificada).
        """
        if train is None or train.empty:
            raise ValueError(
                "El conjunto de entrenamiento para M1 no puede estar vacío."
            )

        required_cols = {"question_text", "intent"}
        missing_cols = sorted(required_cols - set(train.columns))
        if missing_cols:
            raise ValueError(
                f"Faltan columnas requeridas en train para M1: {missing_cols}"
            )

        if "split" in train.columns:
            splits = train["split"].astype(str).str.strip()
            if (splits == SPLIT_TEST).any():
                raise ValueError(
                    "fit() recibió filas con split == 'test'. El entrenamiento y "
                    "la selección del umbral deben usar únicamente datos de 'train'."
                )

        texts = train["question_text"].astype(str).tolist()
        labels = train["intent"].astype(str).str.strip().tolist()

        if "should_escalate" in train.columns:
            should_escalate = np.array(
                [
                    val if isinstance(val, bool) else str(val).strip().lower() == "true"
                    for val in train["should_escalate"].tolist()
                ],
                dtype=bool,
            )
        else:
            should_escalate = np.array(
                [lbl in ESCALATING_INTENTS for lbl in labels], dtype=bool
            )

        counts = Counter(labels)
        if len(counts) < 2:
            raise ValueError(
                "Se requieren al menos 2 intenciones distintas en train para ajustar M1."
            )

        min_class_count = min(counts.values())
        if min_class_count < 2:
            low_classes = sorted(cls for cls, cnt in counts.items() if cnt < 2)
            raise ValueError(
                f"La validación cruzada estratificada no puede ejecutarse porque "
                f"las siguientes intenciones tienen menos de 2 ejemplos en train: "
                f"{low_classes}. Salida propuesta: reúna al menos 2 ejemplos por "
                f"intención en el conjunto de entrenamiento antes de llamar a fit()."
            )

        if min_class_count < self.default_cv_folds:
            n_splits = min_class_count
            small_classes = {
                cls: cnt
                for cls, cnt in sorted(counts.items())
                if cnt < self.default_cv_folds
            }
            msg = (
                f"AVISO M1: Algunas clases tienen menos de {self.default_cv_folds} "
                f"ejemplos en train ({small_classes}); se reduce el número de "
                f"pliegues de validación cruzada a {n_splits}."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            print(msg)
        else:
            n_splits = self.default_cv_folds

        self.cv_folds_ = n_splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        cv_pipeline = self._build_pipeline()
        oof_probas = cross_val_predict(
            cv_pipeline,
            texts,
            labels,
            cv=skf,
            method="predict_proba",
        )

        classes = np.array(sorted(counts.keys()))
        oof_pred_indices = np.argmax(oof_probas, axis=1)
        oof_pred_intents = classes[oof_pred_indices]
        oof_max_probas = np.max(oof_probas, axis=1)

        self.threshold_table_ = self._compute_threshold_table(
            oof_pred_intents=oof_pred_intents,
            oof_max_probas=oof_max_probas,
            should_escalate=should_escalate,
        )
        self.threshold = self._select_best_threshold(self.threshold_table_)

        final_pipeline = self._build_pipeline()
        final_pipeline.fit(texts, labels)
        self._pipeline = final_pipeline

    def _compute_threshold_table(
        self,
        oof_pred_intents: np.ndarray,
        oof_max_probas: np.ndarray,
        should_escalate: np.ndarray,
    ) -> pd.DataFrame:
        """Construye la tabla de compromiso de umbrales sobre predicciones fuera de pliegue."""
        pred_is_escalating = np.array(
            [intent in ESCALATING_INTENTS for intent in oof_pred_intents],
            dtype=bool,
        )
        esc_mask = should_escalate
        resp_mask = ~should_escalate

        rows: list[dict[str, float]] = []
        for thr in self.candidate_thresholds:
            escalated = pred_is_escalating | (oof_max_probas < thr)
            correct_abst = (
                float(escalated[esc_mask].mean()) if esc_mask.any() else float("nan")
            )
            unnecessary_abst = (
                float(escalated[resp_mask].mean()) if resp_mask.any() else float("nan")
            )
            if np.isnan(correct_abst) or np.isnan(unnecessary_abst):
                youden_j = float("nan")
            else:
                youden_j = correct_abst - unnecessary_abst

            rows.append(
                {
                    "threshold": float(thr),
                    "correct_abstention": correct_abst,
                    "unnecessary_abstention": unnecessary_abst,
                    "youden_j": youden_j,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _select_best_threshold(table: pd.DataFrame) -> float:
        """Elige el umbral que maximiza J = abstención_correcta - abstención_innecesaria."""
        valid = table.dropna(subset=["youden_j"])
        if valid.empty:
            return 0.5

        ordered = valid.sort_values(
            by=["youden_j", "correct_abstention", "threshold"],
            ascending=[False, False, True],
        )
        return float(ordered.iloc[0]["threshold"])

    def get_threshold_tradeoff_table(self) -> pd.DataFrame:
        """Devuelve una copia de la tabla de compromiso calculada en el último ``fit()``."""
        return self.threshold_table_.copy()

    def respond(self, message: str) -> EngineResponse:
        """Predice la intención, evalúa el umbral de escalado y genera la respuesta.

        Args:
            message: Texto de la consulta del cliente.

        Returns:
            ``EngineResponse`` con la intención predicha, probabilidad máxima
            como ``confidence``, decisión de escalado, respuesta plantilla y
            latencia en milisegundos.

        Raises:
            RuntimeError: Si se llama a ``respond()`` antes de ``fit()``.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "El motor M1 no ha sido entrenado. Llame a fit(train) antes de respond()."
            )

        start_time = time.perf_counter()
        probas = self._pipeline.predict_proba([str(message)])[0]
        best_idx = int(np.argmax(probas))
        pred_intent = str(self._pipeline.classes_[best_idx])
        confidence = float(probas[best_idx])

        should_escalate = (pred_intent in ESCALATING_INTENTS) or (
            confidence < self.threshold
        )
        answer = None if should_escalate else self._canned_answers.get(pred_intent)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return EngineResponse(
            answer=answer,
            intent=pred_intent,
            confidence=confidence,
            escalate=should_escalate,
            latency_ms=latency_ms,
        )

    def save_threshold_table(
        self,
        path: Path | str | None = None,
    ) -> Path:
        """Guarda la tabla de compromiso de umbrales en un archivo CSV.

        Args:
            path: Ruta destino del CSV (por defecto ``models/m1_threshold_table.csv``).

        Returns:
            Ruta del archivo CSV generado.
        """
        target = (
            DEFAULT_MODEL_PATH.with_name("m1_threshold_table.csv")
            if path is None
            else Path(path)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.threshold_table_.to_csv(target, index=False, encoding="utf-8")
        return target

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
        """Guarda el motor entrenado, su umbral y su tabla de compromiso con ``joblib``.

        Args:
            path: Ruta destino del archivo ``.joblib`` (por defecto en ``models/``).

        Returns:
            Ruta donde quedó almacenado el artefacto.

        Raises:
            RuntimeError: Si el motor aún no ha sido entrenado con ``fit()``.
        """
        if self._pipeline is None:
            raise RuntimeError("No se puede guardar un motor M1 sin entrenar.")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self._pipeline,
            "threshold": self.threshold,
            "cv_folds": self.cv_folds_,
            "threshold_table": self.threshold_table_,
            "canned_answers": self._canned_answers,
            "candidate_thresholds": self.candidate_thresholds,
            "default_cv_folds": self.default_cv_folds,
        }
        joblib.dump(payload, target)
        csv_table_path = target.with_name(f"{target.stem}_threshold_table.csv")
        self.save_threshold_table(csv_table_path)
        return target

    @classmethod
    def load(
        cls,
        path: Path | str = DEFAULT_MODEL_PATH,
        templates_path: Path = DEFAULT_TEMPLATES_PATH,
    ) -> TfidfRlEngine:
        """Carga desde disco un motor M1 previamente guardado con ``joblib``.

        Args:
            path: Ruta al archivo ``.joblib``.
            templates_path: Ruta al archivo JSON de plantillas.

        Returns:
            Instancia de ``TfidfRlEngine`` lista para invocar ``respond()``.
        """
        payload = joblib.load(Path(path))
        instance = cls(
            templates_path=templates_path,
            candidate_thresholds=payload.get(
                "candidate_thresholds", CANDIDATE_THRESHOLDS
            ),
            default_cv_folds=payload.get("default_cv_folds", 5),
        )
        instance._pipeline = payload["pipeline"]
        instance.threshold = float(payload["threshold"])
        instance.cv_folds_ = payload.get("cv_folds")
        instance.threshold_table_ = payload.get(
            "threshold_table", instance.threshold_table_
        )
        if "canned_answers" in payload:
            instance._canned_answers = dict(payload["canned_answers"])
        return instance


TfidfLrEngine = TfidfRlEngine


def _ensure_protocol_compliance() -> Engine:
    """Verificación estática de que TfidfRlEngine cumple el protocolo Engine."""
    return TfidfRlEngine()
