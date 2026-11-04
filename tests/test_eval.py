"""Pruebas unitarias para el módulo de métricas y el arnés de evaluación."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import INTENTS, SPLIT_TEST, SPLIT_TRAIN
from src.engines.base import EngineResponse
from src.eval.metrics import (
    abstention_metrics,
    bootstrap_ci,
    compute_group_metrics,
    confusion_matrix_df,
    hit_at_k,
    intent_accuracy,
    latency_summary,
)
from src.eval.run_eval import (
    build_manual_eval_sheet,
    evaluate,
    main,
    prepare_train_data,
    save_evaluation_outputs,
)


class _SpyEngine:
    """Motor ficticio para inspeccionar los datos que recibe fit() y simular respuestas."""

    def __init__(self, name: str = "M1", retrieved: list[str] | None = None) -> None:
        self.name = name
        self.fitted_df: pd.DataFrame | None = None
        self._retrieved = retrieved or []

    def fit(self, train: pd.DataFrame) -> None:
        self.fitted_df = train.copy()

    def respond(self, message: str) -> EngineResponse:
        if "precio" in message.lower():
            return EngineResponse(
                answer=None,
                intent="precio",
                confidence=0.9,
                escalate=True,
                retrieved=list(self._retrieved),
                latency_ms=1.5,
            )
        return EngineResponse(
            answer="Capacidad confirmada de prueba",
            intent="capacidad",
            confidence=0.85,
            escalate=False,
            retrieved=list(self._retrieved),
            latency_ms=2.5,
        )


@pytest.fixture()
def synthetic_mixed_df() -> pd.DataFrame:
    """Conjunto inventado con mezcla de source (synthetic/real) y split (train/test)."""
    rows = [
        {
            "question_id": "Q001",
            "question_text": "Cuánto vale el alquiler?",
            "intent": "precio",
            "should_escalate": True,
            "reference_answer": "ESCALAR",
            "kb_source": "",
            "split": SPLIT_TRAIN,
            "source": "synthetic",
        },
        {
            "question_id": "Q002",
            "question_text": "Cuántas personas caben en la finca?",
            "intent": "capacidad",
            "should_escalate": False,
            "reference_answer": "[POR CONFIRMAR]",
            "kb_source": "kb_cap_01",
            "split": SPLIT_TRAIN,
            "source": "synthetic",
        },
        {
            "question_id": "Q003",
            "question_text": "Me regala el precio para un evento?",
            "intent": "precio",
            "should_escalate": True,
            "reference_answer": "ESCALAR",
            "kb_source": "",
            "split": SPLIT_TRAIN,
            "source": "real",
        },
        {
            "question_id": "Q004",
            "question_text": "Cuántas habitaciones tiene la finca?",
            "intent": "capacidad",
            "should_escalate": False,
            "reference_answer": "Cuenta con 5 habitaciones.",
            "kb_source": "kb_cap_01",
            "split": SPLIT_TRAIN,
            "source": "real",
        },
        {
            "question_id": "Q005",
            "question_text": "Qué precio tiene el alquiler?",
            "intent": "precio",
            "should_escalate": True,
            "reference_answer": "ESCALAR",
            "kb_source": "",
            "split": SPLIT_TEST,
            "source": "real",
        },
        {
            "question_id": "Q006",
            "question_text": "Cuántas camas hay disponibles para dormir?",
            "intent": "capacidad",
            "should_escalate": False,
            "reference_answer": "[POR CONFIRMAR]",
            "kb_source": "kb_cap_02",
            "split": SPLIT_TEST,
            "source": "synthetic",
        },
    ]
    return pd.DataFrame(rows)


def test_evaluate_blocks_test_split_without_allow_test(
    synthetic_mixed_df: pd.DataFrame,
) -> None:
    """Verifica que evaluate() rechace evaluar sobre test si allow_test es False."""
    with pytest.raises(ValueError, match="allow_test"):
        evaluate(
            eval_df=synthetic_mixed_df,
            engine_names=["M0"],
            split=SPLIT_TEST,
            allow_test=False,
        )


def test_evaluate_allows_test_with_flag_and_prints_warning(
    synthetic_mixed_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Con allow_test=True permite evaluar test e imprime la advertencia de un solo run."""
    result = evaluate(
        eval_df=synthetic_mixed_df,
        engine_names=["M0"],
        split=SPLIT_TEST,
        allow_test=True,
    )
    captured = capsys.readouterr()
    assert "evaluación sobre test: un solo run" in captured.out
    assert set(result.predictions_df["split"].unique()) == {SPLIT_TEST}


def test_never_mixes_source_nor_split(synthetic_mixed_df: pd.DataFrame) -> None:
    """Comprueba que nunca se mezclen source (real/synthetic) ni split (train/test)."""
    result = evaluate(
        eval_df=synthetic_mixed_df,
        engine_names=["M0"],
        split=SPLIT_TRAIN,
        allow_test=False,
    )
    summary = result.summary_df
    assert set(summary["split"].unique()) == {SPLIT_TRAIN}
    assert set(summary["source"].unique()) == {"real", "synthetic"}
    assert len(summary) == 2

    syn_row = summary.loc[summary["source"] == "synthetic"].iloc[0]
    real_row = summary.loc[summary["source"] == "real"].iloc[0]
    assert syn_row["n_total"] == 2
    assert real_row["n_total"] == 2
    assert set(result.predictions_df["question_id"]) == {"Q001", "Q002", "Q003", "Q004"}


def test_fit_never_receives_test_rows_and_respects_train_source(
    synthetic_mixed_df: pd.DataFrame,
) -> None:
    """Verifica que fit() jamás reciba filas de test y filtre según train_source."""
    spy_instances: list[_SpyEngine] = []

    def make_spy() -> _SpyEngine:
        inst = _SpyEngine(name="M1")
        spy_instances.append(inst)
        return inst

    for src_filter, expected_sources in [
        ("synthetic", {"synthetic"}),
        ("real", {"real"}),
        ("both", {"synthetic", "real"}),
    ]:
        evaluate(
            eval_df=synthetic_mixed_df,
            engine_names=["M1"],
            split=SPLIT_TEST,
            allow_test=True,
            train_df=synthetic_mixed_df,
            train_source=src_filter,
            engine_registry={"M1": make_spy},
        )
        fitted = spy_instances[-1].fitted_df
        assert fitted is not None
        assert not fitted.empty
        assert set(fitted["split"].unique()) == {SPLIT_TRAIN}
        assert set(fitted["source"].unique()) == expected_sources


def test_empty_sets_return_nan_and_zero_without_exceptions() -> None:
    """Conjuntos vacíos (n=0) devuelven NaN y n=0 sin lanzar excepciones."""
    low, high, status = bootstrap_ci([])
    assert np.isnan(low) and np.isnan(high)
    assert status == "n=0"

    acc, hits = intent_accuracy([], [])
    assert np.isnan(acc)
    assert hits.size == 0

    abst = abstention_metrics([], [])
    assert abst["n_should_escalate"] == 0
    assert abst["n_respondable"] == 0
    assert np.isnan(abst["correct_abstention"])
    assert np.isnan(abst["unnecessary_abstention"])

    h1, n_kb, _ = hit_at_k(["", None], [["kb_1"], ["kb_2"]], k=1)
    assert n_kb == 0
    assert np.isnan(h1)

    lat = latency_summary([])
    assert lat["n_latency"] == 0
    assert np.isnan(lat["latency_median_ms"])

    empty_df = pd.DataFrame(
        columns=[
            "question_id",
            "question_text",
            "intent",
            "should_escalate",
            "reference_answer",
            "kb_source",
            "split",
            "source",
        ]
    )
    res = evaluate(eval_df=empty_df, engine_names=["M0"], split=SPLIT_TRAIN)
    assert len(res.summary_df) == 1
    row = res.summary_df.iloc[0]
    assert row["n_total"] == 0
    assert np.isnan(row["intent_accuracy"])
    assert row["intent_accuracy_ci_status"] == "n=0"


def test_small_sample_bootstrap_warning_and_hit_at_k_omission() -> None:
    """Verifica la marca low_n cuando n < 10 y que M0/M1 omitan Hit@k pero M2/M3 lo calculen."""
    low, high, status = bootstrap_ci([1.0, 0.0, 1.0, 1.0])
    assert 0.0 <= low <= high <= 1.0
    assert status == "low_n (n=4<10)"

    records = pd.DataFrame(
        [
            {
                "intent": "capacidad",
                "predicted_intent": "capacidad",
                "should_escalate": False,
                "escalate": False,
                "reference_answer": "50 personas",
                "kb_source": "kb_cap_01",
                "retrieved": ["kb_cap_01", "kb_other"],
                "latency_ms": 3.0,
            }
        ]
    )
    m0_metrics = compute_group_metrics(
        records, engine_name="M0", source="synthetic", split="train", hardware="CPU"
    )
    assert np.isnan(m0_metrics["hit_at_1"])
    assert m0_metrics["hit_at_k_status"] == "omitted_for_engine"

    m2_metrics = compute_group_metrics(
        records, engine_name="M2", source="synthetic", split="train", hardware="CPU"
    )
    assert m2_metrics["hit_at_1"] == 1.0
    assert m2_metrics["hit_at_3"] == 1.0
    assert m2_metrics["hit_at_5"] == 1.0
    assert m2_metrics["n_kb"] == 1


def test_manual_eval_sheet_includes_escalar_and_excludes_por_confirmar(
    synthetic_mixed_df: pd.DataFrame,
) -> None:
    """La hoja manual incluye ESCALAR y respuestas confirmadas, pero excluye [POR CONFIRMAR]."""
    result = evaluate(
        eval_df=synthetic_mixed_df,
        engine_names=["M0"],
        split=SPLIT_TRAIN,
    )
    sheet = build_manual_eval_sheet(result.predictions_df)
    assert "[POR CONFIRMAR]" not in set(sheet["reference_answer"])
    assert "ESCALAR" in set(sheet["reference_answer"])
    assert "Cuenta con 5 habitaciones." in set(sheet["reference_answer"])
    assert set(sheet["question_id"]) == {"Q001", "Q003", "Q004"}
    assert "correctness_rater_1" in sheet.columns
    assert "correctness_rater_2" in sheet.columns
    assert "hallucination_rater_1" in sheet.columns
    assert "hallucination_rater_2" in sheet.columns


def test_cli_end_to_end_saves_reports_and_phase2(
    synthetic_mixed_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifica ejecución CLI de punta a punta con --phase2 y --output-name."""
    input_csv = tmp_path / "synthetic_input.csv"
    synthetic_mixed_df.to_csv(input_csv, index=False, encoding="utf-8")
    out_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--eval-input",
            str(input_csv),
            "--train-input",
            str(input_csv),
            "--train-source",
            "synthetic",
            "--engines",
            "M0",
            "--split",
            "train",
            "--hardware",
            "Test CPU",
            "--date",
            "2026-11-04",
            "--phase2",
            "--output-name",
            "custom_eval.csv",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "results_2026-11-04.csv").exists()
    assert (out_dir / "results_fase2.csv").exists()
    assert (out_dir / "custom_eval.csv").exists()
    assert (out_dir / "manual_eval_2026-11-04.csv").exists()
    assert (out_dir / "confusion_matrix_2026-11-04_M0_synthetic_train.csv").exists()
    assert (out_dir / "confusion_matrix_2026-11-04_M0_real_train.csv").exists()

    cm = confusion_matrix_df(["precio"], ["precio"], labels=INTENTS)
    assert cm.loc["precio", "precio"] == 1
    assert len(prepare_train_data(synthetic_mixed_df, "synthetic")) == 2
    saved = save_evaluation_outputs(
        evaluate(synthetic_mixed_df, split=SPLIT_TRAIN),
        output_dir=out_dir,
        date_tag="2026-11-04",
    )
    assert "results" in saved
