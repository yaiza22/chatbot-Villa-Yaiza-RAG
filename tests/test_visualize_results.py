"""Pruebas del módulo de análisis de resultados, solo con datos inventados."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.config import INTENTS
from src.eval import visualize_results as vr

DATE = "2030-01-01"


def _questions() -> pd.DataFrame:
    """Cuatro preguntas inventadas: dos escalan y dos son respondibles."""
    return pd.DataFrame(
        {
            "question_id": ["X001", "X002", "X003", "X004"],
            "question_text": [
                "cuanto vale",
                "hay fecha libre",
                "cuantos caben",
                "donde queda",
            ],
            "intent": ["precio", "disponibilidad", "capacidad", "ubicacion"],
            "should_escalate": [True, True, False, False],
            "kb_source": ["", "", "cap_a", "ubi_a"],
            "reference_answer": ["ESCALAR", "ESCALAR", "texto cap", "texto ubi"],
        }
    )


def _predictions() -> pd.DataFrame:
    """Predicciones inventadas de M0, M2 y M3 sobre las mismas cuatro preguntas."""
    q = _questions()
    rows = []
    # M0: acierta todo menos X004 (dice capacidad y responde).
    for qid, pred, esc in zip(
        q.question_id,
        ["precio", "disponibilidad", "capacidad", "capacidad"],
        [True, True, False, False],
        strict=True,
    ):
        rows.append(
            {
                "engine": "M0",
                "question_id": qid,
                "predicted_intent": pred,
                "escalate": esc,
                "retrieved_str": np.nan,
                "answer": None if esc else "plantilla",
            }
        )
    # M2: responde X001 (debía escalar); top-1 correcto en X003, incorrecto en X004.
    m2 = [
        ("capacidad", False, "cap_a|ubi_a"),
        ("ubicacion", True, "ubi_a|cap_a"),
        ("capacidad", False, "cap_a|ubi_a"),
        ("capacidad", False, "cap_a|ubi_a"),
    ]
    for qid, (pred, esc, ret) in zip(q.question_id, m2, strict=True):
        rows.append(
            {
                "engine": "M2",
                "question_id": qid,
                "predicted_intent": pred,
                "escalate": esc,
                "retrieved_str": ret,
                "answer": None if esc else "fragmento",
            }
        )
    # M3: mismo recuperador; en X001 escribe ESCALAR al final sin escalar.
    m3_answers = [
        "La finca... ESCALAR",
        None,
        "Caben 40.",
        "Comunícate con el encargado.",
    ]
    m3_esc = [False, True, False, False]
    for qid, (pred, _, ret), esc, ans in zip(
        q.question_id, m2, m3_esc, m3_answers, strict=True
    ):
        rows.append(
            {
                "engine": "M3",
                "question_id": qid,
                "predicted_intent": pred,
                "escalate": esc,
                "retrieved_str": ret,
                "answer": ans,
            }
        )
    preds = pd.DataFrame(rows).merge(
        q[
            [
                "question_id",
                "question_text",
                "intent",
                "should_escalate",
                "kb_source",
                "reference_answer",
            ]
        ],
        on="question_id",
    )
    preds["train_source"] = "real"
    preds["source"] = "real"
    preds["split"] = "train"
    return preds


def _write_run(folder: Path, preds: pd.DataFrame, summary: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    preds.to_csv(folder / f"manual_eval_{DATE}.csv", index=False)
    summary.to_csv(folder / f"results_{DATE}.csv", index=False)
    for engine, group in preds.groupby("engine"):
        cm = pd.crosstab(group["intent"], group["predicted_intent"]).reindex(
            index=list(INTENTS), columns=list(INTENTS), fill_value=0
        )
        cm.index.name = "true_intent"
        cm.to_csv(folder / f"confusion_matrix_{DATE}_{engine}_real_train.csv")


def _summary(preds: pd.DataFrame) -> pd.DataFrame:
    out = vr.per_question_outcomes(preds)
    rows = []
    for engine, g in out.groupby("engine"):
        se = g["should_escalate"]
        rows.append(
            {
                "engine": engine,
                "split": "train",
                "source": "real",
                "intent_accuracy": g["intent_hit"].mean(),
                "correct_abstention": g.loc[se, "escalate"].mean(),
                "unnecessary_abstention": g.loc[~se, "escalate"].mean(),
                "hit_at_1": g["hit_at_1"].mean() if g["retrieves"].any() else np.nan,
                "latency_median_ms": 1.0,
                "latency_p95_ms": 2.0,
                "hardware": "equipo de prueba",
                "n_total": len(g),
                "n_should_escalate": int(se.sum()),
                "n_respondable": int((~se).sum()),
                "n_kb": 2,
            }
        )
    return pd.DataFrame(rows)


def test_per_question_outcomes_values() -> None:
    out = vr.per_question_outcomes(_predictions()).set_index(["engine", "question_id"])
    # M0: correcto en X003, incorrecto en X004 (intención equivocada).
    assert out.loc[("M0", "X003"), "auto_correct"] == 1
    assert out.loc[("M0", "X004"), "auto_correct"] == 0
    assert np.isnan(out.loc[("M0", "X001"), "auto_correct"])  # escala: no aplica
    # M2: top-1 = kb_source en X003; en X004 el top-1 es otro fragmento.
    assert out.loc[("M2", "X003"), "auto_correct"] == 1
    assert out.loc[("M2", "X004"), "auto_correct"] == 0
    assert out.loc[("M2", "X004"), "hit_at_1"] == 0
    assert out.loc[("M2", "X004"), "hit_at_3"] == 1
    assert np.isnan(out.loc[("M0", "X003"), "hit_at_1"])  # M0 no recupera
    # M3: texto libre, no se califica automáticamente.
    assert np.isnan(out.loc[("M3", "X003"), "auto_correct"])
    assert out.loc[("M3", "X001"), "m3_token_leak"] == 1
    assert out.loc[("M3", "X004"), "m3_soft_redirect"] == 1


def test_load_and_check_against_summary(tmp_path: Path) -> None:
    preds = _predictions()
    summary = _summary(preds)
    _write_run(
        tmp_path / "a", preds[preds.engine != "M3"], summary[summary.engine != "M3"]
    )
    _write_run(
        tmp_path / "b", preds[preds.engine == "M3"], summary[summary.engine == "M3"]
    )
    run = vr.load_eval_dirs([tmp_path / "a", tmp_path / "b"], split="train")
    assert run.engines == ["M0", "M2", "M3"]
    metrics = vr.compute_metrics(run.predictions, n_boot=200)
    check = vr.check_against_summary(metrics, run.summary)
    assert check["coincide"].all()
    assert vr.check_confusion(run)["coincide"].all()


def test_load_rejects_wrong_split(tmp_path: Path) -> None:
    preds = _predictions()
    _write_run(tmp_path, preds, _summary(preds))
    with pytest.raises(ValueError, match="split"):
        vr.load_eval_dirs([tmp_path], split="test")


def test_load_rejects_different_questions(tmp_path: Path) -> None:
    preds = _predictions()
    preds = preds[~((preds.engine == "M2") & (preds.question_id == "X004"))]
    _write_run(tmp_path, preds, _summary(_predictions()))
    with pytest.raises(ValueError, match="mismas preguntas"):
        vr.load_eval_dirs([tmp_path], split="train")


def test_metric_values_and_small_n_flag() -> None:
    metrics = vr.compute_metrics(_predictions(), n_boot=200).set_index(
        ["engine", "metric"]
    )
    assert metrics.loc[("M2", "correct_abstention"), "value"] == pytest.approx(0.5)
    assert metrics.loc[("M2", "answered_should_escalate"), "value"] == pytest.approx(
        0.5
    )
    assert metrics.loc[("M0", "auto_correct_respondable"), "value"] == pytest.approx(
        0.5
    )
    assert metrics.loc[("M0", "intent_accuracy"), "ci_status"] == "small_n"
    assert ("M0", "hit_at_1") not in metrics.index


def test_escalation_outcomes_counts_add_up() -> None:
    table = vr.escalation_outcomes(_predictions()).set_index("engine")
    parts = ["resp_correct", "resp_wrong", "resp_ungraded", "resp_escalated"]
    assert (table[parts].sum(axis=1) == table["n_respondable"]).all()
    assert (
        table[["esc_ok", "esc_answered"]].sum(axis=1) == table["n_should_escalate"]
    ).all()
    assert table.loc["M3", "resp_ungraded"] == 2


def test_m3_token_sensitivity_only_increases_abstention() -> None:
    sens = vr.m3_token_sensitivity(_predictions())
    values = sens["abstencion_correcta"].tolist()
    assert values == sorted(values)
    assert values[1] == pytest.approx(1.0)


def test_simulate_gate_escalates_if_either_escalates() -> None:
    preds = _predictions()
    combo = vr.simulate_gate(preds, gate="M0", answerer="M2").set_index("question_id")
    assert combo.loc["X001", "escalate"]  # M0 escala aunque M2 responda
    assert not combo.loc["X003", "escalate"]
    assert combo.loc["X003", "answer"] == "fragmento"
    assert (combo["engine"] == "M0+M2").all()


def test_paired_comparison_identical_engines_is_zero() -> None:
    preds = _predictions()
    twin = preds[preds.engine == "M0"].assign(engine="M1")
    comp = vr.paired_comparisons(
        pd.concat([preds, twin]),
        metric_names=["intent_accuracy"],
        pairs=[("M0", "M1")],
        n_boot=200,
    )
    assert comp.loc[0, "diff_a_menos_b"] == 0
    assert comp.loc[0, "ic_incluye_0"]


@dataclass
class _Response:
    answer: str | None
    intent: str | None
    confidence: float
    escalate: bool
    retrieved: list[str] = field(default_factory=list)
    latency_ms: float | None = 0.1


class _FakeEngine:
    """Motor inventado: memoriza la intención más frecuente del train."""

    name = "MF"

    def fit(self, train: pd.DataFrame) -> None:
        self.threshold = 0.5
        self.majority = train["intent"].mode().iloc[0]
        self.seen = set(train["question_text"])

    def respond(self, message: str) -> _Response:
        assert (
            message not in self.seen
        ), "fuga: el pliegue de validación estaba en fit()"
        return _Response(None, self.majority, 0.9, True)


def _train_frame() -> pd.DataFrame:
    intents = ["precio", "capacidad"] * 10
    return pd.DataFrame(
        {
            "question_id": [f"T{i:03d}" for i in range(20)],
            "question_text": [f"pregunta {i}" for i in range(20)],
            "intent": intents,
            "should_escalate": [i == "precio" for i in intents],
            "kb_source": ["" if i == "precio" else "cap_a" for i in intents],
            "split": "train",
            "source": "real",
        }
    )


def test_out_of_fold_never_sees_its_own_rows() -> None:
    preds, choices = vr.out_of_fold_predictions(_FakeEngine, _train_frame(), n_splits=5)
    assert len(preds) == 20
    assert sorted(preds["question_id"]) == sorted(_train_frame()["question_id"])
    assert len(choices) == 5
    assert choices["threshold"].eq(0.5).all()


def test_out_of_fold_rejects_test_rows() -> None:
    frame = _train_frame()
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="train"):
        vr.out_of_fold_predictions(_FakeEngine, frame)


def test_figures_and_tables_are_written(tmp_path: Path) -> None:
    preds = _predictions()
    out = vr.per_question_outcomes(preds)
    metrics = vr.compute_metrics(out, n_boot=100)
    cm = pd.crosstab(preds.intent, preds.predicted_intent)
    vr.plot_confusion(cm, "test", tmp_path / "cm")
    vr.plot_escalation_outcomes(vr.escalation_outcomes(out), tmp_path / "esc")
    vr.plot_metric_panels(metrics, ["correct_abstention"], tmp_path / "met")
    vr.plot_latency(_summary(preds), tmp_path / "lat")
    for name in ("cm", "esc", "met", "lat"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0
    saved = vr.save_table(
        vr.wide_metric_table(metrics, ["correct_abstention"]),
        tmp_path / "tabla",
        "Prueba 100% & más",
        "tab:x",
    )
    tex = saved[1].read_text(encoding="utf-8")
    assert r"100\% \& más" in tex
    assert r"\toprule" in tex


def test_dataset_overview_totals() -> None:
    frame = _train_frame()
    table = vr.dataset_overview(frame)
    assert table.loc["TOTAL", "total"] == 20
    assert table.loc["precio", "escala"]


def test_youden_is_zero_for_always_escalate() -> None:
    preds = _predictions()
    always = preds[preds.engine == "M0"].assign(engine="M1", escalate=True)
    metrics = vr.compute_metrics(always, n_boot=200).set_index(["engine", "metric"])
    assert metrics.loc[("M1", "youden_j"), "value"] == pytest.approx(0.0)
    assert metrics.loc[("M1", "correct_abstention"), "value"] == pytest.approx(1.0)


# ---- Calificación manual de M3 ----------------------------------------------------------
def _sheet(rows: list[tuple], rater: int) -> pd.DataFrame:
    """Hoja inventada con el formato de manual_eval_m3_para_calificar.csv."""
    frame = pd.DataFrame(rows, columns=["tipo", "question_id", "c", "h", "notes"])
    frame["intent"] = "capacidad"
    for n in (1, 2):
        mine = n == rater
        frame[f"correctness_rater_{n}"] = frame["c"] if mine else ""
        frame[f"hallucination_rater_{n}"] = frame["h"] if mine else ""
        frame[f"notes_rater_{n}"] = frame["notes"] if mine else ""
    return frame.drop(columns=["c", "h", "notes"])


RATER_1 = [
    ("respondible", "X003", "2", "0", ""),
    ("respondible", "X004", "1", "0", ""),
    ("debia_escalar", "X001", "", "1", "remite al encargado"),
    ("debia_escalar", "X002", "", "0", ""),
]
RATER_2 = [
    ("respondible", "X003", "2", "0", ""),
    ("respondible", "X004", "0", "0", ""),
    ("debia_escalar", "X001", "", "1", "Remite"),
    ("debia_escalar", "X002", "", "1", ""),
]


def _write_sheets(tmp_path: Path, sep: str = ",") -> tuple[Path, Path]:
    p1, p2 = tmp_path / "r1.csv", tmp_path / "r2.csv"
    _sheet(RATER_1, 1).to_csv(p1, index=False, sep=sep, encoding="utf-8-sig")
    _sheet(RATER_2, 2).to_csv(p2, index=False, sep=sep, encoding="utf-8-sig")
    return p1, p2


def test_load_manual_grades_accepts_semicolon_and_merges(tmp_path: Path) -> None:
    p1, p2 = _write_sheets(tmp_path, sep=";")
    grades = vr.load_manual_grades(p1, p2).set_index("question_id")
    assert grades.loc["X004", "correctness_1"] == 1
    assert grades.loc["X004", "correctness_2"] == 0
    assert np.isnan(grades.loc["X001", "correctness_1"])
    assert grades.loc["X001", "redirect_1"] == 1
    assert grades.loc["X001", "redirect_2"] == 1


def test_load_manual_grades_rejects_invalid_and_missing(tmp_path: Path) -> None:
    bad = [list(r) for r in RATER_1]
    bad[0][2] = "3"
    p1 = tmp_path / "bad.csv"
    _sheet([tuple(r) for r in bad], 1).to_csv(p1, index=False)
    p2 = tmp_path / "r2.csv"
    _sheet(RATER_2, 2).to_csv(p2, index=False)
    with pytest.raises(ValueError, match="no válidos"):
        vr.load_manual_grades(p1, p2)
    empty = [list(r) for r in RATER_1]
    empty[1][3] = ""
    _sheet([tuple(r) for r in empty], 1).to_csv(p1, index=False)
    with pytest.raises(ValueError, match="faltan notas"):
        vr.load_manual_grades(p1, p2)


def test_load_manual_grades_rejects_different_questions(tmp_path: Path) -> None:
    p1, p2 = _write_sheets(tmp_path)
    _sheet(RATER_2[:3], 2).to_csv(p2, index=False)
    with pytest.raises(ValueError, match="mismas preguntas"):
        vr.load_manual_grades(p1, p2)


def test_disagreements_and_resolutions(tmp_path: Path) -> None:
    grades = vr.load_manual_grades(*_write_sheets(tmp_path))
    dis = vr.manual_disagreements(grades)
    assert sorted(dis["question_id"]) == ["X002", "X004"]
    pending = vr.apply_resolutions(grades)
    assert np.isnan(pending.set_index("question_id").loc["X004", "correctness_final"])
    dis["correctness_final"] = (
        ["", "1"] if dis.loc[0, "question_id"] == "X002" else ["1", ""]
    )
    dis["hallucination_final"] = dis["hallucination_1"].astype(int).astype(str)
    final = vr.apply_resolutions(grades, dis).set_index("question_id")
    assert final.loc["X004", "correctness_final"] == 1
    assert final.loc["X003", "correctness_final"] == 2
    assert np.isnan(final.loc["X001", "correctness_final"])  # debia_escalar


def test_agreement_and_summary(tmp_path: Path) -> None:
    grades = vr.apply_resolutions(vr.load_manual_grades(*_write_sheets(tmp_path)))
    agree = vr.manual_agreement(grades).set_index(["medida", "grupo"])
    assert agree.loc[("correccion", "respondible"), "acuerdo_exacto"] == 0.5
    assert agree.loc[("remite_a_persona", "debia_escalar"), "acuerdo_exacto"] == 1.0
    summary = vr.manual_grade_summary(grades, n_boot=100).set_index(
        ["fuente", "metrica", "grupo"]
    )
    assert (
        summary.loc[("calificador_1", "correccion_media", "respondible"), "valor"]
        == 1.5
    )
    assert summary.loc[("promedio", "alucinacion", "debia_escalar"), "valor"] == 0.75
    # Promedio de "correcta completa": X003 (2 y 2) = 1; X004 (1 y 0) = 0 -> 0,5.
    assert summary.loc[("promedio", "correcta_completa", "respondible"), "valor"] == 0.5
    # En consenso, X004 (pendiente) no cuenta: solo X003, con nota 2.
    assert summary.loc[("consenso", "correcta_completa", "respondible"), "n"] == 1


def test_apply_manual_correctness_only_touches_answered_m3_rows(tmp_path: Path) -> None:
    grades = vr.load_manual_grades(*_write_sheets(tmp_path))
    out = vr.per_question_outcomes(_predictions())
    graded = vr.apply_manual_correctness(out, grades, column="correctness_1")
    m3 = graded[graded.engine == "M3"].set_index("question_id")
    assert m3.loc["X003", "auto_correct"] == 1
    assert m3.loc["X004", "auto_correct"] == 0
    assert m3.loc["X004", "manual_partial"] == 1
    m0 = graded[graded.engine == "M0"].set_index("question_id")
    assert m0.loc["X004", "auto_correct"] == 0  # M0 conserva su calificación automática
    table = vr.escalation_outcomes(graded).set_index("engine")
    assert table.loc["M3", "resp_partial"] == 1
    assert table.loc["M3", "resp_correct"] == 1
    assert table.loc["M3", "resp_wrong"] == 0
