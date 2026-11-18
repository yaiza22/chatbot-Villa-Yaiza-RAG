"""Pruebas de la partición incremental (``split.py --only-new``)."""

import csv
from collections import Counter

import pytest

from src.config import SPLIT_TEST, SPLIT_TRAIN
from src.data.split import (
    _ensure_existing_unchanged,
    assign_new_splits,
    assign_splits,
    main,
)

INTENTS_SAMPLE = {
    "capacidad": 40,
    "reglas_servicios": 30,
    "ubicacion": 20,
    "precio": 10,
}


def make_rows(start: int, counts: dict[str, int]) -> list[dict[str, str]]:
    """Filas sintéticas con split vacío; los ids continúan desde ``start``."""
    rows, number = [], start
    for intent, amount in counts.items():
        for _ in range(amount):
            rows.append(
                {
                    "question_id": f"R{number:03d}",
                    "question_text": f"pregunta {number}",
                    "intent": intent,
                    "split": "",
                }
            )
            number += 1
    return rows


def first_batch() -> list[dict[str, str]]:
    return assign_splits(make_rows(1, INTENTS_SAMPLE))


def test_existing_rows_are_not_moved() -> None:
    batch1 = first_batch()
    batch2 = make_rows(101, {"capacidad": 25, "reglas_servicios": 15, "ubicacion": 10})
    result = assign_new_splits(batch1 + batch2)
    assert result[: len(batch1)] == batch1
    assert all(row["split"] in (SPLIT_TRAIN, SPLIT_TEST) for row in result)


def test_cumulative_ratio_is_close_to_70_30() -> None:
    batch1 = first_batch()
    batch2 = make_rows(101, {"capacidad": 25, "reglas_servicios": 15, "ubicacion": 10})
    result = assign_new_splits(batch1 + batch2)
    totals = Counter(row["intent"] for row in result)
    tests = Counter(row["intent"] for row in result if row["split"] == SPLIT_TEST)
    for intent, total in totals.items():
        assert abs(tests[intent] - round(total * 0.3)) <= 1, intent


def test_small_batches_do_not_distort_the_ratio() -> None:
    result = first_batch()
    # Diez lotes de una sola fila: el acumulado debe seguir cerca de 30 %.
    for step in range(10):
        extra = make_rows(500 + step, {"capacidad": 1})
        result = assign_new_splits(result + extra)
    capacity = [row for row in result if row["intent"] == "capacidad"]
    tests = sum(row["split"] == SPLIT_TEST for row in capacity)
    assert abs(tests - round(len(capacity) * 0.3)) <= 1


def test_running_twice_gives_same_result() -> None:
    rows = first_batch() + make_rows(101, {"capacidad": 12, "ubicacion": 7})
    first = assign_new_splits(rows)
    assert first == assign_new_splits(rows)
    # Y una segunda pasada sobre el resultado no cambia nada (ya no hay filas nuevas).
    assert assign_new_splits(first) == first


def test_without_new_rows_nothing_changes() -> None:
    batch1 = first_batch()
    assert assign_new_splits(batch1) == batch1


def test_result_does_not_depend_on_file_order_or_other_intents() -> None:
    old = first_batch()
    new = make_rows(101, {"capacidad": 9, "ubicacion": 6})
    full = assign_new_splits(old + new)
    shuffled = assign_new_splits(list(reversed(old + new)))
    by_id = {row["question_id"]: row["split"] for row in shuffled}
    assert all(by_id[row["question_id"]] == row["split"] for row in full)
    # Quitar una intención no cambia lo que se asigna a las demás.
    only_capacity = assign_new_splits(
        [row for row in old + new if row["intent"] in ("capacidad", "reglas_servicios")]
    )
    capacity_full = {
        r["question_id"]: r["split"] for r in full if r["intent"] == "capacidad"
    }
    capacity_only = {
        r["question_id"]: r["split"]
        for r in only_capacity
        if r["intent"] == "capacidad"
    }
    assert capacity_full == capacity_only


def test_small_classes() -> None:
    rows = make_rows(1, {"capacidad": 1, "ubicacion": 2, "precio": 3})
    result = {row["question_id"]: row for row in assign_new_splits(rows)}
    assert result["R001"]["split"] == SPLIT_TRAIN  # una sola fila: train
    ubicacion = [r["split"] for r in result.values() if r["intent"] == "ubicacion"]
    assert sorted(ubicacion) == [SPLIT_TEST, SPLIT_TRAIN]


def test_new_row_joining_one_existing_row_goes_to_train_when_target_is_met() -> None:
    rows = [
        {
            "question_id": "R001",
            "question_text": "a",
            "intent": "capacidad",
            "split": SPLIT_TEST,
        },
        {
            "question_id": "R002",
            "question_text": "b",
            "intent": "capacidad",
            "split": "",
        },
    ]
    result = assign_new_splits(rows)
    assert result[0]["split"] == SPLIT_TEST
    assert result[1]["split"] == SPLIT_TRAIN


def test_invalid_split_value_fails() -> None:
    rows = first_batch()
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="split"):
        assign_new_splits(rows)


def test_duplicate_ids_and_missing_intent_fail() -> None:
    rows = make_rows(1, {"capacidad": 3})
    with pytest.raises(ValueError, match="repetidos"):
        assign_new_splits(rows + [dict(rows[0])])
    rows[0]["intent"] = ""
    with pytest.raises(ValueError, match="sin intención"):
        assign_new_splits(rows)


def test_guard_detects_changed_existing_row() -> None:
    before = first_batch()
    after = [dict(row) for row in before]
    after[3]["split"] = SPLIT_TEST if before[3]["split"] == SPLIT_TRAIN else SPLIT_TRAIN
    with pytest.raises(ValueError, match=before[3]["question_id"]):
        _ensure_existing_unchanged(before, after)


def test_original_full_split_still_refuses_to_reassign() -> None:
    with pytest.raises(ValueError, match="Ya hay particiones"):
        assign_splits(first_batch())


def write_csv(path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cli_only_new_is_idempotent_and_keeps_old_rows(tmp_path, capsys) -> None:
    path = tmp_path / "labeled.csv"
    batch1 = first_batch()
    write_csv(path, batch1 + make_rows(101, {"capacidad": 10, "ubicacion": 5}))
    assert main(["--input", str(path), "--only-new"]) == 0
    first = read_csv(path)
    assert first[: len(batch1)] == batch1
    assert all(row["split"] for row in first)
    capsys.readouterr()
    assert main(["--input", str(path), "--only-new"]) == 0
    assert "No hay filas nuevas" in capsys.readouterr().out
    assert read_csv(path) == first


def test_cli_rejects_only_new_with_force(tmp_path) -> None:
    path = tmp_path / "labeled.csv"
    write_csv(path, first_batch())
    with pytest.raises(SystemExit):
        main(["--input", str(path), "--only-new", "--force"])
