# Tests de la partición train/test (solo datos sintéticos).

import csv
from collections import Counter
from pathlib import Path

import pytest

from src.config import SPLIT_TEST, SPLIT_TRAIN
from src.data import split as sp

HEADER = ["question_id", "question_text", "intent", "split"]


def _rows(per_intent: dict[str, int]) -> list[dict[str, str]]:
    rows = []
    for intent, amount in per_intent.items():
        for index in range(amount):
            rows.append(
                {
                    "question_id": f"{intent[:3]}{index:03d}",
                    "question_text": f"pregunta {intent} {index}",
                    "intent": intent,
                    "split": "",
                }
            )
    return rows


def _counts(rows: list[dict[str, str]]) -> Counter[tuple[str, str]]:
    return Counter((row["intent"], row["split"]) for row in rows)


def test_split_is_stratified_by_intent() -> None:
    result = sp.assign_splits(_rows({"precio": 11, "capacidad": 12, "ubicacion": 10}))
    counts = _counts(result)
    assert counts[("precio", SPLIT_TEST)] == 3
    assert counts[("precio", SPLIT_TRAIN)] == 8
    assert counts[("capacidad", SPLIT_TEST)] == 4
    assert counts[("capacidad", SPLIT_TRAIN)] == 8
    assert counts[("ubicacion", SPLIT_TEST)] == 3
    assert counts[("ubicacion", SPLIT_TRAIN)] == 7


def test_split_is_reproducible_and_independent_of_row_order() -> None:
    rows = _rows({"precio": 11, "capacidad": 11})
    first = sp.assign_splits(rows)
    second = sp.assign_splits(rows)
    shuffled = sp.assign_splits(list(reversed(rows)))
    by_id = {row["question_id"]: row["split"] for row in first}
    assert first == second
    assert {row["question_id"]: row["split"] for row in shuffled} == by_id


def test_different_seed_gives_a_different_split() -> None:
    rows = _rows({"precio": 30})
    first = {
        r["question_id"]
        for r in sp.assign_splits(rows, seed=1)
        if r["split"] == SPLIT_TEST
    }
    second = {
        r["question_id"]
        for r in sp.assign_splits(rows, seed=2)
        if r["split"] == SPLIT_TEST
    }
    assert first != second


def test_every_row_gets_exactly_one_split_and_input_is_not_modified() -> None:
    rows = _rows({"precio": 11})
    result = sp.assign_splits(rows)
    assert {row["split"] for row in result} == {SPLIT_TRAIN, SPLIT_TEST}
    assert all(row["split"] == "" for row in rows)
    assert [row["question_id"] for row in result] == [
        row["question_id"] for row in rows
    ]


def test_small_groups_keep_one_row_in_each_side() -> None:
    result = sp.assign_splits(_rows({"precio": 2, "capacidad": 1}))
    counts = _counts(result)
    assert counts[("precio", SPLIT_TEST)] == 1
    assert counts[("precio", SPLIT_TRAIN)] == 1
    assert counts[("capacidad", SPLIT_TRAIN)] == 1


def test_existing_splits_are_not_reassigned_without_force() -> None:
    assigned = sp.assign_splits(_rows({"precio": 11}))
    with pytest.raises(ValueError, match="particiones"):
        sp.assign_splits(assigned)
    assert sp.assign_splits(assigned, overwrite=True)


def test_invalid_input_is_rejected() -> None:
    duplicated = _rows({"precio": 2})
    duplicated[1]["question_id"] = duplicated[0]["question_id"]
    with pytest.raises(ValueError, match="repetidos"):
        sp.assign_splits(duplicated)
    no_intent = _rows({"precio": 2})
    no_intent[0]["intent"] = ""
    with pytest.raises(ValueError, match="intención"):
        sp.assign_splits(no_intent)


def test_main_fills_the_split_column_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "labeled.csv"
    sp.write_rows(path, HEADER, _rows({"precio": 11, "capacidad": 11}))
    assert sp.main(["--input", str(path)]) == 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == HEADER
    assert _counts(rows)[("precio", SPLIT_TEST)] == 3
    assert "precio: train=8 test=3" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        sp.main(["--input", str(path)])
