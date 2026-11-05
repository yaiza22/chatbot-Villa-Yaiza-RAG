# Tests de consistencia de ``data/processed/seed.csv`` (preguntas sintéticas).

import csv
from collections import Counter

import pytest

from src.config import (
    ESCALATING_INTENTS,
    INTENTS,
    PROCESSED_DIR,
    SPLIT_TEST,
    SPLIT_TRAIN,
)
from src.data.anonymize import Anonymizer
from src.data.check_pii import check_file

SEED_CSV = PROCESSED_DIR / "seed.csv"
COLUMNS = [
    "question_id",
    "question_text",
    "intent",
    "should_escalate",
    "reference_answer",
    "kb_source",
    "split",
    "source",
]


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with SEED_CSV.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def test_schema_ids_and_sources(rows: list[dict[str, str]]) -> None:
    ids = [row["question_id"] for row in rows]
    assert len(set(ids)) == len(ids)
    assert all(row["source"] == "synthetic" for row in rows)
    assert all(row["question_text"].strip() for row in rows)


def test_every_intent_has_between_10_and_12_questions(
    rows: list[dict[str, str]],
) -> None:
    counts = Counter(row["intent"] for row in rows)
    assert set(counts) == set(INTENTS)
    assert all(10 <= count <= 12 for count in counts.values())


def test_escalation_flag_follows_the_taxonomy(rows: list[dict[str, str]]) -> None:
    for row in rows:
        expected = row["intent"] in ESCALATING_INTENTS
        assert row["should_escalate"] == str(expected), row["question_id"]


def test_reference_answer_follows_the_labeling_conventions(
    rows: list[dict[str, str]],
) -> None:
    for row in rows:
        if row["should_escalate"] == "True":
            assert row["reference_answer"] == "ESCALAR", row["question_id"]
            assert row["kb_source"] == "", row["question_id"]
        else:
            assert row["reference_answer"].strip(), row["question_id"]


def test_split_is_complete_and_stratified(rows: list[dict[str, str]]) -> None:
    assert {row["split"] for row in rows} == {SPLIT_TRAIN, SPLIT_TEST}
    for intent in INTENTS:
        splits = Counter(row["split"] for row in rows if row["intent"] == intent)
        assert splits[SPLIT_TEST] >= 1
        assert splits[SPLIT_TRAIN] > splits[SPLIT_TEST]


def test_questions_are_not_duplicated(rows: list[dict[str, str]]) -> None:
    texts = [" ".join(row["question_text"].lower().split()) for row in rows]
    assert len(set(texts)) == len(texts)


def test_seed_has_no_personal_data() -> None:
    assert check_file(SEED_CSV, Anonymizer()) == []
