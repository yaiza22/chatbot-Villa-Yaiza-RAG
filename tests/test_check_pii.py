# Tests de check_pii (coon datos sintéticos).

import csv
from pathlib import Path

import pytest

from src.data import anonymize as an
from src.data import check_pii as cp

HEADER = ["msg_id", "chat_id", "timestamp", "sender_role", "text"]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _row(number: int, text: str) -> list[str]:
    return [
        f"chat_0001_{number:05d}",
        "chat_0001",
        "2024-03-12 09:15:00",
        "cliente",
        text,
    ]


def test_clean_csv_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_csv(
        tmp_path / "clean.csv",
        HEADER,
        [
            _row(1, "Hola, somos 45 personas y queremos 3 noches"),
            _row(2, "[TEL] [EMAIL]"),
        ],
    )
    assert cp.main([str(path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_dirty_csv_fails_and_never_prints_the_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_csv(
        tmp_path / "dirty.csv",
        HEADER,
        [
            _row(1, "Hola, buenas tardes"),
            _row(2, "llámame al 300 111 2233 o escribe a ana@ejemplo.com"),
        ],
    )
    assert cp.main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "chat_0001_00002" in output
    assert "columna text" in output
    assert "[TEL]" in output
    assert "[EMAIL]" in output
    assert "111 2233" not in output
    assert "ejemplo.com" not in output
    assert "chat_0001_00001" not in output


def test_identifier_and_timestamp_columns_are_not_scanned(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "ids.csv",
        HEADER,
        [["chat_0001_00001", "chat_0001", "2024-03-12 09:15:00", "cliente", "hola"]],
    )
    assert cp.check_file(path, an.Anonymizer()) == []


def test_names_are_checked_only_when_a_list_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_csv(tmp_path / "names.csv", HEADER, [_row(1, "ok Rosita")])
    assert cp.main([str(path)]) == 0
    assert "AVISO" in capsys.readouterr().out

    names = tmp_path / "names.txt"
    names.write_text("Rosita\n", encoding="utf-8")
    assert cp.main([str(path), "--names-file", str(names)]) == 1
    assert "[NOMBRE]" in capsys.readouterr().out


def test_labeled_schema_uses_question_id_and_scans_reference_answer(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path / "labeled.csv",
        [
            "question_id",
            "question_text",
            "intent",
            "should_escalate",
            "reference_answer",
            "kb_source",
            "split",
            "source",
        ],
        [
            [
                "q001",
                "cuanto vale?",
                "precio",
                "True",
                "escala: 3001112233",
                "",
                "train",
                "real",
            ]
        ],
    )
    hits = cp.check_file(path, an.Anonymizer())
    assert [(hit.row_id, hit.column, hit.tag) for hit in hits] == [
        ("q001", "reference_answer", an.TAG_PHONE)
    ]


def test_anonymized_output_passes_the_check(tmp_path: Path) -> None:
    source = _write_csv(
        tmp_path / "interim.csv",
        HEADER,
        [
            _row(1, "soy Rosita, cel 300 111 2233, mail rosita@ejemplo.com"),
            _row(2, "mi cc es 1.234.567.890 y la cuenta 12345678901"),
        ],
    )
    names = tmp_path / "names.txt"
    names.write_text("Rosita\n", encoding="utf-8")
    output = tmp_path / "processed.csv"
    assert (
        an.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--names-file",
                str(names),
            ]
        )
        == 0
    )
    assert cp.main([str(source), "--names-file", str(names)]) == 1
    assert cp.main([str(output), "--names-file", str(names)]) == 0


def test_unknown_column_is_a_usage_error(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "x.csv", HEADER, [_row(1, "hola")])
    with pytest.raises(SystemExit):
        cp.main([str(path), "--columns", "no_existe"])
