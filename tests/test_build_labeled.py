# Pruebas de ``build_labeled`` con hojas sintéticas (no usan chats reales).

import csv
from pathlib import Path

import pytest

from src.data import build_labeled as bl

SHEET_HEADER = [
    "question_id",
    "question_text",
    "intent",
    "should_escalate",
    "dudosa",
    "responsable",
    "pair_id",
]


def row(
    number: int,
    text: str = "cuanto vale el alquiler",
    intent: str = "precio",
    escalate: str = "",
    note: str = "",
) -> list[str]:
    return [f"R{number:03d}", text, intent, escalate, note, "YS", f"P{number:05d}"]


def write_sheet(
    path: Path,
    rows: list[list[str]],
    delimiter: str = ",",
    encoding: str = "utf-8",
    header: list[str] | None = None,
) -> Path:
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(header or SHEET_HEADER)
        writer.writerows(rows)
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def good_rows() -> list[list[str]]:
    return [
        row(1, "cuanto vale el alquiler", "precio"),
        row(2, "caben 80 personas?", "capacidad"),
        row(3, "tienen disponible el 24 de dic?", "disponibilidad"),
        row(4, "27", "", "", "depende del mensaje anterior"),
        row(5, "jajaja ok", "", "", "#descartada"),
    ]


def run_main(sheet: Path, out: Path, *extra: str) -> int:
    return bl.main(["--input", str(sheet), "--output", str(out), *extra])


# --- Lectura: separador, BOM y comas ---------------------------------------------------
def test_commas_inside_quoted_text_are_kept(tmp_path: Path) -> None:
    text = "hola, cuanto vale, para 40 personas, el 24 de dic?"
    sheet = write_sheet(tmp_path / "s.csv", [row(1, text, "disponibilidad")])
    out = tmp_path / "labeled.csv"
    assert run_main(sheet, out) == 0
    result = read_rows(out)
    assert result[0]["question_text"] == text
    assert list(result[0]) == list(bl.LABELED_COLUMNS)


@pytest.mark.parametrize("delimiter", [",", ";", "\t"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_detects_delimiter_and_bom(
    tmp_path: Path, delimiter: str, encoding: str
) -> None:
    sheet = write_sheet(
        tmp_path / "s.csv", good_rows(), delimiter=delimiter, encoding=encoding
    )
    out = tmp_path / "labeled.csv"
    assert run_main(sheet, out) == 0
    assert [r["question_id"] for r in read_rows(out)] == ["R001", "R002", "R003"]
    # La salida siempre es UTF-8 sin BOM y con comas.
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.splitlines()[0].decode() == ",".join(bl.LABELED_COLUMNS)


def test_extra_columns_such_as_answer_text_are_ignored(tmp_path: Path) -> None:
    header = ["question_id", "question_text", "answer_text", "intent", "dudosa"]
    sheet = write_sheet(
        tmp_path / "s.csv",
        [["R001", "cuanto vale", "depende de la fecha", "precio", ""]],
        header=header,
    )
    out = tmp_path / "labeled.csv"
    assert run_main(sheet, out) == 0
    assert "depende" not in out.read_text(encoding="utf-8")


def test_trailing_empty_rows_from_excel_are_ignored(tmp_path: Path) -> None:
    sheet = write_sheet(
        tmp_path / "s.csv", good_rows() + [["", "", "", "", "", "", ""]]
    )
    assert run_main(sheet, tmp_path / "labeled.csv") == 0


def test_non_utf8_file_gives_clear_error(tmp_path: Path) -> None:
    sheet = write_sheet(
        tmp_path / "s.csv",
        [row(1, "cuánto vale la cabaña", "precio")],
        encoding="cp1252",
    )
    with pytest.raises(SystemExit) as excinfo:
        run_main(sheet, tmp_path / "labeled.csv")
    assert excinfo.value.code == 2


def test_missing_required_columns_fail(tmp_path: Path) -> None:
    sheet = write_sheet(
        tmp_path / "s.csv", [["R001", "hola"]], header=["question_id", "x"]
    )
    with pytest.raises(SystemExit) as excinfo:
        run_main(sheet, tmp_path / "labeled.csv")
    assert excinfo.value.code == 2


def test_misaligned_rows_are_reported_by_id_without_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = tmp_path / "s.csv"
    sheet.write_text(
        ",".join(SHEET_HEADER)
        + "\nR001,cuanto vale,precio,,,YS,P00001\n"
        + "R002,hola, cuanto vale el secreto,precio,,,YS,P00002\n",
        encoding="utf-8",
    )
    out = tmp_path / "labeled.csv"
    assert run_main(sheet, out) == 2
    printed = capsys.readouterr().out
    assert "R002" in printed
    assert "secreto" not in printed
    assert not out.exists()


# --- Etiquetas, descartadas y should_escalate -----------------------------------------
def test_should_escalate_is_recomputed_from_intent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        row(1, intent="precio", escalate="VERDADERO"),
        row(2, intent="capacidad", escalate="FALSO"),
        row(3, intent="ubicacion", escalate="VERDADERO"),  # no coincide
        row(4, intent="disponibilidad", escalate=""),
    ]
    out = tmp_path / "labeled.csv"
    assert run_main(write_sheet(tmp_path / "s.csv", rows), out) == 0
    result = {r["question_id"]: r["should_escalate"] for r in read_rows(out)}
    assert result == {"R001": "True", "R002": "False", "R003": "False", "R004": "True"}
    printed = capsys.readouterr().out
    assert "no coincide" in printed
    assert "R003" in printed


def test_reference_answer_and_kb_source_rules(tmp_path: Path) -> None:
    header = SHEET_HEADER + ["reference_answer", "kb_source"]
    rows = [
        row(1, intent="precio") + ["algo que se ignora", "kb_x"],
        row(2, intent="capacidad") + ["", ""],
        row(3, intent="ubicacion") + ["Queda en la vereda X", "ubicacion_finca"],
    ]
    out = tmp_path / "labeled.csv"
    assert run_main(write_sheet(tmp_path / "s.csv", rows, header=header), out) == 0
    result = {r["question_id"]: r for r in read_rows(out)}
    assert result["R001"]["reference_answer"] == "ESCALAR"
    assert result["R001"]["kb_source"] == ""
    assert result["R002"]["reference_answer"] == "[POR CONFIRMAR]"
    assert result["R003"]["reference_answer"] == "Queda en la vereda X"
    assert result["R003"]["kb_source"] == "ubicacion_finca"
    assert {r["source"] for r in result.values()} == {"real"}
    assert {r["split"] for r in result.values()} == {""}


def test_intent_is_normalized_but_typos_block(tmp_path: Path) -> None:
    ok = write_sheet(tmp_path / "ok.csv", [row(1, intent=" Precio ")])
    assert run_main(ok, tmp_path / "labeled.csv") == 0
    assert read_rows(tmp_path / "labeled.csv")[0]["intent"] == "precio"
    bad = write_sheet(tmp_path / "bad.csv", [row(1, intent="preci0")])
    out = tmp_path / "out" / "labeled.csv"
    assert run_main(bad, out) == 2
    assert not out.exists()


def test_discarded_rows_are_split_by_reason_without_question_text(
    tmp_path: Path,
) -> None:
    out = tmp_path / "labeled.csv"
    assert run_main(write_sheet(tmp_path / "s.csv", good_rows()), out) == 0
    discarded_path = tmp_path / "labeled_discarded.csv"
    discarded = read_rows(discarded_path)
    assert list(discarded[0]) == list(bl.DISCARDED_COLUMNS)
    assert {d["question_id"]: d["motivo"] for d in discarded} == {
        "R004": "depende_contexto",
        "R005": "sin_sentido",
    }
    assert "jajaja" not in discarded_path.read_text(encoding="utf-8")
    assert "jajaja" not in out.read_text(encoding="utf-8")
    assert len(read_rows(out)) == 3


@pytest.mark.parametrize(
    "bad_rows",
    [
        [row(1, intent="", note="")],  # sin intención ni motivo: se olvidó
        [row(1, intent="precio", note="#descartada")],  # intención y descarte a la vez
        [row(1, intent="precio", note="¿precio u otro_negocio?")],  # intención y nota
        [row(1), row(1)],  # id repetido
        [
            ["S001", "cuanto vale", "precio", "", "", "YS", "P1"]
        ],  # id con formato inválido
        [row(1, text="   ")],  # texto vacío
    ],
)
def test_blocking_problems_write_nothing(
    tmp_path: Path, bad_rows: list[list[str]]
) -> None:
    out = tmp_path / "labeled.csv"
    assert run_main(write_sheet(tmp_path / "s.csv", bad_rows), out) == 2
    assert not out.exists()
    assert not (tmp_path / "labeled_discarded.csv").exists()


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("#descartada", "sin_sentido"),
        ("Descartar", "sin_sentido"),
        ("¿precio u otro_negocio?", "depende_contexto"),
        ("depende del mensaje anterior", "depende_contexto"),
        ("", "sin_clasificar"),
    ],
)
def test_discard_reason_from_note(note: str, expected: str) -> None:
    assert bl.discard_reason(note) == expected


def test_any_note_without_intent_is_discarded_as_needs_context(
    tmp_path: Path,
) -> None:
    rows = good_rows() + [row(6, "y el otro?", "", "", "¿precio u otro_negocio?")]
    out = tmp_path / "labeled.csv"
    assert run_main(write_sheet(tmp_path / "s.csv", rows), out) == 0
    discarded = {
        d["question_id"]: d["motivo"]
        for d in read_rows(tmp_path / "labeled_discarded.csv")
    }
    assert discarded["R006"] == "depende_contexto"
    assert "R006" not in {r["question_id"] for r in read_rows(out)}


def test_small_classes_are_warned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_main(write_sheet(tmp_path / "s.csv", good_rows()), tmp_path / "l.csv") == 0
    )
    printed = capsys.readouterr().out
    assert "menos de 5 filas" in printed
    assert "no se puede estratificar" in printed


# --- Salida protegida, datos personales y comparación ---------------------------------
def test_existing_output_is_protected_unless_forced(tmp_path: Path) -> None:
    sheet = write_sheet(tmp_path / "s.csv", good_rows())
    out = tmp_path / "labeled.csv"
    assert run_main(sheet, out) == 0
    out.write_text("con particion asignada", encoding="utf-8")
    assert run_main(sheet, out) == 2
    assert out.read_text(encoding="utf-8") == "con particion asignada"
    assert run_main(sheet, out, "--force") == 0
    assert out.read_text(encoding="utf-8") != "con particion asignada"


def test_output_is_reproducible(tmp_path: Path) -> None:
    sheet = write_sheet(tmp_path / "s.csv", good_rows())
    assert run_main(sheet, tmp_path / "a" / "labeled.csv") == 0
    assert run_main(sheet, tmp_path / "b" / "labeled.csv") == 0
    assert (tmp_path / "a" / "labeled.csv").read_bytes() == (
        tmp_path / "b" / "labeled.csv"
    ).read_bytes()


def test_personal_data_blocks_all_output_and_is_never_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = good_rows() + [
        row(6, "escribanme a persona@example.com cuanto vale", "precio")
    ]
    out = tmp_path / "labeled.csv"
    assert run_main(write_sheet(tmp_path / "s.csv", rows), out) == 1
    printed = capsys.readouterr().out
    assert "FALLA" in printed
    assert "persona@example.com" not in printed
    assert not out.exists()
    assert not (tmp_path / "labeled_discarded.csv").exists()


def test_compare_reports_only_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = write_sheet(tmp_path / "s.csv", good_rows())
    manual = tmp_path / "manual.csv"
    with manual.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "question_text", "intent", "should_escalate"])
        writer.writerow(
            ["R001", "cuanto vale el alquiler", "precio", "VERDADERO"]
        )  # igual
        writer.writerow(
            ["R002", "caben 80 personas?", "ubicacion", "FALSO"]
        )  # intención
        writer.writerow(
            ["R003", "texto distinto secreto", "disponibilidad", "True"]
        )  # texto
        writer.writerow(["R099", "fila extra", "precio", "True"])  # solo en el manual
    out = tmp_path / "labeled.csv"
    assert run_main(sheet, out, "--compare-with", str(manual)) == 0
    printed = capsys.readouterr().out
    assert "1 de 3 filas comunes son idénticas" in printed
    assert "R002" in printed and "R003" in printed and "R099" in printed
    assert "secreto" not in printed
