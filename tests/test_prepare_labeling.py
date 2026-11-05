# Pruebas de ``prepare_labeling`` con mensajes sintéticos.

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data import prepare_labeling as pl

BASE = datetime(2025, 3, 1, 9, 0)


def msg(chat: str, minute: float, role: str, text: str) -> dict[str, str]:
    # Una fila de messages.csv; ``minute`` son minutos desde las 9:00.
    stamp = (BASE + timedelta(minutes=minute)).isoformat()
    return {
        "msg_id": f"M{chat}{minute}",
        "chat_id": chat,
        "timestamp": stamp,
        "sender_role": role,
        "text": text,
    }


def write_messages(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pl.REQUIRED_MESSAGE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def many_chats(count: int = 40) -> list[dict[str, str]]:
    rows = []
    for i in range(count):
        chat = f"c{i:03d}"
        rows.append(msg(chat, 0, "cliente", f"cuanto vale para {i + 10} personas"))
        rows.append(msg(chat, 5, "admin", f"el valor depende de la fecha {i}"))
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- Filtro de mensajes ---------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Hola",
        "Buenas tardes",
        "holaaa buenos dias",
        "Gracias",
        "muchas gracias, quedo pendiente",
        "ok",
        "Okkk listo",
        "Perfecto, gracias!",
        "jajaja",
        "👍",
        "[URL]",
        "[NOMBRE] gracias",
        "",
        "   ",
        "<Multimedia omitido>",
        "imagen omitida",
        "Se eliminó este mensaje",
    ],
)
def test_classify_drops_non_questions(text: str) -> None:
    assert pl.classify_message(text) != pl.VERDICT_KEEP


@pytest.mark.parametrize(
    "text",
    [
        "cuanto vale el alquiler?",
        "cuanto vale el alquiler",
        "Hola buenas tardes, me cotiza para 40 personas",
        "ya consigné [CUENTA]",
        "listo?",
        "tienen mascotas",
        "24 de dic",
    ],
)
def test_classify_keeps_possible_questions(text: str) -> None:
    assert pl.classify_message(text) == pl.VERDICT_KEEP


def test_classify_ignores_edited_mark() -> None:
    assert pl.classify_message("<Se editó este mensaje.>") == pl.VERDICT_EMPTY


# --- Emparejamiento -------------------------------------------------------------------


def test_pairs_question_with_next_admin_reply() -> None:
    rows = [
        msg("a", 0, "cliente", "Hola buenas tardes"),
        msg("a", 1, "cliente", "cuanto vale el alquiler"),
        msg("a", 3, "admin", "Buenas tardes, el valor depende de la fecha"),
        msg("a", 4, "admin", "Para cuantas personas?"),
    ]
    pairs = pl.build_pairs(rows)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_id"] == "P00001"
    assert pair["question_text"] == "cuanto vale el alquiler"
    assert pair["answer_text"] == (
        "Buenas tardes, el valor depende de la fecha Para cuantas personas?"
    )
    assert pair["timestamp"] == rows[1]["timestamp"]


def test_consecutive_client_messages_form_one_question() -> None:
    rows = [
        msg("a", 0, "cliente", "queremos reservar"),
        msg("a", 1, "cliente", "para el 24 de dic"),
        msg("a", 2, "cliente", "somos 40"),
        msg("a", 5, "admin", "claro, lo verificamos"),
    ]
    pairs = pl.build_pairs(rows)
    assert [p["question_text"] for p in pairs] == [
        "queremos reservar para el 24 de dic somos 40"
    ]


def test_long_gap_splits_questions_and_leaves_first_unanswered() -> None:
    rows = [
        msg("a", 0, "cliente", "cuanto vale"),
        msg("a", 120, "cliente", "tienen mascotas"),
        msg("a", 125, "admin", "si se permiten"),
    ]
    pairs = pl.build_pairs(rows, max_gap_min=10)
    assert [p["answer_text"] for p in pairs] == ["", "si se permiten"]


def test_thanks_after_answer_closes_the_exchange() -> None:
    rows = [
        msg("a", 0, "cliente", "a que hora es el check in"),
        msg("a", 1, "admin", "a las 3 pm"),
        msg("a", 2, "cliente", "gracias"),
        msg("a", 3, "admin", "con gusto, buen dia"),
    ]
    pairs = pl.build_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["answer_text"] == "a las 3 pm"


def test_admin_first_message_and_media_are_ignored() -> None:
    rows = [
        msg("a", 0, "admin", "recordatorio de su reserva"),
        msg("a", 1, "cliente", "<Multimedia omitido>"),
        msg("a", 2, "cliente", "donde queda la finca"),
        msg("a", 3, "admin", "<Multimedia omitido>"),
    ]
    stats = pl.Stats()
    pairs = pl.build_pairs(rows, stats=stats)
    assert len(pairs) == 1
    assert pairs[0]["answer_text"] == ""
    assert stats.dropped[pl.VERDICT_MEDIA] == 1
    assert stats.answered_pairs == 0


def test_chats_are_never_mixed() -> None:
    rows = [
        msg("a", 0, "cliente", "cuanto vale"),
        msg("b", 1, "cliente", "tienen piscina"),
        msg("b", 2, "admin", "si, hay piscina"),
        msg("a", 3, "admin", "depende de la fecha"),
    ]
    pairs = {p["chat_id"]: p for p in pl.build_pairs(rows)}
    assert pairs["a"]["answer_text"] == "depende de la fecha"
    assert pairs["b"]["answer_text"] == "si, hay piscina"


def test_unparseable_timestamps_are_counted_and_still_merge() -> None:
    rows = [
        msg("a", 0, "cliente", "cuanto vale"),
        msg("a", 1, "cliente", "y la cocina"),
    ]
    for row in rows:
        row["timestamp"] = "01/03/25 9:00 a. m."
    stats = pl.Stats()
    pairs = pl.build_pairs(rows, stats=stats)
    assert len(pairs) == 1
    assert stats.unparsed_timestamps == 2


# --- Muestra y hoja -------------------------------------------------------------------


def test_sample_is_reproducible_and_independent_of_input_order() -> None:
    pairs = pl.build_pairs(many_chats())
    first = pl.sample_pairs(pairs, 15, seed=42)
    again = pl.sample_pairs(list(reversed(pairs)), 15, seed=42)
    other = pl.sample_pairs(pairs, 15, seed=7)
    assert [p["pair_id"] for p in first] == [p["pair_id"] for p in again]
    assert [p["pair_id"] for p in first] != [p["pair_id"] for p in other]
    assert len({p["pair_id"] for p in first}) == 15


def test_sample_larger_than_pool_returns_everything() -> None:
    pairs = pl.build_pairs(many_chats(5))
    assert len(pl.sample_pairs(pairs, 250)) == 5


def test_sheet_columns_ids_and_empty_fields() -> None:
    sheet = pl.build_sheet(pl.sample_pairs(pl.build_pairs(many_chats()), 12))
    assert list(sheet[0]) == list(pl.SHEET_COLUMNS)
    assert [row["question_id"] for row in sheet] == [f"R{i:03d}" for i in range(1, 13)]
    for row in sheet:
        assert row["intent"] == row["should_escalate"] == row["dudosa"] == ""
        assert row["responsable"] == "YS"


def test_sheet_distributes_rows_evenly_among_three() -> None:
    sheet = pl.build_sheet(
        pl.sample_pairs(pl.build_pairs(many_chats()), 25), ["YS", "NG", "CM"]
    )
    counts = [
        sum(1 for r in sheet if r["responsable"] == who) for who in ("YS", "NG", "CM")
    ]
    assert sum(counts) == 25
    assert max(counts) - min(counts) <= 1


# --- Ejecución completa ---------------------------------------------------------------


def run_main(messages: Path, out_dir: Path, *extra: str) -> int:
    return pl.main(
        [
            "--messages",
            str(messages),
            "--out-dir",
            str(out_dir),
            "--sample-size",
            "10",
            *extra,
        ]
    )


def test_end_to_end_is_reproducible(tmp_path: Path) -> None:
    messages = write_messages(tmp_path / "messages.csv", many_chats())
    assert run_main(messages, tmp_path / "out1") == 0
    assert run_main(messages, tmp_path / "out2") == 0
    for name in (pl.QA_FILENAME, pl.SHEET_FILENAME):
        assert (tmp_path / "out1" / name).read_bytes() == (
            tmp_path / "out2" / name
        ).read_bytes()
    qa = read_rows(tmp_path / "out1" / pl.QA_FILENAME)
    sheet = read_rows(tmp_path / "out1" / pl.SHEET_FILENAME)
    assert len(qa) == 40
    assert len(sheet) == 10
    assert {r["pair_id"] for r in sheet} <= {r["pair_id"] for r in qa}
    assert not list((tmp_path / "out1").glob(".tmp_*"))


def test_existing_sheet_is_not_overwritten_without_force(tmp_path: Path) -> None:
    messages = write_messages(tmp_path / "messages.csv", many_chats())
    out = tmp_path / "out"
    assert run_main(messages, out) == 0
    sheet_path = out / pl.SHEET_FILENAME
    sheet_path.write_text("etiquetas hechas a mano", encoding="utf-8")
    assert run_main(messages, out) == 2
    assert sheet_path.read_text(encoding="utf-8") == "etiquetas hechas a mano"
    assert run_main(messages, out, "--force") == 0
    assert sheet_path.read_text(encoding="utf-8") != "etiquetas hechas a mano"


def test_personal_data_blocks_all_output_and_is_never_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = many_chats(10)
    rows.append(msg("zz", 0, "cliente", "escribanme a persona@example.com cuanto vale"))
    messages = write_messages(tmp_path / "messages.csv", rows)
    out = tmp_path / "out"
    assert run_main(messages, out) == 1
    printed = capsys.readouterr().out
    assert "FALLA" in printed
    assert "persona@example.com" not in printed
    assert not (out / pl.QA_FILENAME).exists()
    assert not (out / pl.SHEET_FILENAME).exists()


def test_missing_columns_fail_with_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "messages.csv"
    bad.write_text("msg_id,text\n1,hola\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        run_main(bad, tmp_path / "out")
    assert excinfo.value.code == 2
