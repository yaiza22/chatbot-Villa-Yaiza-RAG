"""Pruebas de la generación incremental de hojas de etiquetado."""

import csv
from types import SimpleNamespace

import pytest

from src.data import prepare_labeling as pl

ANONYMIZER = SimpleNamespace(has_names=True)
N_CHATS = 30


def no_pii(monkeypatch) -> None:
    """Desactiva check_pii (se prueba aparte); aquí solo importa el flujo de lotes."""
    monkeypatch.setattr(pl, "check_file", lambda *args, **kwargs: [])


def write_messages(path, chats: int = N_CHATS, skip_first: int = 0) -> None:
    """Un cliente que pregunta y la administración que responde, por chat."""
    rows = []
    for number in range(1 + skip_first, chats + 1):
        chat = f"chat{number:02d}"
        rows.append(
            {
                "msg_id": f"m{number}a",
                "chat_id": chat,
                "timestamp": "2026-01-01T10:00:00",
                "sender_role": "cliente",
                "text": f"cuanto vale el alquiler del plan {number}?",
            }
        )
        rows.append(
            {
                "msg_id": f"m{number}b",
                "chat_id": chat,
                "timestamp": "2026-01-01T10:05:00",
                "sender_role": "admin",
                "text": f"respuesta numero {number}",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["msg_id", "chat_id", "timestamp", "sender_role", "text"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def go(tmp_path, **kwargs) -> int:
    return pl.run(tmp_path / "messages.csv", tmp_path, anonymizer=ANONYMIZER, **kwargs)


def test_first_run_creates_qa_and_batch1(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    assert go(tmp_path, sample_size=10) == 0
    assert len(read_csv(tmp_path / "qa_pairs.csv")) == N_CHATS
    sheet = read_csv(tmp_path / "labeling_sheet_batch1.csv")
    assert [row["question_id"] for row in sheet] == [f"R{n:03d}" for n in range(1, 11)]
    assert {"intent", "kb_source", "dudosa"} <= set(sheet[0])
    assert all(row["intent"] == row["kb_source"] == row["dudosa"] == "" for row in sheet)


def test_second_batch_is_disjoint_and_continues_numbering(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    first_bytes = (tmp_path / "labeling_sheet_batch1.csv").read_bytes()
    assert go(tmp_path, sample_size=8) == 0
    batch1 = read_csv(tmp_path / "labeling_sheet_batch1.csv")
    batch2 = read_csv(tmp_path / "labeling_sheet_batch2.csv")
    assert not {r["pair_id"] for r in batch1} & {r["pair_id"] for r in batch2}
    assert [r["question_id"] for r in batch2] == [f"R{n:03d}" for n in range(11, 19)]
    assert (tmp_path / "labeling_sheet_batch1.csv").read_bytes() == first_bytes


def test_all_remaining_takes_the_rest(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    assert go(tmp_path, all_remaining=True) == 0
    batch2 = read_csv(tmp_path / "labeling_sheet_batch2.csv")
    assert len(batch2) == N_CHATS - 10
    used = {r["pair_id"] for r in read_csv(tmp_path / "labeling_sheet_batch1.csv")}
    assert not used & {r["pair_id"] for r in batch2}


def test_nothing_left_fails_with_clear_message(tmp_path, monkeypatch, capsys) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, all_remaining=True)
    capsys.readouterr()
    assert go(tmp_path, sample_size=5) == 2
    assert "no queda nada" in capsys.readouterr().out
    assert not (tmp_path / "labeling_sheet_batch2.csv").exists()


def test_sample_larger_than_pool_takes_everything_with_notice(tmp_path, monkeypatch, capsys) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=25)
    capsys.readouterr()
    assert go(tmp_path, sample_size=100) == 0
    assert "solo quedan 5" in capsys.readouterr().out
    assert len(read_csv(tmp_path / "labeling_sheet_batch2.csv")) == 5


def test_existing_qa_is_read_and_messages_are_not(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    qa_before = (tmp_path / "qa_pairs.csv").read_bytes()
    (tmp_path / "messages.csv").unlink()  # si lo leyera, fallaría
    assert go(tmp_path, sample_size=5) == 0
    assert (tmp_path / "qa_pairs.csv").read_bytes() == qa_before


def test_rebuild_qa_recreates_it_when_ids_still_match(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    (tmp_path / "qa_pairs.csv").write_text("basura", encoding="utf-8")
    assert go(tmp_path, rebuild_qa=True) == 0
    assert len(read_csv(tmp_path / "qa_pairs.csv")) == N_CHATS
    assert not (tmp_path / "labeling_sheet_batch2.csv").exists()  # sin cantidad, no hay hoja


def test_rebuild_qa_aborts_if_ids_would_shift(tmp_path, monkeypatch, capsys) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    qa_before = (tmp_path / "qa_pairs.csv").read_bytes()
    write_messages(tmp_path / "messages.csv", skip_first=1)  # cambian los pair_id
    capsys.readouterr()
    assert go(tmp_path, rebuild_qa=True) == 2
    assert "No se escribió ningún archivo" in capsys.readouterr().out
    assert (tmp_path / "qa_pairs.csv").read_bytes() == qa_before


def test_sheet_that_does_not_match_qa_aborts(tmp_path, monkeypatch, capsys) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    sheet_path = tmp_path / "labeling_sheet_batch1.csv"
    text = sheet_path.read_text(encoding="utf-8").replace("plan 1", "plan 999", 1)
    sheet_path.write_text(text, encoding="utf-8")
    capsys.readouterr()
    assert go(tmp_path, sample_size=5) == 2
    out = capsys.readouterr().out
    assert "texto distinto" in out and "labeling_sheet_batch1.csv" in out
    assert not (tmp_path / "labeling_sheet_batch2.csv").exists()


def test_unknown_pair_id_in_a_sheet_aborts(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    sheet_path = tmp_path / "labeling_sheet_batch1.csv"
    sheet_path.write_text(
        sheet_path.read_text(encoding="utf-8").replace("P00001", "P99999"), encoding="utf-8"
    )
    assert go(tmp_path, sample_size=5) == 2


def test_same_seed_gives_same_sample(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    ids = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        write_messages(folder / "messages.csv")
        pl.run(folder / "messages.csv", folder, anonymizer=ANONYMIZER, sample_size=12)
        ids.append([r["pair_id"] for r in read_csv(folder / "labeling_sheet_batch1.csv")])
    assert ids[0] == ids[1]


def test_etiquetadas_file_is_not_counted_as_a_sheet(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    sheet = tmp_path / "labeling_sheet_batch1.csv"
    (tmp_path / "labeling_sheet_batch1_etiquetadas.csv").write_bytes(sheet.read_bytes())
    assert go(tmp_path, sample_size=5) == 0
    assert (tmp_path / "labeling_sheet_batch2.csv").exists()


def test_old_sheet_without_kb_source_column_is_still_readable(tmp_path, monkeypatch) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    go(tmp_path, sample_size=10)
    sheet = tmp_path / "labeling_sheet_batch1.csv"
    rows = read_csv(sheet)
    columns = [c for c in rows[0] if c != "kb_source"]
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:  # Excel suele agregar BOM
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    assert go(tmp_path, sample_size=5) == 0
    #assert [r["question_id"] for r in read_csv(tmp_path / "labeling_sheet_batch2.csv")][0] == "R011"
    assert read_csv(tmp_path / "labeling_sheet_batch2.csv")[0]["question_id"] == "R011"


def test_incompatible_or_missing_options(tmp_path, monkeypatch, capsys) -> None:
    no_pii(monkeypatch)
    write_messages(tmp_path / "messages.csv")
    assert go(tmp_path, sample_size=5, all_remaining=True) == 2
    go(tmp_path, sample_size=5)
    capsys.readouterr()
    assert go(tmp_path) == 2  # qa existe y no se pidió hoja ni --rebuild-qa
    assert "Nada que hacer" in capsys.readouterr().out
    other = tmp_path / "otra"
    assert pl.run(other / "messages.csv", other, anonymizer=ANONYMIZER, sample_size=5) == 2


def test_pii_hit_writes_nothing(tmp_path, monkeypatch) -> None:
    hit = SimpleNamespace(file="x.csv", row=1, row_id="R001", column="question_text", tag="[TEL]", count=1)
    monkeypatch.setattr(pl, "check_file", lambda *args, **kwargs: [hit])
    write_messages(tmp_path / "messages.csv")
    assert go(tmp_path, sample_size=5) == 1
    assert not (tmp_path / "qa_pairs.csv").exists()
    assert not (tmp_path / "labeling_sheet_batch1.csv").exists()


def test_build_sheet_continues_numbering_and_rotates_responsables() -> None:
    sample = [{"pair_id": f"P{n}", "question_text": "q", "answer_text": ""} for n in range(4)]
    sheet = pl.build_sheet(sample, ("YS", "CM"), start=251)
    assert [r["question_id"] for r in sheet] == ["R251", "R252", "R253", "R254"]
    assert [r["responsable"] for r in sheet] == ["YS", "CM", "YS", "CM"]
    with pytest.raises(ValueError):
        pl.build_sheet(sample, start=0)
