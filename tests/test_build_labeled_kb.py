"""Pruebas de build_labeled con base de conocimiento, filas pendientes y fusión."""

import csv
from types import SimpleNamespace

import pytest

from src.data import build_labeled as bl

ANONYMIZER = SimpleNamespace(has_names=True)

KB_FILES = {
    "reglas_mascotas": ("reglas_servicios", "# Mascotas\n\nSe permiten mascotas pequeñas.\nDeben ir con correa.\n"),
    "reglas_licor": ("reglas_servicios", "# Licor\n\nSe puede llevar licor propio."),
    "capacidad_personas": ("capacidad", "# Capacidad\n\nCaben 80 personas."),
    "ubicacion_direccion": ("ubicacion", "# Dirección\n\nVereda ejemplo, Ibagué."),
    "reglas_cancelaciones": ("reglas_servicios", "# Cancelaciones\n\nAnticipo del 30 %. [POR CONFIRMAR] plazo."),
    "precio_general": ("precio", "# Precio\n\nDepende de la fecha."),
}


def make_kb(folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for kb_id, (intent, body) in KB_FILES.items():
        (folder / f"{kb_id}.md").write_text(
            f"---\nid: {kb_id}\nintent: {intent}\nvigente_desde: 2026-11-03\n---\n\n{body}",
            encoding="utf-8",
        )


def sheet_row(number: int, intent: str = "", kb_source: str = "", dudosa: str = "") -> dict[str, str]:
    return {
        "question_id": f"R{number:03d}",
        "question_text": f"pregunta {number}",
        "intent": intent,
        "kb_source": kb_source,
        "should_escalate": "",
        "dudosa": dudosa,
        "pair_id": f"P{number:05d}",
    }


def write_sheet(path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def go(tmp_path, sheet_name="hoja.csv", **kwargs) -> int:
    kwargs.setdefault("kb_dir", tmp_path / "kb")
    return bl.run(
        tmp_path / sheet_name,
        tmp_path / "labeled.csv",
        anonymizer=ANONYMIZER,
        **kwargs,
    )


def setup(tmp_path, monkeypatch, rows, name="hoja.csv") -> None:
    monkeypatch.setattr(bl, "check_file", lambda *args, **kwargs: [])
    make_kb(tmp_path / "kb")
    write_sheet(tmp_path / name, rows)


# ---------- base de conocimiento ----------

def test_load_kb_parses_header_and_strips_title(tmp_path) -> None:
    make_kb(tmp_path)
    kb = bl.load_kb(tmp_path)
    assert kb["reglas_mascotas"].intent == "reglas_servicios"
    assert kb["reglas_mascotas"].since == "2026-11-03"
    assert kb["reglas_mascotas"].body == "Se permiten mascotas pequeñas. Deben ir con correa."


def test_load_kb_reports_malformed_fragments(tmp_path) -> None:
    (tmp_path / "sin_encabezado.md").write_text("# Solo texto\n\nhola", encoding="utf-8")
    (tmp_path / "id_distinto.md").write_text("---\nid: otro\nintent: capacidad\n---\ntexto", encoding="utf-8")
    (tmp_path / "intencion_rara.md").write_text("---\nid: intencion_rara\nintent: magia\n---\ntexto", encoding="utf-8")
    (tmp_path / "vacio.md").write_text("---\nid: vacio\nintent: capacidad\n---\n# Título\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        bl.load_kb(tmp_path)
    message = str(error.value)
    for name in ("sin_encabezado.md", "id_distinto.md", "intencion_rara.md", "vacio.md"):
        assert name in message


def test_load_kb_missing_or_empty_folder(tmp_path) -> None:
    with pytest.raises(ValueError, match="no se encontró"):
        bl.load_kb(tmp_path / "no_existe")
    (tmp_path / "vacia").mkdir()
    with pytest.raises(ValueError, match="no hay fragmentos"):
        bl.load_kb(tmp_path / "vacia")


# ---------- reglas por fila ----------

def kb_of(tmp_path):
    make_kb(tmp_path)
    return bl.load_kb(tmp_path)


def test_kb_source_derives_intent_and_reference(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, kb_source="capacidad_personas")], kb_of(tmp_path))
    row = built.labeled[0]
    assert (row["intent"], row["should_escalate"], row["kb_source"]) == ("capacidad", "False", "capacidad_personas")
    assert row["reference_answer"] == "Caben 80 personas."
    assert not built.intent_kb_mismatch


def test_kb_source_accepts_md_extension_and_spaces(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, kb_source=" reglas_licor.md ")], kb_of(tmp_path))
    assert built.labeled[0]["kb_source"] == "reglas_licor"


def test_other_intent_with_kb_source_warns_and_uses_fragment(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, "ubicacion", "reglas_licor")], kb_of(tmp_path))
    assert built.labeled[0]["intent"] == "reglas_servicios"
    assert built.intent_kb_mismatch == ["R001"]


def test_unknown_kb_source_is_blocking(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, kb_source="no_existe")], kb_of(tmp_path))
    assert not built.labeled and [q for q, _ in built.kb_unknown] == ["R001"]
    assert bl._blocking_problems(built)


def test_fragment_that_escalates_is_rejected(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, kb_source="precio_general")], kb_of(tmp_path))
    assert built.kb_escalating == ["R001"] and not built.labeled


def test_escalating_intent_with_kb_source_is_a_conflict(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, "precio", "reglas_licor")], kb_of(tmp_path))
    assert built.intent_kb_conflict == ["R001"] and not built.labeled


def test_note_with_kb_source_is_a_conflict(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, kb_source="reglas_licor", dudosa="depende")], kb_of(tmp_path))
    assert built.conflicts == ["R001"]


def test_escalating_intent_gets_escalar_and_no_kb_source(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, "disponibilidad")], kb_of(tmp_path))
    row = built.labeled[0]
    assert (row["reference_answer"], row["kb_source"], row["should_escalate"]) == ("ESCALAR", "", "True")


def test_answerable_without_kb_source_is_incomplete(tmp_path) -> None:
    built = bl.build_labeled([sheet_row(1, "capacidad")], kb_of(tmp_path))
    assert built.incomplete == ["R001"] and not built.labeled
    assert bl._blocking_problems(built)
    assert not bl._blocking_problems(built, allow_pending=True)


def test_empty_row_blocks_unless_pending_allowed(tmp_path) -> None:
    kb = kb_of(tmp_path)
    strict = bl.build_labeled([sheet_row(1)], kb)
    assert strict.unlabeled == ["R001"] and bl._blocking_problems(strict)
    lenient = bl.build_labeled([sheet_row(1)], kb, allow_pending=True)
    assert lenient.unlabeled == ["R001"] and not lenient.discarded
    assert not bl._blocking_problems(lenient, allow_pending=True)


def test_notes_still_discard_with_reason(tmp_path) -> None:
    built = bl.build_labeled(
        [sheet_row(1, dudosa="#descartada"), sheet_row(2, dudosa="depende del chat anterior")],
        kb_of(tmp_path),
        allow_pending=True,
    )
    assert {r["question_id"]: r["motivo"] for r in built.discarded} == {
        "R001": "sin_sentido",
        "R002": "depende_contexto",
    }


def test_without_kb_the_previous_behavior_is_kept() -> None:
    built = bl.build_labeled([sheet_row(1, "capacidad")])
    assert built.labeled[0]["reference_answer"] == "[POR CONFIRMAR]"
    assert not built.incomplete


def test_compact_ids() -> None:
    assert bl.compact_ids(["R251", "R252", "R253", "R300", "R10", "R11"]) == "R10-R11, R251-R253, R300"
    assert bl.compact_ids([]) == ""


# ---------- run: archivos y avisos ----------

def test_run_writes_files_and_reports_pending(tmp_path, monkeypatch, capsys) -> None:
    rows = [
        sheet_row(1, kb_source="capacidad_personas"),
        sheet_row(2, "precio"),
        sheet_row(3, dudosa="#descartada"),
        sheet_row(4),
        sheet_row(5),
        sheet_row(6),
        sheet_row(8, "reglas_servicios"),
    ]
    setup(tmp_path, monkeypatch, rows)
    assert go(tmp_path) == 2  # sin --allow-pending se detiene
    assert not (tmp_path / "labeled.csv").exists()
    capsys.readouterr()
    assert go(tmp_path, allow_pending=True) == 0
    out = capsys.readouterr().out
    assert "Vacías (sin intención, sin kb_source y sin nota): 3 (R004-R006)" in out
    assert "Incompletas (respondibles sin kb_source): 1 (R008)" in out
    labeled = read_csv(tmp_path / "labeled.csv")
    assert [r["question_id"] for r in labeled] == ["R001", "R002"]
    assert [r["question_id"] for r in read_csv(tmp_path / "labeled_discarded.csv")] == ["R003"]


def test_run_refuses_existing_output_without_merge(tmp_path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch, [sheet_row(1, "precio")])
    assert go(tmp_path) == 0
    assert go(tmp_path) == 2


def test_run_rejects_incompatible_flags(tmp_path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch, [sheet_row(1, "precio")])
    assert go(tmp_path, merge=True, force=True) == 2
    assert go(tmp_path, allow_relabel=True) == 2


def test_pii_hit_writes_nothing(tmp_path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch, [sheet_row(1, "precio")])
    hit = SimpleNamespace(file="x", row=1, row_id="R001", column="question_text", tag="[TEL]", count=1)
    monkeypatch.setattr(bl, "check_file", lambda *args, **kwargs: [hit])
    assert go(tmp_path) == 1
    assert not (tmp_path / "labeled.csv").exists()


# ---------- fusión ----------

def with_splits(tmp_path) -> list[dict[str, str]]:
    """Lote 1 ya fusionado y con split asignado a mano."""
    labeled = read_csv(tmp_path / "labeled.csv")
    for index, row in enumerate(labeled):
        row["split"] = "test" if index % 2 else "train"
    with (tmp_path / "labeled.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bl.LABELED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(labeled)
    return labeled


def first_batch(tmp_path, monkeypatch) -> list[dict[str, str]]:
    rows = [sheet_row(1, "otro_negocio"), sheet_row(2, "precio"), sheet_row(3, "disponibilidad")]
    setup(tmp_path, monkeypatch, rows)
    assert go(tmp_path) == 0
    return with_splits(tmp_path)


def test_merge_adds_new_rows_without_touching_split(tmp_path, monkeypatch, capsys) -> None:
    before = first_batch(tmp_path, monkeypatch)
    write_sheet(tmp_path / "lote2.csv", [sheet_row(4, kb_source="reglas_licor"), sheet_row(5, "ubicacion"), sheet_row(6)])
    capsys.readouterr()
    assert go(tmp_path, "lote2.csv", merge=True, allow_pending=True) == 0
    after = read_csv(tmp_path / "labeled.csv")
    assert after[:3] == before
    assert [r["question_id"] for r in after[3:]] == ["R004"]  # R005 es respondible sin fragmento
    assert after[3]["split"] == "" and after[3]["intent"] == "reglas_servicios"
    out = capsys.readouterr().out
    assert "nuevas 1" in out and "Filas sin split" in out and "R004" in out
    assert "Incompletas (respondibles sin kb_source): 1 (R005)" in out
    assert "Vacías (sin intención, sin kb_source y sin nota): 1 (R006)" in out


def test_merge_rerun_on_same_sheet_only_adds_what_is_new(tmp_path, monkeypatch, capsys) -> None:
    first_batch(tmp_path, monkeypatch)
    sheet = [sheet_row(4, kb_source="reglas_licor"), sheet_row(5)]
    write_sheet(tmp_path / "lote2.csv", sheet)
    go(tmp_path, "lote2.csv", merge=True, allow_pending=True)
    sheet[1]["intent"] = "precio"  # se etiqueta R005 y se vuelve a ejecutar
    write_sheet(tmp_path / "lote2.csv", sheet)
    capsys.readouterr()
    assert go(tmp_path, "lote2.csv", merge=True, allow_pending=True) == 0
    assert "nuevas 1; actualizadas 0; sin cambios 1" in capsys.readouterr().out
    assert [r["question_id"] for r in read_csv(tmp_path / "labeled.csv")][-2:] == ["R004", "R005"]


def test_merge_blocks_label_changes_without_allow_relabel(tmp_path, monkeypatch, capsys) -> None:
    first_batch(tmp_path, monkeypatch)
    snapshot = (tmp_path / "labeled.csv").read_bytes()
    write_sheet(tmp_path / "lote1b.csv", [sheet_row(1, kb_source="reglas_cancelaciones")])
    capsys.readouterr()
    assert go(tmp_path, "lote1b.csv", merge=True) == 2
    assert "falta --allow-relabel" in capsys.readouterr().out
    assert (tmp_path / "labeled.csv").read_bytes() == snapshot


def test_merge_with_allow_relabel_updates_only_label_fields(tmp_path, monkeypatch, capsys) -> None:
    before = first_batch(tmp_path, monkeypatch)
    write_sheet(tmp_path / "lote1b.csv", [sheet_row(1, kb_source="reglas_cancelaciones")])
    capsys.readouterr()
    assert go(tmp_path, "lote1b.csv", merge=True, allow_relabel=True) == 0
    out = capsys.readouterr().out
    assert "R001: intent otro_negocio -> reglas_servicios" in out
    assert "kb_source (vacío) -> reglas_cancelaciones" in out
    assert "[POR CONFIRMAR]" in out  # el fragmento de cancelaciones lo trae: debe avisar
    after = read_csv(tmp_path / "labeled.csv")
    assert after[1:] == before[1:]  # las demás filas, intactas
    changed = after[0]
    assert changed["split"] == before[0]["split"] and changed["source"] == before[0]["source"]
    assert changed["question_text"] == before[0]["question_text"]
    assert changed["should_escalate"] == "False" and changed["reference_answer"].startswith("Anticipo")


def test_merge_never_changes_question_text(tmp_path, monkeypatch, capsys) -> None:
    first_batch(tmp_path, monkeypatch)
    row = sheet_row(1, "otro_negocio")
    row["question_text"] = "texto cambiado"
    write_sheet(tmp_path / "lote1b.csv", [row])
    capsys.readouterr()
    assert go(tmp_path, "lote1b.csv", merge=True, allow_relabel=True) == 2
    assert "texto de la pregunta no coincide" in capsys.readouterr().out


def test_merge_rejects_row_that_is_now_discarded(tmp_path, monkeypatch) -> None:
    first_batch(tmp_path, monkeypatch)
    write_sheet(tmp_path / "lote1b.csv", [sheet_row(1, dudosa="#descartada"), sheet_row(9, "precio")])
    assert go(tmp_path, "lote1b.csv", merge=True, allow_relabel=True) == 2


def test_merge_updates_discarded_file(tmp_path, monkeypatch, capsys) -> None:
    rows = [sheet_row(1, "precio"), sheet_row(2, dudosa="#descartada")]
    setup(tmp_path, monkeypatch, rows)
    assert go(tmp_path) == 0
    # R002 estaba descartada y ahora se etiqueta; R003 es una descartada nueva.
    write_sheet(
        tmp_path / "lote2.csv",
        [sheet_row(2, kb_source="capacidad_personas"), sheet_row(3, dudosa="necesita contexto")],
    )
    capsys.readouterr()
    assert go(tmp_path, "lote2.csv", merge=True) == 0
    assert "Salieron de las descartadas" in capsys.readouterr().out
    assert [r["question_id"] for r in read_csv(tmp_path / "labeled.csv")] == ["R001", "R002"]
    discarded = read_csv(tmp_path / "labeled_discarded.csv")
    assert [(r["question_id"], r["motivo"]) for r in discarded] == [("R003", "depende_contexto")]


def test_merge_discarded_helper() -> None:
    old = [{"question_id": "R002", "motivo": "sin_sentido", "dudosa": "x", "pair_id": "P2"}]
    new = [{"question_id": "R003", "motivo": "depende_contexto", "dudosa": "y", "pair_id": "P3"}]
    rows, recovered = bl.merge_discarded(old, new, {"R002"})
    assert [r["question_id"] for r in rows] == ["R003"] and recovered == ["R002"]


def test_merge_creates_labeled_if_it_does_not_exist(tmp_path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch, [sheet_row(1, "precio")])
    assert go(tmp_path, merge=True) == 0
    assert [r["question_id"] for r in read_csv(tmp_path / "labeled.csv")] == ["R001"]


def test_merge_keeps_rows_missing_from_the_sheet(tmp_path, monkeypatch) -> None:
    before = first_batch(tmp_path, monkeypatch)
    write_sheet(tmp_path / "lote2.csv", [sheet_row(4, "ubicacion")])
    go(tmp_path, "lote2.csv", merge=True, kb_dir=None)
    assert read_csv(tmp_path / "labeled.csv")[:3] == before


def test_merge_treats_excel_booleans_as_equal(tmp_path, monkeypatch, capsys) -> None:
    first_batch(tmp_path, monkeypatch)
    text = (tmp_path / "labeled.csv").read_text(encoding="utf-8").replace(",True,", ",VERDADERO,")
    (tmp_path / "labeled.csv").write_text(text, encoding="utf-8")
    write_sheet(tmp_path / "lote1b.csv", [sheet_row(1, "otro_negocio")])
    capsys.readouterr()
    assert go(tmp_path, "lote1b.csv", merge=True) == 0
    assert "sin cambios 1" in capsys.readouterr().out
