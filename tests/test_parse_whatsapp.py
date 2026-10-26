# Tests del parser de exports de WhatsApp (solo datos sintéticos).

import csv
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

from src.data import parse_whatsapp as pw

FIXTURES = Path(__file__).parent / "fixtures"
ANDROID = FIXTURES / "chat_android_sintetico.txt"
IOS = FIXTURES / "chat_ios_sintetico.txt"
ADMIN = ["Admin Prueba"]


def _parse(path: Path) -> pw.ParsedChat:
    return pw.parse_chat_lines(pw.read_chat_lines(path))


def _kinds(chat: pw.ParsedChat) -> Counter[str]:
    return Counter(message.kind for message in chat.messages)


def _texts(chat: pw.ParsedChat) -> list[str]:
    return [m.text for m in chat.messages if m.kind == pw.KIND_TEXT]


# --- Detección de formato ----------------------------------------------------
def test_detect_format_android_and_ios() -> None:
    assert pw.detect_format(pw.read_chat_lines(ANDROID)) == pw.FORMAT_ANDROID
    assert pw.detect_format(pw.read_chat_lines(IOS)) == pw.FORMAT_IOS


def test_detect_format_unknown_raises() -> None:
    with pytest.raises(ValueError, match="formato"):
        pw.detect_format(["esto no es un chat", "ni esto"])


# --- Clasificación por tipo --------------------------------------------------
def test_android_counts_by_kind() -> None:
    kinds = _kinds(_parse(ANDROID))
    assert kinds == {
        pw.KIND_TEXT: 12,
        pw.KIND_SYSTEM: 1,
        pw.KIND_IMAGE: 1,
        pw.KIND_AUDIO: 1,
        pw.KIND_MEDIA: 1,
        pw.KIND_CALL: 1,
        pw.KIND_DELETED: 1,
    }


def test_ios_counts_by_kind() -> None:
    kinds = _kinds(_parse(IOS))
    assert kinds == {
        pw.KIND_TEXT: 8,
        pw.KIND_SYSTEM: 1,
        pw.KIND_AUDIO: 1,
        pw.KIND_IMAGE: 2,
        pw.KIND_VIDEO: 1,
        pw.KIND_CALL: 2,
        pw.KIND_DOCUMENT: 1,
        pw.KIND_DELETED: 1,
    }


def test_ios_system_message_with_sender_prefix_is_system() -> None:
    first = _parse(IOS).messages[0]
    assert first.kind == pw.KIND_SYSTEM
    assert first.text == ""


def test_text_that_mentions_a_call_or_a_file_is_still_text() -> None:
    lines = [
        "[01/05/2024, 10:00:00] Cliente: Llamada por favor, necesito información",
        "[01/05/2024, 10:01:00] Cliente: te mando el .pdf mañana",
    ]
    chat = pw.parse_chat_lines(lines)
    assert _kinds(chat) == {pw.KIND_TEXT: 2}


# --- Multilínea, pie de foto y mensajes editados -----------------------------
def test_multiline_messages_are_joined() -> None:
    texts = _texts(_parse(ANDROID))
    assert "Somos como 40 personas\npara un cumpleaños" in texts
    assert "Una boda,\nmás o menos 120 invitados" in _texts(_parse(IOS))


def test_caption_of_an_attachment_is_kept_as_text() -> None:
    assert "Así está el grupo" in _texts(_parse(ANDROID))


def test_edited_marker_is_removed() -> None:
    texts = _texts(_parse(ANDROID))
    assert "Hola, claro, le confirmo ese detalle" in texts
    assert not any("Se editó" in text for text in texts)


def test_edited_marker_with_period_is_removed() -> None:
    lines = [
        "[20/04/2024, 8:03:40 a. m.] Cliente: Hola, me interesa la finca",
        "Gracias. <Se editó este mensaje.>",
    ]
    message = pw.parse_chat_lines(lines).messages[0]
    assert message.kind == pw.KIND_TEXT
    assert message.text == "Hola, me interesa la finca\nGracias."


# --- Fechas y horas ----------------------------------------------------------
def test_twelve_hour_clock_conversion() -> None:
    messages = {m.text: m.timestamp for m in _parse(ANDROID).messages if m.text}
    assert messages["Buenos días, con gusto. ¿Para cuántas personas sería?"] == (
        datetime(2024, 3, 12, 9, 17)
    )
    assert messages["gracias, quedo atento"] == datetime(2024, 3, 12, 23, 50)
    assert messages["Con gusto, buenas noches"] == datetime(2024, 3, 13, 0, 5)
    assert messages["Así está el grupo"] == datetime(2024, 3, 12, 12, 30)


def test_24_hour_clock_and_two_digit_year_in_ios() -> None:
    messages = {m.text: m.timestamp for m in _parse(IOS).messages if m.text}
    assert messages["ok gracias"] == datetime(2024, 4, 20, 14, 10, 0)
    assert messages["buen día, ya tienen respuesta?"] == datetime(2024, 4, 21, 9, 0, 0)


def test_hard_spaces_before_am_pm_are_supported() -> None:
    line = "12/03/2024, 3:40\u202fp.\u00a0m. - Cliente: hola"
    message = pw.parse_chat_lines([line]).messages[0]
    assert message.timestamp == datetime(2024, 3, 12, 15, 40)
    assert message.text == "hola"


def test_invisible_marks_at_line_start_are_ignored() -> None:
    lines = ["\u200e[20/04/2024, 8:03:40 a. m.] Cliente: \u200eimagen omitida"]
    message = pw.parse_chat_lines(lines).messages[0]
    assert message.kind == pw.KIND_IMAGE


def test_invalid_date_raises_without_leaking_content() -> None:
    lines = ["[20/04/2024, 8:03:40] Cliente: hola", "[31/02/2024, 8:03:40] Cliente: x"]
    with pytest.raises(ValueError, match="línea 2") as error:
        pw.parse_chat_lines(lines)
    assert "Cliente" not in str(error.value)


def test_lines_before_first_header_are_counted_as_orphans() -> None:
    lines = ["texto suelto", "", "[20/04/2024, 8:03:40] Cliente: hola"]
    chat = pw.parse_chat_lines(lines)
    assert chat.orphan_lines == 1
    assert len(chat.messages) == 1


# --- Roles y filas de salida -------------------------------------------------
def test_roles_use_configurable_names_case_insensitive() -> None:
    chat = _parse(ANDROID)
    rows = pw.messages_to_rows("chat_0001", chat.messages, ["  admin   PRUEBA "])
    roles = Counter(row["sender_role"] for row in rows)
    assert roles == {pw.ROLE_CLIENT: 7, pw.ROLE_ADMIN: 5}


def test_unknown_sender_is_client_by_default() -> None:
    chat = _parse(IOS)
    rows = pw.messages_to_rows("chat_0001", chat.messages, ["Otro Nombre"])
    assert {row["sender_role"] for row in rows} == {pw.ROLE_CLIENT}


def test_rows_only_include_text_messages_with_schema_columns() -> None:
    chat = _parse(IOS)
    rows = pw.messages_to_rows("chat_0007", chat.messages, ADMIN)
    assert len(rows) == 8
    assert all(tuple(row) == pw.CSV_COLUMNS for row in rows)
    assert rows[0]["msg_id"] == "chat_0007_00001"
    assert len({row["msg_id"] for row in rows}) == len(rows)


# --- Zip, dataset completo y CSV ---------------------------------------------
def test_zip_reads_only_the_txt(tmp_path: Path) -> None:
    archive_path = tmp_path / "chat.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(ANDROID, arcname="chat.txt")
        archive.writestr("IMG-20240312-WA0001.jpg", b"no-es-una-imagen-real")
    chat = pw.parse_chat_lines(pw.read_chat_lines(archive_path))
    assert _kinds(chat)[pw.KIND_TEXT] == 12


def test_zip_without_txt_raises(tmp_path: Path) -> None:
    archive_path = tmp_path / "vacio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("foto.jpg", b"x")
    with pytest.raises(ValueError, match=".txt"):
        pw.read_chat_lines(archive_path)


def test_build_dataset_assigns_neutral_sequential_chat_ids(tmp_path: Path) -> None:
    (tmp_path / "Chat de WhatsApp con Persona Inventada.txt").write_text(
        ANDROID.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "zz_otro.txt").write_text(IOS.read_text(encoding="utf-8"), "utf-8")
    dataset = pw.build_dataset(tmp_path, ADMIN)
    assert [chat_id for chat_id, _ in dataset.chat_index] == ["chat_0001", "chat_0002"]
    assert {row["chat_id"] for row in dataset.rows} == {"chat_0001", "chat_0002"}
    assert len(dataset.rows) == 20
    assert not any(
        "Persona Inventada" in value for row in dataset.rows for value in row.values()
    )


def test_build_dataset_reports_failed_chat_and_continues(tmp_path: Path) -> None:
    (tmp_path / "a_malo.txt").write_text("esto no es un chat\n", encoding="utf-8")
    (tmp_path / "b_bueno.txt").write_text(
        ANDROID.read_text(encoding="utf-8"), encoding="utf-8"
    )
    dataset = pw.build_dataset(tmp_path, ADMIN)
    assert len(dataset.stats.failed_chats) == 1
    assert dataset.stats.failed_chats[0].startswith("chat_0001")
    assert len(dataset.rows) == 12


def test_main_writes_csv_without_sender_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "interim" / "messages.csv"
    code = pw.main(
        ["--input", str(FIXTURES), "--output", str(output), "--admin", "Admin Prueba"]
    )
    assert code == 0
    raw = output.read_text(encoding="utf-8")
    assert "Admin Prueba" not in raw
    assert "Cliente Uno" not in raw
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == pw.CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 20
    assert "Somos como 40 personas\npara un cumpleaños" in {r["text"] for r in rows}
    assert (output.parent / "chat_index.csv").exists()
    summary = capsys.readouterr().out
    assert "Filas escritas: 20" in summary
    assert "Somos como" not in summary  # el resumen no muestra contenido


def test_main_requires_admin_name(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        pw.main(["--input", str(FIXTURES), "--output", str(tmp_path / "m.csv")])


def test_main_accepts_admin_file(tmp_path: Path) -> None:
    admin_file = tmp_path / "admin.txt"
    admin_file.write_text("# nombres de la finca\nAdmin Prueba\n", encoding="utf-8")
    output = tmp_path / "m.csv"
    code = pw.main(
        [
            "--input",
            str(ANDROID),
            "--output",
            str(output),
            "--admin-file",
            str(admin_file),
        ]
    )
    assert code == 0
    with output.open(encoding="utf-8", newline="") as handle:
        roles = Counter(row["sender_role"] for row in csv.DictReader(handle))
    assert roles == {pw.ROLE_CLIENT: 7, pw.ROLE_ADMIN: 5}
