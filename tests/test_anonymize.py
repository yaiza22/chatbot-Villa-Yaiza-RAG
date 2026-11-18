# Tests del anonimizador (con datos sintéticos).

import csv
from collections import Counter
from pathlib import Path

import pytest

from src.data import anonymize as an
from src.data.anonymize import Anonymizer


def _anon(text: str, chat_id: str | None = None, **kwargs: object) -> str:
    return Anonymizer(**kwargs).anonymize(text, chat_id)[0]  # type: ignore[arg-type]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- Teléfonos ---------------------------------------------------------------


@pytest.mark.parametrize(
    "number",
    [
        "3001112233",
        "300 111 2233",
        "300-111-2233",
        "300.111.2233",
        "+57 300 111 2233",
        "+573001112233",
        "(300) 111 2233",
        "57 3001112233",
        "0057 300 111 22 33",
        "(608) 277 1234",
        "+1 305 555 0100",
    ],
)
def test_phone_formats_are_masked(number: str) -> None:
    assert _anon(f"mi cel es {number} gracias") == "mi cel es [TEL] gracias"


def test_prices_dates_and_quantities_are_preserved() -> None:
    text = (
        "Somos 45 personas, 3 noches, del 15 al 17 de diciembre de 2026. "
        "Cuesta $1.500.000 la noche, el depósito es de 300.000 y el total "
        "3.500.000. Llegamos a las 3:30 pm. Hay 120 invitados y 2 cabañas. "
        "Nos cobran 3500000 por noche."
    )
    result, counts = Anonymizer().anonymize(text)
    assert result == text
    assert not counts


# --- Correos, URLs y usuarios ------------------------------------------------


def test_emails_are_masked_without_leaving_a_dangling_url() -> None:
    text = "escríbeme a ana.prueba@correo-ejemplo.com o a otra@ejemplo.co."
    result, counts = Anonymizer().anonymize(text)
    assert result == "escríbeme a [EMAIL] o a [EMAIL]."
    assert counts == {an.TAG_EMAIL: 2}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mira https://ejemplo.com/mapa?x=1.", "mira [URL]."),
        ("ver (https://ejemplo.com/x)", "ver ([URL])"),
        ("web www.ejemplo.co/finca ahí", "web [URL] ahí"),
        ("síguenos en ejemplo.com ya", "síguenos en [URL] ya"),
        ("escríbeme wa.me/573001112233", "escríbeme [URL]"),
        ("sígueme en @finca_ejemplo", "sígueme en [URL]"),
    ],
)
def test_urls_and_handles_are_masked(text: str, expected: str) -> None:
    assert _anon(text) == expected


def test_url_with_phone_inside_counts_as_one_url_and_no_phone() -> None:
    _, counts = Anonymizer().anonymize("wa.me/573001112233")
    assert counts == {an.TAG_URL: 1}


def test_sentences_without_space_after_period_are_not_urls() -> None:
    text = "Hola.Gracias por todo.Nos vemos. Ya.Me avisan."
    assert _anon(text) == text


# --- Documentos y cuentas ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mi cc es 1.234.567.890", "mi cc es [DOC]"),
        ("Cédula: 12345678", "Cédula: [DOC]"),
        ("NIT 900.123.456-7", "NIT [DOC]"),
    ],
)
def test_document_numbers_need_a_keyword(text: str, expected: str) -> None:
    assert _anon(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Cuenta de ahorros Bancolombia 12345678901",
            "Cuenta de ahorros Bancolombia [CUENTA]",
        ),
        ("consigné a la cuenta 123-456789-01", "consigné a la cuenta [CUENTA]"),
        ("referencia 1234567890123", "referencia [CUENTA]"),
        ("Nequi 3001112233", "Nequi [TEL]"),
    ],
)
def test_account_numbers_are_masked(text: str, expected: str) -> None:
    assert _anon(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "la finca cuenta con piscina para 45 personas",
        "la cuenta es $350.000",
        "la cuenta es de 1.500.000 en total",
        "somos 12 en el documento de reserva",
    ],
)
def test_account_and_document_keywords_do_not_mask_ordinary_text(text: str) -> None:
    assert _anon(text) == text


# --- Nombres -----------------------------------------------------------------


def test_names_ignore_case_and_accents_and_merge_full_names() -> None:
    anonymizer = Anonymizer(names=["María Ejemplo"])
    result, counts = anonymizer.anonymize("Hola, soy maría ejemplo y quiero cotizar")
    assert result == "Hola, soy [NOMBRE] y quiero cotizar"
    assert counts == {an.TAG_NAME: 1}
    assert anonymizer.anonymize("MARIA llegó")[0] == "[NOMBRE] llegó"


def test_names_only_match_whole_words() -> None:
    anonymizer = Anonymizer(names=["Ana"])
    assert anonymizer.anonymize("banana y Anabel")[0] == "banana y Anabel"
    assert anonymizer.anonymize("dile a Ana")[0] == "dile a [NOMBRE]"


def test_tilde_on_n_is_not_folded_away() -> None:
    anonymizer = Anonymizer(names=["Peña"])
    assert anonymizer.anonymize("me da pena")[0] == "me da pena"
    assert anonymizer.anonymize("hola Peña")[0] == "hola [NOMBRE]"


def test_farm_name_is_kept_even_if_a_name_matches() -> None:
    anonymizer = Anonymizer(names=["Yaiza", "Carlos"])
    text = "Alojamientos Villa Yaiza: hola Yaiza, soy Carlos"
    expected = "Alojamientos Villa Yaiza: hola [NOMBRE], soy [NOMBRE]"
    assert anonymizer.anonymize(text)[0] == expected


def test_existing_tags_are_not_masked_as_names() -> None:
    anonymizer = Anonymizer(names=["tel"])
    result, counts = anonymizer.anonymize("llámame 3001112233")
    assert result == "llámame [TEL]"
    assert an.TAG_NAME not in counts


def test_without_a_name_list_names_are_untouched() -> None:
    anonymizer = Anonymizer()
    assert anonymizer.anonymize("soy Juan")[0] == "soy Juan"
    assert not anonymizer.has_names


def test_chat_specific_names_only_apply_to_their_chat() -> None:
    anonymizer = Anonymizer(chat_names={"chat_0001": {"rosita"}})
    assert anonymizer.anonymize("hola rosita", "chat_0001")[0] == "hola [NOMBRE]"
    assert anonymizer.anonymize("hola rosita", "chat_0002")[0] == "hola rosita"
    assert anonymizer.anonymize("hola rosita")[0] == "hola rosita"


@pytest.mark.parametrize(
    ("source_file", "expected"),
    [
        ("Chat de WhatsApp con Pedro Pérez.zip", {"pedro", "perez"}),
        ("WhatsApp Chat - Pedro.zip", {"pedro"}),
        ("Chat de WhatsApp con +57 300 111 2233.zip", set()),
        ("Chat de WhatsApp con Finca Rosa.zip", {"rosa"}),
        ("Chat de WhatsApp con Sr Mario.txt", {"mario"}),
    ],
)
def test_contact_tokens_from_file_name(source_file: str, expected: set[str]) -> None:
    assert an.contact_tokens(source_file) == expected


# --- Propiedades generales ---------------------------------------------------


def test_anonymizing_twice_changes_nothing_the_second_time() -> None:
    text = (
        "Hola soy Juan Ejemplo, mi cel 300 111 2233, correo juan@ejemplo.com, "
        "cc 1.234.567.890, cuenta de ahorros 12345678901, mira "
        "https://ejemplo.com/x y @juan_ejemplo"
    )
    anonymizer = Anonymizer(names=["Juan Ejemplo"])
    first, counts = anonymizer.anonymize(text)
    assert first == (
        "Hola soy [NOMBRE], mi cel [TEL], correo [EMAIL], cc [DOC], "
        "cuenta de ahorros [CUENTA], mira [URL] y [URL]"
    )
    assert counts == Counter(
        {
            an.TAG_NAME: 1,
            an.TAG_PHONE: 1,
            an.TAG_EMAIL: 1,
            an.TAG_DOC: 1,
            an.TAG_ACCOUNT: 1,
            an.TAG_URL: 2,
        }
    )
    second, second_counts = anonymizer.anonymize(first)
    assert second == first
    assert not second_counts


# --- CSV de punta a punta ----------------------------------------------------

HEADER = ["msg_id", "chat_id", "timestamp", "sender_role", "text"]


def _sample_input(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "interim" / "messages.csv"
    source.parent.mkdir()
    _write_csv(
        source,
        HEADER,
        [
            [
                "chat_0001_00001",
                "chat_0001",
                "2024-03-12 09:15:00",
                "cliente",
                "Hola, soy Rosita, mi cel 300 111 2233",
            ],
            [
                "chat_0001_00002",
                "chat_0001",
                "2024-03-12 09:17:00",
                "admin",
                "Con gusto, 45 personas son $1.500.000",
            ],
            [
                "chat_0002_00001",
                "chat_0002",
                "2024-03-12 10:00:00",
                "cliente",
                "ok Rosita",
            ],
        ],
    )
    index = tmp_path / "interim" / "chat_index.csv"
    _write_csv(
        index,
        ["chat_id", "source_file"],
        [
            ["chat_0001", "Chat de WhatsApp con Rosita Ejemplo.zip"],
            ["chat_0002", "Chat de WhatsApp con Otro Nombre.zip"],
        ],
    )
    return source, index


def test_main_anonymizes_csv_and_writes_a_counts_only_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, index = _sample_input(tmp_path)
    output = tmp_path / "processed" / "messages.csv"
    code = an.main(
        ["--input", str(source), "--output", str(output), "--index", str(index)]
    )
    assert code == 0
    rows = _read_csv(output)
    assert [row["text"] for row in rows] == [
        "Hola, soy [NOMBRE], mi cel [TEL]",
        "Con gusto, 45 personas son $1.500.000",
        "ok Rosita",  # el nombre es del chat 1, no del chat 2
    ]
    assert list(rows[0]) == HEADER
    assert rows[0]["timestamp"] == "2024-03-12 09:15:00"

    report = {
        row["metric"]: row["count"] for row in _read_csv(output.parent / an.REPORT_NAME)
    }
    assert report == {
        "rows_total": "3",
        "rows_changed": "1",
        "[TEL]": "1",
        "[EMAIL]": "0",
        "[URL]": "0",
        "[NOMBRE]": "1",
        "[DOC]": "0",
        "[CUENTA]": "0",
    }
    assert "AVISO" not in capsys.readouterr().out


def test_output_folder_has_no_mapping_and_no_original_values(tmp_path: Path) -> None:
    source, index = _sample_input(tmp_path)
    output = tmp_path / "processed" / "messages.csv"
    an.main(["--input", str(source), "--output", str(output), "--index", str(index)])
    files = sorted(path.name for path in output.parent.iterdir())
    assert files == [an.REPORT_NAME, "messages.csv"]
    for path in output.parent.iterdir():
        content = path.read_text(encoding="utf-8")
        assert "111 2233" not in content
        assert "Rosita" not in content.replace("ok Rosita", "")


def test_main_warns_when_there_is_no_name_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _ = _sample_input(tmp_path)
    an.main(["--input", str(source), "--output", str(tmp_path / "out" / "m.csv")])
    assert "AVISO" in capsys.readouterr().out


def test_main_reads_names_and_keep_files(tmp_path: Path) -> None:
    source, _ = _sample_input(tmp_path)
    names = tmp_path / "names.txt"
    names.write_text("# nombres\nRosita\n", encoding="utf-8")
    output = tmp_path / "out" / "m.csv"
    an.main(
        ["--input", str(source), "--output", str(output), "--names-file", str(names)]
    )
    assert [row["text"] for row in _read_csv(output)][2] == "ok [NOMBRE]"


def test_main_refuses_to_overwrite_its_input(tmp_path: Path) -> None:
    source, _ = _sample_input(tmp_path)
    with pytest.raises(SystemExit):
        an.main(["--input", str(source), "--output", str(source)])


def test_main_rejects_unknown_column(tmp_path: Path) -> None:
    source, _ = _sample_input(tmp_path)
    with pytest.raises(SystemExit):
        an.main(
            [
                "--input",
                str(source),
                "--output",
                str(tmp_path / "out.csv"),
                "--columns",
                "no_existe",
            ]
        )
