import os
from pathlib import Path

import pytest

from brag.parsing import CHUNK_SEPARATOR, DocumentParser, file_doc_id, source_label


@pytest.fixture
def parser(settings):
    return DocumentParser(settings)


def test_docx_text_and_small_table(parser, make_docx):
    path = make_docx(table=[["Region", "Revenue"], ["EMEA", "18.4"], ["APAC", "11.0"]])
    parsed = parser.parse(path)

    assert parsed.doc_id == file_doc_id(path)
    assert parsed.stats["text_chunks"] >= 1 and parsed.stats["tables"] == 1
    text = next(i for i in parsed.content_list if i["type"] == "text")
    assert text["text"].startswith("Quarterly Report > Summary")  # breadcrumb
    assert "Revenue grew 18% in Q3." in text["text"]
    table = next(i for i in parsed.content_list if i["type"] == "table")
    assert "EMEA" in table["table_body"]


def test_separator_ends_every_text_item_but_the_last(parser, make_docx):
    path = make_docx(paragraphs=tuple(f"Paragraph {i}. " + "word " * 600 for i in range(4)))
    texts = [i["text"] for i in parser.parse(path).content_list if i["type"] == "text"]
    assert len(texts) > 1
    assert all(t.endswith(CHUNK_SEPARATOR) for t in texts[:-1])
    assert not texts[-1].endswith(CHUNK_SEPARATOR)


def test_large_sheet_is_row_chunked_with_a_preview(parser, make_xlsx, settings):
    path = make_xlsx(rows=settings.table_modal_max_rows + 30)
    parsed = parser.parse(path)

    assert parsed.stats["table_row_chunks"] >= 1
    preview = next(i for i in parsed.content_list if i["type"] == "table")
    assert "Preview" in preview["table_footnote"][0]
    assert "Sheet: Sales" in preview["table_caption"]
    row_chunk = next(i for i in parsed.content_list if i["type"] == "text")
    assert row_chunk["text"].startswith("[Sheet 1]")
    assert "| Region" in row_chunk["text"]  # header repeated in the chunk


def test_pptx_keeps_slide_numbers_and_speaker_notes(parser, make_pptx):
    parsed = parser.parse(make_pptx())
    joined = "\n".join(i["text"] for i in parsed.content_list if i["type"] == "text")
    assert "[Slide 1]" in joined and "[Slide 2]" in joined
    assert "Speaker notes: Mention partner Acme" in joined


def test_unsupported_extension(parser, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hi")
    with pytest.raises(ValueError):
        parser.parse(path)


def test_doc_id_follows_content(tmp_path):
    a, b = tmp_path / "a.docx", tmp_path / "b.docx"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert file_doc_id(a) == file_doc_id(b)
    b.write_bytes(b"edited")
    assert file_doc_id(a) != file_doc_id(b)


def test_source_label_is_relative_inside_cwd(tmp_path, monkeypatch):
    (tmp_path / "q3").mkdir()
    monkeypatch.chdir(tmp_path)
    assert source_label("q3/report.docx") == "q3/report.docx"
    outside = Path(os.path.abspath(os.sep)) / "elsewhere" / "report.docx"
    assert source_label(outside) == outside.as_posix()
