"""Shared fixtures. Nothing here calls the Claude API or loads the embedding model."""

from pathlib import Path

import pytest

from brag.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        working_dir=tmp_path / "rag_storage",
        output_dir=tmp_path / "output",
        ontology_file=tmp_path / "ontology.yaml",
    )


@pytest.fixture
def make_docx(tmp_path: Path):
    def make(name: str = "report.docx", paragraphs: tuple[str, ...] = ("Revenue grew 18% in Q3.",), table=None) -> Path:
        import docx

        d = docx.Document()
        d.add_heading("Quarterly Report", level=1)
        d.add_heading("Summary", level=2)
        for text in paragraphs:
            d.add_paragraph(text)
        if table:
            t = d.add_table(rows=len(table), cols=len(table[0]))
            for r, row in enumerate(table):
                for c, value in enumerate(row):
                    t.cell(r, c).text = str(value)
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        d.save(path)
        return path

    return make


@pytest.fixture
def make_xlsx(tmp_path: Path):
    def make(name: str = "sales.xlsx", rows: int = 5) -> Path:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sales"
        ws.append(["Region", "Product", "Revenue"])
        for i in range(rows):
            ws.append([f"Region {i % 4}", f"Widget {i}", 1000 + i])
        path = tmp_path / name
        wb.save(path)
        return path

    return make


@pytest.fixture
def make_pptx(tmp_path: Path):
    def make(name: str = "deck.pptx") -> Path:
        from pptx import Presentation

        prs = Presentation()
        for i, title in enumerate(["Widget Pro launch", "Pricing"], 1):
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = title
            slide.placeholders[1].text = f"Point {i}: Widget Pro ships in July."
            slide.notes_slide.notes_text_frame.text = f"Mention partner Acme on slide {i}."
        path = tmp_path / name
        prs.save(path)
        return path

    return make
