"""Docling parsing + Chonkie chunking -> RAG-Anything content list.

Docling turns PDF/DOCX/PPTX/XLSX into a DoclingDocument. We walk it in reading
order and emit RAG-Anything (MinerU-format) content items:

* prose (paragraphs, list items, speaker notes) is buffered per section and
  chunked with Chonkie's RecursiveChunker; each chunk carries a section
  breadcrumb so it stays meaningful on its own
* small tables become ``table`` items that RAG-Anything analyses with the LLM
* large tables (typical for Excel) are split by Chonkie's TableChunker into
  row windows with the header repeated, and stored as text chunks; a preview
  of the first rows is also kept as a ``table`` item so the table itself
  becomes a graph entity
* pictures are saved to disk and become ``image`` items for the vision model
* formulas become ``equation`` items

All text is Unicode-normalised and chunked with rules that work across
European languages (see multilingual.py).

Chunk boundaries are preserved through LightRAG by ending each text item with
CHUNK_SEPARATOR and inserting with ``split_by_character_only=True``.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chonkie import RecursiveChunker, TableChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    ContentLayer,
    DoclingDocument,
    FormulaItem,
    GroupItem,
    GroupLabel,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from raganything.utils import extract_neighbor_text_from_content_list

from .config import SUPPORTED_EXTENSIONS, Settings
from .multilingual import EUROPEAN_RULES, normalize_text

logger = logging.getLogger(__name__)

# Printable "symbol for record separator" (LightRAG strips control characters,
# so the real ASCII RS would vanish before splitting). Never produced by Docling.
CHUNK_SEPARATOR = "\u241e"
TABLE_PREVIEW_ROWS = 20
MAX_BREADCRUMB_CHARS = 200


def file_doc_id(path: str | Path) -> str:
    """Document ID from the file's bytes: unchanged files keep their ID, edited ones get a new one."""
    return "doc-" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32]


def source_label(path: str | Path) -> str:
    """How a file is named in the index and in citations: relative to the working
    directory when it is inside it, so two report.docx files in different folders
    stay distinguishable; absolute otherwise."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class ParsedDocument:
    source: Path
    doc_id: str
    content_list: list[dict[str, Any]]
    stats: dict[str, int] = field(default_factory=dict)


class DocumentParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        pdf_options = PdfPipelineOptions(
            do_ocr=settings.pdf_ocr,
            # Keep picture crops, so figures in PDFs reach the vision model like
            # embedded images in Office files do.
            generate_picture_images=True,
            images_scale=2.0,
        )
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.XLSX],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
        )
        # min_characters_per_chunk=1: Chonkie glues any split shorter than this
        # onto the previous one, so the default (24) disables word-level
        # splitting entirely (nearly every word is shorter) and long runs fall
        # through to raw token cuts mid-word. Splits are still merged up to
        # chunk_size afterwards, so this does not produce tiny chunks.
        self.text_chunker = RecursiveChunker(
            tokenizer=settings.tokenizer, chunk_size=settings.chunk_size,
            rules=EUROPEAN_RULES, min_characters_per_chunk=1,
        )
        self.table_chunker = TableChunker(tokenizer=settings.tokenizer, chunk_size=settings.chunk_size)

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path).resolve()
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.name}")

        doc = self.converter.convert(path).document
        doc_id = file_doc_id(path)
        image_dir = (self.settings.image_dir / doc_id).resolve()

        builder = _ContentListBuilder(self, doc, image_dir, unit=_location_unit(path))
        builder.walk()
        return ParsedDocument(source=path, doc_id=doc_id, content_list=builder.finish(), stats=builder.stats)


class _ContentListBuilder:
    def __init__(self, parser: DocumentParser, doc: DoclingDocument, image_dir: Path, unit: str):
        self.p = parser
        self.doc = doc
        self.image_dir = image_dir
        self.unit = unit
        self.items: list[dict[str, Any]] = []
        self.headings: list[tuple[int, str]] = []  # (level, text) stack
        self.buffer: list[str] = []
        self.buffer_page = 0
        self.stats = {"text_chunks": 0, "tables": 0, "table_row_chunks": 0, "images": 0, "equations": 0}

    # -- traversal ---------------------------------------------------------

    def walk(self) -> None:
        layers = {ContentLayer.BODY, ContentLayer.NOTES}
        for item, _level in self.doc.iterate_items(included_content_layers=layers):
            page = _page_idx(item)
            if page != self.buffer_page:
                self._flush_text()
                self.buffer_page = page

            if isinstance(item, (TitleItem, SectionHeaderItem)):
                self._flush_text()
                level = 0 if isinstance(item, TitleItem) else item.level
                self.headings = [h for h in self.headings if h[0] < level] + [(level, normalize_text(item.text).strip())]
            elif isinstance(item, TableItem):
                self._flush_text()
                self._add_table(item, page)
            elif isinstance(item, PictureItem):
                self._flush_text()
                self._add_picture(item, page)
            elif isinstance(item, FormulaItem):
                self._flush_text()
                self._add_equation(item, page)
            elif isinstance(item, TextItem) and item.text.strip():
                text = normalize_text(item.text).strip()
                if isinstance(item, ListItem):
                    text = f"{item.marker or '-'} {text}"
                if item.content_layer == ContentLayer.NOTES:
                    text = f"Speaker notes: {text}"
                self.buffer.append(text)

    def finish(self) -> list[dict[str, Any]]:
        self._flush_text()
        # RAG-Anything copies neighbouring item text into image prompts; compute
        # it now, before the separators below are appended.
        for i, it in enumerate(self.items):
            if it["type"] == "image":
                it["_neighbor_text"] = extract_neighbor_text_from_content_list(self.items, i)
        # Terminate every text item but the last with the separator, so the
        # "\n\n"-joined text RAG-Anything hands LightRAG splits back exactly.
        text_items = [it for it in self.items if it["type"] == "text"]
        for it in text_items[:-1]:
            it["text"] += f"\n{CHUNK_SEPARATOR}"
        return self.items

    # -- emitters ----------------------------------------------------------

    def _location(self, page: int) -> str:
        """"[Slide 3]", "[Sheet 2]" or "[Page 5]"; empty for DOCX, which has no pages."""
        return f"[{self.unit} {page + 1}]" if self.doc.pages else ""

    def _breadcrumb(self, page: int, extra: str | None = None) -> str:
        parts = [h[1] for h in self.headings]
        if extra:
            parts.append(extra)
        crumb = " > ".join(parts)[:MAX_BREADCRUMB_CHARS]
        return " ".join(x for x in (self._location(page), crumb) if x)

    def _add_text_chunk(self, text: str, page: int, context: str) -> None:
        body = f"{context}\n\n{text}" if context else text
        self.items.append({"type": "text", "text": body, "page_idx": page})

    def _flush_text(self) -> None:
        if not self.buffer:
            return
        text = "\n\n".join(self.buffer)
        self.buffer = []
        context = self._breadcrumb(self.buffer_page)
        for chunk in self.p.text_chunker.chunk(text):
            if chunk.text.strip():
                self._add_text_chunk(chunk.text.strip(), self.buffer_page, context)
                self.stats["text_chunks"] += 1

    def _add_table(self, item: TableItem, page: int) -> None:
        markdown = normalize_text(item.export_to_markdown(doc=self.doc)).strip()
        if not markdown:
            return
        caption = normalize_text(item.caption_text(self.doc)).strip()
        sheet = _sheet_name(item, self.doc)
        # The location goes into the caption so the chunk RAG-Anything writes for
        # the table can be cited by slide, sheet or page like text chunks can.
        captions = [c for c in (self._location(page), f"Sheet: {sheet}" if sheet else "", caption) if c]
        rows = item.data.num_rows

        if rows <= self.p.settings.table_modal_max_rows:
            self.items.append({
                "type": "table", "table_body": markdown, "table_caption": captions,
                "table_footnote": [], "page_idx": page,
            })
            self.stats["tables"] += 1
            return

        # Large table: full coverage as row-window text chunks...
        label = " | ".join(c for c in captions if c != self._location(page)) or "Table"  # breadcrumb has the location
        pieces = self.p.table_chunker.chunk(markdown)
        for i, piece in enumerate(pieces, 1):
            context = self._breadcrumb(page, f"{label} (part {i}/{len(pieces)}, {rows - 1} data rows total)")
            self._add_text_chunk(piece.text.strip(), page, context)
            self.stats["table_row_chunks"] += 1

        # ...plus a preview so the table is still analysed as one entity.
        preview = "\n".join(markdown.splitlines()[: TABLE_PREVIEW_ROWS + 2])
        self.items.append({
            "type": "table", "table_body": preview, "page_idx": page,
            "table_caption": captions,
            "table_footnote": [f"Preview: first {TABLE_PREVIEW_ROWS} of {rows - 1} data rows."],
        })
        self.stats["tables"] += 1

    def _add_picture(self, item: PictureItem, page: int) -> None:
        image = item.get_image(self.doc)
        min_px = self.p.settings.min_image_px
        if image is None or min(image.size) < min_px:
            return
        self.image_dir.mkdir(parents=True, exist_ok=True)
        img_path = self.image_dir / f"image_{self.stats['images'] + 1:03d}.png"
        image.convert("RGB").save(img_path)
        caption = normalize_text(item.caption_text(self.doc)).strip()
        self.items.append({
            "type": "image", "img_path": str(img_path),
            "image_caption": [c for c in (self._location(page), caption) if c],
            "image_footnote": [], "page_idx": page,
            # RAG-Anything only derives this from MinerU heading items, which we don't emit.
            "_section_path": " > ".join(h[1] for h in self.headings),
        })
        self.stats["images"] += 1

    def _add_equation(self, item: FormulaItem, page: int) -> None:
        latex = (item.text or item.orig or "").strip()
        if latex:
            self.items.append({"type": "equation", "latex": latex, "text": latex, "page_idx": page})
            self.stats["equations"] += 1


def _location_unit(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "Page"
    return "Slide" if ext.startswith(".p") else "Sheet" if ext.startswith(".x") else "Page"


def _page_idx(item) -> int:
    """0-based page/slide/sheet index; DOCX has no pages, so it is always 0."""
    return item.prov[0].page_no - 1 if item.prov else 0


def _sheet_name(item, doc: DoclingDocument) -> str | None:
    parent = item.parent.resolve(doc) if item.parent else None
    if isinstance(parent, GroupItem) and parent.label == GroupLabel.SHEET:
        return parent.name
    return None
