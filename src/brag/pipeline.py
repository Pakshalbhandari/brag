"""Ingestion and query pipeline: Docling -> Chonkie -> RAG-Anything (LightRAG)."""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import DEFAULT_ENTITY_TYPES
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import get_env_value
from raganything import RAGAnything, RAGAnythingConfig

from .config import SUPPORTED_EXTENSIONS, Settings
from .manifest import Manifest
from .models import ClaudeModels, build_embedding_func
from .ontology import load_entity_types
from .parsing import CHUNK_SEPARATOR, DocumentParser, file_doc_id, source_label

logger = logging.getLogger(__name__)

# Appended to LightRAG's answer prompt. Every chunk starts with its location
# (see parsing._breadcrumb), so the model can cite more than the file name.
CITATION_PROMPT = (
    "Each document chunk starts with its location: a bracketed slide, sheet or page "
    "number such as [Slide 3], followed by its section headings. In the References "
    "section, after each file name, list the locations the cited facts came from, "
    "for example: `- [1] decks/q3.pptx (Slide 3, Slide 7)`. Use the section heading "
    "when a chunk has no bracketed number."
)


def collect_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
            if p.exists():
                files.append(p)
            else:
                logger.warning("File not found: %s", p)
        else:
            logger.warning("Skipping unsupported file: %s", p)
    # Office lock files ("~$report.docx") are not real documents.
    return [f for f in files if not f.name.startswith("~$")]


@dataclass
class IngestResult:
    source: Path
    status: Literal["indexed", "updated", "unchanged", "empty", "failed"]
    doc_id: str | None = None
    stats: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RemoveResult:
    label: str
    ok: bool
    error: str | None = None


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.parser = DocumentParser(self.settings)
        self.models = ClaudeModels(self.settings)
        self.manifest = Manifest.load(self.settings.manifest_file)
        self.entity_types = load_entity_types(self.settings) or get_env_value(
            "ENTITY_TYPES", DEFAULT_ENTITY_TYPES, list
        )
        self._initialized = False
        models = self.models

        # LightRAG deep-copies its config; bound methods would drag the
        # (unpicklable) Anthropic client along, plain closures do not.
        async def llm_func(*args, **kwargs):
            return await models.complete(*args, **kwargs)

        async def vision_func(*args, **kwargs):
            return await models.vision(*args, **kwargs)

        embedding_func = build_embedding_func(self.settings)

        # Built here rather than left to RAG-Anything, so initialising, deleting
        # and querying go through LightRAG's public API.
        self.lightrag = LightRAG(
            working_dir=str(self.settings.working_dir),
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            chunk_token_size=self.settings.lightrag_chunk_token_size,
            chunk_overlap_token_size=0,
            llm_model_name=self.settings.llm_model,
            llm_model_max_async=self.settings.llm_max_async,
            addon_params={"language": self.settings.kg_language, "entity_types": self.entity_types},
        )
        self.rag = RAGAnything(
            lightrag=self.lightrag,
            config=RAGAnythingConfig(
                working_dir=str(self.settings.working_dir),
                parser_output_dir=str(self.settings.output_dir),
                parser="docling",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            ),
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embedding_func,
        )

    async def initialize(self) -> None:
        if not self._initialized:
            Path(self.settings.working_dir).mkdir(parents=True, exist_ok=True)
            await self.lightrag.initialize_storages()
            await initialize_pipeline_status()
            # RAG-Anything's record of which documents had their tables and images
            # processed. Created here (RAG-Anything keeps one it is handed) so
            # removal can clear it without reaching into RAG-Anything's internals.
            if self.rag.multimodal_status_cache is None:
                cache = self.lightrag.key_string_value_json_storage_cls(
                    namespace="multimodal_status",
                    workspace=self.lightrag.workspace,
                    global_config=self.lightrag.__dict__,
                    embedding_func=self.lightrag.embedding_func,
                )
                await cache.initialize()
                self.rag.multimodal_status_cache = cache
            self._initialized = True

    # -- ingest ------------------------------------------------------------

    def _check_entity_types(self) -> None:
        used = self.manifest.entity_types
        if self.manifest.files and used is not None and used != list(self.entity_types):
            logger.warning(
                "Entity types differ from the ones this index was built with. New documents "
                "will use the new types; re-index everything for a consistent graph."
            )
        self.manifest.entity_types = list(self.entity_types)

    async def _is_indexed(self, doc_id: str) -> bool:
        record = await self.lightrag.doc_status.get_by_id(doc_id)
        return bool(record) and getattr(record.get("status"), "value", record.get("status")) == DocStatus.PROCESSED.value

    async def ingest_file(self, path: str | Path) -> IngestResult:
        await self.initialize()
        path = Path(path).resolve()
        doc_id = file_doc_id(path)
        previous = self.manifest.get(path)
        stats: dict[str, int] = {}

        # Already indexed: this file is unchanged, or an identical copy was indexed
        # from another path (or before brag kept a manifest).
        if not await self._is_indexed(doc_id):
            parsed = self.parser.parse(path)
            stats = parsed.stats
            logger.info("Parsed %s: %s", path.name, stats)
            if not parsed.content_list:
                logger.warning("No content extracted from %s", path.name)
                if previous:  # the file was emptied: its old version should go too
                    self.manifest.pop(path)
                    if not self.manifest.is_referenced(previous.doc_id):
                        await self._delete_document(previous.doc_id)
                    self.manifest.save()
                return IngestResult(path, "empty", doc_id, stats)
            await self.rag.insert_content_list(
                content_list=parsed.content_list,
                file_path=source_label(path),
                doc_id=doc_id,
                split_by_character=CHUNK_SEPARATOR,
                split_by_character_only=True,
            )
            if not await self._is_indexed(doc_id):
                raise RuntimeError(f"LightRAG did not finish indexing {doc_id}")

        self.manifest.set(path, doc_id, source_label(path))
        if previous is None:
            status = "indexed"
        elif previous.doc_id == doc_id:
            status = "unchanged"
        else:
            # The new version is in; now drop the old one, unless another file shares it.
            status = "updated"
            if not self.manifest.is_referenced(previous.doc_id):
                await self._delete_document(previous.doc_id)
        self._check_entity_types()
        self.manifest.save()
        return IngestResult(path, status, doc_id, stats)

    async def ingest(self, paths: Iterable[str | Path]) -> list[IngestResult]:
        results = []
        for f in collect_files(paths):
            try:
                results.append(await self.ingest_file(f))
            except Exception as e:
                logger.debug("Failed to ingest %s", f, exc_info=True)
                results.append(IngestResult(Path(f).resolve(), "failed", error=f"{type(e).__name__}: {e}"))
        return results

    # -- remove ------------------------------------------------------------

    async def _delete_document(self, doc_id: str) -> None:
        result = await self.lightrag.adelete_by_doc_id(doc_id)
        if result.status not in ("success", "not_found"):
            raise RuntimeError(f"Could not delete {doc_id}: {result.message}")
        # RAG-Anything remembers that a document's tables and images were processed;
        # forget it, or re-adding the same file later would skip them.
        cache = self.rag.multimodal_status_cache
        await cache.delete([doc_id])
        await cache.index_done_callback()
        shutil.rmtree(Path(self.settings.output_dir) / "images" / doc_id, ignore_errors=True)

    async def remove(self, paths: Iterable[str | Path]) -> list[RemoveResult]:
        """Remove files (or every indexed file under folders) from the index.

        The files do not need to exist any more.
        """
        await self.initialize()
        results = []
        for key in self.manifest.match(list(paths)):
            entry = self.manifest.files.pop(key)
            try:
                if not self.manifest.is_referenced(entry.doc_id):
                    await self._delete_document(entry.doc_id)
                results.append(RemoveResult(entry.label, True))
            except Exception as e:
                self.manifest.files[key] = entry
                results.append(RemoveResult(entry.label, False, f"{type(e).__name__}: {e}"))
            self.manifest.save()
        return results

    def indexed_files(self) -> dict[str, str]:
        """Indexed files (absolute path -> doc_id)."""
        return {k: v.doc_id for k, v in self.manifest.files.items()}

    # -- query -------------------------------------------------------------

    async def query(self, question: str, mode: str = "mix", **kwargs) -> str:
        working_dir = Path(self.settings.working_dir)
        if not working_dir.is_dir() or not any(working_dir.glob("kv_store_*.json")):
            raise RuntimeError(f"No index found in {working_dir}/. Run `brag ingest <paths>` first.")
        await self.initialize()
        kwargs.setdefault("user_prompt", CITATION_PROMPT)
        return await self.rag.aquery(question, mode=mode, **kwargs)

    async def close(self) -> None:
        if self._initialized:
            await self.rag.finalize_storages()
