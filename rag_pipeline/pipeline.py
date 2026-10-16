"""Ingestion and query pipeline: Docling -> Chonkie -> RAG-Anything (LightRAG)."""

import logging
from pathlib import Path
from typing import Iterable

from lightrag.constants import DEFAULT_ENTITY_TYPES
from lightrag.utils import get_env_value
from raganything import RAGAnything, RAGAnythingConfig

from .config import SUPPORTED_EXTENSIONS, Settings
from .models import ClaudeModels, build_embedding_func
from .parsing import CHUNK_SEPARATOR, DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


def collect_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
        else:
            logger.warning("Skipping unsupported file: %s", p)
    # Office lock files ("~$report.docx") are not real documents.
    return [f for f in files if not f.name.startswith("~$")]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.parser = DocumentParser(self.settings)
        models = ClaudeModels(self.settings)

        # LightRAG deep-copies its config; bound methods would drag the
        # (unpicklable) Anthropic client along, plain closures do not.
        async def llm_func(*args, **kwargs):
            return await models.complete(*args, **kwargs)

        async def vision_func(*args, **kwargs):
            return await models.vision(*args, **kwargs)

        self.rag = RAGAnything(
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
            embedding_func=build_embedding_func(self.settings),
            lightrag_kwargs={
                "chunk_token_size": self.settings.lightrag_chunk_token_size,
                "chunk_overlap_token_size": 0,
                "llm_model_name": self.settings.llm_model,
                "llm_model_max_async": self.settings.llm_max_async,
                "addon_params": {
                    "language": self.settings.kg_language,
                    "entity_types": get_env_value("ENTITY_TYPES", DEFAULT_ENTITY_TYPES, list),
                },
            },
        )

    async def ingest_file(self, path: str | Path) -> ParsedDocument:
        parsed = self.parser.parse(path)
        logger.info("Parsed %s: %s", parsed.source.name, parsed.stats)
        if not parsed.content_list:
            logger.warning("No content extracted from %s", parsed.source.name)
            return parsed
        await self.rag.insert_content_list(
            content_list=parsed.content_list,
            file_path=parsed.source.name,
            doc_id=parsed.doc_id,
            split_by_character=CHUNK_SEPARATOR,
            split_by_character_only=True,
        )
        return parsed

    async def ingest(self, paths: Iterable[str | Path]) -> list[ParsedDocument]:
        results = []
        for f in collect_files(paths):
            try:
                results.append(await self.ingest_file(f))
            except Exception:
                logger.exception("Failed to ingest %s", f)
        return results

    async def query(self, question: str, mode: str = "mix", **kwargs) -> str:
        return await self.rag.aquery(question, mode=mode, **kwargs)

    async def close(self) -> None:
        await self.rag.finalize_storages()
