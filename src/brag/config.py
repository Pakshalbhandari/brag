"""Runtime settings, read from environment variables.

The `brag` CLI loads a .env file from the current directory first; library users
can call `dotenv.load_dotenv()` themselves before creating `Settings`."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# File types Docling handles for the three Office formats this pipeline targets.
SUPPORTED_EXTENSIONS = {
    ".docx", ".dotx", ".docm", ".dotm",
    ".pptx", ".potx", ".ppsx", ".pptm", ".potm", ".ppsm",
    ".xlsx", ".xlsm", ".xltx", ".xltm",
}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # Storage
    working_dir: Path = field(default_factory=lambda: Path(_env("WORKING_DIR", "./rag_storage")))
    output_dir: Path = field(default_factory=lambda: Path(_env("OUTPUT_DIR", "./output")))

    # Claude (LLM + vision)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-opus-5-5"))
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "medium"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "16000")))
    llm_max_async: int = field(default_factory=lambda: int(_env("LLM_MAX_ASYNC", "4")))

    # Local embeddings (sentence-transformers). bge-m3 is multilingual (100+
    # languages, all EU ones) and accepts 8192 tokens, so 1k-token chunks fit.
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-m3"))

    # Language the knowledge graph (entity names, descriptions) is written in.
    # One canonical language lets "Europäische Union" in a German file and
    # "Union européenne" in a French one merge into the same entity.
    # Answers are always given in the language of the question.
    kg_language: str = field(default_factory=lambda: _env("KG_LANGUAGE", "English"))

    # Chonkie chunking. Token counts use o200k_base, the same encoding LightRAG uses.
    tokenizer: str = field(default_factory=lambda: _env("CHUNK_TOKENIZER", "o200k_base"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1024")))

    # Tables with more rows than this are split by rows into text chunks
    # (header repeated in each); smaller ones go to RAG-Anything's table processor.
    table_modal_max_rows: int = field(default_factory=lambda: int(_env("TABLE_MODAL_MAX_ROWS", "40")))

    # Images smaller than this (either side, px) are treated as icons and skipped.
    min_image_px: int = field(default_factory=lambda: int(_env("MIN_IMAGE_PX", "64")))

    @property
    def lightrag_chunk_token_size(self) -> int:
        # Headroom for the section breadcrumb prepended to each Chonkie chunk.
        return self.chunk_size + 256
