"""Runtime settings, read from environment variables.

The `brag` CLI loads a .env file from the current directory first; library users
can call `dotenv.load_dotenv()` themselves before creating `Settings`."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# File types Docling handles: PDF, HTML (e.g. Confluence pages) and the three Office formats.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".html", ".htm",
    ".docx", ".dotx", ".docm", ".dotm",
    ".pptx", ".potx", ".ppsx", ".pptm", ".potm", ".ppsm",
    ".xlsx", ".xlsm", ".xltx", ".xltm",
}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


PROVIDERS = ("anthropic", "bedrock", "vertex", "foundry")

# Model IDs differ by platform: Bedrock prefixes "anthropic.", Vertex uses the
# bare ID, and Foundry takes the name of your deployment (set LLM_MODEL).
_DEFAULT_MODELS = {
    "anthropic": ("claude-opus-5-5", "claude-opus-4-8"),
    "bedrock": ("anthropic.claude-opus-5-5", "anthropic.claude-opus-4-8"),
    "vertex": ("claude-opus-5-5", "claude-opus-4-8"),
    "foundry": ("claude-opus-5-5", "claude-opus-4-8"),
}


def _provider() -> str:
    provider = _env("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, not {provider!r}")
    return provider


@dataclass
class Settings:
    # Storage
    working_dir: Path = field(default_factory=lambda: Path(_env("WORKING_DIR", "./rag_storage")))
    output_dir: Path = field(default_factory=lambda: Path(_env("OUTPUT_DIR", "./output")))

    # Entity types for the knowledge graph, written by `brag ontology propose`.
    # Used when the file exists; otherwise ENTITY_TYPES or LightRAG's defaults.
    ontology_file: Path = field(default_factory=lambda: Path(_env("ONTOLOGY_FILE", "./ontology.yaml")))

    # Claude (LLM + vision). LLM_PROVIDER picks the platform: the Claude API, or
    # Amazon Bedrock, Google Vertex AI or Microsoft Foundry for data-residency
    # rules. Each platform client reads its usual credentials from the environment.
    llm_provider: str = field(default_factory=_provider)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", _DEFAULT_MODELS[_provider()][0]))
    # Retried on this model when a safety classifier declines a request. On the
    # Claude API the server picks the fallback ("default") unless this is set.
    llm_fallback_model: str = field(default_factory=lambda: _env(
        "LLM_FALLBACK_MODEL", "" if _provider() == "anthropic" else _DEFAULT_MODELS[_provider()][1]
    ))
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "medium"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "16000")))
    llm_max_async: int = field(default_factory=lambda: int(_env("LLM_MAX_ASYNC", "4")))

    # Local embeddings (sentence-transformers). bge-m3 is multilingual (100+
    # languages, all EU ones) and accepts 8192 tokens, so 1k-token chunks fit.
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-m3"))

    # Optional local cross-encoder that re-scores retrieved chunks before they
    # reach the LLM, e.g. BAAI/bge-reranker-v2-m3 (multilingual). Off when empty.
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", ""))

    # LightRAG storage backends (class names, e.g. PGKVStorage, PGVectorStorage,
    # Neo4JStorage, QdrantVectorDBStorage). Each reads its connection settings
    # from LightRAG's own environment variables (POSTGRES_HOST, NEO4J_URI, ...).
    kv_storage: str = field(default_factory=lambda: _env("KV_STORAGE", "JsonKVStorage"))
    vector_storage: str = field(default_factory=lambda: _env("VECTOR_STORAGE", "NanoVectorDBStorage"))
    graph_storage: str = field(default_factory=lambda: _env("GRAPH_STORAGE", "NetworkXStorage"))
    doc_status_storage: str = field(default_factory=lambda: _env("DOC_STATUS_STORAGE", "JsonDocStatusStorage"))
    # Separates several indexes in one set of backends (LightRAG's WORKSPACE).
    workspace: str = field(default_factory=lambda: _env("WORKSPACE", ""))

    # PDF: run OCR on scanned pages. Needs an OCR engine Docling supports
    # (e.g. `pip install rapidocr-onnxruntime`); text-based PDFs do not need it.
    pdf_ocr: bool = field(default_factory=lambda: _env_bool("PDF_OCR", False))

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
    def index_dir(self) -> Path:
        """Local folder for this index: the working dir, or its workspace subfolder
        (where LightRAG's file storages put a workspace too)."""
        return Path(self.working_dir) / self.workspace if self.workspace else Path(self.working_dir)

    @property
    def image_dir(self) -> Path:
        """Where extracted images go, per document ID (and per workspace)."""
        base = Path(self.output_dir) / self.workspace if self.workspace else Path(self.output_dir)
        return base / "images"

    @property
    def manifest_file(self) -> Path:
        """Which file is indexed under which document ID (see manifest.py)."""
        return self.index_dir / "brag_manifest.json"

    @property
    def lightrag_chunk_token_size(self) -> int:
        # Headroom for the section breadcrumb prepended to each Chonkie chunk.
        return self.chunk_size + 256
