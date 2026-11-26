"""RAG over DOCX/PPTX/XLSX with Docling, Chonkie and RAG-Anything."""

__version__ = "0.4.0"

__all__ = ["RAGPipeline", "Settings", "collect_files", "__version__"]


def __getattr__(name: str):
    # Lazy so `import brag` (and `brag --help`) stays fast: the pipeline pulls
    # in Docling, LightRAG and sentence-transformers, which take seconds to import.
    if name in {"RAGPipeline", "collect_files"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name == "Settings":
        from .config import Settings

        return Settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
