import pytest

from brag.config import Settings


def test_defaults(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK_MODEL", "KV_STORAGE", "RERANK_MODEL", "PDF_OCR"):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert (s.llm_provider, s.llm_model, s.llm_fallback_model) == ("anthropic", "claude-opus-5-5", "")
    assert s.kv_storage == "JsonKVStorage" and s.rerank_model == "" and s.pdf_ocr is False


def test_bedrock_model_ids(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Bedrock")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    s = Settings()
    assert s.llm_model == "anthropic.claude-opus-5-5" and s.llm_fallback_model == "anthropic.claude-opus-4-8"


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings()


def test_storage_and_flags_from_env(monkeypatch):
    monkeypatch.setenv("VECTOR_STORAGE", "PGVectorStorage")
    monkeypatch.setenv("WORKSPACE", "finance")
    monkeypatch.setenv("PDF_OCR", "yes")
    s = Settings()
    assert s.vector_storage == "PGVectorStorage" and s.workspace == "finance" and s.pdf_ocr is True
