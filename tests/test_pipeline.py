"""Ingest/update/remove logic, with LightRAG and RAG-Anything replaced by fakes."""

import asyncio
from types import SimpleNamespace

import pytest

from brag.manifest import Manifest
from brag.models import TokenUsage
from brag.parsing import DocumentParser
from brag.pipeline import CITATION_PROMPT, RAGPipeline, collect_files


class FakeKV:
    def __init__(self):
        self.data = {}

    async def get_by_id(self, key):
        return self.data.get(key)

    async def delete(self, keys):
        for k in keys:
            self.data.pop(k, None)

    async def index_done_callback(self):
        pass


class FakeDocStatus(FakeKV):
    async def get_status_counts(self):
        processed = sum(1 for v in self.data.values() if v.get("status") == "processed")
        return {"processed": processed}


class FakeLightRAG:
    def __init__(self):
        self.doc_status = FakeDocStatus()
        self.deleted = []
        self.data_queries = []

    async def aquery_data(self, query, param):
        self.data_queries.append(param)
        return {"status": "success", "data": {"chunks": [{"file_path": "report.docx", "content": "x"}]}}

    async def adelete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)
        self.doc_status.data.pop(doc_id, None)
        return SimpleNamespace(status="success", message="")


class FakeRAG:
    def __init__(self, lightrag, fail_for=()):
        self.lightrag = lightrag
        self.multimodal_status_cache = FakeKV()
        self.inserted = []
        self.fail_for = fail_for
        self.queries = []

    async def insert_content_list(self, content_list, file_path, doc_id, **kwargs):
        if any(name in file_path for name in self.fail_for):
            raise RuntimeError("boom")
        self.inserted.append((file_path, doc_id))
        self.lightrag.doc_status.data[doc_id] = {"status": "processed"}
        self.multimodal_status_cache.data[doc_id] = {"multimodal_processed": True}

    async def aquery(self, question, mode, **kwargs):
        self.queries.append(kwargs)
        return "answer"


@pytest.fixture
def make_pipeline(settings, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def make(fail_for=()):
        p = object.__new__(RAGPipeline)  # skip __init__: no embedding model, no API client
        p.settings = settings
        p.parser = DocumentParser(settings)
        p.models = SimpleNamespace(usage=TokenUsage())
        p.manifest = Manifest.load(settings.manifest_file)
        p.entity_types = ["Product"]
        p.lightrag = FakeLightRAG()
        p.rag = FakeRAG(p.lightrag, fail_for)
        p.rerank_func = None
        p._initialized = True
        return p

    return make


def run(coro):
    return asyncio.run(coro)


def test_ingest_then_unchanged(make_pipeline, make_docx):
    path = make_docx()
    p = make_pipeline()
    assert [r.status for r in run(p.ingest([path]))] == ["indexed"]
    assert p.rag.inserted == [("report.docx", p.manifest.get(path).doc_id)]

    assert [r.status for r in run(p.ingest([path]))] == ["unchanged"]
    assert len(p.rag.inserted) == 1  # not parsed or inserted again


def test_edited_file_replaces_old_version(make_pipeline, make_docx, settings):
    path = make_docx()
    p = make_pipeline()
    run(p.ingest([path]))
    old_id = p.manifest.get(path).doc_id
    (settings.output_dir / "images" / old_id).mkdir(parents=True)

    make_docx(paragraphs=("Revenue grew 25% in Q3.",))
    (result,) = run(p.ingest([path]))

    assert result.status == "updated" and result.doc_id != old_id
    assert p.lightrag.deleted == [old_id]
    assert old_id not in p.rag.multimodal_status_cache.data
    assert not (settings.output_dir / "images" / old_id).exists()
    assert Manifest.load(settings.manifest_file).get(path).doc_id == result.doc_id


def test_failed_file_is_reported_and_old_version_kept(make_pipeline, make_docx, make_xlsx):
    good, bad = make_docx(), make_xlsx()
    p = make_pipeline(fail_for=("sales",))
    results = {r.source.name: r for r in run(p.ingest([good, bad]))}

    assert results["report.docx"].status == "indexed"
    assert results["sales.xlsx"].status == "failed" and "boom" in results["sales.xlsx"].error
    assert p.manifest.get(bad) is None


def test_identical_copies_share_a_document(make_pipeline, make_docx):
    a = make_docx("a/report.docx")
    b = make_docx("b/report.docx")
    p = make_pipeline()
    run(p.ingest([a, b]))
    assert len(p.rag.inserted) == 1  # same content, indexed once

    run(p.remove([a]))
    assert p.lightrag.deleted == []  # b still uses it
    run(p.remove([b]))
    assert len(p.lightrag.deleted) == 1


def test_remove_folder_and_deleted_files(make_pipeline, make_docx, make_xlsx, tmp_path):
    a = make_docx("q3/report.docx")
    p = make_pipeline()
    run(p.ingest([a, make_xlsx()]))
    a.unlink()  # the file is gone from disk but still in the index

    results = run(p.remove([tmp_path / "q3"]))
    assert [(r.label, r.ok) for r in results] == [("q3/report.docx", True)]
    assert list(p.indexed_files()) == [p.manifest.key(tmp_path / "sales.xlsx")]
    assert run(p.remove([tmp_path / "nothing.docx"])) == []


def test_query_asks_for_locations_and_skips_rerank_without_model(make_pipeline, make_docx):
    p = make_pipeline()
    run(p.ingest([make_docx()]))
    assert run(p.query("q?")) == "answer"
    assert p.rag.queries[0]["user_prompt"] == CITATION_PROMPT
    assert p.rag.queries[0]["enable_rerank"] is False


def test_query_reranks_when_a_model_is_set(make_pipeline, make_docx):
    p = make_pipeline()
    p.rerank_func = object()
    run(p.ingest([make_docx()]))
    run(p.query("q?"))
    assert p.rag.queries[0]["enable_rerank"] is True


def test_query_without_index(make_pipeline):
    with pytest.raises(RuntimeError, match="brag ingest"):
        run(make_pipeline().query("q?"))


def test_retrieve_returns_ranked_chunks(make_pipeline, make_docx):
    p = make_pipeline()
    run(p.ingest([make_docx()]))
    data = run(p.retrieve("q?", mode="naive"))
    assert data["chunks"][0]["file_path"] == "report.docx"
    assert p.lightrag.data_queries[0].mode == "naive"


def test_rag_anything_keeps_full_labels(settings):
    from brag.pipeline import rag_anything_config

    assert rag_anything_config(settings).use_full_path is True


def test_collect_files_skips_lock_files_and_missing(tmp_path, make_docx):
    make_docx("docs/a.docx")
    (tmp_path / "docs" / "~$a.docx").write_bytes(b"")
    (tmp_path / "docs" / "notes.txt").write_text("x")
    files = collect_files([tmp_path / "docs", tmp_path / "missing.docx"])
    assert [f.name for f in files] == ["a.docx"]
