"""Connectors against mocked HTTP APIs, and the sync engine against a fake index."""

import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from brag.connectors import load_sources
from brag.connectors.base import ConnectorError, RemoteDoc, clean_path
from brag.connectors.confluence import ConfluenceConnector
from brag.connectors.google import DriveConnector, GCSConnector
from brag.connectors.microsoft import GraphConnector
from brag.connectors.sync import sync_source
from brag.manifest import Manifest
from brag.pipeline import IngestResult, RemoveResult


def run(coro):
    return asyncio.run(coro)


def client(handler, **kwargs):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


# -- shared --------------------------------------------------------------------


def test_clean_path():
    assert clean_path("Reports/Q3: draft?.docx") == "Reports/Q3_ draft_.docx"
    assert clean_path("../../etc/passwd") == "etc/passwd"
    assert clean_path("/abs//a\\b.docx") == "abs/a/b.docx"
    with pytest.raises(ConnectorError):
        clean_path("../..")


def test_load_sources(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "sources:\n"
        "  - {name: sp, type: sharepoint, site_url: 'https://c.sharepoint.com/sites/F', index: finance}\n"
        "  - {name: wiki, type: confluence, base_url: 'https://a.atlassian.net/wiki', space: ENG}\n"
    )
    sp, wiki = load_sources(path)
    assert (sp.type, sp.index, sp.options["site_url"]) == ("sharepoint", "finance", "https://c.sharepoint.com/sites/F")
    assert wiki.index is None


@pytest.mark.parametrize("body,error", [
    ("sources: []\n", "no sources"),
    ("sources:\n  - {name: x, type: dropbox}\n", "type must be"),
    ("sources:\n  - {name: x, type: gcs}\n", "missing bucket"),
    ("sources:\n  - {name: x, type: gcs, bucket: b, colour: red}\n", "unknown option"),
    ("sources:\n  - {name: 'bad name', type: gcs, bucket: b}\n", "Source name"),
    ("sources:\n  - {name: x, type: gcs, bucket: b}\n  - {name: x, type: gcs, bucket: c}\n", "Duplicate"),
])
def test_invalid_sources(tmp_path, body, error):
    path = tmp_path / "sources.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=error):
        load_sources(path)


# -- Microsoft Graph -----------------------------------------------------------


def graph_handler(calls):
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        url, path = str(request.url), request.url.path
        if request.url.host == "login.microsoftonline.com":
            form = parse_qs(request.content.decode())
            assert form["grant_type"] == ["client_credentials"]
            assert form["scope"] == ["https://graph.microsoft.com/.default"]
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        if request.url.host == "download.example.com":
            return httpx.Response(200, content=b"file bytes")
        assert request.headers["Authorization"] == "Bearer tok"
        if path == "/v1.0/sites/contoso.sharepoint.com:/sites/Finance":
            return httpx.Response(200, json={"id": "contoso.sharepoint.com,s1,w1"})
        if path == "/v1.0/sites/contoso.sharepoint.com,s1,w1/drives":
            return httpx.Response(200, json={"value": [{"id": "d0", "name": "Archive"}, {"id": "d1", "name": "Documents"}]})
        if path == "/v1.0/drives/d1/root:/Reports:/children":
            return httpx.Response(200, json={
                "value": [{"id": "fo1", "name": "2026", "folder": {"childCount": 1}},
                          {"id": "f1", "name": "q3.docx", "file": {}, "cTag": "c1", "size": 10}],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/drives/d1/root:/Reports:/children?$skiptoken=2",
            } if "skiptoken" not in url else {"value": [{"id": "f2", "name": "notes.txt", "file": {}, "cTag": "c2"}]})
        if path == "/v1.0/drives/d1/items/fo1/children":
            return httpx.Response(200, json={"value": [{"id": "f3", "name": "q4.xlsx", "file": {}, "eTag": "e3"}]})
        if path == "/v1.0/drives/d1/items/f1/content":
            return httpx.Response(302, headers={"Location": "https://download.example.com/f1?sig=abc"})
        return httpx.Response(404, json={"error": url})
    return handle


@pytest.fixture
def ms_env(monkeypatch):
    for k, v in {"MS_TENANT_ID": "t", "MS_CLIENT_ID": "c", "MS_CLIENT_SECRET": "s"}.items():
        monkeypatch.setenv(k, v)


def test_sharepoint_lists_recursively_and_follows_paging(ms_env):
    calls = []
    conn = GraphConnector("sp", site_url="https://contoso.sharepoint.com/sites/Finance", folder="/Reports/",
                          client=client(graph_handler(calls)))
    docs = run(conn.list())
    assert sorted((d.path, d.version) for d in docs) == [
        ("2026/q4.xlsx", "e3"), ("notes.txt", "c2"), ("q3.docx", "c1")]  # unsupported types are filtered by sync
    assert sum(1 for c in calls if c.url.host == "login.microsoftonline.com") == 1  # token reused


def test_sharepoint_download_does_not_leak_the_token(ms_env, tmp_path):
    calls = []
    conn = GraphConnector("sp", site_url="https://contoso.sharepoint.com/sites/Finance",
                          client=client(graph_handler(calls)))
    run(conn.download(RemoteDoc("f1", "q3.docx", "c1"), tmp_path / "out" / "q3.docx"))
    assert (tmp_path / "out" / "q3.docx").read_bytes() == b"file bytes"
    download = next(c for c in calls if c.url.host == "download.example.com")
    assert "Authorization" not in download.headers


def test_sharepoint_unknown_library(ms_env):
    conn = GraphConnector("sp", site_url="https://contoso.sharepoint.com/sites/Finance", drive="Legal",
                          client=client(graph_handler([])))
    with pytest.raises(ConnectorError, match="no library named 'Legal'"):
        run(conn.list())


def test_graph_needs_exactly_one_target():
    with pytest.raises(ConnectorError):
        GraphConnector("x")


def test_onedrive_uses_the_users_drive(ms_env):
    seen = []

    def handle(request):
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        seen.append(request.url.path)
        if request.url.path == "/v1.0/users/ceo%40contoso.com/drives" or request.url.path == "/v1.0/users/ceo@contoso.com/drives":
            return httpx.Response(200, json={"value": [{"id": "od", "name": "OneDrive"}]})
        if request.url.path == "/v1.0/drives/od/root/children":
            return httpx.Response(200, json={"value": [{"id": "x", "name": "a.pdf", "file": {}, "cTag": "1"}]})
        return httpx.Response(404)

    docs = run(GraphConnector("od", user="ceo@contoso.com", client=client(handle)).list())
    assert [d.path for d in docs] == ["a.pdf"]


# -- Google --------------------------------------------------------------------


class FakeGoogleAuth:
    async def headers(self):
        return {"Authorization": "Bearer g"}


def test_drive_recurses_exports_and_downloads(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        q = parse_qs(request.url.query.decode())
        if request.url.path == "/drive/v3/files":
            folder = q["q"][0].split("'")[1]
            assert "trashed = false" in q["q"][0] and q["supportsAllDrives"] == ["true"]
            files = {
                "root": [
                    {"id": "sub", "name": "Policies", "mimeType": "application/vnd.google-apps.folder"},
                    {"id": "doc", "name": "Handbook", "mimeType": "application/vnd.google-apps.document", "version": "7"},
                    {"id": "form", "name": "Survey", "mimeType": "application/vnd.google-apps.form"},
                ],
                "sub": [{"id": "pdf", "name": "travel.pdf", "mimeType": "application/pdf", "md5Checksum": "m1", "size": "5"}],
            }[folder]
            return httpx.Response(200, json={"files": files})
        if request.url.path == "/drive/v3/files/doc/export":
            assert q["mimeType"] == ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
            return httpx.Response(200, content=b"docx")
        if request.url.path == "/drive/v3/files/pdf":
            assert q["alt"] == ["media"]
            return httpx.Response(200, content=b"pdf")
        return httpx.Response(404)

    conn = DriveConnector("drive", folder_id="root", client=client(handle), auth=FakeGoogleAuth())
    docs = {d.path: d for d in run(conn.list())}
    assert set(docs) == {"Handbook.docx", "Policies/travel.pdf"}  # forms are skipped
    assert docs["Handbook.docx"].version == "7" and docs["Policies/travel.pdf"].version == "m1"
    run(conn.download(docs["Handbook.docx"], tmp_path / "h.docx"))
    run(conn.download(docs["Policies/travel.pdf"], tmp_path / "t.pdf"))
    assert (tmp_path / "h.docx").read_bytes() == b"docx" and (tmp_path / "t.pdf").read_bytes() == b"pdf"
    assert all(r.headers["Authorization"] == "Bearer g" for r in requests)


def test_gcs_pages_strips_prefix_and_encodes_names(tmp_path):
    paths = []

    def handle(request):
        paths.append(request.url.raw_path.decode())
        q = parse_qs(request.url.query.decode())
        if request.url.raw_path.decode().startswith("/storage/v1/b/acme-docs/o?"):
            assert q["prefix"] == ["reports/"]
            if "pageToken" not in q:
                return httpx.Response(200, json={"items": [{"name": "reports/", "generation": "1"},
                                                           {"name": "reports/2026/q3.docx", "generation": "11", "size": "3"}],
                                                 "nextPageToken": "p2"})
            return httpx.Response(200, json={"items": [{"name": "reports/q4.pdf", "generation": "12"}]})
        return httpx.Response(200, content=b"obj")

    conn = GCSConnector("gcs", bucket="acme-docs", prefix="reports/", client=client(handle), auth=FakeGoogleAuth())
    docs = run(conn.list())
    assert [(d.id, d.path, d.version) for d in docs] == [
        ("reports/2026/q3.docx", "2026/q3.docx", "11"), ("reports/q4.pdf", "q4.pdf", "12")]
    run(conn.download(docs[0], tmp_path / "q3.docx"))
    assert paths[-1] == "/storage/v1/b/acme-docs/o/reports%2F2026%2Fq3.docx?alt=media"


# -- Confluence ----------------------------------------------------------------


def test_confluence_pages_attachments_and_paging(tmp_path):
    def handle(request):
        path, q = request.url.path, parse_qs(request.url.query.decode())
        if path == "/wiki/api/v2/spaces":
            assert q["keys"] == ["ENG"]
            return httpx.Response(200, json={"results": [{"id": "77", "key": "ENG"}]})
        if path == "/wiki/api/v2/spaces/77/pages":
            if "cursor" not in q:
                assert q["body-format"] == ["storage"]
                return httpx.Response(200, json={
                    "results": [{"id": "1", "title": "Onboarding", "version": {"number": 3},
                                 "body": {"storage": {"value": "<p>Welcome & <b>hello</b></p>"}}}],
                    "_links": {"next": "/wiki/api/v2/spaces/77/pages?cursor=abc&body-format=storage"},
                })
            return httpx.Response(200, json={"results": [{"id": "2", "title": "Deploys", "version": {"number": 1},
                                                          "body": {"storage": {"value": "<p>Deploy on Tuesdays.</p>"}}}]})
        if path == "/wiki/api/v2/pages/1/attachments":
            return httpx.Response(200, json={
                "results": [{"id": "a1", "title": "checklist.pdf", "downloadLink": "/download/attachments/1/checklist.pdf",
                             "version": {"number": 2}, "fileSize": 4},
                            {"id": "a2", "title": "photo.png", "downloadLink": "/download/attachments/1/photo.png"}],
                "_links": {"base": "https://acme.atlassian.net/wiki"},
            })
        if path == "/wiki/api/v2/pages/2/attachments":
            return httpx.Response(200, json={"results": []})
        if path == "/wiki/download/attachments/1/checklist.pdf":
            return httpx.Response(200, content=b"pdf")
        return httpx.Response(404, json={"path": path})

    conn = ConfluenceConnector("wiki", base_url="https://acme.atlassian.net/wiki/", space="ENG", client=client(handle))
    docs = {d.path: d for d in run(conn.list())}
    assert set(docs) == {"Onboarding.html", "Onboarding/checklist.pdf", "Deploys.html"}
    assert docs["Onboarding.html"].version == "3" and docs["Onboarding/checklist.pdf"].version == "2"

    run(conn.download(docs["Onboarding.html"], tmp_path / "p.html"))
    page = (tmp_path / "p.html").read_text()
    assert "<h1>Onboarding</h1>" in page and "<p>Welcome & <b>hello</b></p>" in page
    run(conn.download(docs["Onboarding/checklist.pdf"], tmp_path / "c.pdf"))
    assert (tmp_path / "c.pdf").read_bytes() == b"pdf"


def test_confluence_needs_credentials(monkeypatch):
    monkeypatch.delenv("CONFLUENCE_EMAIL", raising=False)
    with pytest.raises(ConnectorError, match="CONFLUENCE_EMAIL"):
        ConfluenceConnector("wiki", base_url="https://a.atlassian.net/wiki", space="X")


# -- sync engine ---------------------------------------------------------------


class FakeSource:
    name = "src"

    def __init__(self, docs):
        self.docs = docs
        self.fail_listing = False
        self.downloads = []

    async def list(self):
        if self.fail_listing:
            raise ConnectorError("listing failed")
        return list(self.docs)

    async def download(self, doc, dest):
        self.downloads.append(doc.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"{doc.path}@{doc.version}")

    async def aclose(self):
        pass


class FakeIndex:
    def __init__(self, path):
        self.manifest = Manifest(path)
        self.fail = set()

    async def ingest_file(self, path, label=None):
        if label in self.fail:
            return IngestResult(path, "failed", error="parse error")
        self.manifest.set(path, "doc-" + Path(path).read_text(), label)
        return IngestResult(path, "indexed")

    async def remove(self, paths):
        return [RemoveResult(str(p), self.manifest.pop(p) is not None) for p in paths]


def labels(index):
    return sorted(e.label for e in index.manifest.files.values())


def test_sync_adds_updates_renames_and_removes(tmp_path):
    source = FakeSource([RemoteDoc("1", "a.docx", "v1"), RemoteDoc("2", "dir/b.pdf", "v1"), RemoteDoc("3", "c.txt", "v1")])
    index = FakeIndex(tmp_path / "m.json")
    mirror, state = tmp_path / "mirror", tmp_path / "state.json"

    r = run(sync_source(index, source, mirror, state))
    assert r.added == ["src/a.docx", "src/dir/b.pdf"] and labels(index) == ["src/a.docx", "src/dir/b.pdf"]

    r = run(sync_source(index, source, mirror, state))
    assert r.unchanged == 2 and not r.added and source.downloads.count("a.docx") == 1

    source.docs = [RemoteDoc("1", "a.docx", "v2"), RemoteDoc("2", "moved/b.pdf", "v1")]
    r = run(sync_source(index, source, mirror, state))
    assert r.updated == ["src/a.docx", "src/moved/b.pdf"]
    assert labels(index) == ["src/a.docx", "src/moved/b.pdf"]
    assert not (mirror / "dir" / "b.pdf").exists() and (mirror / "moved" / "b.pdf").exists()

    source.docs = [RemoteDoc("1", "a.docx", "v2")]
    r = run(sync_source(index, source, mirror, state))
    assert r.removed == ["src/moved/b.pdf"] and labels(index) == ["src/a.docx"]
    assert json.loads(state.read_text())["docs"].keys() == {"1"}


def test_failed_listing_removes_nothing(tmp_path):
    source = FakeSource([RemoteDoc("1", "a.docx", "v1")])
    index = FakeIndex(tmp_path / "m.json")
    run(sync_source(index, source, tmp_path / "mirror", tmp_path / "state.json"))
    source.fail_listing = True
    with pytest.raises(ConnectorError):
        run(sync_source(index, source, tmp_path / "mirror", tmp_path / "state.json"))
    assert labels(index) == ["src/a.docx"]


def test_failed_document_is_retried_next_time(tmp_path):
    source = FakeSource([RemoteDoc("1", "a.docx", "v1")])
    index = FakeIndex(tmp_path / "m.json")
    index.fail = {"src/a.docx"}
    r = run(sync_source(index, source, tmp_path / "mirror", tmp_path / "state.json"))
    assert r.failed == {"src/a.docx": "RuntimeError: parse error"}
    index.fail = set()
    r = run(sync_source(index, source, tmp_path / "mirror", tmp_path / "state.json"))
    assert r.added == ["src/a.docx"]
