"""The HTTP service, with a locally signed OIDC token and fake pipelines."""

import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
jwt = pytest.importorskip("jwt")
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from brag.manifest import Manifest  # noqa: E402
from brag.pipeline import IngestResult, RemoveResult  # noqa: E402
from brag.service.access import IndexAccess, load_access  # noqa: E402
from brag.service.app import ServiceConfig, create_app  # noqa: E402
from brag.service.audit import AuditLog  # noqa: E402
from brag.service.auth import AuthError, OIDCVerifier, claim, groups_from  # noqa: E402

ISSUER, AUDIENCE = "https://idp.example.com", "api://brag"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def token(sub="alice", groups=("finance",), key=KEY, **overrides):
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": sub, "iat": now, "exp": now + 300,
              "preferred_username": f"{sub}@example.com", "groups": list(groups), **overrides}
    return jwt.encode(claims, key, algorithm="RS256")


def verifier(**kwargs):
    return OIDCVerifier(ISSUER, AUDIENCE, key_resolver=lambda _: KEY.public_key(), **kwargs)


# -- auth ----------------------------------------------------------------------


def test_valid_token():
    user = verifier().verify(token())
    assert user.subject == "alice" and user.name == "alice@example.com" and user.groups == {"finance"}


@pytest.mark.parametrize("bad", [
    lambda: token(key=OTHER_KEY),
    lambda: token(exp=int(time.time()) - 3600),
    lambda: token(aud="api://other"),
    lambda: token(iss="https://evil.example.com"),
    lambda: "not-a-jwt",
])
def test_rejected_tokens(bad):
    with pytest.raises(AuthError):
        verifier().verify(bad())


def test_unsigned_token_is_rejected():
    unsigned = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": int(time.time()) + 60}, None, algorithm="none")
    with pytest.raises(AuthError):
        verifier().verify(unsigned)


def test_groups_claim_paths():
    claims = {"realm_access": {"roles": ["a", "b"]}, "scp": "read write"}
    assert claim(claims, "realm_access.roles") == ["a", "b"]
    assert claim(claims, "missing.path") is None
    assert groups_from("read write") == {"read", "write"} and groups_from(None) == frozenset()
    user = verifier(groups_claim="realm_access.roles").verify(token(realm_access={"roles": ["hr"]}))
    assert user.groups == {"hr"}


# -- access --------------------------------------------------------------------


def test_load_access(tmp_path):
    path = tmp_path / "access.yaml"
    path.write_text("indexes:\n  finance:\n    readers: [fin]\n    writers: [fin-admin]\n  handbook:\n    readers: ['*']\n")
    access = load_access(path)
    assert access["finance"].can_read(frozenset({"fin"})) and not access["finance"].can_write(frozenset({"fin"}))
    assert access["finance"].can_read(frozenset({"fin-admin"}))  # writers can read
    assert access["handbook"].can_read(frozenset()) and not access["handbook"].can_write(frozenset({"x"}))


@pytest.mark.parametrize("body,error", [
    ("indexes: {}\n", "no indexes"),
    ("indexes:\n  Bad Name:\n    readers: [a]\n", "Index name"),
    ("indexes:\n  x:\n    readers: []\n", "no readers"),
])
def test_invalid_access(tmp_path, body, error):
    path = tmp_path / "access.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=error):
        load_access(path)


# -- app -----------------------------------------------------------------------


class FakePipeline:
    def __init__(self, settings):
        self.settings = settings
        self.manifest = Manifest(settings.manifest_file)
        self.closed = False

    async def retrieve(self, question, mode):
        if not self.manifest.files:
            raise RuntimeError("The index is empty. Run `brag ingest <paths>` first.")
        return {"chunks": [{"file_path": e.label, "content": "text"} for e in self.manifest.files.values()]}

    async def query(self, question, mode):
        return f"answer to {question}"

    async def ingest_file(self, path, label=None):
        if "broken" in path.name:
            raise RuntimeError("parse error")
        self.manifest.set(path, f"doc-{path.name}", label)
        return IngestResult(path, "indexed", f"doc-{path.name}", {"text_chunks": 1})

    async def remove(self, paths):
        out = []
        for p in paths:
            entry = self.manifest.pop(p)
            out.append(RemoveResult(entry.label if entry else p.name, entry is not None))
        return out

    async def close(self):
        self.closed = True


@pytest.fixture
def setup(tmp_path, settings):
    audit_path = tmp_path / "audit.jsonl"
    config = ServiceConfig(
        access={
            "finance": IndexAccess("finance", frozenset({"finance"}), frozenset({"finance-admins"})),
            "handbook": IndexAccess("handbook", frozenset({"*"}), frozenset({"hr"})),
        },
        verifier=verifier(),
        audit=AuditLog(audit_path),
        data_dir=tmp_path / "data",
        base_settings=settings,
        max_upload_bytes=1024,
    )
    app = create_app(config, pipeline_factory=FakePipeline)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, audit=audit_path, config=config)


def auth(sub="alice", groups=("finance",)):
    return {"Authorization": f"Bearer {token(sub, groups)}"}


def audit_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_for(client, job_id, headers, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def upload(client, index, files, headers):
    return client.post(
        f"/v1/indexes/{index}/documents",
        files=[("files", (name, data, "application/octet-stream")) for name, data in files],
        headers=headers,
    )


def test_health_needs_no_token(setup):
    assert setup.client.get("/healthz").json()["status"] == "ok"


def test_missing_and_bad_tokens(setup):
    assert setup.client.get("/v1/me").status_code == 401
    r = setup.client.get("/v1/me", headers={"Authorization": f"Bearer {token(key=OTHER_KEY)}"})
    assert r.status_code == 401
    assert audit_events(setup.audit)[-1]["event"] == "auth_failed"


def test_me_lists_only_readable_indexes(setup):
    body = setup.client.get("/v1/me", headers=auth(groups=("finance",))).json()
    assert body["indexes"] == [{"name": "finance", "write": False}, {"name": "handbook", "write": False}]
    body = setup.client.get("/v1/me", headers=auth(groups=("sales",))).json()
    assert body["indexes"] == [{"name": "handbook", "write": False}]


def test_unreadable_index_looks_missing(setup):
    r = setup.client.post("/v1/indexes/finance/query", json={"question": "q"}, headers=auth(groups=("sales",)))
    assert r.status_code == 404
    assert setup.client.post("/v1/indexes/nope/query", json={"question": "q"}, headers=auth()).status_code == 404
    assert audit_events(setup.audit)[-1]["event"] == "denied"


def test_readers_cannot_upload(setup):
    r = upload(setup.client, "finance", [("a.docx", b"x")], auth(groups=("finance",)))
    assert r.status_code == 403


def test_upload_query_list_remove(setup):
    admin = auth("bob", ("finance-admins",))
    r = upload(setup.client, "finance", [("q3.docx", b"one"), ("q4.xlsx", b"two")], admin)
    assert r.status_code == 202
    job = wait_for(setup.client, r.json()["job_id"], admin)
    assert job["status"] == "done" and [x["file"] for x in job["results"]] == ["q3.docx", "q4.xlsx"]

    reader = auth("alice", ("finance",))
    docs = setup.client.get("/v1/indexes/finance/documents", headers=reader).json()["documents"]
    assert [d["name"] for d in docs] == ["q3.docx", "q4.xlsx"]

    r = setup.client.post("/v1/indexes/finance/query", json={"question": "What grew?", "mode": "naive"}, headers=reader)
    assert r.json() == {"answer": "answer to What grew?", "sources": ["q3.docx", "q4.xlsx"]}
    event = audit_events(setup.audit)[-1]
    assert event["event"] == "query" and event["user"] == "alice" and event["question"] == "What grew?"
    assert event["sources"] == ["q3.docx", "q4.xlsx"] and event["index"] == "finance"

    assert setup.client.delete("/v1/indexes/finance/documents/q3.docx", headers=reader).status_code == 403
    r = setup.client.delete("/v1/indexes/finance/documents/q3.docx", headers=admin)
    assert wait_for(setup.client, r.json()["job_id"], admin)["status"] == "done"
    assert not (setup.config.data_dir / "finance" / "q3.docx").exists()
    docs = setup.client.get("/v1/indexes/finance/documents", headers=reader).json()["documents"]
    assert [d["name"] for d in docs] == ["q4.xlsx"]
    assert setup.client.delete("/v1/indexes/finance/documents/q3.docx", headers=admin).status_code == 404


def test_failed_file_marks_the_job_failed(setup):
    admin = auth("bob", ("finance-admins",))
    r = upload(setup.client, "finance", [("broken.docx", b"x"), ("ok.docx", b"y")], admin)
    job = wait_for(setup.client, r.json()["job_id"], admin)
    assert job["status"] == "failed" and job["error"] == "1 file(s) failed"
    assert {x["file"]: x["status"] for x in job["results"]} == {"broken.docx": "failed", "ok.docx": "indexed"}


def test_jobs_are_private_to_index_readers(setup):
    admin = auth("bob", ("finance-admins",))
    job_id = upload(setup.client, "finance", [("a.docx", b"x")], admin).json()["job_id"]
    assert setup.client.get(f"/v1/jobs/{job_id}", headers=auth(groups=("sales",))).status_code == 404


@pytest.mark.parametrize("name", ["../evil.docx", ".hidden.docx", "~$lock.docx", "notes.txt", "a\\b.docx"])
def test_unsafe_or_unsupported_names(setup, name):
    r = upload(setup.client, "finance", [(name, b"x")], auth("bob", ("finance-admins",)))
    assert r.status_code == 400
    assert not list((setup.config.data_dir).rglob("evil.docx"))


def test_upload_size_limit(setup):
    r = upload(setup.client, "finance", [("big.docx", b"x" * 2048)], auth("bob", ("finance-admins",)))
    assert r.status_code == 413
    assert not (setup.config.data_dir / "finance" / "big.docx").exists()


def test_empty_index_query(setup):
    r = setup.client.post("/v1/indexes/handbook/query", json={"question": "q"}, headers=auth(groups=()))
    assert r.status_code == 409


def test_each_index_gets_its_own_workspace(setup):
    admin = auth("bob", ("finance-admins", "hr"))
    for index in ("finance", "handbook"):
        wait_for(setup.client, upload(setup.client, index, [("a.docx", b"x")], admin).json()["job_id"], admin)
    pipelines = setup.app.state.manager.pipelines
    assert pipelines["finance"].settings.workspace == "finance"
    assert pipelines["finance"].settings.manifest_file != pipelines["handbook"].settings.manifest_file


def test_question_logging_can_be_turned_off(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl", include_questions=False)
    log.write("query", user="u", question="secret?")
    assert "question" not in audit_events(tmp_path / "a.jsonl")[0]
