"""FastAPI application.

Endpoints (all under /v1 need a bearer token):

    GET    /healthz                                   liveness, no auth
    GET    /v1/me                                     the caller and their indexes
    POST   /v1/indexes/{index}/query                  {question, mode} -> answer + sources
    POST   /v1/indexes/{index}/retrieve               {question, mode} -> ranked chunks, no answer
    GET    /v1/indexes/{index}/documents              indexed documents
    POST   /v1/indexes/{index}/documents              multipart upload -> 202 + job (writers)
    DELETE /v1/indexes/{index}/documents/{name}       remove -> 202 + job (writers)
    POST   /v1/indexes/{index}/sync                   pull the index's sources -> 202 + job (writers)
    GET    /v1/indexes/{index}/usage                  token budget and the caller's query count
    GET    /v1/jobs/{job_id}                          ingest/remove/sync job status

An index the caller cannot read answers 404, so its existence is not
revealed. Uploads and removals run in the background, one at a time per
index, because LightRAG's pipeline has a single writer per workspace.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .. import __version__
from ..config import SUPPORTED_EXTENSIONS, Settings
from .access import IndexAccess
from .audit import AuditLog
from .auth import AuthError, OIDCVerifier, User
from .usage import LimitExceeded, UsageStore

logger = logging.getLogger(__name__)

Mode = Literal["mix", "hybrid", "local", "global", "naive"]
LOCAL_USER = User("local", "local", frozenset())


@dataclass
class ServiceConfig:
    access: dict[str, IndexAccess]
    verifier: OIDCVerifier | None  # None: no authentication (loopback development only)
    audit: AuditLog
    data_dir: Path
    base_settings: Settings
    max_upload_bytes: int = 100 * 1024 * 1024
    sources: list[Any] = field(default_factory=list)  # connectors.SourceConfig, each with an index
    sync_interval_minutes: int = 0  # 0: sync only on request


@dataclass
class Job:
    id: str
    index: str
    kind: Literal["ingest", "remove", "sync"]
    files: list[str]
    user: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_at: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Mode = "mix"


class IndexManager:
    """One pipeline per index (LightRAG workspace), created on first use, plus a
    background worker per index that runs ingest and remove jobs in order."""

    def __init__(self, config: ServiceConfig, pipeline_factory: Callable[[Settings], Any]):
        self.config = config
        self.factory = pipeline_factory
        self.pipelines: dict[str, Any] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        self.workers: dict[str, asyncio.Task] = {}
        self.jobs: dict[str, Job] = {}
        self._create_lock = asyncio.Lock()
        self.usage = UsageStore(config.data_dir / "usage.json")

    def record_usage(self, index: str) -> None:
        pipeline = self.pipelines.get(index)
        if pipeline is not None:
            self.usage.record_tokens(index, pipeline.models.usage)

    def sources_for(self, index: str) -> list[Any]:
        return [src for src in self.config.sources if src.index == index]

    def upload_dir(self, index: str) -> Path:
        return self.config.data_dir / index

    async def pipeline(self, index: str) -> Any:
        async with self._create_lock:
            if index not in self.pipelines:
                settings = replace(self.config.base_settings, workspace=index)
                self.pipelines[index] = await asyncio.to_thread(self.factory, settings)
            return self.pipelines[index]

    def submit(self, job: Job) -> Job:
        self.jobs[job.id] = job
        if job.index not in self.queues:
            self.queues[job.index] = asyncio.Queue()
            self.workers[job.index] = asyncio.create_task(self._worker(job.index))
        self.queues[job.index].put_nowait(job)
        return job

    async def _worker(self, index: str) -> None:
        queue = self.queues[index]
        while True:
            job: Job = await queue.get()
            job.status = "running"
            try:
                pipeline = await self.pipeline(index)
                paths = [self.upload_dir(index) / name for name in job.files]
                if job.kind == "sync":
                    from ..connectors import sync

                    job.results, failed = [], 0
                    for source in self.sources_for(index):
                        try:
                            r = await sync(pipeline, source)
                            job.results.append(asdict(r))
                            failed += len(r.failed)
                        except Exception as e:
                            job.results.append({"source": source.name, "error": f"{type(e).__name__}: {e}"})
                            failed += 1
                    job.status = "failed" if failed else "done"
                    job.error = f"{failed} problem(s); see results" if failed else None
                elif job.kind == "ingest":
                    results = [await _ingest_one(pipeline, p) for p in paths]
                    job.results = results
                    failed = [r for r in results if r["status"] == "failed"]
                    job.status = "failed" if failed else "done"
                    job.error = f"{len(failed)} file(s) failed" if failed else None
                else:
                    removed = await pipeline.remove(paths)
                    job.results = [{"file": r.label, "removed": r.ok, "error": r.error} for r in removed]
                    for p, r in zip(paths, removed):
                        if r.ok:
                            p.unlink(missing_ok=True)
                    job.status = "done" if all(r.ok for r in removed) else "failed"
            except Exception as e:
                logger.exception("Job %s failed", job.id)
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                self.record_usage(index)
                job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.config.audit.write(
                    "job_finished", job_id=job.id, index=index, kind=job.kind, files=job.files,
                    status=job.status, error=job.error, user=job.user,
                )
                queue.task_done()

    async def sync_periodically(self, minutes: int) -> None:
        while True:
            await asyncio.sleep(minutes * 60)
            for index in sorted({src.index for src in self.config.sources}):
                busy = any(j.index == index and j.kind == "sync" and j.status in ("queued", "running")
                           for j in self.jobs.values())
                if not busy:
                    self.submit(Job(uuid.uuid4().hex, index, "sync", [], "scheduler"))

    async def close(self) -> None:
        for task in self.workers.values():
            task.cancel()
        for pipeline in self.pipelines.values():
            try:
                await pipeline.close()
            except Exception:
                logger.exception("Error closing a pipeline")


async def _ingest_one(pipeline: Any, path: Path) -> dict[str, Any]:
    try:
        r = await pipeline.ingest_file(path, label=path.name)
        return {"file": path.name, "status": r.status, "stats": r.stats, "error": r.error}
    except Exception as e:
        return {"file": path.name, "status": "failed", "error": f"{type(e).__name__}: {e}"}


def _safe_filename(name: str | None) -> str:
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name or name.startswith((".", "~$")) or name != Path(name).name:
        raise HTTPException(400, f"Invalid file name: {name!r}")
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {name!r}")
    return name


def _sources(chunks: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(c.get("file_path", "")) for c in chunks if c.get("file_path")))


def create_app(config: ServiceConfig, pipeline_factory: Callable[[Settings], Any] | None = None) -> FastAPI:
    if pipeline_factory is None:
        from ..pipeline import RAGPipeline

        pipeline_factory = RAGPipeline
    manager = IndexManager(config, pipeline_factory)
    audit = config.audit

    unknown = sorted({src.index for src in config.sources} - set(config.access))
    if unknown:
        raise ValueError(f"Sources point at indexes not in access.yaml: {', '.join(map(str, unknown))}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ticker = None
        if config.sync_interval_minutes > 0 and config.sources:
            ticker = asyncio.create_task(manager.sync_periodically(config.sync_interval_minutes))
        yield
        if ticker:
            ticker.cancel()
        await manager.close()

    app = FastAPI(title="brag", version=__version__, lifespan=lifespan)
    app.state.manager = manager
    bearer = HTTPBearer(auto_error=False)

    def current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> User:
        if config.verifier is None:
            return LOCAL_USER
        if creds is None:
            raise HTTPException(401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        try:
            return config.verifier.verify(creds.credentials)
        except AuthError as e:
            audit.write("auth_failed", path=request.url.path, reason=str(e)[:200])
            raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})

    def can_read(user: User, index: str) -> bool:
        return config.verifier is None or (index in config.access and config.access[index].can_read(user.groups))

    def can_write(user: User, index: str) -> bool:
        return config.verifier is None or (index in config.access and config.access[index].can_write(user.groups))

    def require(user: User, index: str, write: bool = False) -> None:
        if index not in config.access or not can_read(user, index):
            audit.write("denied", user=user.subject, index=index, need="write" if write else "read")
            raise HTTPException(404, "Index not found")
        if write and not can_write(user, index):
            audit.write("denied", user=user.subject, index=index, need="write")
            raise HTTPException(403, "You may read this index but not change it")

    def too_many(e: LimitExceeded) -> HTTPException:
        return HTTPException(429, str(e), headers={"Retry-After": str(max(e.retry_after, 1))})

    def admit_query(user: User, index: str) -> None:
        limits = config.access[index].limits
        try:
            manager.usage.check_tokens(index, limits.monthly_tokens)
            manager.usage.admit_query(index, user.subject, limits.user_queries_per_minute, limits.user_queries_per_day)
        except LimitExceeded as e:
            audit.write("limited", user=user.subject, index=index, reason=str(e))
            raise too_many(e)

    def admit_job(user: User, index: str) -> None:
        try:
            manager.usage.check_tokens(index, config.access[index].limits.monthly_tokens)
        except LimitExceeded as e:
            audit.write("limited", user=user.subject, index=index, reason=str(e))
            raise too_many(e)

    def index_error(e: RuntimeError) -> HTTPException:
        if "index is empty" in str(e).lower():
            return HTTPException(409, "This index has no documents yet")
        return HTTPException(500, "Query failed")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/me")
    async def me(user: User = Depends(current_user)) -> dict[str, Any]:
        indexes = [
            {"name": name, "write": can_write(user, name)}
            for name in sorted(config.access)
            if can_read(user, name)
        ]
        return {"subject": user.subject, "name": user.name, "groups": sorted(user.groups), "indexes": indexes}

    @app.post("/v1/indexes/{index}/query")
    async def query(index: str, body: QueryRequest, user: User = Depends(current_user)) -> dict[str, Any]:
        require(user, index)
        admit_query(user, index)
        pipeline = await manager.pipeline(index)
        started = time.monotonic()
        try:
            # Retrieval first, to record which documents the answer is built
            # from; the query reuses LightRAG's cached keyword extraction.
            data = await pipeline.retrieve(body.question, mode=body.mode)
            answer = await pipeline.query(body.question, mode=body.mode)
        except RuntimeError as e:
            raise index_error(e)
        finally:
            manager.record_usage(index)
        sources = _sources(data.get("chunks", []))
        audit.write(
            "query", user=user.subject, name=user.name, index=index, mode=body.mode,
            question=body.question, sources=sources, duration_ms=round((time.monotonic() - started) * 1000),
        )
        return {"answer": answer, "sources": sources}

    @app.post("/v1/indexes/{index}/retrieve")
    async def retrieve(index: str, body: QueryRequest, user: User = Depends(current_user)) -> dict[str, Any]:
        require(user, index)
        admit_query(user, index)
        pipeline = await manager.pipeline(index)
        try:
            data = await pipeline.retrieve(body.question, mode=body.mode)
        except RuntimeError as e:
            raise index_error(e)
        finally:
            manager.record_usage(index)
        chunks = [{"file": c.get("file_path"), "content": c.get("content")} for c in data.get("chunks", [])]
        audit.write(
            "retrieve", user=user.subject, name=user.name, index=index, mode=body.mode,
            question=body.question, sources=_sources(data.get("chunks", [])),
        )
        return {"chunks": chunks}

    @app.get("/v1/indexes/{index}/documents")
    async def list_documents(index: str, user: User = Depends(current_user)) -> dict[str, Any]:
        require(user, index)
        pipeline = await manager.pipeline(index)
        docs = sorted(
            ({"name": e.label, "indexed_at": e.indexed_at} for e in pipeline.manifest.files.values()),
            key=lambda d: d["name"],
        )
        return {"documents": docs}

    @app.post("/v1/indexes/{index}/documents", status_code=status.HTTP_202_ACCEPTED)
    async def upload(index: str, files: list[UploadFile] = File(...), user: User = Depends(current_user)) -> dict:
        require(user, index, write=True)
        admit_job(user, index)
        names = [_safe_filename(f.filename) for f in files]
        if len(set(names)) != len(names):
            raise HTTPException(400, "Duplicate file names in one upload")
        target = manager.upload_dir(index)
        target.mkdir(parents=True, exist_ok=True)
        for f, name in zip(files, names):
            tmp = target / f".upload-{uuid.uuid4().hex}"
            size = 0
            try:
                with tmp.open("wb") as out:
                    while chunk := await f.read(1024 * 1024):
                        size += len(chunk)
                        if size > config.max_upload_bytes:
                            raise HTTPException(413, f"{name} is larger than {config.max_upload_bytes // 2**20} MB")
                        out.write(chunk)
                tmp.replace(target / name)  # same name: the new version replaces the old on ingest
            finally:
                tmp.unlink(missing_ok=True)
        job = manager.submit(Job(uuid.uuid4().hex, index, "ingest", names, user.subject))
        audit.write("upload", user=user.subject, name=user.name, index=index, files=names, job_id=job.id)
        return {"job_id": job.id, "status": job.status}

    @app.delete("/v1/indexes/{index}/documents/{name}", status_code=status.HTTP_202_ACCEPTED)
    async def remove(index: str, name: str, user: User = Depends(current_user)) -> dict:
        require(user, index, write=True)
        name = _safe_filename(name)
        pipeline = await manager.pipeline(index)
        if pipeline.manifest.get(manager.upload_dir(index) / name) is None:
            raise HTTPException(404, "Document not found")
        job = manager.submit(Job(uuid.uuid4().hex, index, "remove", [name], user.subject))
        audit.write("remove", user=user.subject, name=user.name, index=index, files=[name], job_id=job.id)
        return {"job_id": job.id, "status": job.status}

    @app.post("/v1/indexes/{index}/sync", status_code=status.HTTP_202_ACCEPTED)
    async def sync_index(index: str, user: User = Depends(current_user)) -> dict:
        require(user, index, write=True)
        if not manager.sources_for(index):
            raise HTTPException(404, "This index has no sources configured")
        admit_job(user, index)
        job = manager.submit(Job(uuid.uuid4().hex, index, "sync", [], user.subject))
        audit.write("sync", user=user.subject, name=user.name, index=index, job_id=job.id,
                    sources=[s.name for s in manager.sources_for(index)])
        return {"job_id": job.id, "status": job.status}

    @app.get("/v1/indexes/{index}/usage")
    async def usage(index: str, user: User = Depends(current_user)) -> dict[str, Any]:
        require(user, index)
        limits = config.access[index].limits
        return {
            "month_tokens": manager.usage.tokens_this_month(index),
            "monthly_tokens_limit": limits.monthly_tokens,
            "your_queries_today": manager.usage.queries_today(index, user.subject),
            "user_queries_per_day_limit": limits.user_queries_per_day,
            "user_queries_per_minute_limit": limits.user_queries_per_minute,
        }

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str, user: User = Depends(current_user)) -> dict[str, Any]:
        job = manager.jobs.get(job_id)
        if job is None or not can_read(user, job.index):
            raise HTTPException(404, "Job not found")
        return asdict(job)

    return app
