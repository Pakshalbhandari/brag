"""Mirror a remote source into an index.

Each sync lists the source, downloads documents that are new or whose version
changed into a local mirror folder, indexes them (an edited document replaces
its old version), and removes documents that disappeared from the source. A
state file per source remembers which version of each document is indexed.

Deletions only happen after a complete listing: if listing fails, nothing is
removed.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Connector, RemoteDoc, clean_path, supported

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    source: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: int = 0
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.source}: {len(self.added)} added, {len(self.updated)} updated, "
            f"{self.unchanged} unchanged, {len(self.removed)} removed, {len(self.failed)} failed"
        )


class SyncState:
    def __init__(self, path: Path):
        self.path = path
        self.docs: dict[str, dict[str, Any]] = {}
        if path.exists():
            self.docs = json.loads(path.read_text(encoding="utf-8")).get("docs", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".sync-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"docs": self.docs}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


async def sync_source(pipeline, connector: Connector, mirror_dir: Path, state_file: Path) -> SyncResult:
    """Bring the index in line with the source. ``mirror_dir`` holds this source's local copies."""
    name = connector.name
    result = SyncResult(name)
    state = SyncState(state_file)
    remote = [d for d in await connector.list() if supported(d.path.rsplit("/", 1)[-1])]
    seen: set[str] = set()

    for doc in remote:
        seen.add(doc.id)
        label = f"{name}/{clean_path(doc.path)}"
        local = mirror_dir / clean_path(doc.path)
        previous = state.docs.get(doc.id)
        try:
            if (
                previous
                and previous["version"] == doc.version
                and previous["local"] == str(local)
                and local.exists()
                and pipeline.manifest.get(local) is not None
            ):
                result.unchanged += 1
                continue
            if previous and previous["local"] != str(local):  # moved or renamed in the source
                await _forget(pipeline, Path(previous["local"]))
            await connector.download(doc, local)
            ingested = await pipeline.ingest_file(local, label=label)
            if ingested.status == "failed":
                raise RuntimeError(ingested.error or "indexing failed")
            (result.updated if previous else result.added).append(label)
            state.docs[doc.id] = {"version": doc.version, "local": str(local), "label": label}
        except Exception as e:
            logger.debug("Sync of %s failed", label, exc_info=True)
            result.failed[label] = f"{type(e).__name__}: {e}"
        state.save()

    for doc_id in [d for d in state.docs if d not in seen]:
        entry = state.docs[doc_id]
        try:
            await _forget(pipeline, Path(entry["local"]))
            result.removed.append(entry["label"])
            del state.docs[doc_id]
        except Exception as e:
            result.failed[entry["label"]] = f"{type(e).__name__}: {e}"
        state.save()
    return result


async def _forget(pipeline, local: Path) -> None:
    removed = await pipeline.remove([local])
    for r in removed:
        if not r.ok:
            raise RuntimeError(r.error or "removal failed")
    local.unlink(missing_ok=True)


def mirror_paths(settings, source: str) -> tuple[Path, Path]:
    """(mirror folder, state file) for a source, inside the index's own folder."""
    base = settings.index_dir / "sources" / source
    return base / "files", base / "state.json"

