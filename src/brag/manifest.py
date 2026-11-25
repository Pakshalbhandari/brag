"""Which source file is indexed under which document ID.

LightRAG only knows document IDs, which are content hashes, so on its own it
cannot tell that a changed report.docx replaces an earlier version. The manifest
records path -> doc_id, so re-ingesting an edited file removes the old version,
and `brag remove` can find what to delete.

Two paths with identical contents share one doc_id; a document is only deleted
once no path refers to it any more.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1


@dataclass
class ManifestEntry:
    doc_id: str
    label: str
    indexed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


@dataclass
class Manifest:
    path: Path
    files: dict[str, ManifestEntry] = field(default_factory=dict)
    entity_types: list[str] | None = None

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        files = {k: ManifestEntry(**v) for k, v in data.get("files", {}).items()}
        return cls(path, files, data.get("entity_types"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": MANIFEST_VERSION,
            "entity_types": self.entity_types,
            "files": {k: asdict(v) for k, v in sorted(self.files.items())},
        }
        # Write-then-rename so an interrupted run never leaves a truncated manifest.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".manifest-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    @staticmethod
    def key(path: str | Path) -> str:
        return Path(path).resolve().as_posix()

    def get(self, path: str | Path) -> ManifestEntry | None:
        return self.files.get(self.key(path))

    def set(self, path: str | Path, doc_id: str, label: str) -> None:
        self.files[self.key(path)] = ManifestEntry(doc_id=doc_id, label=label)

    def pop(self, path: str | Path) -> ManifestEntry | None:
        return self.files.pop(self.key(path), None)

    def is_referenced(self, doc_id: str) -> bool:
        return any(e.doc_id == doc_id for e in self.files.values())

    def match(self, paths: list[str | Path]) -> list[str]:
        """Manifest keys for the given files, or for every file under the given folders.

        Works for files that no longer exist on disk, so deleted files can be removed.
        """
        keys = []
        for p in paths:
            key = self.key(p)
            prefix = key.rstrip("/") + "/"
            keys += [k for k in self.files if k == key or k.startswith(prefix)]
        return sorted(set(keys))
