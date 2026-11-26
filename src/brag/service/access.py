"""Which identity-provider groups may read or write which index.

access.yaml:

    indexes:
      finance:
        readers: [finance-team, executives]   # may query and list documents
        writers: [finance-admins]             # may also upload and remove
      handbook:
        readers: ["*"]                        # any signed-in user
        writers: [hr-admins]

Each index is a separate LightRAG workspace, so documents never mix across
indexes: a knowledge graph merges facts from every document it holds, so
per-document filtering inside one shared graph would leak through entity
summaries. Writers can always read.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

INDEX_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ANY = "*"


@dataclass(frozen=True)
class IndexAccess:
    name: str
    readers: frozenset[str]
    writers: frozenset[str]

    def can_read(self, groups: frozenset[str]) -> bool:
        return ANY in self.readers or bool(self.readers & groups) or self.can_write(groups)

    def can_write(self, groups: frozenset[str]) -> bool:
        return ANY in self.writers or bool(self.writers & groups)


def load_access(path: str | Path) -> dict[str, IndexAccess]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    indexes = data.get("indexes") or {}
    if not indexes:
        raise ValueError(f"{path} defines no indexes")
    access = {}
    for name, spec in indexes.items():
        name = str(name)
        if not INDEX_NAME.match(name):
            raise ValueError(f"Index name {name!r}: use lowercase letters, digits, '-' and '_' (max 64)")
        spec = spec or {}
        readers = frozenset(str(g) for g in spec.get("readers") or [])
        writers = frozenset(str(g) for g in spec.get("writers") or [])
        if not readers and not writers:
            raise ValueError(f"Index {name!r} has no readers or writers")
        access[name] = IndexAccess(name, readers, writers)
    return access
