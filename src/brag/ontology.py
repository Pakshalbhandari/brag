"""Corpus-specific entity types for the knowledge graph.

LightRAG extracts entities with a fixed, generic list of types (person,
organization, location, event, concept, ...). `brag ontology propose` samples
the corpus (prose chunks, plus table headers, which carry most of the schema in
spreadsheets) and asks Claude for entity types that fit it. The result is
written to ontology.yaml for a person to review and edit; ingestion then uses
its type names.

Re-running `propose` only adds types that are not in the file yet, so edits are
kept. To stop a deleted type from being proposed again, add its name under
`rejected`.
"""

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Settings
from .parsing import CHUNK_SEPARATOR, DocumentParser

logger = logging.getLogger(__name__)

MAX_SAMPLE_CHARS = 60_000
MAX_CHUNK_CHARS = 1_500
MAX_TYPES = 20

HEADER = """\
# Entity types for the brag knowledge graph.
# Proposed by `brag ontology propose`; edit freely. Only `name` is passed to
# entity extraction; descriptions and examples are for review.
# To delete a type for good, remove it here and add its name under `rejected`,
# so re-running `propose` does not add it back.
# Changing types after ingesting means re-indexing (delete the index folder and ingest again).
"""

SYSTEM_PROMPT = """You design entity-type schemas for knowledge graphs built from business documents.

You get samples from one document collection: text chunks and table headers. \
Propose the entity types an extraction model should tag in this collection.

Rules:
- Propose between 6 and {max_types} types.
- Prefer types specific to the domain (Product, Region, FinancialMetric) over \
generic ones, but keep generic types the documents clearly need (Person, Organization).
- Each type must cover entities that recur across the collection, not one document.
- Names: singular, PascalCase, ASCII letters only, written in English.
- Descriptions: one sentence saying what belongs in the type and what does not.
- Examples: 1 to 4 entity names copied from the samples, in their original language.
- Do not propose types for document structure (Section, Slide, Sheet, Table) \
or for bare values (Number, Date, Percentage)."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "examples": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description", "examples"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entity_types"],
    "additionalProperties": False,
}


@dataclass
class EntityType:
    name: str
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class Ontology:
    entity_types: list[EntityType] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def type_names(self) -> list[str]:
        return [t.name for t in self.entity_types]

    @classmethod
    def load(cls, path: Path) -> "Ontology":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        types = []
        for raw in data.get("entity_types") or []:
            if isinstance(raw, str):
                raw = {"name": raw}
            name = _clean_name(str(raw.get("name", "")))
            if name:
                types.append(EntityType(name, str(raw.get("description") or ""), list(raw.get("examples") or [])))
        return cls(types, [str(r) for r in data.get("rejected") or []])

    def save(self, path: Path) -> None:
        data = {
            "entity_types": [asdict(t) for t in self.entity_types],
            "rejected": sorted(set(self.rejected), key=str.lower),
        }
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
        path.write_text(HEADER + "\n" + body, encoding="utf-8")

    def merge(self, proposed: list[EntityType]) -> list[EntityType]:
        """Add proposed types not already present or rejected; return the ones added."""
        known = {t.name.lower() for t in self.entity_types} | {r.lower() for r in self.rejected}
        added = []
        for t in proposed:
            if t.name.lower() not in known:
                self.entity_types.append(t)
                known.add(t.name.lower())
                added.append(t)
        return added


def _clean_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name.strip())


def load_entity_types(settings: Settings) -> list[str] | None:
    """Type names from the ontology file, or None when there is no usable file."""
    path = Path(settings.ontology_file)
    if not path.exists():
        return None
    names = Ontology.load(path).type_names
    if not names:
        logger.warning("%s lists no entity types; using the defaults", path)
        return None
    return names


def collect_samples(parser: DocumentParser, files: list[Path], seed: int = 0) -> str:
    """Table headers from every file, then text chunks spread evenly across files."""
    headers: list[str] = []
    chunks_per_file: list[list[str]] = []
    for f in files:
        parsed = parser.parse(f)
        chunks = []
        for item in parsed.content_list:
            if item["type"] == "table":
                lines = item["table_body"].splitlines()[:3]  # header, rule, first row
                caption = " | ".join(item.get("table_caption") or [])
                headers.append(f"[{parsed.source.name}{' | ' + caption if caption else ''}]\n" + "\n".join(lines))
            elif item["type"] == "text":
                text = item["text"].replace(CHUNK_SEPARATOR, "").strip()
                chunks.append(f"[{parsed.source.name}]\n{text[:MAX_CHUNK_CHARS]}")
        random.Random(seed).shuffle(chunks)
        chunks_per_file.append(chunks)

    parts, size = [], 0

    def add(part: str) -> bool:
        nonlocal size
        if size + len(part) > MAX_SAMPLE_CHARS:
            return False
        parts.append(part)
        size += len(part)
        return True

    for h in headers:
        if not add("TABLE HEADER " + h):
            break
    # Round-robin so one huge file cannot crowd out the rest.
    depth = max((len(c) for c in chunks_per_file), default=0)
    for i in range(depth):
        for chunks in chunks_per_file:
            if i < len(chunks) and not add("TEXT " + chunks[i]):
                return "\n\n---\n\n".join(parts)
    return "\n\n---\n\n".join(parts)


def parse_proposal(raw: dict[str, Any]) -> list[EntityType]:
    types, seen = [], set()
    for t in raw.get("entity_types", []):
        name = _clean_name(t.get("name", ""))
        if name and name.lower() not in seen:
            seen.add(name.lower())
            types.append(EntityType(name, t.get("description", "").strip(), [str(e) for e in t.get("examples", [])][:4]))
    return types[:MAX_TYPES]


async def propose(models, samples: str) -> list[EntityType]:
    """Ask Claude for entity types that fit the sampled corpus."""
    text = await models.complete_json(
        system=SYSTEM_PROMPT.format(max_types=MAX_TYPES),
        prompt=f"Samples from the document collection:\n\n{samples}",
        schema=RESPONSE_SCHEMA,
    )
    return parse_proposal(json.loads(text))
