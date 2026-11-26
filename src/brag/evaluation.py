"""Evaluation against a golden question set.

Retrieval metrics (no answer generation; in graph modes one small keyword
call per question):

* recall@k  share of a question's expected sources found in the top k chunks
* hit@k     whether at least one expected source is in the top k
* MRR       1 / rank of the first chunk from an expected source

Answer metrics (``--answers``; generates an answer per question):

* facts         share of expected facts that appear in the answer
* groundedness  share of the answer's claims that the retrieved context
                supports, judged by Claude

A golden file is YAML:

    cases:
      - id: q3-growth
        question: How much did revenue grow in Q3?
        sources:
          - samples/report.docx                       # any chunk of this file
          - {file: samples/sales.xlsx, location: Sheet 2}   # only that sheet
        facts:
          - "18%"
          - ["cautiously optimistic", "vorsichtig optimistisch"]   # any of these

Files are matched against the labels used at ingest (paths relative to where
`brag ingest` ran), so run both from the same folder.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

MAX_JUDGE_CONTEXT_CHARS = 80_000

JUDGE_SYSTEM = """You check answers from a document question-answering system for groundedness.

You get the context the system retrieved and its answer. Split the answer into \
its factual claims (ignore greetings, hedges and the list of references) and \
decide for each whether the context supports it. A claim is supported only if \
the context states it or it follows directly from what the context states; \
general knowledge does not count. Translations of supported facts count as supported."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


@dataclass
class ExpectedSource:
    file: str
    location: str | None = None  # "Slide 3" matches chunks starting "[Slide 3]"


@dataclass
class Case:
    id: str
    question: str
    sources: list[ExpectedSource] = field(default_factory=list)
    facts: list[list[str]] = field(default_factory=list)  # each fact: accepted alternatives


@dataclass
class CaseResult:
    id: str
    question: str
    recall: float | None
    hit: bool | None
    reciprocal_rank: float | None
    retrieved: list[str]
    answer: str | None = None
    fact_recall: float | None = None
    missing_facts: list[str] = field(default_factory=list)
    groundedness: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    error: str | None = None


def load_golden(path: str | Path) -> list[Case]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cases = []
    for i, raw in enumerate(data.get("cases") or [], 1):
        if not raw.get("question"):
            raise ValueError(f"Case {i} in {path} has no question")
        sources = [
            ExpectedSource(s) if isinstance(s, str) else ExpectedSource(str(s["file"]), s.get("location"))
            for s in raw.get("sources") or []
        ]
        facts = [[str(f)] if isinstance(f, (str, int, float)) else [str(a) for a in f] for f in raw.get("facts") or []]
        cases.append(Case(str(raw.get("id") or f"case-{i}"), str(raw["question"]), sources, facts))
    if not cases:
        raise ValueError(f"{path} has no cases")
    return cases


# -- matching ------------------------------------------------------------------


def _same_file(label: str, expected: str) -> bool:
    a, b = label.replace("\\", "/").strip("/"), expected.replace("\\", "/").strip("/")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def chunk_matches(chunk: dict[str, Any], source: ExpectedSource) -> bool:
    if not _same_file(str(chunk.get("file_path", "")), source.file):
        return False
    if source.location:
        return f"[{source.location.strip('[] ')}]".lower() in str(chunk.get("content", "")).lower()
    return True


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def fact_found(answer: str, alternatives: list[str]) -> bool:
    norm, compact = _normalize(answer), re.sub(r"\s+", "", answer.lower())
    return any(_normalize(a) in norm or re.sub(r"\s+", "", a.lower()) in compact for a in alternatives)


def strip_references(answer: str) -> str:
    return re.split(r"^#+\s*references\b", answer, flags=re.I | re.M)[0].strip()


# -- scoring -------------------------------------------------------------------


def score_retrieval(case: Case, chunks: list[dict[str, Any]], k: int) -> dict[str, Any]:
    top = chunks[:k]
    retrieved = [str(c.get("file_path", "")) for c in top]
    if not case.sources:
        return {"recall": None, "hit": None, "reciprocal_rank": None, "retrieved": retrieved}
    ranks = []
    for source in case.sources:
        rank = next((i for i, c in enumerate(top) if chunk_matches(c, source)), None)
        if rank is not None:
            ranks.append(rank)
    return {
        "recall": len(ranks) / len(case.sources),
        "hit": bool(ranks),
        "reciprocal_rank": 1 / (min(ranks) + 1) if ranks else 0.0,
        "retrieved": retrieved,
    }


def judge_context(data: dict[str, Any]) -> str:
    parts = [f"[{c.get('file_path', '')}]\n{c.get('content', '')}" for c in data.get("chunks", [])]
    parts += [f"Entity {e.get('entity_name')}: {e.get('description', '')}" for e in data.get("entities", [])]
    parts += [
        f"Relation {r.get('src_id')} -> {r.get('tgt_id')}: {r.get('description', '')}"
        for r in data.get("relationships", [])
    ]
    return "\n\n".join(parts)[:MAX_JUDGE_CONTEXT_CHARS]


async def judge_groundedness(models, context: str, answer: str) -> tuple[float | None, list[str]]:
    body = strip_references(answer)
    if not body:
        return None, []
    raw = await models.complete_json(
        system=JUDGE_SYSTEM,
        prompt=f"<context>\n{context}\n</context>\n\n<answer>\n{body}\n</answer>",
        schema=JUDGE_SCHEMA,
    )
    claims = json.loads(raw).get("claims", [])
    if not claims:
        return None, []
    unsupported = [c["claim"] for c in claims if not c.get("supported")]
    return 1 - len(unsupported) / len(claims), unsupported


async def evaluate_case(pipeline, case: Case, mode: str, k: int, answers: bool) -> CaseResult:
    data = await pipeline.retrieve(case.question, mode=mode)
    result = CaseResult(case.id, case.question, **score_retrieval(case, data.get("chunks", []), k))
    if answers:
        result.answer = await pipeline.query(case.question, mode=mode)
        if case.facts:
            missing = [alts[0] for alts in case.facts if not fact_found(result.answer, alts)]
            result.fact_recall = 1 - len(missing) / len(case.facts)
            result.missing_facts = missing
        result.groundedness, result.unsupported_claims = await judge_groundedness(
            pipeline.models, judge_context(data), result.answer
        )
    return result


@dataclass
class EvalReport:
    config: dict[str, Any]
    results: list[CaseResult]

    def metric(self, name: str) -> float | None:
        values = [getattr(r, name) for r in self.results if getattr(r, name) is not None and not r.error]
        return mean(float(v) for v in values) if values else None

    @property
    def summary(self) -> dict[str, float | None]:
        return {
            "recall@k": self.metric("recall"),
            "hit@k": self.metric("hit"),
            "mrr": self.metric("reciprocal_rank"),
            "facts": self.metric("fact_recall"),
            "groundedness": self.metric("groundedness"),
            "errors": sum(1 for r in self.results if r.error),
        }

    def to_json(self) -> str:
        return json.dumps(
            {"config": self.config, "summary": self.summary, "results": [asdict(r) for r in self.results]},
            indent=2,
            ensure_ascii=False,
        )

    def failures(self, thresholds: dict[str, float | None]) -> list[str]:
        """Metrics below their minimum, as readable messages."""
        out = []
        summary = self.summary
        for name, minimum in thresholds.items():
            if minimum is None:
                continue
            value = summary.get(name)
            if value is None:
                out.append(f"{name}: not measured (minimum {minimum:.2f})")
            elif value < minimum:
                out.append(f"{name}: {value:.3f} < {minimum:.2f}")
        if summary["errors"]:
            out.append(f"{summary['errors']} case(s) failed with errors")
        return out


async def run_eval(pipeline, cases: list[Case], mode: str = "mix", k: int = 5, answers: bool = False) -> EvalReport:
    settings = pipeline.settings
    config = {
        "mode": mode,
        "k": k,
        "answers": answers,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model or None,
        "vector_storage": settings.vector_storage,
        "graph_storage": settings.graph_storage,
        "chunk_size": settings.chunk_size,
    }
    results = []
    for case in cases:
        try:
            results.append(await evaluate_case(pipeline, case, mode, k, answers))
        except Exception as e:
            results.append(CaseResult(case.id, case.question, None, None, None, [], error=f"{type(e).__name__}: {e}"))
    return EvalReport(config, results)
