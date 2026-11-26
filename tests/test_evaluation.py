import asyncio
import json
from types import SimpleNamespace

import pytest

from brag.evaluation import (
    Case,
    ExpectedSource,
    chunk_matches,
    fact_found,
    judge_groundedness,
    load_golden,
    run_eval,
    score_retrieval,
    strip_references,
)


def test_load_golden(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text(
        """
cases:
  - id: a
    question: Q one?
    sources: [docs/a.docx, {file: b.xlsx, location: Sheet 2}]
    facts: ["18%", ["cautiously optimistic", "vorsichtig optimistisch"], 45]
  - question: Q two?
"""
    )
    a, b = load_golden(path)
    assert a.sources == [ExpectedSource("docs/a.docx"), ExpectedSource("b.xlsx", "Sheet 2")]
    assert a.facts == [["18%"], ["cautiously optimistic", "vorsichtig optimistisch"], ["45"]]
    assert b.id == "case-2" and b.sources == [] and b.facts == []


def test_load_golden_rejects_empty_and_missing_questions(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("cases: []\n")
    with pytest.raises(ValueError, match="no cases"):
        load_golden(path)
    path.write_text("cases:\n  - id: x\n")
    with pytest.raises(ValueError, match="no question"):
        load_golden(path)


def test_shipped_sample_golden_set_loads():
    assert len(load_golden("eval/samples.yaml")) >= 8


def test_chunk_matches_file_and_location():
    chunk = {"file_path": "samples/sales.xlsx", "content": "Table Analysis: Caption: [Sheet 2], Sheet: Headcount"}
    assert chunk_matches(chunk, ExpectedSource("samples/sales.xlsx"))
    assert chunk_matches(chunk, ExpectedSource("sales.xlsx"))  # label suffix
    assert chunk_matches(chunk, ExpectedSource("samples/sales.xlsx", "Sheet 2"))
    assert chunk_matches(chunk, ExpectedSource("samples/sales.xlsx", "[sheet 2]"))
    assert not chunk_matches(chunk, ExpectedSource("samples/sales.xlsx", "Sheet 1"))
    assert not chunk_matches(chunk, ExpectedSource("other/sales.xlsx"))
    assert not chunk_matches(chunk, ExpectedSource("les.xlsx"))  # whole path segments only


def test_fact_found():
    answer = "Revenue grew 18 %.\nThe CFO is  Vorsichtig optimistisch."
    assert fact_found(answer, ["18%"])  # spacing differences
    assert fact_found(answer, ["cautiously optimistic", "vorsichtig optimistisch"])
    assert not fact_found(answer, ["25%"])


def test_strip_references():
    assert strip_references("Answer text.\n\n### References\n- [1] a.docx") == "Answer text."
    assert strip_references("No refs here.") == "No refs here."


def test_score_retrieval():
    case = Case("c", "q", [ExpectedSource("a.docx"), ExpectedSource("b.docx")])
    chunks = [{"file_path": "x.docx"}, {"file_path": "b.docx"}, {"file_path": "x.docx"}, {"file_path": "a.docx"}]
    assert score_retrieval(case, chunks, k=5) == {
        "recall": 1.0, "hit": True, "reciprocal_rank": 0.5,
        "retrieved": ["x.docx", "b.docx", "x.docx", "a.docx"],
    }
    top2 = score_retrieval(case, chunks, k=2)
    assert top2["recall"] == 0.5 and top2["reciprocal_rank"] == 0.5
    miss = score_retrieval(case, chunks, k=1)
    assert miss["recall"] == 0.0 and miss["hit"] is False and miss["reciprocal_rank"] == 0.0
    assert score_retrieval(Case("c", "q"), chunks, k=5)["recall"] is None


class FakeJudge:
    def __init__(self, claims):
        self.claims = claims
        self.prompts = []

    async def complete_json(self, system, prompt, schema):
        self.prompts.append(prompt)
        return json.dumps({"claims": self.claims})


def test_judge_groundedness():
    judge = FakeJudge([{"claim": "a", "supported": True}, {"claim": "b", "supported": False}])
    score, unsupported = asyncio.run(judge_groundedness(judge, "ctx", "Body.\n### References\n- [1] x"))
    assert score == 0.5 and unsupported == ["b"]
    assert "References" not in judge.prompts[0]
    assert asyncio.run(judge_groundedness(FakeJudge([]), "ctx", "Body.")) == (None, [])
    assert asyncio.run(judge_groundedness(judge, "ctx", "### References\n- x")) == (None, [])


class FakePipeline:
    def __init__(self, settings):
        self.settings = settings
        self.models = FakeJudge([{"claim": "growth was 18%", "supported": True}])

    async def retrieve(self, question, mode):
        if "boom" in question:
            raise RuntimeError("retrieval failed")
        return {"chunks": [{"file_path": "samples/report.docx", "content": "Revenue grew 18%."}]}

    async def query(self, question, mode):
        return "Revenue grew 18%.\n### References\n- [1] samples/report.docx"


def test_run_eval_with_answers_and_thresholds(settings):
    cases = [
        Case("ok", "growth?", [ExpectedSource("samples/report.docx")], [["18%"], ["Widget Pro"]]),
        Case("bad", "boom?", [ExpectedSource("samples/report.docx")]),
    ]
    report = asyncio.run(run_eval(FakePipeline(settings), cases, mode="naive", k=3, answers=True))

    ok, bad = report.results
    assert ok.recall == 1.0 and ok.fact_recall == 0.5 and ok.missing_facts == ["Widget Pro"]
    assert ok.groundedness == 1.0
    assert bad.error == "RuntimeError: retrieval failed"
    summary = report.summary
    assert summary["recall@k"] == 1.0 and summary["facts"] == 0.5 and summary["errors"] == 1
    assert report.config["mode"] == "naive" and report.config["k"] == 3

    failures = report.failures({"recall@k": 0.9, "facts": 0.8, "groundedness": None, "mrr": 0.5})
    assert failures == ["facts: 0.500 < 0.80", "1 case(s) failed with errors"]
    assert json.loads(report.to_json())["summary"]["facts"] == 0.5


def test_unmeasured_metric_with_threshold_fails(settings):
    report = asyncio.run(run_eval(FakePipeline(settings), [Case("ok", "q", [ExpectedSource("x.docx")])], answers=False))
    assert report.failures({"groundedness": 0.9}) == ["groundedness: not measured (minimum 0.90)"]
