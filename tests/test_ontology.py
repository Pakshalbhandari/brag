import asyncio
import json

from brag.ontology import (
    MAX_SAMPLE_CHARS,
    EntityType,
    Ontology,
    collect_samples,
    load_entity_types,
    parse_proposal,
    propose,
)
from brag.parsing import DocumentParser


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "ontology.yaml"
    Ontology([EntityType("Product", "Sold goods.", ["Widget Pro"])], rejected=["Concept"]).save(path)

    text = path.read_text()
    assert text.startswith("# Entity types")
    loaded = Ontology.load(path)
    assert loaded.type_names == ["Product"]
    assert loaded.entity_types[0].examples == ["Widget Pro"]
    assert loaded.rejected == ["Concept"]


def test_hand_written_names_only_file(tmp_path):
    path = tmp_path / "ontology.yaml"
    path.write_text("entity_types:\n  - Product\n  - name: Sales Region\n")
    assert Ontology.load(path).type_names == ["Product", "SalesRegion"]


def test_merge_keeps_edits_and_skips_rejected():
    ontology = Ontology([EntityType("Product", "my edited description")], rejected=["concept"])
    added = ontology.merge([EntityType("product", "new"), EntityType("Concept"), EntityType("Region")])
    assert [t.name for t in added] == ["Region"]
    assert ontology.entity_types[0].description == "my edited description"


def test_parse_proposal_cleans_and_dedupes():
    raw = {"entity_types": [
        {"name": "Sales Region", "description": " Areas. ", "examples": ["EMEA", "APAC", "NA", "LATAM", "ANZ"]},
        {"name": "sales-region", "description": "dupe", "examples": []},
        {"name": "!!!", "description": "empty after cleaning", "examples": []},
    ]}
    types = parse_proposal(raw)
    assert [t.name for t in types] == ["SalesRegion"]
    assert types[0].description == "Areas." and len(types[0].examples) == 4


def test_load_entity_types(settings):
    assert load_entity_types(settings) is None
    settings.ontology_file.write_text("entity_types: []\n")
    assert load_entity_types(settings) is None
    settings.ontology_file.write_text("entity_types: [Product, Region]\n")
    assert load_entity_types(settings) == ["Product", "Region"]


def test_collect_samples_includes_headers_and_every_file(settings, make_docx, make_xlsx):
    files = [make_docx("a.docx"), make_docx("b.docx", paragraphs=("Acme opened an office in Kraków.",)), make_xlsx()]
    samples = collect_samples(DocumentParser(settings), files)
    assert "TABLE HEADER" in samples and "| Region" in samples
    assert "[a.docx]" in samples and "Kraków" in samples
    assert len(samples) <= MAX_SAMPLE_CHARS + 100


def test_propose_uses_structured_output():
    class FakeModels:
        async def complete_json(self, system, prompt, schema):
            self.schema = schema
            return json.dumps({"entity_types": [{"name": "Product", "description": "d", "examples": ["Widget"]}]})

    models = FakeModels()
    types = asyncio.run(propose(models, "TEXT sample"))
    assert [t.name for t in types] == ["Product"]
    assert models.schema["required"] == ["entity_types"]
