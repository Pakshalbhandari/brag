from brag.manifest import Manifest


def test_round_trip(tmp_path):
    m = Manifest(tmp_path / "m.json", entity_types=["Product"])
    m.set(tmp_path / "a.docx", "doc-1", "a.docx")
    m.save()

    loaded = Manifest.load(tmp_path / "m.json")
    assert loaded.get(tmp_path / "a.docx").doc_id == "doc-1"
    assert loaded.entity_types == ["Product"]


def test_missing_file_loads_empty(tmp_path):
    assert Manifest.load(tmp_path / "nope.json").files == {}


def test_match_files_folders_and_deleted_files(tmp_path):
    m = Manifest(tmp_path / "m.json")
    for name in ["q3/a.docx", "q3/sub/b.xlsx", "q3x/c.pptx", "d.docx"]:
        m.set(tmp_path / name, f"doc-{name}", name)  # none of these exist on disk

    folder = {k.rsplit("/", 2)[-2] + "/" + k.rsplit("/", 1)[-1] for k in m.match([tmp_path / "q3"])}
    assert folder == {"q3/a.docx", "sub/b.xlsx"}  # not q3x/
    assert m.match([tmp_path / "d.docx"]) == [Manifest.key(tmp_path / "d.docx")]
    assert m.match([tmp_path / "other.docx"]) == []


def test_shared_doc_ids_are_reference_counted(tmp_path):
    m = Manifest(tmp_path / "m.json")
    m.set(tmp_path / "a.docx", "doc-same", "a.docx")
    m.set(tmp_path / "copy/a.docx", "doc-same", "copy/a.docx")
    m.pop(tmp_path / "a.docx")
    assert m.is_referenced("doc-same")
    m.pop(tmp_path / "copy/a.docx")
    assert not m.is_referenced("doc-same")
