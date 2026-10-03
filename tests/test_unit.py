"""Hermetic tests: no download, no CUAD. Run on a fresh clone."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from contractreader import corpus, reader, sample, split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy(tmp_path):
    return str(sample.write(tmp_path / "cuad.json"))


def test_env_var_overrides_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(corpus.ENV, str(tmp_path))
    assert corpus.cuad_path() == tmp_path / "cuad.json"
    assert not corpus.available()
    sample.write(tmp_path / "cuad.json")
    assert corpus.available()
    assert len(corpus.load()) == 40 * len(sample.CATEGORIES)


def test_missing_corpus_names_the_fetch_script(tmp_path, monkeypatch):
    monkeypatch.setenv(corpus.ENV, str(tmp_path / "nowhere"))
    with pytest.raises(corpus.CorpusMissingError, match="fetch_data.py"):
        corpus.load()


def test_corrupt_corpus_is_a_clear_error(tmp_path):
    bad = tmp_path / "cuad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a CUAD"):
        corpus.load(bad)
    bad2 = tmp_path / "other.json"
    bad2.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a CUAD"):
        corpus.load(bad2)


def test_parser_reads_categories_and_spans(toy):
    qs = corpus.load(toy)
    assert corpus.categories(toy) == sorted(sample.CATEGORIES)
    assert all(q.spans_are_real for q in qs)
    assert all(q.answerable for q in qs if q.category == "Document Name")
    assert corpus.unanswerable(toy)


def test_category_of_unknown_phrasing():
    assert corpus._category_of("What is this?") == "unknown"


def test_split_is_disjoint_and_covers_everything(toy):
    s = split.make(toy)
    assert split.disjoint(s)
    assert len(s.train) + len(s.val) + len(s.test) == 40
    assert s.test and s.val
    assert split.make(toy).test == s.test


def test_window_arithmetic(toy):
    qs = corpus.load(toy)
    small = reader.window(qs, 200)
    big = reader.window(qs, 10**7)
    assert big.kept == big.total == len(corpus.answerable(toy))
    # Only the title fits in 200 chars; the clauses sit at the back.
    assert small.kept == 40
    assert reader.window([], 8_000).share == 0.0


def test_detector_learns_the_planted_clause(toy):
    s = split.make(toy)
    model = reader.CuePresence().fit(s.questions("train", toy)).tune(s.questions("val", toy))
    law = [q for q in corpus.load(toy) if q.category == "Governing Law"]
    correct = sum(model.predict(q) == q.answerable for q in law)
    always_absent = sum(not q.answerable for q in law)
    assert correct / len(law) >= 0.85
    assert correct > always_absent


def test_predict_rejects_wrong_type():
    with pytest.raises(TypeError, match="contract_text"):
        reader.CuePresence().predict(None)


def test_bag_matches_words():
    text = "Governing law: the laws of Delaware shall govern."
    assert reader.bag(text) == reader.words(text)
    assert "the" not in reader.words(text)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_resumes_a_part_file(tmp_path, monkeypatch):
    fetch = _load_script("fetch_data")
    payload = b"0123456789" * 1000
    dest = tmp_path / "x.parquet"
    (tmp_path / "x.parquet.part").write_bytes(payload[:4000])
    seen = {}

    class Response:
        status = 206

        def __init__(self, body):
            self.body = body

        def read(self, n):
            chunk, self.body = self.body[:n], self.body[n:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        seen["range"] = request.get_header("Range")
        return Response(payload[4000:])

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    fetch.download("http://example.invalid/x", dest)
    assert seen["range"] == "bytes=4000-"
    assert dest.read_bytes() == payload
    assert not (tmp_path / "x.parquet.part").exists()


def test_demo_runs_without_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(corpus.ENV, str(tmp_path))
    spec = importlib.util.spec_from_file_location("demo", ROOT / "demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    monkeypatch.setattr(sys, "argv", ["demo.py"])
    demo.main()
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out
    assert "fetch_data.py" in out
