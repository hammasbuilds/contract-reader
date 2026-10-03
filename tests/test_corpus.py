"""What has to be true about CUAD before any number computed from it means anything.

These run against the real 40 MB corpus, not a fixture. A fixture would test
the parser and nothing else, and every claim in the README is a claim about the
real file.
"""

from __future__ import annotations

import pytest
from conftest import NEEDS_CORPUS

from contractreader import corpus

pytestmark = NEEDS_CORPUS


def test_corpus_size(questions):
    """The counts the README quotes."""
    assert len(questions) == 20_910
    assert len(corpus.contracts()) == 510
    assert len(corpus.categories()) == 41


def test_every_contract_is_asked_every_category(questions):
    """The premise of the abstention task.

    If some contracts were asked fewer questions, the 68% unanswerable rate
    would be partly an artefact of which questions got asked, not a fact about
    contracts.
    """
    per_contract = {}
    for q in questions:
        per_contract.setdefault(q.contract, set()).add(q.category)
    sizes = {len(v) for v in per_contract.values()}
    assert sizes == {41}


def test_answerable_share(questions):
    """68% unanswerable, which is the number both degenerate baselines rest on."""
    answerable = corpus.answerable()
    share = 1 - len(answerable) / len(questions)
    assert 0.67 < share < 0.69


def test_every_labelled_span_is_really_in_its_contract():
    """Asserted over the whole corpus, not sampled.

    A benchmark whose gold answers do not occur in its own contexts would make
    every extraction look wrong for a reason that has nothing to do with the
    reader. Checking it is cheap; assuming it is not.
    """
    broken = [q for q in corpus.answerable() if not q.spans_are_real]
    assert broken == []


def test_unanswerable_questions_have_no_spans():
    assert all(not q.answers for q in corpus.unanswerable())


def test_category_parsed_from_every_question(questions):
    """The category is pulled out of the question text with a regex.

    If CUAD ever rephrases its questions, this fails loudly instead of quietly
    lumping everything into 'unknown' and reporting one enormous category.
    """
    assert not [q for q in questions if q.category == "unknown"]


@pytest.mark.parametrize(
    "category",
    ["Cap On Liability", "Governing Law", "Change Of Control", "Anti-Assignment"],
)
def test_the_commercially_interesting_categories_exist(category):
    """The README names these four. If a rename silently drops one, the
    headline finding would be computed over a different set than it claims."""
    assert category in corpus.categories()
