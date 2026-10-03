"""Shared fixtures. Corpus-dependent tests skip cleanly when CUAD is not on disk."""

from __future__ import annotations

import pytest

from contractreader import corpus, split

NEEDS_CORPUS = pytest.mark.skipif(
    not corpus.available(),
    reason=(
        f"CUAD not found at {corpus.cuad_path()}; run `python scripts/fetch_data.py` "
        f"(pip install -e .[fetch]) or set {corpus.ENV}"
    ),
)


@pytest.fixture(scope="session")
def questions():
    return corpus.load()


@pytest.fixture(scope="session")
def the_split():
    return split.make()
