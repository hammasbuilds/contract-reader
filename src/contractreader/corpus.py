"""CUAD — 510 commercial contracts, read by lawyers, clause by clause.

The Contract Understanding Atticus Dataset is unusual among legal corpora in
that the labels were made by attorneys rather than crowdworkers, over 41 clause
categories that matter commercially: auto-renewal, change of control, most
favoured nation, exclusivity, cap on liability, termination for convenience.

What makes it the right corpus for this project is not the extraction task.
It is the **abstention** task hiding inside it:

    20,910 questions
     6,702 have an answer span in the contract
    14,208 do not  <-- 68%

Every contract is asked about every category, so most questions are about a
clause that simply is not there. A reader that always finds something scores
catastrophically, and a reader that never finds anything scores 68%. Neither
number is about reading contracts, which is precisely the trap this repository
is built to measure.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ENV = "CONTRACTREADER_DATA"
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"


def data_dir() -> Path:
    """Where the corpus lives: $CONTRACTREADER_DATA if set, else <repo>/data."""
    override = os.environ.get(ENV)
    return Path(override) if override else DEFAULT_DATA


def cuad_path() -> Path:
    return data_dir() / "cuad.json"


def available() -> bool:
    """Whether the real corpus is on disk (tests and demo.py use this to decide)."""
    return cuad_path().is_file()


# CUAD phrases every question the same way; the category is in quotes.
_CATEGORY = re.compile(r'related to "([^"]+)"')


class CorpusMissingError(FileNotFoundError):
    """CUAD is not on disk."""


@dataclass(frozen=True)
class Question:
    """One clause category, asked of one contract."""

    contract: str
    category: str
    question: str
    answers: tuple[str, ...]
    contract_text: str

    @property
    def answerable(self) -> bool:
        """Whether this contract contains the clause at all."""
        return bool(self.answers)

    @property
    def spans_are_real(self) -> bool:
        """Every labelled span is genuinely a substring of the contract.

        Asserted rather than assumed: a benchmark whose answers do not occur in
        its own contexts would quietly make every extraction look wrong.
        """
        return all(a in self.contract_text for a in self.answers)


def _category_of(question: str) -> str:
    found = _CATEGORY.search(question)
    return found.group(1) if found else "unknown"


def load(path: str | os.PathLike[str] | None = None) -> tuple[Question, ...]:
    """Every question in the corpus. Cached per resolved path."""
    target = Path(path) if path else cuad_path()
    return _load(str(target.resolve()))


@lru_cache(maxsize=2)
def _load(target_name: str) -> tuple[Question, ...]:
    target = Path(target_name)
    if not target.is_file():
        raise CorpusMissingError(
            f"{target} is missing. Run `python scripts/fetch_data.py` (needs "
            f"`pip install -e .[fetch]`), or point {ENV} at a directory holding cuad.json."
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        documents = raw["data"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"{target} is not a CUAD SQuAD-style JSON file ({exc}). "
            "Delete it and re-run scripts/fetch_data.py."
        ) from exc

    out: list[Question] = []
    for document in documents:
        title = document.get("title", "")
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                spans = tuple(a["text"] for a in qa.get("answers", []))
                out.append(
                    Question(
                        contract=title,
                        category=_category_of(qa["question"]),
                        question=qa["question"],
                        answers=spans,
                        contract_text=context,
                    )
                )
    return tuple(out)


def contracts(path: str | None = None) -> dict[str, str]:
    """Contract title -> full text."""
    return {q.contract: q.contract_text for q in load(path)}


def categories(path: str | None = None) -> list[str]:
    return sorted({q.category for q in load(path)})


def answerable(path: str | None = None) -> list[Question]:
    return [q for q in load(path) if q.answerable]


def unanswerable(path: str | None = None) -> list[Question]:
    """The majority, and the half that any honest evaluation has to keep."""
    return [q for q in load(path) if not q.answerable]
