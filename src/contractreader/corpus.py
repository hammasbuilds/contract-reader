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
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
CUAD = DATA / "cuad.json"
MASTER = DATA / "master_clauses.csv"

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


@lru_cache(maxsize=1)
def load(path: str | None = None) -> tuple[Question, ...]:
    target = Path(path) if path else CUAD
    if not target.exists():
        raise CorpusMissingError(
            f"{target} is missing. Run scripts/fetch_data.sh, which pulls CUAD "
            "from the HuggingFace mirror in byte ranges."
        )
    raw = json.loads(target.read_text(encoding="utf-8"))

    out: list[Question] = []
    for document in raw["data"]:
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
