"""Reading a contract without a model, to find out how much needs one.

Two things live here.

**`window`** — what a fixed context window can even see. A 14B model reading a
52,000-character contract does not read it; it reads the front of it. This
measures exactly what falls off the back, which turns out to be the whole point
of the repository.

**`CuePresence`** — a model-free detector for "does this contract contain a
clause of this category". It learns, from the training contracts only, which
terms occur in contracts that have the clause and not in contracts that do not,
then scores an unseen contract by how many of those terms it contains.

The detector exists to set a floor, not to win. If counting words gets most of
the way, the interesting question is what the remaining distance is made of;
if it does not, that is worth knowing before anyone spends an hour of GPU on
the same task.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

TOKEN = re.compile(r"[a-z][a-z'-]{2,}")

# Terms this common carry no signal and only slow the count down.
STOP = frozenset(
    """the and for any all not that this with shall may other such than
    from which are was were been have has had its his her their they them
    will would can could should each per off out upon into under over
    agreement party parties section hereof herein hereto thereof""".split()
)


def words(text: str) -> set[str]:
    return {w for w in TOKEN.findall(text.lower()) if w not in STOP}


# Every contract is asked all 41 questions, so a naive pass tokenises the same
# 52,000 characters 41 times over. Keyed by id() of the shared string object,
# which is safe because `corpus.load` is cached and hands out the same object
# to every question drawn from one contract; the text is also kept so the entry
# cannot outlive what it describes.
_BAGS: dict[int, tuple[str, set[str]]] = {}


def bag(text: str) -> set[str]:
    """`words`, memoised per contract. Same answer, 41x less work."""
    key = id(text)
    hit = _BAGS.get(key)
    if hit is not None and hit[0] is text:
        return hit[1]
    made = words(text)
    _BAGS[key] = (text, made)
    return made


@dataclass(frozen=True)
class Window:
    """What survives a context window, and what it costs."""

    chars: int
    kept: int
    total: int

    @property
    def share(self) -> float:
        return self.kept / self.total if self.total else 0.0


def window(questions, chars: int) -> Window:
    """How many gold spans fit inside the first `chars` of their contract.

    Counted on answerable questions only: an unanswerable one has no span to
    lose, and including them would let a big unanswerable majority hide the
    loss behind a number that never moves.
    """
    answerable = [q for q in questions if q.answerable]
    kept = 0
    for q in answerable:
        start = q.contract_text.find(q.answers[0])
        if 0 <= start and start + len(q.answers[0]) <= chars:
            kept += 1
    return Window(chars=chars, kept=kept, total=len(answerable))


@dataclass(frozen=True)
class Cues:
    """The terms that distinguish contracts having a clause from those without."""

    category: str
    terms: tuple[tuple[str, float], ...]

    def score(self, text: str) -> float:
        present = bag(text)
        return sum(weight for term, weight in self.terms if term in present)


class CuePresence:
    """Model-free 'is this clause in this contract', fitted on train only."""

    def __init__(self, top: int = 40) -> None:
        self.top = top
        self.cues: dict[str, Cues] = {}
        self.thresholds: dict[str, float] = {}

    def fit(self, questions) -> CuePresence:
        """Learn per-category cue terms by log-odds of occurrence.

        A term scores highly when it appears in many contracts that have the
        clause and few that do not. Both counts are smoothed, because a term
        appearing in one contract and no others is noise that would otherwise
        get an infinite weight.
        """
        have: dict[str, Counter] = defaultdict(Counter)
        havent: dict[str, Counter] = defaultdict(Counter)
        n_have: Counter = Counter()
        n_havent: Counter = Counter()

        for q in questions:
            terms = bag(q.contract_text)
            if q.answerable:
                have[q.category].update(terms)
                n_have[q.category] += 1
            else:
                havent[q.category].update(terms)
                n_havent[q.category] += 1

        for category in set(have) | set(havent):
            pos, neg = have[category], havent[category]
            n_p = max(1, n_have[category])
            n_n = max(1, n_havent[category])
            scored = []
            for term in set(pos) | set(neg):
                p = (pos[term] + 1) / (n_p + 2)
                n = (neg[term] + 1) / (n_n + 2)
                scored.append((term, math.log(p / n)))
            # Tie-break on the term itself. Without it, equal-weight terms are
            # ordered by however the set iterated, which Python randomises per
            # process, so the top-40 cut fell in a different place on every run
            # and test accuracy wandered between 0.668 and 0.734 with no code
            # change. Many terms tie here: any term appearing in exactly the
            # same number of positive and negative contracts gets the same
            # weight, and in a 41-category corpus that is thousands of them.
            scored.sort(key=lambda t: (-t[1], t[0]))
            self.cues[category] = Cues(category, tuple(scored[: self.top]))
        return self

    def tune(self, questions) -> CuePresence:
        """Pick one threshold per category, on validation data.

        Chosen to maximise balanced accuracy rather than accuracy: 68% of
        questions are unanswerable, so plain accuracy is maximised by a
        threshold that says 'absent' to everything, which is not a detector.
        """
        by_category: dict[str, list] = defaultdict(list)
        for q in questions:
            by_category[q.category].append(q)

        for category, group in by_category.items():
            cues = self.cues.get(category)
            if cues is None:
                continue
            scored = [(cues.score(q.contract_text), q.answerable) for q in group]
            best, best_at = -1.0, 0.0
            # Sorted and deduplicated so the candidate cuts are examined in the
            # same order every time, and so a tie between two equally good
            # thresholds always resolves to the lower one.
            for cut in sorted({s for s, _ in scored}):
                tp = sum(1 for s, gold in scored if s >= cut and gold)
                fn = sum(1 for s, gold in scored if s < cut and gold)
                tn = sum(1 for s, gold in scored if s < cut and not gold)
                fp = sum(1 for s, gold in scored if s >= cut and not gold)
                sens = tp / (tp + fn) if tp + fn else 0.0
                spec = tn / (tn + fp) if tn + fp else 0.0
                balanced = (sens + spec) / 2
                if balanced > best:
                    best, best_at = balanced, cut
            self.thresholds[category] = best_at
        return self

    def predict(self, question) -> bool:
        cues = self.cues.get(question.category)
        if cues is None:
            return False
        cut = self.thresholds.get(question.category, 0.0)
        return cues.score(question.contract_text) >= cut
