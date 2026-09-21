"""Train/validation/test, split by contract and never by question.

CUAD ships no split, so this makes one. Two decisions matter and both are easy
to get wrong.

**Split by contract.** Every contract is asked all 41 questions, so splitting
the 20,910 questions at random would put the same contract on both sides of the
line. Anything learned from its text would then be tested on its text. The unit
of independence here is the document, not the question.

**80/15/5**, because 510 contracts is small: a 5% test set is 26 contracts and
roughly a thousand questions, which is enough to report and small enough to
leave real data for fitting. Validation is where thresholds get chosen; test is
looked at once, at the end.

The split is seeded and derived from a sorted list, so it is the same split on
every machine and after every re-download.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import corpus

TRAIN, VAL, TEST = "train", "val", "test"
SHARES = {TRAIN: 0.80, VAL: 0.15, TEST: 0.05}
SEED = 7


@dataclass(frozen=True)
class Split:
    """Which contracts are in each part, and the questions that follow."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def of(self, name: str) -> tuple[str, ...]:
        return {TRAIN: self.train, VAL: self.val, TEST: self.test}[name]

    def questions(self, name: str, path: str | None = None) -> list:
        wanted = set(self.of(name))
        return [q for q in corpus.load(path) if q.contract in wanted]


def make(path: str | None = None, seed: int = SEED) -> Split:
    """The split. Deterministic given the corpus."""
    titles = sorted({q.contract for q in corpus.load(path)})
    rng = random.Random(seed)
    rng.shuffle(titles)

    n = len(titles)
    n_train = int(n * SHARES[TRAIN])
    n_val = int(n * SHARES[VAL])
    return Split(
        train=tuple(titles[:n_train]),
        val=tuple(titles[n_train : n_train + n_val]),
        test=tuple(titles[n_train + n_val :]),
    )


def disjoint(split: Split) -> bool:
    """No contract appears in two parts. Asserted in the tests."""
    parts = [set(split.train), set(split.val), set(split.test)]
    return all(not (a & b) for i, a in enumerate(parts) for b in parts[i + 1 :])
