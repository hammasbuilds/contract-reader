"""The split, the window arithmetic, and the one bug that made this repo wrong.

The determinism test is the important one here. Before it existed, three runs
of the same code on the same data gave test accuracies of 0.673, 0.668 and
0.734, because equal-weight cue terms were ordered by set iteration and Python
randomises string hashing per process. A repository that prints a different
number every time it runs cannot support a claim about anything.
"""

from __future__ import annotations

import pytest

from contractreader import corpus, reader, split

SPLIT = split.make()


def test_split_is_by_contract_and_disjoint():
    assert split.disjoint(SPLIT)
    assert len(SPLIT.train) + len(SPLIT.val) + len(SPLIT.test) == 510


def test_no_contract_appears_in_two_splits_via_its_questions():
    """The leak this project most needs to avoid.

    Every contract is asked 41 questions. Splitting questions at random would
    put the same contract text on both sides, and a cue fitted on it would be
    tested on it.
    """
    seen = {}
    for name in ("train", "val", "test"):
        for q in SPLIT.questions(name):
            assert seen.setdefault(q.contract, name) == name


def test_split_is_stable_across_calls():
    assert split.make().test == SPLIT.test


def test_window_counts_only_answerable():
    """An unanswerable question has no span to lose.

    Counting it as 'kept' would let the 68% majority hold the number up no
    matter how much truncation actually costs.
    """
    w = reader.window(corpus.load(), 8_000)
    assert w.total == len(corpus.answerable())


def test_window_is_monotonic():
    """A bigger window cannot see fewer spans."""
    kept = [reader.window(corpus.load(), n).kept for n in (4_000, 8_000, 16_000, 32_000)]
    assert kept == sorted(kept)


def test_truncation_hits_the_valuable_clauses_hardest():
    """The headline finding, asserted rather than just printed.

    The four header fields survive an 8k window almost intact; everything else
    is mostly lost. If that ever stops being true the README is wrong.
    """
    header = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
    kept_head, kept_rest = [], []
    for q in corpus.answerable():
        start = q.contract_text.find(q.answers[0])
        if start < 0:
            continue
        survives = start + len(q.answers[0]) <= 8_000
        (kept_head if q.category in header else kept_rest).append(survives)

    head = sum(kept_head) / len(kept_head)
    rest = sum(kept_rest) / len(kept_rest)
    assert head > 0.90, head
    assert rest < 0.30, rest
    assert head - rest > 0.60


def test_bag_is_memoised_but_still_correct():
    """The 41x speedup must not change the answer."""
    text = next(iter(corpus.contracts().values()))
    assert reader.bag(text) == reader.words(text)
    assert reader.bag(text) is reader.bag(text)


def test_cue_weights_are_deterministic():
    """Two fits on the same data give the same cues, in the same order.

    Guards the tie-break added to the log-odds sort. Without it the top-40 cut
    fell among equal-weight terms in whatever order the set iterated.
    """
    train = SPLIT.questions("train")
    a = reader.CuePresence().fit(train)
    b = reader.CuePresence().fit(train)
    assert a.cues.keys() == b.cues.keys()
    for category in a.cues:
        assert a.cues[category].terms == b.cues[category].terms


def test_whole_detector_is_deterministic():
    train, val, test = (SPLIT.questions(n) for n in ("train", "val", "test"))
    runs = []
    for _ in range(2):
        model = reader.CuePresence().fit(train).tune(val)
        runs.append(sum(1 for q in test if model.predict(q) == q.answerable))
    assert runs[0] == runs[1]


def test_detector_beats_refusing_to_read():
    """On test, not on train. A floor, and an honest one."""
    model = reader.CuePresence().fit(SPLIT.questions("train")).tune(SPLIT.questions("val"))
    test = SPLIT.questions("test")
    acc = sum(1 for q in test if model.predict(q) == q.answerable) / len(test)
    always_absent = sum(1 for q in test if not q.answerable) / len(test)
    assert acc > always_absent


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_every_split_has_both_classes(name):
    """A split with no answerable questions would make sensitivity undefined
    and the balanced accuracy silently meaningless."""
    group = SPLIT.questions(name)
    assert any(q.answerable for q in group)
    assert any(not q.answerable for q in group)
