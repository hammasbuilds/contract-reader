"""Every model-free number in the README.

    python scripts/measure.py

Nothing here uses a language model. That is the point: the repository is about
what a contract reader's score is actually made of, and most of it turns out to
be made of things you can count.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contractreader import corpus, reader, split  # noqa: E402

WINDOWS = (4_000, 8_000, 16_000, 32_000, 64_000)


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def the_corpus() -> None:
    rule("the corpus")
    questions = corpus.load()
    answerable = corpus.answerable()
    texts = corpus.contracts()
    print(f"questions            {len(questions):>7,}")
    print(f"contracts            {len(texts):>7,}")
    print(f"clause categories    {len(corpus.categories()):>7}")
    print(f"answerable           {len(answerable):>7,}  ({len(answerable) / len(questions):.0%})")
    print(f"unanswerable         {len(questions) - len(answerable):>7,}"
          f"  ({1 - len(answerable) / len(questions):.0%})")
    print(f"median contract      {int(median(len(t) for t in texts.values())):>7,} chars")
    bad = [q for q in answerable if not q.spans_are_real]
    print(f"spans not in context {len(bad):>7}   <- asserted, not assumed")


def the_baselines() -> None:
    rule("finding 1 — the two scores that are not about reading contracts")
    questions = corpus.load()
    absent = sum(1 for q in questions if not q.answerable) / len(questions)
    print(f"  answer nothing, ever   accuracy {absent:.3f}")
    print(f"  answer something, ever accuracy {1 - absent:.3f}")
    print("\n  ^ 68% is the score for refusing to read. Any reader that reports")
    print("    plain accuracy is reporting mostly this. Everything below uses")
    print("    balanced accuracy, which costs a detector for both mistakes.")


def the_window() -> None:
    rule("finding 2 — what a context window can even see")
    questions = corpus.load()
    print(f"{'window':>10}{'gold spans kept':>18}")
    for chars in WINDOWS:
        w = reader.window(questions, chars)
        print(f"{chars:>10,}{w.kept:>10,} / {w.total:,}   {w.share:>6.1%}")
    print("\n  ^ 8,000 characters is what a 14B model comfortably takes. It sees")
    print("    41% of the answers. That number is the average, and the average")
    print("    is the trap — see finding 3.")


def the_split_of_the_window() -> None:
    rule("finding 3 — the average is carried by four trivial fields")
    questions = [q for q in corpus.load() if q.answerable]
    by: dict[str, list[bool]] = defaultdict(list)
    where: dict[str, list[float]] = defaultdict(list)
    for q in questions:
        start = q.contract_text.find(q.answers[0])
        if start < 0:
            continue
        by[q.category].append(start + len(q.answers[0]) <= 8_000)
        where[q.category].append(start / max(1, len(q.contract_text)))

    rows = [
        (sum(k) / len(k), median(where[c]), c, len(k))
        for c, k in by.items()
        if len(k) >= 20
    ]
    rows.sort(reverse=True)
    print(f"{'clause category':<38}{'n':>5}{'median pos':>12}{'kept by 8k':>12}")
    for keep, pos, category, n in rows[:5]:
        print(f"{category:<38}{n:>5}{pos:>11.0%}{keep:>11.0%}")
    print(f"{'  ...':<38}")
    for keep, pos, category, n in rows[-8:]:
        print(f"{category:<38}{n:>5}{pos:>11.0%}{keep:>11.0%}")

    header = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
    head = [k for c, k in by.items() if c in header for k in k]
    rest = [k for c, k in by.items() if c not in header for k in k]
    print(f"\n  the four header fields   n={len(head):>5}   kept by 8k {sum(head) / len(head):.0%}")
    print(f"  every other category     n={len(rest):>5}   kept by 8k {sum(rest) / len(rest):.0%}")
    print("\n  ^ truncation does not lose a random 59% of clauses. It keeps the")
    print("    four that are written on the first page and loses the ones anyone")
    print("    pays a lawyer to find: cap on liability, change of control,")
    print("    anti-assignment, governing law. A reader truncating to 8k scores")
    print("    respectably on average while being blind to the commercial terms.")


def the_floor() -> None:
    rule("finding 4 — how much of this is just word counting")
    parts = split.make()
    train = parts.questions("train")
    val = parts.questions("val")
    test = parts.questions("test")
    print(f"contracts  train {len(parts.train)}  val {len(parts.val)}  test {len(parts.test)}"
          f"   (split by contract, never by question)")
    print(f"questions  train {len(train):,}  val {len(val):,}  test {len(test):,}")

    model = reader.CuePresence().fit(train).tune(val)
    print(f"\n{'split':<8}{'n':>7}{'accuracy':>11}{'always-absent':>15}{'balanced':>11}"
          f"{'sens':>8}{'spec':>8}")
    got = {}
    for name, group in (("train", train), ("val", val), ("test", test)):
        tp = sum(1 for q in group if model.predict(q) and q.answerable)
        fn = sum(1 for q in group if not model.predict(q) and q.answerable)
        tn = sum(1 for q in group if not model.predict(q) and not q.answerable)
        fp = sum(1 for q in group if model.predict(q) and not q.answerable)
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        base = sum(1 for q in group if not q.answerable) / len(group)
        acc = (tp + tn) / len(group)
        got[name] = (acc, base, (sens + spec) / 2)
        print(f"{name:<8}{len(group):>7,}{acc:>11.3f}{base:>15.3f}"
              f"{(sens + spec) / 2:>11.3f}{sens:>8.3f}{spec:>8.3f}")

    # Written from the numbers above rather than typed in. An earlier version
    # quoted them by hand and they went stale the first time the detector
    # changed, which is the same failure this repository is about.
    train_acc, test_acc = got["train"][0], got["test"][0]
    t_acc, t_base, t_bal = got["test"]
    print(f"\n  ^ all three splits on purpose. Train {train_acc:.3f} and test"
          f" {test_acc:.3f} is the")
    print(f"    same detector; the {train_acc - test_acc:.3f} gap is what fitting 41"
          " categories on 408")
    print("    contracts buys you, and quoting the first number would be quoting")
    print("    the overfit.")
    print(f"    On test, counting words beats refusing to read by"
          f" {t_acc - t_base:.3f} accuracy")
    print(f"    and {t_bal - 0.5:.3f} balanced accuracy over chance. That is the floor a")
    print("    model has to clear before it has earned its hour on the card.")


def main() -> None:
    the_corpus()
    the_baselines()
    the_window()
    the_split_of_the_window()
    the_floor()
    print()


if __name__ == "__main__":
    main()
