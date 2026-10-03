"""Demo: the window finding and the word-count detector, end to end.

    python demo.py                      # real CUAD if fetched, else a synthetic toy corpus
    python demo.py --json               # same, as JSON
    python demo.py --contract my.txt    # which clause categories the detector flags in your file

With no data on disk it runs on `contractreader.sample`, a generated toy corpus,
and says so loudly; none of those numbers are CUAD numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from contractreader import corpus, reader, sample, split  # noqa: E402
from contractreader.corpus import Question  # noqa: E402


def run(path: str | None) -> tuple[dict, reader.CuePresence]:
    qs = corpus.load(path)
    parts = split.make(path)
    train, val, test = (parts.questions(n, path) for n in ("train", "val", "test"))
    model = reader.CuePresence().fit(train).tune(val)
    tp = sum(1 for q in test if model.predict(q) and q.answerable)
    tn = sum(1 for q in test if not model.predict(q) and not q.answerable)
    pos = sum(1 for q in test if q.answerable)
    neg = len(test) - pos
    sens = tp / pos if pos else 0.0
    spec = tn / neg if neg else 0.0
    w = reader.window(qs, 8_000)
    return {
        "questions": len(qs),
        "contracts": len(parts.train) + len(parts.val) + len(parts.test),
        "unanswerable_share": round(1 - len(corpus.answerable(path)) / len(qs), 3),
        "window_8k_spans_kept": round(w.share, 3),
        "test": {
            "n": len(test),
            "accuracy": round((tp + tn) / len(test), 3),
            "always_absent": round(neg / len(test), 3),
            "balanced_accuracy": round((sens + spec) / 2, 3),
        },
    }, model


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--contract", type=Path, help="a UTF-8 text file to run the detector on")
    args = ap.parse_args()

    real = corpus.available()
    with tempfile.TemporaryDirectory() as tmp:
        path = None if real else str(sample.write(Path(tmp) / "cuad.json"))
        result, model = run(path)
    result["corpus"] = "CUAD" if real else "SYNTHETIC toy corpus (not CUAD)"

    if args.contract is not None:
        try:
            text = args.contract.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"cannot read {args.contract}: {exc}") from None
        if not text.strip():
            raise SystemExit(f"{args.contract} is empty")
        result["your_contract"] = {
            c: model.predict(Question(args.contract.name, c, "", (), text))
            for c in sorted(model.cues)
        }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if not real:
        print("NOTE: CUAD is not on disk, so this ran on a SYNTHETIC toy corpus.")
        print("      Run `python scripts/fetch_data.py` for the real numbers.\n")
    print(f"corpus                {result['corpus']}")
    print(f"questions / contracts {result['questions']:,} / {result['contracts']}")
    print(f"unanswerable share    {result['unanswerable_share']:.3f}")
    print(f"8k window keeps       {result['window_8k_spans_kept']:.1%} of gold spans")
    t = result["test"]
    print(f"detector on test      acc {t['accuracy']:.3f}  always-absent {t['always_absent']:.3f}"
          f"  balanced {t['balanced_accuracy']:.3f}  (n={t['n']:,})")
    if "your_contract" in result:
        flagged = [c for c, v in result["your_contract"].items() if v]
        print(f"\n{args.contract.name}: detector flags {len(flagged)} categories")
        for c in flagged:
            print(f"  - {c}")


if __name__ == "__main__":
    main()
