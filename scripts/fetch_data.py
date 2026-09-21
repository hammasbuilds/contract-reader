"""Rebuild data/cuad.json from HuggingFace's auto-converted parquet.

    python scripts/fetch_data.py

CUAD's canonical distribution is `data.zip` in the GitHub repository, which is
served through a redirect that hands back HTML rather than the archive, and the
organisation has since been renamed. The HuggingFace mirror
`theatticusproject/cuad-qa` is a loading *script*, not data — but HuggingFace
auto-converts every dataset to parquet on the `refs/convert/parquet` branch, and
those files are ungated, small (4 MB each) and directly addressable.

The parquet is in HuggingFace's flat QA shape, one row per question. This
rewrites it into the SQuAD shape `corpus.py` reads, grouping rows back into
contracts, so the corpus on disk is the same one the tests and the README are
written against. The script verifies that grouping produced the expected counts
and refuses to write a file that does not match.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "cuad.json"

BASE = (
    "https://huggingface.co/datasets/theatticusproject/cuad-qa/"
    "resolve/refs%2Fconvert%2Fparquet/default"
)
PARTS = (
    "train/0000.parquet",
    "train/0001.parquet",
    "train/0002.parquet",
    "test/0000.parquet",
)

# What the corpus must come out as. Every number in the README is computed from
# a corpus of this shape, so a fetch that produces a different one is a fetch
# that silently invalidates the results.
EXPECT_QUESTIONS = 20_910
EXPECT_CONTRACTS = 510


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "contract-reader"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def rows():
    for part in PARTS:
        print(f"  {part} ...", end="", flush=True)
        blob = get(f"{BASE}/{part}")
        table = pq.read_table(io.BytesIO(blob))
        print(f" {table.num_rows:,} rows")
        yield from table.to_pylist()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    print("fetching CUAD parquet from the HuggingFace convert branch")

    # The two splits are not shaped the same. `test` is the canonical form:
    # exactly 41 rows per contract, each carrying the whole contract as its
    # context. `train` is chunked and augmented — one contract appears over a
    # hundred times, each row holding a slice of the text.
    #
    # So the corpus is the deduplication: one entry per (contract, category),
    # against the longest context seen for that contract, which is the full
    # document. Answer spans are merged as text and then filtered to those that
    # actually occur in the chosen context, because the `answer_start` offsets
    # in a chunked row are relative to that chunk and mean nothing here.
    # `corpus.py` reads only the text, never the offset.
    longest: dict[str, str] = {}
    answers: dict[tuple[str, str], set[str]] = defaultdict(set)
    asked: dict[tuple[str, str], str] = {}

    for row in rows():
        title, context = row["title"], row["context"]
        if len(context) > len(longest.get(title, "")):
            longest[title] = context
        key = (title, row["question"])
        asked[key] = row["question"]
        answers[key].update(row["answers"]["text"])

    documents = []
    for title, context in sorted(longest.items()):
        qas = []
        for (doc, question), text in sorted(asked.items()):
            if doc != title:
                continue
            spans = sorted(s for s in answers[(doc, question)] if s in context)
            qas.append(
                {
                    "question": text,
                    "answers": [{"text": s, "answer_start": context.find(s)} for s in spans],
                }
            )
        documents.append({"title": title, "paragraphs": [{"context": context, "qas": qas}]})

    questions = sum(len(q["qas"]) for d in documents for q in d["paragraphs"])
    print(f"\ncontracts {len(documents):,}   questions {questions:,}")

    if (questions, len(documents)) != (EXPECT_QUESTIONS, EXPECT_CONTRACTS):
        print(
            f"\nREFUSING TO WRITE: expected {EXPECT_QUESTIONS:,} questions over "
            f"{EXPECT_CONTRACTS} contracts.\nThe upstream dataset has changed shape. "
            "Every number in the README was computed on the expected corpus, so "
            "writing this one would invalidate them silently.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    OUT.write_text(json.dumps({"version": "cuad-qa", "data": documents}), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
