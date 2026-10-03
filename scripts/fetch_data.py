"""Rebuild data/cuad.json from HuggingFace's auto-converted parquet.

    pip install -e .[fetch]          # pyarrow
    python scripts/fetch_data.py     # writes $CONTRACTREADER_DATA/cuad.json or data/cuad.json

Downloads are resumable: each parquet part is streamed to `<name>.part` under
`<data>/parquet/`, resumed with an HTTP Range request if interrupted, and only
renamed into place once complete. `cuad.json` itself is written to a temporary
file and swapped in atomically, so a crash never leaves a half-written corpus.

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

import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contractreader.corpus import cuad_path, data_dir  # noqa: E402

CHUNK = 1 << 20

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


def download(url: str, dest: Path, attempts: int = 5) -> Path:
    """Stream `url` to `dest`, resuming a previous `.part` and renaming on completion."""
    if dest.is_file():
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "contract-reader"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores Range answers 200 with the whole file.
                mode = "ab" if have and response.status == 206 else "wb"
                with open(part, mode) as out:
                    while block := response.read(CHUNK):
                        out.write(block)
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # .part already holds the whole file
                os.replace(part, dest)
                return dest
            if attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == attempts:
                raise
        print(f" (retry {attempt}, resuming)", end="", flush=True)
    raise RuntimeError("unreachable")


def rows(cache: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(
            "fetch_data.py needs pyarrow to read the parquet files: pip install -e .[fetch]"
        ) from None
    cache.mkdir(parents=True, exist_ok=True)
    for name in PARTS:
        print(f"  {name} ...", end="", flush=True)
        local = download(f"{BASE}/{name}", cache / name.replace("/", "-"))
        table = pq.read_table(local)
        print(f" {table.num_rows:,} rows")
        yield from table.to_pylist()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    data = data_dir()
    out = cuad_path()
    data.mkdir(parents=True, exist_ok=True)
    print(f"fetching CUAD parquet from the HuggingFace convert branch into {data}")

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

    for row in rows(data / "parquet"):
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

    write_atomic(out, json.dumps({"version": "cuad-qa", "data": documents}))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
