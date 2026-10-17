# contract-reader

[![ci](https://github.com/hammasbuilds/contract-reader/actions/workflows/ci.yml/badge.svg)](https://github.com/hammasbuilds/contract-reader/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

We built a measurement of what a contract reader can and cannot see on
[CUAD](https://github.com/TheAtticusProject/cuad): 510 commercial contracts labelled clause
by clause by attorneys over 41 categories. It is for anyone building an LLM contract-review
tool who has to decide how much of a contract to show the model. **A reader that truncates
to 8,000 characters sees 41% of the answers, and those are almost entirely the four fields
printed on the first page**; the clauses people pay lawyers to find are mostly cut off.

## What it does

- Rebuilds CUAD from Hugging Face's parquet conversion into one file: 510 contracts × 41
  categories = 20,910 questions, every gold span verified to be a substring of its contract.
  The fetch refuses to write a corpus of any other shape.
- Reports the two scores that need no reading (always "absent", always "present"), so every
  other number can be read against them.
- Measures, for each context window from 4k to 64k characters, how many gold spans fall
  inside it, overall and per clause category, with each category's median position.
- Trains a model-free cue detector per category (terms that separate contracts with the
  clause from those without), split 80/15/5 **by contract**, and reports train, val and test.
- `demo.py --contract FILE` runs that detector on your own contract text.

## Sample input / output

A short hand-written software licence (`contract.txt`, five clauses: grant, renewal,
assignment, liability cap, governing law) through `python demo.py --contract contract.txt`
with CUAD fetched:

```text
corpus                CUAD
questions / contracts 20,910 / 510
unanswerable share    0.679
8k window keeps       40.5% of gold spans
detector on test      acc 0.689  always-absent 0.653  balanced 0.668  (n=1,066)

contract.txt: detector flags 9 categories
  - Document Name
  - License Grant
  - Most Favored Nation
  - No-Solicit Of Employees
  - Non-Transferable License
  - Parties
  - Price Restrictions
  - Renewal Term
  - Unlimited/All-You-Can-Eat-License
```

It finds the licence grant, renewal term and header fields, misses anti-assignment,
liability cap and governing law, and raises three false alarms: what a 0.668
balanced-accuracy word counter looks like on one contract. That is the floor a model reader
has to clear.

## Results

From `python scripts/measure.py` on the current Hugging Face fetch (median contract 33,143
characters).

**1. Refusing to read scores 0.679.** 14,208 of 20,910 questions (68%) ask about a clause
that is not in that contract, so plain accuracy mostly measures this. Everything below uses
balanced accuracy.

**2. What a context window can see:**

| Window (chars) | Gold spans kept | |
|---:|---:|---:|
| 4,000 | 2,129 / 6,702 | 31.8% |
| **8,000** | **2,714 / 6,702** | **40.5%** |
| 16,000 | 3,613 / 6,702 | 53.9% |
| 32,000 | 4,774 / 6,702 | 71.2% |
| 64,000 | 5,834 / 6,702 | 87.0% |

**3. The average is carried by four trivial fields.**

| Clause category | n | Median position | Kept by 8k |
|---|---:|---:|---:|
| Document Name | 510 | 0% | **100%** |
| Parties | 509 | 1% | 98% |
| Agreement Date | 470 | 1% | 93% |
| Effective Date | 390 | 1% | 84% |
| … | | | |
| **Governing Law** | 437 | 83% | **8%** |
| **Anti-Assignment** | 374 | 79% | **7%** |
| **Change Of Control** | 121 | 67% | **6%** |
| **Cap On Liability** | 275 | 69% | **5%** |
| Insurance | 166 | 61% | 4% |
| Uncapped Liability | 111 | 69% | 2% |

| | n | Kept by 8k |
|---|---:|---:|
| The four header fields | 1,879 | **94%** |
| Every other category | 4,823 | **20%** |

An 8k reader scores respectably on average while missing nearly every commercial term, so
a model reader should be compared against a retrieval step that chooses which 8k to show it.

**4. How much is word counting.** The cue detector, split by contract (408 / 76 / 26):

| Split | n | Accuracy | Always-absent | Balanced | Sens | Spec |
|---|---:|---:|---:|---:|---:|---:|
| train | 16,728 | 0.817 | 0.681 | 0.795 | 0.733 | 0.857 |
| val | 3,116 | 0.730 | 0.682 | 0.713 | 0.664 | 0.761 |
| **test** | **1,066** | **0.689** | 0.653 | **0.668** | 0.603 | 0.734 |

On test, counting words beats refusing to read by 0.036 accuracy and 0.168 balanced
accuracy over chance. Everything here is clause *presence*, not span extraction.

## Tech stack

- Python 3.11+; the package is standard library only
- PyArrow (only to fetch CUAD's parquet and build `data/cuad.json`)
- pytest, ruff, GitHub Actions (Ubuntu and Windows)

## Run it

```bash
git clone https://github.com/hammasbuilds/contract-reader
cd contract-reader
pip install -e ".[dev,fetch]"

python -m pytest -q                 # fresh clone: hermetic tests pass, corpus tests skip
python demo.py                      # runs on a small built-in toy corpus until CUAD is fetched
python scripts/fetch_data.py        # build data/cuad.json (38 MB, not in git, ~1 min)
python scripts/measure.py           # every table above
python demo.py --json               # headline numbers as JSON
python demo.py --contract my_contract.txt
```

Set `CONTRACTREADER_DATA=/some/dir` to keep `cuad.json` elsewhere. The fetch resumes an
interrupted download with HTTP Range requests and writes `cuad.json` atomically.

```
src/contractreader/corpus.py    CUAD parsed: 20,910 questions, spans verified
src/contractreader/split.py     80/15/5 by contract, seeded and stable
src/contractreader/reader.py    window arithmetic + the model-free cue detector
src/contractreader/sample.py    small toy corpus for demo.py and hermetic tests (not CUAD)
scripts/measure.py              every table above
demo.py                         end-to-end demo, --json, --contract FILE
```

## Licence

Code: MIT, see [LICENSE](LICENSE). CUAD is not committed; `scripts/fetch_data.py`
downloads it, and it stays under its own licence (CC BY 4.0, The Atticus Project).
