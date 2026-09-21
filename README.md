# contract-reader

> A contract reader that truncates to 8,000 characters sees 41% of the answers — and the 41% is almost entirely the four fields printed on the first page.

**Status:** the model-free half is complete and measured on all 510 contracts. The 14B
reader is not written yet; the findings below do not need it, and they change what it
should be asked to do. See [What is not measured yet](#what-is-not-measured-yet).

## The corpus

[CUAD](https://github.com/TheAtticusProject/cuad) — the Contract Understanding Atticus
Dataset. 510 commercial contracts, labelled clause by clause **by attorneys** rather than
crowdworkers, over 41 categories that matter commercially.

| | |
|---|---:|
| Questions | **20,910** |
| Contracts | 510 |
| Clause categories | 41 |
| Answerable | 6,702 (32%) |
| **Unanswerable** | **14,208 (68%)** |
| Median contract | 52,563 chars |

Every contract is asked every category, so most questions are about a clause that simply
is not in that contract. That 68% is not a quirk of the sampling — it is asserted in
[tests/test_corpus.py](tests/test_corpus.py), along with the fact that every gold span
really is a substring of its own contract.

```
python scripts/fetch_data.py  # rebuild data/cuad.json (38 MB, not in git)
python scripts/measure.py     # prints every table below
python -m pytest              # 23 tests, against the real corpus
```

CUAD's canonical `data.zip` now 404s behind an organisation rename, so the fetch rebuilds
the corpus from HuggingFace's parquet conversion. The two splits are not shaped alike —
`test` carries whole contracts, `train` is chunked and augmented — so the script
deduplicates to one entry per (contract, category) against the longest context. It verifies
it produced 510 × 41 = 20,910 questions and **refuses to write** anything else, because a
fetch that quietly produces a different corpus invalidates every number below.

## Finding 1 — two scores that are not about reading contracts

| Strategy | Accuracy |
|---|---:|
| Answer nothing, ever | **0.681** |
| Answer something, ever | 0.319 |

Refusing to read scores 68%. Any reader reporting plain accuracy on CUAD is reporting
mostly this, which is why everything below uses **balanced accuracy** instead.

## Finding 2 — what a context window can even see

| Window | Gold spans kept | |
|---:|---:|---:|
| 4,000 | 2,115 / 6,702 | 31.6% |
| **8,000** | **2,719 / 6,702** | **40.6%** |
| 16,000 | 3,603 / 6,702 | 53.8% |
| 32,000 | 4,776 / 6,702 | 71.3% |
| 64,000 | 5,829 / 6,702 | 87.0% |

A 14B model does not read a 52,000-character contract. It reads the front of it.

## Finding 3 — the average is carried by four trivial fields

This is the one worth the repository. Truncation does **not** lose a random 59% of clauses.

| Clause category | n | Median position | Kept by 8k |
|---|---:|---:|---:|
| Document Name | 510 | 0% | **100%** |
| Parties | 509 | 1% | 98% |
| Agreement Date | 470 | 1% | 93% |
| Effective Date | 390 | 1% | 83% |
| … | | | |
| Insurance | 166 | 61% | 5% |
| **Cap On Liability** | 275 | 67% | **5%** |
| **Change Of Control** | 121 | 68% | **6%** |
| **Anti-Assignment** | 374 | 79% | **6%** |
| **Governing Law** | 437 | 84% | **8%** |
| Uncapped Liability | 111 | 69% | 2% |

| | n | Kept by 8k |
|---|---:|---:|
| The four header fields | 1,879 | **94%** |
| Every other category | 4,823 | **20%** |

The clauses that survive truncation are the ones you could find with your eye: the title,
the parties, the date. The clauses that do not survive are the ones anyone pays a lawyer
to find. A reader truncating to 8k scores respectably on average while being **blind to
every commercial term in the contract**.

## Finding 4 — how much of this is just word counting

A detector with no model in it: per category, learn from the training contracts which
terms distinguish contracts that have the clause from those that do not, then score an
unseen contract by how many of those terms it contains.

Split **by contract, never by question** — every contract is asked 41 questions, so
splitting questions at random would put the same contract text on both sides of the line.

| Split | n | Accuracy | Always-absent | Balanced | Sens | Spec |
|---|---:|---:|---:|---:|---:|---:|
| train | 16,728 | 0.817 | 0.681 | 0.795 | 0.733 | 0.857 |
| val | 3,116 | 0.730 | 0.682 | 0.713 | 0.664 | 0.761 |
| **test** | **1,066** | **0.689** | 0.653 | **0.668** | 0.603 | 0.734 |

All three splits are reported on purpose. The 0.129 train-to-test gap is what fitting 41
categories on 408 contracts buys you, and quoting the train number would be quoting the
overfit.

On test, counting words beats refusing to read by 0.036 accuracy and 0.168 balanced
accuracy over chance. **That is the floor a model has to clear before it has earned its
hour on the card.**

## The bug that made this repository wrong for an hour

Three runs of the same code on the same data gave test accuracies of **0.673, 0.668 and
0.734**.

Cue terms are ranked by log-odds and cut at the top 40. Thousands of terms tie — any term
appearing in equally many positive and negative contracts gets an identical weight — and
ties were broken by whatever order the `set` iterated. Python randomises string hashing
per process, so the top-40 cut fell in a different place on every run.

Fixed by tie-breaking on the term itself, and pinned by
`test_whole_detector_is_deterministic`. A repository that prints a different number every
time it runs cannot support a claim about anything.

The same pass found the detector re-tokenising each 52,000-character contract 41 times,
once per question. Memoising by contract took the fit from over two minutes to five
seconds; `test_bag_is_memoised_but_still_correct` asserts the speedup did not change the
answer.

## What is not measured yet

**The 14B reader.** Not written. Findings 2 and 3 say what it should be asked, and it is
not "read this contract": at 8k it cannot see the clauses that matter, so the honest
comparison is against a retrieval step that picks which 8k to show it. Measuring a
truncating reader against these numbers would mostly measure truncation.

**Span extraction.** Everything here is the presence question — *is this clause in this
contract* — not *which words are it*. Presence is the half that the 68% base rate makes
treacherous, so it is the half worth doing first.

## Layout

```
src/contractreader/corpus.py    CUAD, parsed. 20,910 questions, spans verified
src/contractreader/split.py     80/15/5 by contract, seeded and stable
src/contractreader/reader.py    window arithmetic + the model-free cue detector
scripts/measure.py              every table above
tests/                          23 tests against the real corpus
```
