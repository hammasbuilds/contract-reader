"""A tiny, synthetic, CUAD-shaped corpus for the demo and the hermetic tests.

It is generated, not sampled from CUAD, and no number in the README comes from
it. It exists so that `demo.py` and the unit tests exercise the real parser,
split, window arithmetic and detector without the 38 MB download.

Each contract is a header (title, parties, date) followed by filler and, for
some contracts, a governing-law or cap-on-liability clause placed near the end,
which mirrors the shape of finding 3 on purpose.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES = ("Document Name", "Governing Law", "Cap On Liability")
QUESTION = (
    'Highlight the parts (if any) of this contract related to "{c}" that should be '
    "reviewed by a lawyer."
)
FILLER = (
    "The Supplier shall deliver the Products in accordance with the specifications "
    "set out in Schedule A and shall maintain records of each shipment. "
)
CLAUSES = {
    "Governing Law": "This Agreement shall be governed by the laws of the State of {s}.",
    "Cap On Liability": (
        "In no event shall either party's aggregate liability exceed the fees paid "
        "in the twelve months preceding the claim."
    ),
}
STATES = ("New York", "Delaware", "California", "Texas")


def build(n_contracts: int = 40, seed: int = 0) -> dict:
    """SQuAD-shaped dict, the same shape `scripts/fetch_data.py` writes."""
    rng = random.Random(seed)
    documents = []
    for i in range(n_contracts):
        title = f"SYNTHETIC_SUPPLY_AGREEMENT_{i:03d}"
        head = f"{title.replace('_', ' ')}\nBetween Acme Corp and Vendor {i} Ltd.\n"
        body = FILLER * rng.randint(60, 140)  # roughly 9k-21k characters
        present = {c: rng.random() < 0.5 for c in CLAUSES}
        tail = ""
        spans: dict[str, list[str]] = {"Document Name": [head.splitlines()[0]]}
        for category, has in present.items():
            if has:
                text = CLAUSES[category].format(s=rng.choice(STATES))
                tail += text + " " + FILLER
                spans[category] = [text]
        context = head + body + tail
        qas = [
            {
                "question": QUESTION.format(c=c),
                "answers": [
                    {"text": s, "answer_start": context.find(s)} for s in spans.get(c, [])
                ],
            }
            for c in CATEGORIES
        ]
        documents.append({"title": title, "paragraphs": [{"context": context, "qas": qas}]})
    return {"version": "synthetic", "data": documents}


def write(path: str | Path, **kwargs) -> Path:
    target = Path(path)
    target.write_text(json.dumps(build(**kwargs)), encoding="utf-8")
    return target
