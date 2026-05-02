"""Build held-out evaluation set BEFORE training (paper §2.5 Eq. 7).

E = E_pos ∪ E_neg ∪ E_boundary

  - E_pos: correct (input, gold) pairs covering full label / entity range
  - E_neg: negatives — inputs that should NOT trigger any label (or for
    generation tasks: adversarial / out-of-domain prompts)
  - E_boundary: confusable pairs at decision boundaries; designed to test
    fine-grained discrimination

For generation tasks E_neg becomes adversarial inputs and E_boundary holds
multi-step / edge-case items. Negatives + boundaries are LLM-synthesized
when not present in the source dataset.
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from typing import Optional

from ..llm import LLMClient
from ..data.curate import Example
from .harness import EvalSet, EvalExample


NEG_SYSTEM = """Generate negative evaluation examples.

For classification / NER: inputs that should NOT trigger any of the given labels.
For generation: adversarial / out-of-domain / ill-posed prompts.

Output JSON: {"examples": [{"input": str, "expected": "no_label" | str}]}.
"""

BOUNDARY_SYSTEM = """Generate boundary-case evaluation examples.

Confusable pairs at decision boundaries — inputs that test fine-grained
discrimination (e.g., easily-confused labels, near-miss formats, multi-step
reasoning required).

Output JSON: {"examples": [{"input": str, "gold": str, "rationale": str}]}.
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.split("```", 1)[-1].split("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _stratified_sample(examples: list[Example], k: int,
                      seed: int = 42) -> list[Example]:
    by_lab: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_lab[ex.label or "_unlabeled"].append(ex)
    rng = random.Random(seed)
    per = max(1, k // max(1, len(by_lab)))
    out: list[Example] = []
    for items in by_lab.values():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out[:k]


def _synth_negatives(client: LLMClient, task_type: str,
                    labels: Optional[list[str]], k: int) -> list[EvalExample]:
    payload = {"task_type": task_type, "labels": labels, "n": k}
    try:
        resp = client.complete(
            [{"role": "user", "content": json.dumps(payload)}],
            system=NEG_SYSTEM,
        )
        d = _parse_json(resp.content)
        out = []
        for e in d.get("examples", [])[:k]:
            out.append(EvalExample(
                input=str(e.get("input", "")),
                gold=str(e.get("expected", "no_label")),
                slice="neg",
                metadata={"synthetic": True},
            ))
        return [e for e in out if e.input]
    except Exception:
        return []


def _synth_boundaries(client: LLMClient, examples: list[Example],
                     task_type: str, labels: Optional[list[str]],
                     k: int) -> list[EvalExample]:
    sample = examples[: min(20, len(examples))]
    payload = {
        "task_type": task_type,
        "labels": labels,
        "samples": [{"input": e.input, "gold": e.output, "label": e.label}
                    for e in sample],
        "n": k,
    }
    try:
        resp = client.complete(
            [{"role": "user", "content": json.dumps(payload, default=str)}],
            system=BOUNDARY_SYSTEM,
        )
        d = _parse_json(resp.content)
        out = []
        for e in d.get("examples", [])[:k]:
            out.append(EvalExample(
                input=str(e.get("input", "")),
                gold=str(e.get("gold", "")),
                slice="boundary",
                metadata={"synthetic": True,
                          "rationale": str(e.get("rationale", ""))[:300]},
            ))
        return [e for e in out if e.input and e.gold]
    except Exception:
        return []


def build_holdout(
    examples: list[Example],
    task_type: str,
    client: Optional[LLMClient] = None,
    labels: Optional[list[str]] = None,
    k_pos: int = 200,
    k_neg: int = 50,
    k_boundary: int = 50,
    seed: int = 42,
) -> tuple[EvalSet, list[Example]]:
    """Partition examples + synthesize neg/boundary slices.

    Returns (eval_set, train_pool). The held-out indices are removed from
    train_pool so they're never used for training (paper §2.5).
    """
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)

    pos_take = _stratified_sample(pool, k_pos, seed=seed)
    pos_ids = {id(e) for e in pos_take}
    train_pool = [e for e in pool if id(e) not in pos_ids]

    pos_eval = [
        EvalExample(input=e.input, gold=e.output, slice="pos",
                   label=e.label, metadata={"source": "held_out"})
        for e in pos_take
    ]

    neg_eval: list[EvalExample] = []
    bound_eval: list[EvalExample] = []
    if client and k_neg > 0:
        neg_eval = _synth_negatives(client, task_type, labels, k_neg)
    if client and k_boundary > 0:
        bound_eval = _synth_boundaries(client, examples, task_type, labels, k_boundary)

    es = EvalSet(pos=pos_eval, neg=neg_eval, boundary=bound_eval)
    return es, train_pool
