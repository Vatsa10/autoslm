"""Eval harness. Held-out E (positive/negative/boundary slices) + regression set R.

Implements paper Eq.7: E = E_pos U E_neg U E_boundary.
Implements paper Eq.16: a(pi) >= tau and r(pi) <= eps.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from ..llm import LLMClient
from ..search.pipeline import LearningStrategy
from .metrics import exact_match, token_f1, rouge_l, rouge_2, code_pass_at_1
from .llm_judge import llm_judge_score


@dataclass
class EvalExample:
    input: str
    gold: str
    slice: str = "pos"               # 'pos' | 'neg' | 'boundary'
    label: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalSet:
    pos: list[EvalExample] = field(default_factory=list)
    neg: list[EvalExample] = field(default_factory=list)
    boundary: list[EvalExample] = field(default_factory=list)

    def all(self) -> list[EvalExample]:
        return self.pos + self.neg + self.boundary

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        es = cls()
        for r in rows:
            ex = EvalExample(
                input=r["input"], gold=r["gold"],
                slice=r.get("slice", "pos"),
                label=r.get("label"), metadata=r.get("metadata", {}),
            )
            getattr(es, ex.slice).append(ex)
        return es

    def save(self, path: str | Path) -> None:
        rows = [{**asdict(ex)} for ex in self.all()]
        Path(path).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


@dataclass
class EvalResult:
    score: float
    per_slice: dict
    per_example: list[dict]
    n: int
    method: str


# ---------- scoring ----------

def _score_one(method: str, pred: str, gold: str, judge_client: Optional[LLMClient],
               input_text: str, criteria: Optional[str], extra: dict) -> tuple[float, dict]:
    info: dict = {}
    if method == "exact_match":
        s = exact_match(pred, gold)
    elif method == "f1":
        s = token_f1(pred, gold)
    elif method == "rouge":
        s = rouge_l(pred, gold)
    elif method == "rouge_2":
        s = rouge_2(pred, gold)
    elif method == "pass_at_k":
        test = extra.get("test_code") or gold
        s = code_pass_at_1(pred, test)
    elif method == "llm_judge":
        if judge_client is None:
            raise ValueError("llm_judge selected but no judge_client provided")
        j = llm_judge_score(judge_client, input_text, gold, pred, criteria)
        s = j["score"]
        info = j
    else:
        raise ValueError(f"unknown eval method {method}")
    return s, info


def evaluate(
    examples: list[EvalExample],
    predictor: Callable[[list[str]], list[str]],
    method: str = "exact_match",
    judge_client: Optional[LLMClient] = None,
    criteria: Optional[str] = None,
) -> EvalResult:
    inputs = [ex.input for ex in examples]
    preds = predictor(inputs)
    per_ex: list[dict] = []
    per_slice: dict[str, list[float]] = {"pos": [], "neg": [], "boundary": []}
    for ex, pred in zip(examples, preds):
        s, info = _score_one(method, pred, ex.gold, judge_client, ex.input,
                             criteria, ex.metadata)
        per_slice.setdefault(ex.slice, []).append(s)
        per_ex.append({
            "input": ex.input, "gold": ex.gold, "pred": pred, "slice": ex.slice,
            "score": s, "info": info,
        })
    n = len(examples) or 1
    overall = sum(d["score"] for d in per_ex) / n
    slice_scores = {k: (sum(v) / len(v) if v else None) for k, v in per_slice.items()}
    return EvalResult(score=overall, per_slice=slice_scores, per_example=per_ex,
                      n=n, method=method)


def regression_count(
    regression_set: list[EvalExample],
    predictor: Callable[[list[str]], list[str]],
    method: str = "exact_match",
    judge_client: Optional[LLMClient] = None,
    criteria: Optional[str] = None,
    pass_threshold: float = 0.7,
) -> int:
    """Count previously-passing examples now failing.

    Per paper Section 2.2: 'r(pi; R) counts the number of previously correct
    examples that the new model answers incorrectly on a held-out regression set'.
    """
    res = evaluate(regression_set, predictor, method, judge_client, criteria)
    return sum(1 for d in res.per_example if d["score"] < pass_threshold)


def score_pipeline(
    eval_set: EvalSet,
    regression_set: Optional[list[EvalExample]],
    predictor: Callable[[list[str]], list[str]],
    strategy: LearningStrategy,
    judge_client: Optional[LLMClient] = None,
) -> tuple[float, int, EvalResult]:
    """One-call scoring used by MCGS evaluator. Returns (a(pi), r(pi), full_result)."""
    method = strategy.eval_method
    e = evaluate(eval_set.all(), predictor, method, judge_client, strategy.judge_criteria)
    r = regression_count(regression_set or [], predictor, method, judge_client,
                         strategy.judge_criteria) if regression_set else 0
    return e.score, r, e
