"""LLM-driven EXPAND and FUSE operators (paper Eq. 2 and Eq. 4).

The orchestrator inspects the parent trajectory + failure analysis and proposes
a hypothesis-driven modification. We constrain the LLM to a JSON schema so the
mutation is a structured edit on (D, H, S) rather than free-form code.
"""
from __future__ import annotations
import copy
import json
from dataclasses import asdict
from typing import Optional

from ..llm import LLMClient
from .pipeline import Pipeline, DatasetSpec, HyperParams, LearningStrategy
from .mcgs import MCGS, Node


EXPAND_SYSTEM = """You are the search controller for Pioneer Agent (paper arXiv:2604.09791).
You navigate a graph of training pipelines pi=(D,H,S). Each node is a completed train+eval.
Given the parent pipeline, its score, recent failure modes, and lineage, propose ONE
hypothesis-driven modification that targets the dominant failure mode.

Iteration policy (Section 2.4):
  - score < 0.80: dataset rework (gaps, hard-neg shortage, distribution mismatch)
  - 0.80 <= score < 0.95: tune H (epochs, lr, lora_rank, base_model, sys_prompt)
  - score >= 0.95: surgical adds (2-3 examples per remaining failure pattern)
  - regression from previous iter: ROLL BACK (output `rollback: true`)

Do NOT bulk-add data above 0.95. Do NOT change all three of D,H,S in one step.
Output strict JSON matching the schema. Include a brief `hypothesis` explaining why.
"""

EXPAND_SCHEMA = {
    "type": "object",
    "required": ["hypothesis", "edits"],
    "properties": {
        "hypothesis": {"type": "string"},
        "rollback": {"type": "boolean"},
        "edits": {
            "type": "object",
            "properties": {
                "D": {"type": "object"},
                "H": {"type": "object"},
                "S": {"type": "object"},
            },
        },
    },
}


def _apply_edits(pi: Pipeline, edits: dict) -> Pipeline:
    new = copy.deepcopy(pi)
    for k, v in (edits.get("D") or {}).items():
        if hasattr(new.D, k):
            setattr(new.D, k, v)
    for k, v in (edits.get("H") or {}).items():
        if hasattr(new.H, k):
            setattr(new.H, k, v)
    for k, v in (edits.get("S") or {}).items():
        if hasattr(new.S, k):
            setattr(new.S, k, v)
    new.parent_id = pi.fingerprint()
    return new


def llm_expander(client: LLMClient, failure_summary_provider=None):
    """Returns Expander callable. failure_summary_provider(node) -> str (optional)."""

    def _expand(parent: Node, mcgs: MCGS) -> Pipeline:
        ancestors = []
        cur = parent
        while cur is not None and len(ancestors) < 6:
            ancestors.append({
                "id": cur.id,
                "score": cur.result.score if cur.result else None,
                "regressions": cur.result.regressions if cur.result else None,
                "notes": cur.pipeline.notes,
                "H": asdict(cur.pipeline.H),
                "S": asdict(cur.pipeline.S),
                "D_summary": {
                    "name": cur.pipeline.D.name,
                    "ratios": [cur.pipeline.D.gold_ratio, cur.pipeline.D.hard_neg_ratio,
                               cur.pipeline.D.replay_ratio],
                    "max_examples": cur.pipeline.D.max_examples,
                },
            })
            cur = mcgs.nodes.get(cur.parent) if cur.parent else None

        failure = ""
        if failure_summary_provider:
            try:
                failure = failure_summary_provider(parent)
            except Exception as e:
                failure = f"<failure summary error: {e}>"

        prompt = json.dumps({
            "lineage": ancestors,
            "best_score": (mcgs.best().result.score if mcgs.best() else None),
            "iteration": mcgs.iteration,
            "score_threshold": mcgs.cfg.score_threshold,
            "regression_eps": mcgs.cfg.regression_epsilon,
            "schema": EXPAND_SCHEMA,
            "failure_summary": failure[:8000],
        }, default=str)

        resp = client.complete(
            [{"role": "user", "content": prompt}],
            system=EXPAND_SYSTEM,
        )
        text = resp.content.strip()
        # extract JSON
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        try:
            data = json.loads(text)
        except Exception:
            # fall back: minor lr nudge
            data = {"hypothesis": "fallback: halve lr",
                    "edits": {"H": {"learning_rate": parent.pipeline.H.learning_rate * 0.5}}}

        if data.get("rollback"):
            # rollback = re-emit parent (caller will detect score==parent and prune; or use best ancestor)
            best = mcgs.best()
            return copy.deepcopy(best.pipeline if best else parent.pipeline)

        child = _apply_edits(parent.pipeline, data.get("edits") or {})
        child.notes = data.get("hypothesis", "")[:500]
        return child

    return _expand


FUSE_SYSTEM = """You merge two or more high-scoring training pipelines into one candidate.
Combine complementary strategies (e.g. branch A: hard-negs fixed precision; branch B: more epochs fixed convergence -> fused: both).
Output the same JSON schema as EXPAND but applied to the top branch as base."""


def llm_fuser(client: LLMClient):
    def _fuse(branches: list[Node], mcgs: MCGS) -> Pipeline:
        payload = {
            "branches": [{
                "id": b.id, "score": b.result.score,
                "H": asdict(b.pipeline.H), "S": asdict(b.pipeline.S),
                "notes": b.pipeline.notes,
            } for b in branches],
            "schema": EXPAND_SCHEMA,
        }
        resp = client.complete(
            [{"role": "user", "content": json.dumps(payload, default=str)}],
            system=FUSE_SYSTEM,
        )
        text = resp.content.strip()
        if "```" in text:
            text = text.split("```", 1)[-1].split("```", 1)[0]
            if text.startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except Exception:
            return copy.deepcopy(branches[0].pipeline)
        return _apply_edits(branches[0].pipeline, data.get("edits") or {})

    return _fuse
