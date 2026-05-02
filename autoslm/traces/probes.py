"""Live confirmation probes (paper Section 2.6 step 3).

Before constructing training data, the agent verifies each identified weakness
is systematic rather than sampling noise. For each fixable cluster:
  1. synthesize a probe set targeting the hypothesized failure mode
  2. evaluate the deployed model on the probes
  3. classify cluster as `confirmed` (pass_rate < threshold) or demote to `external`
  4. failing probe inputs become additional D_gold entries

Probe construction depends on the failure mode:
  - label confusion -> boundary inputs distinguishing the two labels
  - long-input failures -> extended versions of previously-correct short inputs
  - missed entity types -> synthetic examples with explicit entity inclusion
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..llm import LLMClient
from ..eval.harness import EvalExample
from .taxonomy import Cluster


PROBE_SYSTEM = """You design probe inputs to confirm a deployed model's hypothesized weakness.

Given a failure cluster (label, description, samples), generate K boundary-case inputs
likely to trigger the same failure. Each probe must include the expected gold output.

Strategy by failure pattern:
  - Label confusion (A↔B): boundary inputs disambiguating A from B; gold = correct one.
  - Long-context failure: extend short correct inputs to longer length.
  - Missed entity / class: explicit examples that cleanly invoke the missing type.
  - Format / output mode: inputs with explicit format demands.

Output strict JSON: {"probes": [{"input": str, "gold": str, "rationale": str}]}.
"""


@dataclass
class ProbeResult:
    cluster_id: str
    pass_rate: float
    probes: list[EvalExample]
    failing_probes: list[EvalExample]
    confirmed: bool                  # True if pass_rate < threshold (weakness confirmed)
    rationale: str = ""


def synthesize_probes(
    client: LLMClient, cluster: Cluster, k: int = 12
) -> list[EvalExample]:
    """LLM-generate K probe examples targeting the cluster's failure mode."""
    payload = {
        "cluster": {
            "label": cluster.label,
            "description": cluster.description,
            "fixability": cluster.fixability,
            "size": cluster.size,
            "samples": cluster.representative_examples[:5],
        },
        "k": k,
    }
    resp = client.complete(
        [{"role": "user", "content": json.dumps(payload, default=str)[:20_000]}],
        system=PROBE_SYSTEM,
    )
    text = resp.content.strip()
    if "```" in text:
        text = text.split("```", 1)[-1].split("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        out: list[EvalExample] = []
        for p in data.get("probes", [])[:k]:
            out.append(EvalExample(
                input=str(p.get("input", "")),
                gold=str(p.get("gold", "")),
                slice="boundary",
                metadata={"probe_for_cluster": cluster.cluster_id,
                          "rationale": str(p.get("rationale", ""))[:300]},
            ))
        return out
    except Exception:
        return []


def confirm_weakness(
    cluster: Cluster,
    probes: list[EvalExample],
    predictor: Callable[[list[str]], list[str]],
    score_fn: Callable[[str, str], float],
    pass_threshold: float = 0.7,
    confirm_pass_rate_max: float = 0.7,
) -> ProbeResult:
    """Run probes against deployed model. Confirm cluster as systematic if model
    fails on >= (1 - confirm_pass_rate_max) fraction of probes.

    Args:
      predictor: maps list of input strings -> list of model predictions
      score_fn: per-example scorer (e.g. exact_match or token_f1)
      pass_threshold: per-example pass cutoff
      confirm_pass_rate_max: cluster confirmed iff observed pass rate is BELOW this
    """
    if not probes:
        return ProbeResult(cluster_id=cluster.cluster_id, pass_rate=1.0,
                           probes=[], failing_probes=[], confirmed=False,
                           rationale="no probes synthesized")
    inputs = [p.input for p in probes]
    preds = predictor(inputs)
    scores = [score_fn(pred, ex.gold) for pred, ex in zip(preds, probes)]
    pass_n = sum(1 for s in scores if s >= pass_threshold)
    pass_rate = pass_n / len(probes)
    failing = [ex for ex, s in zip(probes, scores) if s < pass_threshold]
    confirmed = pass_rate < confirm_pass_rate_max
    return ProbeResult(
        cluster_id=cluster.cluster_id, pass_rate=pass_rate,
        probes=probes, failing_probes=failing, confirmed=confirmed,
        rationale=f"pass_rate={pass_rate:.2f}, threshold<{confirm_pass_rate_max}",
    )


def confirm_all(
    client: LLMClient,
    fixable_clusters: list[Cluster],
    predictor: Callable[[list[str]], list[str]],
    score_fn: Callable[[str, str], float],
    k_per_cluster: int = 12,
) -> dict[str, ProbeResult]:
    """Probe + confirm every fixable cluster. Returns map cluster_id -> ProbeResult."""
    out: dict[str, ProbeResult] = {}
    for c in fixable_clusters:
        probes = synthesize_probes(client, c, k=k_per_cluster)
        out[c.cluster_id] = confirm_weakness(c, probes, predictor, score_fn)
    return out
