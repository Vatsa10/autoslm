"""Trace Analyzer sub-agent (paper Section 2.1).

Data-heavy SQL + clustering work delegated to a sub-agent with a much higher
output token allowance (~100K). It runs a tool-use loop with `query_traces`,
`bash`, and `edit_file`. Outputs go to disk; the main agent reads only a brief.

Why a sub-agent: paper notes the main orchestrator's context becomes
crowded if it loads tens of thousands of inference rows. The sub-agent has
its own context, persists rich artifacts, and returns a short summary.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import AutoSLMConfig
from ..llm import LLMClient, chat_with_tools
from ..traces import TraceStore
from ..tools.registry import tool_specs, default_handlers


TRACE_ANALYZER_SYSTEM = """You are the Trace Analyzer sub-agent of autoslm.

Your job: given a deployed model_id and an analysis task, query the inference
store, group failures into actionable clusters, label each cluster's
fixability (`fixable` | `external` | `poison`), and persist artifacts to disk.

Definitions:
  - fixable: original intent + correct answer recoverable (typos, casing,
    truncation, distribution shift)
  - external: not addressable by training (schema/prompt errors, label
    ambiguity)
  - poison: training on raw teaches WRONG behavior (false premise, label flip,
    off-domain, prompt injection, jailbreak, gibberish, empty)

Workflow:
  1. Use `query_traces` (SQL + bash_pipeline) to inspect failures. Persist
     full result sets to disk; keep only summaries in your context.
  2. Group failures into 4-12 clusters by input pattern, prediction error,
     or judge_reason.
  3. For each cluster: name, description, fixability, size, ~5 representative
     ids.
  4. Use `edit_file` to write `clusters.json` and a markdown `summary.md`.
  5. Reply with a one-paragraph brief and the artifact paths.

Constraints:
  - Never load >50 raw rows into your context. Always use bash_pipeline.
  - Each cluster description: <= 1 sentence, 200 chars max.
  - Be conservative on `fixable`: when in doubt, prefer `external`.
"""


@dataclass
class TraceAnalysisResult:
    summary_path: str
    clusters_path: str
    brief: str
    n_clusters: int
    fixable: int
    external: int
    poison: int


class TraceAnalyzer:
    """Sub-agent with own LLM, own context. Reads via SQL, writes via files."""

    def __init__(self, cfg: AutoSLMConfig, store: TraceStore,
                 model: Optional[str] = None,
                 max_tokens: int = 100_000,
                 max_turns: int = 30):
        self.cfg = cfg
        self.store = store
        # Distinct client: paper says sub-agents have 10x token limit.
        self.client = LLMClient(
            model=model or cfg.orchestrator_model,
            max_tokens=max_tokens,
            thinking_budget=cfg.thinking_budget,
        )
        self.max_turns = max_turns

    def analyze(self, task: str, deployed_model_id: str,
                workdir: str | Path) -> TraceAnalysisResult:
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        tools = tool_specs()
        handlers = default_handlers(self.cfg, self.store, sandbox_dir=wd)
        # forbid spawning further sub-agents from inside the sub-agent
        handlers["delegate_task"] = lambda args: {"error": "nested delegate not allowed"}

        seed = {
            "task": task,
            "deployed_model_id": deployed_model_id,
            "artifacts_dir": str(wd),
            "expected_output_files": ["clusters.json", "summary.md"],
            "schema_hint": (
                "inferences(id, model_id, task, input, prediction, gold, verdict, "
                "judge_model, judge_reason, judge_score, metadata, deployment_stage)"
            ),
        }

        chat_with_tools(
            self.client,
            messages=[{"role": "user", "content": json.dumps(seed, default=str)}],
            tools=tools,
            tool_handlers=handlers,
            system=TRACE_ANALYZER_SYSTEM,
            max_iters=self.max_turns,
        )

        clusters_path = wd / "clusters.json"
        summary_path = wd / "summary.md"
        clusters: list[dict] = []
        if clusters_path.exists():
            try:
                clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
            except Exception:
                clusters = []

        fix = sum(1 for c in clusters if c.get("fixability") == "fixable")
        ext = sum(1 for c in clusters if c.get("fixability") == "external")
        poi = sum(1 for c in clusters if c.get("fixability") == "poison")
        brief = (
            f"{len(clusters)} clusters: {fix} fixable, {ext} external, {poi} poison. "
            f"Artifacts: {summary_path.name}, {clusters_path.name}"
        )
        return TraceAnalysisResult(
            summary_path=str(summary_path),
            clusters_path=str(clusters_path),
            brief=brief,
            n_clusters=len(clusters),
            fixable=fix, external=ext, poison=poi,
        )

    @staticmethod
    def clusters_to_taxonomy(clusters_json_path: str | Path) -> list:
        """Convert sub-agent's clusters.json into autoslm.traces.taxonomy.Cluster."""
        from ..traces.taxonomy import Cluster
        rows = json.loads(Path(clusters_json_path).read_text(encoding="utf-8"))
        out: list[Cluster] = []
        for i, c in enumerate(rows):
            out.append(Cluster(
                cluster_id=str(c.get("cluster_id", f"c{i}")),
                label=str(c.get("label", f"cluster_{i}"))[:120],
                fixability=str(c.get("fixability", "fixable")),
                size=int(c.get("size", 0)),
                description=str(c.get("description", ""))[:1000],
                representative_ids=list(c.get("representative_ids", []) or []),
                representative_examples=list(c.get("representative_examples", []) or []),
            ))
        return out
