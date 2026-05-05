"""Monte Carlo Graph Search over training pipelines (paper Section 2.2).

Graph G=(V,E). Each node = (pi_i, f(pi_i)). Edges encode causal lineage.
- EXPAND: orchestrator LLM proposes child config given parent + diagnosis
- Score: full train+eval (no surrogate; paper rejects predictors)
- UCT: time-decaying exploration coef
- FUSE: merge top-K complementary branches
- Stagnation -> evolution OR fusion
- Rollback: regression -> revert to best ancestor

This module is execution-engine agnostic. The caller supplies an `evaluator`
callable that maps Pipeline -> NodeResult (score, regressions, artifacts).
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import networkx as nx

from .pipeline import Pipeline


@dataclass
class NodeResult:
    score: float                    # f(pi) on held-out E
    regressions: int = 0            # r(pi; R)
    metrics: dict = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    eval_artifacts_path: Optional[str] = None
    failed: bool = False
    error: Optional[str] = None


@dataclass
class Node:
    id: str
    pipeline: Pipeline
    result: Optional[NodeResult] = None
    visits: int = 0
    children: list[str] = field(default_factory=list)
    parent: Optional[str] = None
    pruned: bool = False
    fused_from: list[str] = field(default_factory=list)


# ---------- protocols (caller supplies) ----------

Evaluator = Callable[[Pipeline], NodeResult]
Expander = Callable[[Node, "MCGS"], Pipeline]   # propose child given parent
Fuser = Callable[[list[Node], "MCGS"], Pipeline]  # merge complementary branches
StagnationDetector = Callable[["MCGS", Node], bool]


# ---------- engine ----------

@dataclass
class MCGSConfig:
    score_threshold: float = 0.96
    regression_epsilon: int = 2
    enforce_regression: bool = True   # production mode True; cold-start False
    explore_coef_init: float = 1.4
    explore_coef_final: float = 0.2
    decay_horizon: int = 30           # iterations until c(t) reaches final
    fusion_top_k: int = 3
    stagnation_window: int = 3
    stagnation_eps: float = 0.005
    max_iterations: int = 50
    parallel_branches: int = 1       # paper Eq.3: 2-3 configs trained in parallel


class MCGS:
    def __init__(self, cfg: MCGSConfig, evaluator: Evaluator,
                 expander: Expander, fuser: Optional[Fuser] = None,
                 stagnation_detector: Optional[StagnationDetector] = None,
                 persist_dir: Optional[Path] = None):
        self.cfg = cfg
        self.evaluator = evaluator
        self.expander = expander
        self.fuser = fuser
        self.stagnation_detector = stagnation_detector or self._default_stagnation
        self.G = nx.DiGraph()
        self.nodes: dict[str, Node] = {}
        self.iteration = 0
        self.persist_dir = persist_dir
        self.history: list[dict] = []

    # ---- core ops ----

    def add_node(self, pipeline: Pipeline, parent_id: Optional[str] = None,
                 fused_from: Optional[list[str]] = None) -> Node:
        nid = pipeline.fingerprint()
        if nid in self.nodes:
            return self.nodes[nid]
        node = Node(id=nid, pipeline=pipeline, parent=parent_id,
                    fused_from=fused_from or [])
        self.nodes[nid] = node
        self.G.add_node(nid)
        if parent_id:
            self.G.add_edge(parent_id, nid)
            self.nodes[parent_id].children.append(nid)
        for src in fused_from or []:
            self.G.add_edge(src, nid, fusion=True)
        return node

    def evaluate(self, node: Node) -> NodeResult:
        """Run full train+eval. Trust evaluator to persist artifacts."""
        result = self.evaluator(node.pipeline)
        node.result = result
        node.visits += 1
        self.history.append({
            "iter": self.iteration, "id": node.id, "score": result.score,
            "regressions": result.regressions, "failed": result.failed,
            "parent": node.parent, "fused_from": node.fused_from,
            "notes": node.pipeline.notes,
        })
        if self.persist_dir:
            self._persist()
        return result

    def is_acceptable(self, result: NodeResult) -> bool:
        if result.failed:
            return False
        if result.score < self.cfg.score_threshold:
            return False
        if self.cfg.enforce_regression and result.regressions > self.cfg.regression_epsilon:
            return False
        return True

    # ---- selection ----

    def explore_coef(self) -> float:
        ratio = min(1.0, self.iteration / max(1, self.cfg.decay_horizon))
        return (1 - ratio) * self.cfg.explore_coef_init + ratio * self.cfg.explore_coef_final

    def _mean_descendant_score(self, node: Node) -> float:
        scores: list[float] = []
        stack = [node.id]
        seen = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            n = self.nodes[nid]
            if n.result and not n.result.failed:
                scores.append(n.result.score)
            stack.extend(n.children)
        return sum(scores) / len(scores) if scores else 0.0

    def uct(self, node: Node, total_evals: int) -> float:
        if node.visits == 0:
            return float("inf")
        mean = self._mean_descendant_score(node)
        c = self.explore_coef()
        return mean + c * math.sqrt(math.log(max(1, total_evals)) / node.visits)

    def select_leaf(self) -> Node:
        candidates = [n for n in self.nodes.values()
                      if not n.pruned and n.result and not n.result.failed]
        if not candidates:
            # fall back to any non-pruned
            candidates = [n for n in self.nodes.values() if not n.pruned]
        total = sum(n.visits for n in candidates) or 1
        return max(candidates, key=lambda n: self.uct(n, total))

    def top_k(self, k: int) -> list[Node]:
        scored = [n for n in self.nodes.values()
                  if n.result and not n.result.failed and not n.pruned]
        scored.sort(key=lambda n: n.result.score, reverse=True)
        return scored[:k]

    def best(self) -> Optional[Node]:
        top = self.top_k(1)
        return top[0] if top else None

    def best_acceptable(self) -> Optional[Node]:
        for n in self.top_k(len(self.nodes)):
            if self.is_acceptable(n.result):
                return n
        return None

    # ---- stagnation ----

    def _default_stagnation(self, mcgs: "MCGS", node: Node) -> bool:
        # stagnant if last `window` descendant scores improve by < eps over best
        scores = [h["score"] for h in self.history[-self.cfg.stagnation_window:]
                  if not h["failed"]]
        if len(scores) < self.cfg.stagnation_window:
            return False
        best = max(h["score"] for h in self.history if not h["failed"])
        return all(best - s < self.cfg.stagnation_eps for s in scores)

    # ---- main loop ----

    def _evaluate_parallel(self, nodes: list[Node]) -> list[NodeResult]:
        """Evaluate multiple nodes in parallel using ProcessPoolExecutor or Modal."""
        if self.cfg.parallel_branches <= 1 or len(nodes) <= 1:
            return [self.evaluate(n) for n in nodes]

        # Try Modal first (if configured)
        try:
            import modal
            return self._evaluate_modal(nodes)
        except ImportError:
            pass

        # Fallback: local ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor
        pipelines = [n.pipeline for n in nodes]
        with ProcessPoolExecutor(max_workers=min(len(nodes), self.cfg.parallel_branches)) as ex:
            results = list(ex.map(self.evaluator, pipelines))
        for node, result in zip(nodes, results):
            node.result = result
            node.visits += 1
        return results

    def _evaluate_modal(self, nodes: list[Node]) -> list[NodeResult]:
        """Evaluate branches via Modal sandbox (paper Section 2.5 / 2.6)."""
        from . import modal_runner  # lazy import
        return modal_runner.evaluate_batch([n.pipeline for n in nodes])

    def run(self, root: Pipeline) -> Node:
        root_node = self.add_node(root)
        root_result = self.evaluate(root_node)
        if self.is_acceptable(root_result):
            return root_node

        for it in range(self.cfg.max_iterations):
            self.iteration = it + 1
            parent = self.select_leaf()

            # stagnation -> fuse if possible, else continue with evolution flag
            if self.stagnation_detector(self, parent):
                if self.fuser and len(self.top_k(self.cfg.fusion_top_k)) >= 2:
                    branches = self.top_k(self.cfg.fusion_top_k)
                    fused_pi = self.fuser(branches, self)
                    fused_pi.iteration = self.iteration
                    fused_pi.notes = f"FUSE({','.join(n.id for n in branches)})"
                    new = self.add_node(fused_pi, parent_id=branches[0].id,
                                        fused_from=[n.id for n in branches])
                    self.evaluate(new)
                    if self.is_acceptable(new.result):
                        return new
                    if new.result.failed or new.result.score < parent.result.score:
                        new.pruned = True   # rollback
                    continue

            # Parallel branch evaluation (paper Eq.3)
            children: list[Node] = []
            for _ in range(max(1, self.cfg.parallel_branches)):
                child_pi = self.expander(parent, self)
                child_pi.parent_id = parent.id
                child_pi.iteration = self.iteration
                child = self.add_node(child_pi, parent_id=parent.id)
                children.append(child)

            # Evaluate in parallel
            results = self._evaluate_parallel(children)
            for child, result in zip(children, results):
                if result.failed:
                    child.pruned = True
                    continue
                # rollback rule (paper Section 2.4)
                if parent.result and result.score < parent.result.score:
                    child.pruned = True
                    continue
                # regression gate (production)
                if self.cfg.enforce_regression and result.regressions > self.cfg.regression_epsilon:
                    child.pruned = True
                    continue
                if self.is_acceptable(result):
                    return child

        # exhausted budget; return best acceptable or just best
        return self.best_acceptable() or self.best() or root_node

    # ---- persistence ----

    def _persist(self) -> None:
        if not self.persist_dir:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        nodes_dump = {
            nid: {
                "pipeline": n.pipeline.to_dict(),
                "result": asdict(n.result) if n.result else None,
                "parent": n.parent,
                "children": n.children,
                "pruned": n.pruned,
                "fused_from": n.fused_from,
                "visits": n.visits,
            }
            for nid, n in self.nodes.items()
        }
        (self.persist_dir / "graph.json").write_text(json.dumps({
            "iteration": self.iteration,
            "history": self.history,
            "nodes": nodes_dump,
            "edges": list(self.G.edges(data=True)),
        }, indent=2, default=str))
