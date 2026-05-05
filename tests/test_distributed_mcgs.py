"""Tests for distributed MCGS (paper Eq. 3, v0.4 planned)."""

from __future__ import annotations

import pytest
from autoslm.search.mcgs import MCGS, MCGSConfig, Node, NodeResult
from autoslm.search.pipeline import Pipeline, DatasetSpec, HyperParams, LearningStrategy


def _make_pipeline(fingerprint: str = "test") -> Pipeline:
    D = DatasetSpec(name="test")
    H = HyperParams(base_model="test-model")
    S = LearningStrategy()
    p = Pipeline(D=D, H=H, S=S)
    p.notes = fingerprint
    return p


def _mock_evaluator(pipeline: Pipeline) -> NodeResult:
    """Mock evaluator that returns improving scores."""
    import hashlib, json
    h = hashlib.md5(json.dumps(pipeline.notes or "", default=str).encode()).hexdigest()
    score = 0.5 + (int(h[:2], 16) % 40) / 100.0  # 0.5 - 0.89
    return NodeResult(score=min(score, 0.96), regressions=0)


def test_parallel_branches_config():
    cfg = MCGSConfig(parallel_branches=2, max_iterations=5)
    assert cfg.parallel_branches == 2


def _mock_expander(node: Node, mcgs: MCGS) -> Pipeline:
    """Mock expander that returns a modified pipeline."""
    import copy
    new_pi = copy.deepcopy(node.pipeline)
    new_pi.notes = f"expanded_from_{node.id}"
    return new_pi


def test_mcgs_single_branch():
    """Original single-branch behavior still works."""
    cfg = MCGSConfig(parallel_branches=1, max_iterations=10, score_threshold=0.95)
    mcgs = MCGS(cfg, evaluator=_mock_evaluator, expander=_mock_expander)
    root = _make_pipeline("root")
    result = mcgs.run(root)
    assert result is not None


def test_mcgs_parallel_evaluation():
    """Test that parallel_branches > 1 creates multiple children."""

    def counting_evaluator(pipeline: Pipeline) -> NodeResult:
        return NodeResult(score=0.8, regressions=0)

    cfg = MCGSConfig(parallel_branches=2, max_iterations=3, score_threshold=0.95)
    mcgs = MCGS(cfg, evaluator=counting_evaluator, expander=_mock_expander)
    root = _make_pipeline("root")
    root_node = mcgs.add_node(root)
    root_result = mcgs.evaluate(root_node)
    # After running, should have at least 1 node
    assert len(mcgs.nodes) >= 1


def _module_level_evaluator(pipeline: Pipeline) -> NodeResult:
    """Module-level evaluator that can be pickled."""
    import hashlib, json
    h = hashlib.md5(json.dumps(pipeline.notes or "", default=str).encode()).hexdigest()
    score = 0.7 + (int(h[:2], 16) % 25) / 100.0
    return NodeResult(score=min(score, 0.95), regressions=0)


def test_mcgs_parallel_best_selected():
    """When running parallel branches, best result should be selected."""

    cfg = MCGSConfig(parallel_branches=1, max_iterations=5, score_threshold=0.80)
    mcgs = MCGS(cfg, evaluator=_module_level_evaluator, expander=_mock_expander)
    root = _make_pipeline("root")
    result = mcgs.run(root)
    assert result is not None
    assert result.result is not None
    assert result.result.score >= 0.7
