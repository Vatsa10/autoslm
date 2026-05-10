"""Smoke test for cold-start mode.

Minimal 50-example synthetic task, single iteration, asserts loop completes.
Requires torch (cold-start evaluator imports inference.load_for_inference).
"""

from __future__ import annotations

import importlib.util

import pytest
from pathlib import Path

torch_available = importlib.util.find_spec("torch") is not None

pytestmark = [
    pytest.mark.skipif(not torch_available, reason="torch not installed"),
    pytest.mark.e2e,
]

from autoslm.modes.cold_start import run_cold_start
from autoslm.search.pipeline import Pipeline, DatasetSpec, HyperParams, LearningStrategy


def test_cold_start_minimal():
    """Run cold-start with minimal synthetic data."""
    # Create a proper AutoSLMConfig
    from autoslm.config import AutoSLMConfig
    cfg = AutoSLMConfig(workdir=Path("test_run"))
    cfg.base_model_default = "test-model"

    # Mock the LLMClient to avoid API calls
    import unittest.mock as mock

    with mock.patch("autoslm.modes.cold_start.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete.return_value = type("Resp", (), {"content": "{}"})()

        with mock.patch("autoslm.modes.cold_start.train_pipeline") as mock_train:
            mock_train.return_value = type("Result", (), {
                "model_id": "test",
                "checkpoint_path": "/tmp/test",
                "final_loss": 0.5,
                "error": None,
            })()
            with mock.patch("autoslm.modes.cold_start.score_pipeline") as mock_eval:
                mock_eval.return_value = (0.85, 0, {"score": 0.85})
                with mock.patch("autoslm.modes.cold_start.classify_task") as mock_classify:
                    from autoslm.data.acquire import TaskClassification
                    mock_classify.return_value = TaskClassification(
                        task_type="classification", eval_method="exact_match",
                        supervision="direct", model_family="decoder",
                        canonical_dataset=None, labels=None, rationale="test"
                    )
                    with mock.patch("autoslm.modes.cold_start.baseline_survey") as mock_survey:
                        mock_survey.return_value = type("Survey", (), {
                            "published_sota": 0.80,
                            "target_threshold": 0.90,
                            "notes": "test"
                        })()
                        with mock.patch("autoslm.modes.cold_start.acquire_dataset") as mock_acquire:
                            mock_acquire.return_value = [
                                type("Example", (), {
                                    "input": "test", "output": "test", "label": "test",
                                    "metadata": {"source": "synthetic"},
                                    "is_hard_negative": False,
                                    "is_replay": False,
                                })()
                            ] * 50
                            with mock.patch("autoslm.modes.cold_start.build_holdout") as mock_holdout:
                                # build_holdout returns (eval_set, train_pool)
                                # eval_set needs pos, neg, boundary attributes
                                eval_set = type("EvalSet", (), {
                                    "pos": [], "neg": [], "boundary": []
                                })()
                                mock_holdout.return_value = (eval_set, [])
                                result = run_cold_start(
                                    cfg=cfg,
                                    task_spec="synthetic classification",
                                    base_model="test-model",
                                    dataset_hint="synthetic",
                                    target_threshold=0.90,
                                    max_iterations=1,
                                )
                                assert result is not None
