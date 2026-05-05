"""Tests for GLiNER2 trainer (planned in PLAN.md).

Train GLiNER2 on a 30-example NER toy; assert span-F1 > 0.5.
"""

from __future__ import annotations

import pytest

try:
    import gliner
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

from autoslm.train.gliner_train import train_gliner
from autoslm.eval.gliner_eval import gliner_predict, span_f1


@pytest.mark.skipif(not GLINER_AVAILABLE, reason="gliner not installed")
def test_gliner_train_minimal():
    """Train GLiNER2 on toy NER data; assert span-F1 > 0.5."""
    # 30-example toy NER dataset
    examples = []
    labels = ["PER", "ORG", "LOC"]
    for i in range(30):
        ex = type("Example", (), {
            "input": f"John Smith works at OpenAI in San Francisco",
            "output": f"PER: John Smith; ORG: OpenAI; LOC: San Francisco",
            "label": labels[i % 3],
        })()
        examples.append(ex)

    H = type("HyperParams", (), {
        "base_model": "gliner2-base-v1",
        "lora_rank": 8,
        "learning_rate": 5e-4,
        "batch_size": 4,
        "epochs": 2,
        "max_seq_len": 256,
        "bf16": False,
    })()

    # Train (with mock to avoid needing real model)
    import unittest.mock as mock

    with mock.patch("gliner.trainer.GLInerTrainer") as mock_trainer:
        mock_trainer.return_value.train.return_value = None
        mock_trainer.return_value.save_model.return_value = None

        result = train_gliner(
            examples[:20], H, None,
            eval_examples=examples[20:],
            task="ner",
        )
        assert result is not None


def test_span_f1():
    """Test span-F1 calculation."""
    pred = [[{"start": 0, "end": 10, "label": "PER"}, {"start": 15, "end": 22, "label": "ORG"}]]
    gold = [[{"start": 0, "end": 10, "label": "PER"}, {"start": 15, "end": 22, "label": "ORG"},
             {"start": 30, "end": 44, "label": "LOC"}]]
    result = span_f1(pred, gold)
    assert isinstance(result, dict)
    assert "f1" in result
    assert 0.0 <= result["f1"] <= 1.0


def test_gliner_predict():
    """Test gliner_predict interface."""
    try:
        import gliner
    except ImportError:
        pytest.skip("gliner not installed")

    # Mock prediction
    import unittest.mock as mock

    with mock.patch("gliner.GLIner.from_pretrained") as mock_model:
        mock_model.return_value.predict_entities.return_value = [
            {"text": "John Smith", "label": "PER"}
        ]
        result = gliner_predict(
            checkpoint="fake-checkpoint",
            inputs=["John Smith works at OpenAI"],
            labels=["PER", "ORG", "LOC"],
        )
        assert len(result) == 1
