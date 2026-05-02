"""Dispatcher: routes a Pipeline to the correct trainer (decoder vs encoder).

Paper §2.1 supports two model families:
  - decoder: Llama / Qwen via peft + trl SFT
  - gliner2: encoder NER + classification, full FT or LoRA
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from ..search.pipeline import HyperParams, LearningStrategy
from ..data.curate import Example
from .lora_sft import train_lora_sft, TrainResult


def train_pipeline(
    examples: list[Example],
    H: HyperParams,
    S: LearningStrategy,
    output_dir: str | Path,
    eval_examples: Optional[list[Example]] = None,
    task: str = "generation",
) -> TrainResult:
    if H.model_family == "gliner2":
        from .gliner_train import train_gliner
        gliner_task = "ner" if task in {"ner", "extraction"} else "classification"
        return train_gliner(examples, H, S, output_dir,
                           eval_examples=eval_examples, task=gliner_task)
    return train_lora_sft(examples, H, S, output_dir,
                         eval_examples=eval_examples)
