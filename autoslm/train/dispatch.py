"""Dispatcher: routes a Pipeline to the correct trainer (decoder vs encoder, SFT vs DPO/KTO).

Paper §2.1 supports two model families:
  - decoder: Llama / Qwen via peft + trl SFT
  - gliner2: encoder NER + classification, full FT or LoRA

Paper Section 6.3: SFT is default; DPO/KTO used when preference data available.
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
    # GLiNER2 encoder path
    if H.model_family == "gliner2":
        from .gliner_train import train_gliner
        gliner_task = "ner" if task in {"ner", "extraction"} else "classification"
        return train_gliner(examples, H, S, output_dir,
                           eval_examples=eval_examples, task=gliner_task)

    # DPO/KTO preference optimization (paper Section 6.3)
    if S.objective == "dpo":
        from .dpo import train_dpo
        # Convert Example to dict for DPO pair construction
        ex_dicts = [{"input": e.input, "output": e.output,
                     "judge_score": getattr(e, "judge_score", 0.5)} for e in examples]
        return train_dpo(ex_dicts, H, S, output_dir,
                         eval_examples=eval_examples)
    if S.objective == "kto":
        from .dpo import train_kto
        ex_dicts = [{"input": e.input, "output": e.output,
                     "judge_score": getattr(e, "judge_score", 0.5)} for e in examples]
        return train_kto(ex_dicts, H, S, output_dir,
                         eval_examples=eval_examples)

    # Default: SFT
    return train_lora_sft(examples, H, S, output_dir,
                           eval_examples=eval_examples)
