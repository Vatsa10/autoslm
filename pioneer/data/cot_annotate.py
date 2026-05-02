"""Chain-of-thought annotation by teacher model (paper Section 2.3 quality control 5).

Teacher generates step-by-step reasoning. Selection (Section 2.5):
- DeepSeek-R1: math/scientific reasoning (GSM8K, ARC)
- GPT-4.1: code, general knowledge (HumanEval, TriviaQA)
"""
from __future__ import annotations
import json
from typing import Iterable

from ..llm import LLMClient
from .curate import Example


COT_SYSTEM = """You add a chain-of-thought to a (input, gold-output) pair.
Produce reasoning that arrives at the gold output. Then concatenate:
<think> ... </think>\n<answer>{gold output}</answer>
Output JSON: {"cot_output": str}.
"""


def annotate_cot(client: LLMClient, examples: Iterable[Example],
                 wrap_format: str = "tagged") -> list[Example]:
    out: list[Example] = []
    for ex in examples:
        try:
            resp = client.complete(
                [{"role": "user", "content": json.dumps(
                    {"input": ex.input, "gold": ex.output}, default=str)}],
                system=COT_SYSTEM,
            )
            text = resp.content.strip()
            if "```" in text:
                text = text.split("```", 1)[-1].split("```", 1)[0]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            new = Example(
                input=ex.input,
                output=str(data["cot_output"]),
                label=ex.label,
                is_hard_negative=ex.is_hard_negative,
                is_replay=ex.is_replay,
                metadata={**ex.metadata, "cot": True, "wrap_format": wrap_format},
            )
            out.append(new)
        except Exception:
            out.append(ex)
    return out


def teacher_for_task(task: str, cfg) -> str:
    """Pick teacher model per paper Section 2.5."""
    reasoning = {"math", "reasoning", "qa_reasoning", "arc", "gsm8k"}
    return cfg.teacher_reasoning_model if task.lower() in reasoning else cfg.teacher_general_model
