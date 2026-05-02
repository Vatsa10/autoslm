"""LLM-as-judge (paper Section 2.2 + 2.6).

Default judge: GPT-4o-mini or DeepSeek-Chat (cheap). Returns score in [0,1].
"""
from __future__ import annotations
import json
from typing import Optional

from ..llm import LLMClient


JUDGE_SYSTEM = """You are a strict evaluator. Given (input, gold, prediction[, criteria]),
score the prediction on [0, 1] for correctness against the gold output.

Use the criteria if provided. Be conservative: partial credit only when the
prediction captures the core correct content. Off-topic, hallucinated, or
incorrectly-formatted predictions get 0.

Output JSON: {"score": float, "verdict": "pass"|"fail", "reasoning": str}.
"""


def llm_judge_score(client: LLMClient, input_text: str, gold: str, prediction: str,
                    criteria: Optional[str] = None,
                    pass_threshold: float = 0.7) -> dict:
    payload = {
        "input": input_text, "gold": gold, "prediction": prediction,
        "criteria": criteria or "Correctness against gold; format may vary.",
    }
    resp = client.complete(
        [{"role": "user", "content": json.dumps(payload, default=str)[:30_000]}],
        system=JUDGE_SYSTEM,
    )
    text = resp.content.strip()
    if "```" in text:
        text = text.split("```", 1)[-1].split("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        score = float(data.get("score", 0.0))
        verdict = data.get("verdict") or ("pass" if score >= pass_threshold else "fail")
        return {"score": score, "verdict": verdict,
                "reasoning": data.get("reasoning", "")[:1000]}
    except Exception:
        return {"score": 0.0, "verdict": "fail", "reasoning": "judge parse error"}
