"""Cold-start data acquisition (paper §2.5).

Stages:
  1. classify_task — LLM determines task type, eval method, supervision format
  2. acquire_dataset — HF benchmark download (preferred) or teacher-LLM synthesis
  3. baseline_survey — pulls published SOTA + sets initial target threshold
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional

from ..llm import LLMClient
from .curate import Example


CLASSIFY_SYSTEM = """You classify an autoslm cold-start task spec.

Output strict JSON:
  {"task_type": "classification" | "ner" | "generation" | "qa_reasoning" | "summarization" | "code",
   "eval_method": "exact_match" | "f1" | "rouge" | "rouge_2" | "pass_at_k" | "llm_judge",
   "supervision": "direct" | "cot",
   "model_family": "decoder" | "gliner2",
   "canonical_dataset": str | null,   // HuggingFace datasets ID if a known benchmark applies
   "labels": list[str] | null,        // for classification or NER, leave null otherwise
   "rationale": str}

Heuristics:
  - GLiNER2 only when task is NER or short-text classification with fixed label set.
  - cot for math/reasoning/multi-step QA (GSM8K, ARC, etc.). direct otherwise.
  - rouge_2 for short summarization (XSum, SAMSum). rouge for general summarization.
  - pass_at_k for code generation (HumanEval, MBPP).
"""


@dataclass
class TaskClassification:
    task_type: str
    eval_method: str
    supervision: str
    model_family: str
    canonical_dataset: Optional[str]
    labels: Optional[list[str]]
    rationale: str


def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.split("```", 1)[-1].split("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def classify_task(spec: str, client: LLMClient) -> TaskClassification:
    resp = client.complete([{"role": "user", "content": spec}], system=CLASSIFY_SYSTEM)
    try:
        d = _parse_json(resp.content)
    except Exception:
        d = {"task_type": "generation", "eval_method": "llm_judge",
             "supervision": "direct", "model_family": "decoder",
             "canonical_dataset": None, "labels": None,
             "rationale": "fallback"}
    return TaskClassification(
        task_type=str(d.get("task_type", "generation")),
        eval_method=str(d.get("eval_method", "exact_match")),
        supervision=str(d.get("supervision", "direct")),
        model_family=str(d.get("model_family", "decoder")),
        canonical_dataset=d.get("canonical_dataset"),
        labels=d.get("labels"),
        rationale=str(d.get("rationale", ""))[:1000],
    )


# ---------- dataset acquisition ----------

HF_FIELD_GUESSES = {
    "input": ["question", "prompt", "text", "input", "article", "dialogue"],
    "gold":  ["answer", "label", "output", "target", "summary", "highlights",
              "labels", "answers", "completion", "canonical_solution"],
}


def _autodetect_fields(row: dict) -> tuple[Optional[str], Optional[str]]:
    inp = next((k for k in HF_FIELD_GUESSES["input"] if k in row), None)
    gld = next((k for k in HF_FIELD_GUESSES["gold"] if k in row), None)
    return inp, gld


def acquire_from_huggingface(dataset_id: str,
                            split: str = "train",
                            limit: int = 2000) -> list[Example]:
    """Try to load `dataset_id` from HuggingFace via `datasets`; auto-map fields."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
    except Exception:
        try:
            # some datasets need a subset name
            ds = load_dataset(dataset_id, "main", split=split, streaming=True)
        except Exception:
            return []
    out: list[Example] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        inp_k, gld_k = _autodetect_fields(row)
        if not inp_k or not gld_k:
            continue
        v = row[gld_k]
        if isinstance(v, dict) and "text" in v:
            v = v["text"]
        if isinstance(v, list):
            v = v[0] if v else ""
        out.append(Example(
            input=str(row[inp_k]),
            output=str(v),
            label=str(row.get("label")) if "label" in row else None,
            metadata={"source": dataset_id, "split": split},
        ))
    return out


SYNTH_SYSTEM = """You synthesize seed training examples for an autoslm cold-start task.

Given a task description and target labels (if any), produce N diverse examples.
Output strict JSON: {"examples": [{"input": str, "gold": str, "label": str|null}]}.

Constraints:
  - Each example must be realistic and unambiguous.
  - For classification: cover all labels roughly equally.
  - For NER: include the entity types from `labels` and use bracketed
    annotation in `gold`, e.g. "John (PER) lives in [Paris](LOC)".
  - Avoid duplicates and trivial templates.
"""


def acquire_synth(client: LLMClient, spec: str, classification: TaskClassification,
                 n: int = 200) -> list[Example]:
    """Teacher-LLM seed synthesis fallback."""
    payload = {
        "task_spec": spec,
        "task_type": classification.task_type,
        "labels": classification.labels,
        "n": n,
    }
    resp = client.complete(
        [{"role": "user", "content": json.dumps(payload, default=str)}],
        system=SYNTH_SYSTEM,
    )
    try:
        d = _parse_json(resp.content)
        out: list[Example] = []
        for ex in d.get("examples", [])[:n]:
            out.append(Example(
                input=str(ex.get("input", "")),
                output=str(ex.get("gold", "")),
                label=ex.get("label"),
                metadata={"source": "synth"},
            ))
        return [e for e in out if e.input and e.output]
    except Exception:
        return []


def acquire_dataset(spec: str, client: LLMClient,
                   classification: TaskClassification,
                   hint: Optional[str] = None,
                   limit: int = 2000) -> list[Example]:
    candidate = hint or classification.canonical_dataset
    if candidate:
        for split in ("train", "validation", "test"):
            data = acquire_from_huggingface(candidate, split=split, limit=limit)
            if data:
                return data
    # synthesis fallback
    return acquire_synth(client, spec, classification, n=min(limit, 200))


# ---------- baseline survey ----------

BASELINE_SYSTEM = """Estimate published SOTA + set an initial target threshold for autoslm cold-start.

Given a task spec and base model, return JSON:
  {"published_sota": float | null,        // 0..1
   "target_threshold": float,             // 0..1, where to stop iterating
   "notes": str}

Set target_threshold conservatively below published_sota when known
(e.g. 0.96 default if no SOTA known). For 3B-class models on knowledge-heavy
tasks, set lower (e.g. 0.5-0.7).
"""


@dataclass
class BaselineSurvey:
    published_sota: Optional[float]
    target_threshold: float
    notes: str


def baseline_survey(spec: str, base_model: str, client: LLMClient) -> BaselineSurvey:
    payload = {"task_spec": spec, "base_model": base_model}
    resp = client.complete(
        [{"role": "user", "content": json.dumps(payload)}],
        system=BASELINE_SYSTEM,
    )
    try:
        d = _parse_json(resp.content)
        return BaselineSurvey(
            published_sota=d.get("published_sota"),
            target_threshold=float(d.get("target_threshold", 0.96)),
            notes=str(d.get("notes", ""))[:1000],
        )
    except Exception:
        return BaselineSurvey(published_sota=None, target_threshold=0.96, notes="fallback")
