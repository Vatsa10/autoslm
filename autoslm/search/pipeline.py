"""Training pipeline as paper Section 2.2: pi = (D, H, S).

Each pipeline is one full train+eval attempt. Hashable for graph keys.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


@dataclass
class DatasetSpec:
    """D: dataset composition + curation constraints."""
    name: str
    gold_path: Optional[str] = None
    hard_neg_path: Optional[str] = None
    replay_path: Optional[str] = None
    # target proportions sum ~ 1.0; replay omitted in cold-start
    gold_ratio: float = 0.65
    hard_neg_ratio: float = 0.35
    replay_ratio: float = 0.0
    max_examples: Optional[int] = None
    label_balance_max_ratio: float = 3.0  # paper: no label > 3x another
    context_length_match: bool = True
    twofor_one: bool = True
    surface_patterns_per_label: int = 4


@dataclass
class HyperParams:
    """H: optimization config."""
    base_model: str
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 3
    warmup_ratio: float = 0.03
    max_seq_len: int = 2048
    quant: Literal["none", "8bit", "4bit"] = "8bit"
    bf16: bool = True
    grad_checkpoint: bool = True
    system_prompt: Optional[str] = None
    full_finetune: bool = False  # encoder path or "big" tier
    distributed: bool = False     # FSDP for "big" tier
    # Model family — `decoder` for Llama/Qwen (peft+trl); `gliner2` for the
    # encoder NER + classification path (paper §2.1).
    model_family: Literal["decoder", "gliner2"] = "decoder"
    # GLiNER2 label set (entity types or class labels)
    gliner_labels: Optional[list[str]] = None
    seed: int = 42


@dataclass
class LearningStrategy:
    """S: supervision shape."""
    supervision: Literal["direct", "cot"] = "direct"
    teacher_model: Optional[str] = None  # used for synth / CoT annotation
    eval_method: Literal["exact_match", "f1", "rouge", "pass_at_k", "llm_judge"] = "exact_match"
    judge_model: Optional[str] = None
    judge_criteria: Optional[str] = None


@dataclass
class Pipeline:
    """pi = (D, H, S). One node = one full train+eval attempt."""
    D: DatasetSpec
    H: HyperParams
    S: LearningStrategy
    notes: str = ""
    parent_id: Optional[str] = None
    iteration: int = 0
    extra: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        blob = json.dumps(
            {"D": asdict(self.D), "H": asdict(self.H), "S": asdict(self.S)},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "id": self.fingerprint(),
            "D": asdict(self.D),
            "H": asdict(self.H),
            "S": asdict(self.S),
            "notes": self.notes,
            "parent_id": self.parent_id,
            "iteration": self.iteration,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        return cls(
            D=DatasetSpec(**d["D"]),
            H=HyperParams(**d["H"]),
            S=LearningStrategy(**d["S"]),
            notes=d.get("notes", ""),
            parent_id=d.get("parent_id"),
            iteration=d.get("iteration", 0),
            extra=d.get("extra", {}),
        )
