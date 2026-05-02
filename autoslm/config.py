from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel, Field


class HardwareTier(BaseModel):
    """Preset for compute envelope. Drives quantization + LoRA defaults."""
    name: str
    quant: str = "none"
    lora_rank: int = 16
    max_seq_len: int = 2048
    bf16: bool = True
    grad_checkpoint: bool = False


TIERS: dict[str, HardwareTier] = {
    "edge": HardwareTier(name="edge", quant="4bit", lora_rank=8, max_seq_len=1024, grad_checkpoint=True),
    "mid": HardwareTier(name="mid", quant="8bit", lora_rank=32, max_seq_len=2048, grad_checkpoint=True),
    "big": HardwareTier(name="big", quant="none", lora_rank=64, max_seq_len=4096, grad_checkpoint=False),
}


class AutoSLMConfig(BaseModel):
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    workdir: Path = Field(default_factory=lambda: Path.cwd() / ".autoslm")

    # Orchestrator LLM
    orchestrator_model: str = os.getenv("AUTOSLM_ORCH_MODEL", "anthropic/claude-sonnet-4-6")
    orchestrator_max_turns: int = 500  # production default; cold-start uses 1500
    thinking_budget: int = 32_000

    # Teacher / judge LLMs
    judge_model: str = os.getenv("AUTOSLM_JUDGE_MODEL", "openai/gpt-4o-mini")
    teacher_reasoning_model: str = os.getenv("AUTOSLM_TEACHER_REASONING", "deepseek/deepseek-reasoner")
    teacher_general_model: str = os.getenv("AUTOSLM_TEACHER_GENERAL", "openai/gpt-4.1")

    # Trainer
    base_model_default: str = "meta-llama/Llama-3.2-3B"
    hardware_tier: str = "mid"

    # Search
    score_threshold: float = 0.96
    regression_epsilon: int = 2
    explore_coef_init: float = 1.4
    explore_coef_final: float = 0.2
    parallel_branches: int = 2

    # Trace store
    trace_db: str = "duckdb"
    trace_db_path: Path = Field(default_factory=lambda: Path.cwd() / ".autoslm" / "traces.duckdb")

    def tier(self) -> HardwareTier:
        return TIERS[self.hardware_tier]

    def ensure_dirs(self) -> None:
        for sub in ("runs", "datasets", "checkpoints", "logs", "graphs"):
            (self.workdir / sub).mkdir(parents=True, exist_ok=True)


DEFAULT = AutoSLMConfig()
