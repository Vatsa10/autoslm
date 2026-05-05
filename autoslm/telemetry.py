"""Cost telemetry (paper Section 6.1).

Per-run tracking:
  - LLM tokens by role (orchestrator, judge, teacher) via litellm callbacks
  - GPU minutes per training step (timed wall-clock)
  - Sandbox hours (Modal or local)
  - Output: runs/<id>/cost.json and rolled-up cost.md
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    total: int = 0

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input += other.input
        self.output += other.output
        self.total += other.total
        return self

    @property
    def cost_usd(self) -> float:
        """Rough cost estimate (adjust per-model)."""
        input_cost = self.input * 1.5e-6  # $1.5/1M tokens
        output_cost = self.output * 2.0e-6  # $2.0/1M tokens
        return input_cost + output_cost


@dataclass
class CostRecord:
    run_id: str
    # LLM token usage by role
    tokens: dict[str, TokenUsage] = field(default_factory=dict)  # role -> TokenUsage
    # Training time
    gpu_minutes: float = 0.0
    training_steps: int = 0
    # Sandbox
    sandbox_hours: float = 0.0
    sandbox_provider: str = "local"  # "local" | "modal"
    # Totals
    total_llm_cost_usd: float = 0.0
    total_compute_cost_usd: float = 0.0
    # Metadata
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def finish(self):
        self.end_time = time.time()

    @property
    def wall_clock_minutes(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) / 60.0
        return (time.time() - self.start_time) / 60.0

    @property
    def total_cost_usd(self) -> float:
        return self.total_llm_cost_usd + self.total_compute_cost_usd


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Tracks costs for one run and persists to disk."""

    def __init__(self, run_id: str, output_dir: str | Path = "runs"):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.record = CostRecord(run_id=run_id)
        self._training_start: Optional[float] = None

    # ---- LLM tokens ----

    def log_llm_tokens(
        self,
        role: str,  # "orchestrator" | "judge" | "teacher"
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        if role not in self.record.tokens:
            self.record.tokens[role] = TokenUsage()
        usage = TokenUsage(input=input_tokens, output=output_tokens,
                           total=input_tokens + output_tokens)
        self.record.tokens[role] += usage

    def log_litellm_response(self, role: str, response) -> None:
        """Extract token usage from a litellm response object."""
        try:
            usage = response.usage
            self.log_llm_tokens(
                role=role,
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            )
        except Exception:
            pass

    # ---- Training time ----

    def training_start(self) -> None:
        self._training_start = time.time()

    def training_end(self) -> None:
        if self._training_start is not None:
            elapsed = (time.time() - self._training_start) / 60.0
            self.record.gpu_minutes += elapsed
            self._training_start = None

    def log_training_time(self, minutes: float) -> None:
        self.record.gpu_minutes += minutes

    def log_training_step(self) -> None:
        self.record.training_steps += 1

    # ---- Sandbox ----

    def log_sandbox_time(self, hours: float, provider: str = "local") -> None:
        self.record.sandbox_hours += hours
        self.record.sandbox_provider = provider

    # ---- Compute cost ----

    def _compute_llm_cost(self) -> float:
        total = 0.0
        for role, usage in self.record.tokens.items():
            total += usage.cost_usd
        return total

    def _compute_gpu_cost(self) -> float:
        # A10G ~$0.40/hr, A100 ~$1.50/hr
        if self.record.sandbox_provider == "modal":
            hourly = 0.40  # A10G on Modal
        else:
            hourly = 0.20  # local GPU estimate
        return self.record.gpu_minutes / 60.0 * hourly

    # ---- Persistence ----

    def save(self) -> Path:
        self.record.total_llm_cost_usd = self._compute_llm_cost()
        self.record.total_compute_cost_usd = self._compute_gpu_cost()
        self.record.finish()

        run_dir = self.output_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # JSON dump
        cost_path = run_dir / "cost.json"
        data = self._to_dict()
        cost_path.write_text(json.dumps(data, indent=2, default=str))

        # Markdown summary
        md_path = run_dir / "cost.md"
        md_path.write_text(self._to_markdown())

        return cost_path

    def _to_dict(self) -> dict:
        tokens_dict = {}
        for role, u in self.record.tokens.items():
            tokens_dict[role] = {
                "input": u.input,
                "output": u.output,
                "total": u.total,
                "cost_usd": u.cost_usd,
            }
        return {
            "run_id": self.record.run_id,
            "tokens": tokens_dict,
            "gpu_minutes": self.record.gpu_minutes,
            "training_steps": self.record.training_steps,
            "sandbox_hours": self.record.sandbox_hours,
            "sandbox_provider": self.record.sandbox_provider,
            "total_llm_cost_usd": self.record.total_llm_cost_usd,
            "total_compute_cost_usd": self.record.total_compute_cost_usd,
            "total_cost_usd": self.record.total_cost_usd,
            "wall_clock_minutes": self.record.wall_clock_minutes,
            "start_time": self.record.start_time,
            "end_time": self.record.end_time,
        }

    def _to_markdown(self) -> str:
        lines = [
            f"# Cost Report: {self.run_id}",
            "",
            "## LLM Token Usage",
            "",
            "| Role | Input | Output | Total | Cost (USD) |",
            "|------|-------|--------|-------|------------|",
        ]
        for role, u in self.record.tokens.items():
            lines.append(f"| {role} | {u.input} | {u.output} | {u.total} | ${u.cost_usd:.4f} |")
        lines += [
            "",
            f"**Total LLM Cost:** ${self.record.total_llm_cost_usd:.4f}",
            "",
            "## Compute",
            "",
            f"- GPU minutes: {self.record.gpu_minutes:.1f}",
            f"- Training steps: {self.record.training_steps}",
            f"- Sandbox: {self.record.sandbox_hours:.2f} hours ({self.record.sandbox_provider})",
            f"- Wall-clock: {self.record.wall_clock_minutes:.1f} min",
            "",
            f"**Total Compute Cost:** ${self.record.total_compute_cost_usd:.4f}",
            "",
            "## Total",
            "",
            f"**Total Cost:** ${self.record.total_cost_usd:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregate telemetry across runs
# ---------------------------------------------------------------------------

def aggregate_costs(runs_dir: str | Path) -> dict:
    """Roll up cost.json from all runs into a summary dict."""
    runs_dir = Path(runs_dir)
    total_llm = 0.0
    total_compute = 0.0
    run_costs: list[dict] = []
    for cost_file in runs_dir.rglob("cost.json"):
        try:
            data = json.loads(cost_file.read_text())
            total_llm += data.get("total_llm_cost_usd", 0.0)
            total_compute += data.get("total_compute_cost_usd", 0.0)
            run_costs.append(data)
        except Exception:
            continue
    return {
        "n_runs": len(run_costs),
        "total_llm_cost_usd": total_llm,
        "total_compute_cost_usd": total_compute,
        "total_cost_usd": total_llm + total_compute,
        "runs": run_costs,
    }


def save_aggregate_report(runs_dir: str | Path, output: str | Path = "cost.md") -> Path:
    """Generate a rolled-up cost.md for all runs."""
    agg = aggregate_costs(runs_dir)
    lines = [
        "# Aggregate Cost Report",
        "",
        f"## Summary ({agg['n_runs']} runs)",
        "",
        f"- Total LLM Cost: ${agg['total_llm_cost_usd']:.4f}",
        f"- Total Compute Cost: ${agg['total_compute_cost_usd']:.4f}",
        f"- **Grand Total: ${agg['total_cost_usd']:.4f}**",
        "",
        "## Per-Run",
        "",
    ]
    for run in agg.get("runs", []):
        lines.append(
            f"- {run.get('run_id', 'unknown')}: "
            f"${run.get('total_cost_usd', 0):.4f} "
            f"({run.get('wall_clock_minutes', 0):.1f} min)"
        )
    out = Path(output)
    out.write_text("\n".join(lines))
    return out
