"""Shared helpers for benchmark replication scripts.

Each `bench/repro/<scenario>.py` exports a uniform `run(max_iter, tier, base_model,
out_dir, dry_run) -> dict` so they can be invoked the same way from the CLI
(`autoslm repro <scenario>`) or directly (`python -m bench.repro.<scenario>`).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from autoslm.config import AutoSLMConfig


def make_cfg(tier: str, out_dir: str | Path) -> AutoSLMConfig:
    """Build an AutoSLMConfig with workdir scoped to the run output dir."""
    out = Path(out_dir)
    workdir = out / "autoslm"
    cfg = AutoSLMConfig(hardware_tier=tier, workdir=workdir,
                       trace_db_path=workdir / "traces.duckdb")
    cfg.ensure_dirs()
    return cfg


def stamp(prefix: str = "") -> str:
    return f"{prefix}{time.strftime('%Y%m%d-%H%M%S')}"


def write_result(out_dir: str | Path, scenario: str, result: dict,
                target: float) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_score = result.get("best_score") or result.get("final_score") or 0.0
    payload = {
        "scenario": scenario,
        "target": target,
        "final_score": final_score,
        "passed_target": float(final_score) >= target * 0.95,
        "result": result,
    }
    p = out / "result.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def announce(scenario: str, base_model: str, target: float, max_iter: int,
            tier: str, out_dir: Path) -> None:
    print(f"[{scenario}] base_model={base_model} target={target:.3f} "
          f"max_iter={max_iter} tier={tier} out={out_dir}")


def dry_run_print(scenario: str, **kwargs) -> dict:
    info = {"scenario": scenario, "dry_run": True, **kwargs}
    print(json.dumps(info, indent=2, default=str))
    return info
