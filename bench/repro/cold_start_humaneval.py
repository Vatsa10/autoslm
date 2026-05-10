"""HumanEval cold-start replication (paper Table 2; target 92.7% pass@1 on Qwen3-8B)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autoslm.modes.cold_start import run_cold_start
from bench.repro._common import (
    announce, dry_run_print, make_cfg, stamp, write_result,
)

SCENARIO = "humaneval"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TARGET = 0.927


def run(max_iter: int = 10, tier: str = "mid",
        base_model: Optional[str] = None,
        out_dir: str | Path = None, dry_run: bool = False,
        target: float = DEFAULT_TARGET) -> dict:
    base_model = base_model or DEFAULT_BASE_MODEL
    out_dir = Path(out_dir or f"runs/repro/{SCENARIO}/{stamp()}")
    if dry_run:
        return dry_run_print(SCENARIO, base_model=base_model, tier=tier,
                            max_iter=max_iter, target=target,
                            out_dir=str(out_dir))
    cfg = make_cfg(tier, out_dir)
    announce(SCENARIO, base_model, target, max_iter, tier, out_dir)
    result = run_cold_start(
        cfg=cfg,
        task_spec="HumanEval: Python function completion. Generate correct function body given signature + docstring.",
        base_model=base_model,
        dataset_hint="openai/openai_humaneval",
        target_threshold=target,
        max_iterations=max_iter,
        run_id=f"humaneval-{stamp()}",
    )
    p = write_result(out_dir, SCENARIO, result, target)
    print(f"\nresult: {p}\nfinal_score: {result.get('best_score')} target: {target}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--tier", default="mid", choices=["edge", "mid", "big"])
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(max_iter=args.max_iter, tier=args.tier, base_model=args.base_model,
        out_dir=args.out, dry_run=args.dry_run, target=args.target)


if __name__ == "__main__":
    main()
