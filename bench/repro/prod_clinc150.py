"""CLINC150 production-mode replication (paper Table 8; target 99.3% intent acc).

Bootstraps a deployed-model trace set from the CLINC150 benchmark, then runs
production-mode adaptation against those traces. For a real paper-replication
run, point at your own production trace store via env var AUTOSLM_DB_PATH.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autoslm.modes.production import run_production
from autoslm.traces import TraceStore, TraceRecord
from bench.repro._common import (
    announce, dry_run_print, make_cfg, stamp, write_result,
)

SCENARIO = "clinc150"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_TARGET = 0.993


def _bootstrap_traces(store: TraceStore, deployed_id: str, base_model: str,
                     n: int = 200, fail_rate: float = 0.30,
                     seed: int = 42) -> int:
    """Create a synthetic deployment trace set from CLINC150 (or fall back to
    a minimal hand-rolled intent set when HF datasets is unavailable)."""
    rng = random.Random(seed)
    rows: list[TraceRecord] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("clinc_oos", "small", split="validation",
                         streaming=True, trust_remote_code=True)
        for i, ex in enumerate(ds):
            if i >= n:
                break
            text = ex.get("text") or ex.get("input") or ""
            label_id = ex.get("intent")
            gold = str(label_id)
            wrong = str((label_id + 1) % 150)
            failed = rng.random() < fail_rate
            rows.append(TraceRecord(
                id=str(uuid.uuid4()),
                input=text, prediction=wrong if failed else gold,
                gold=gold, verdict="fail" if failed else "pass",
                model_id=deployed_id, task="intent_classification",
                judge_model="exact_match",
                judge_score=0.0 if failed else 1.0,
                metadata={"label": gold, "source": "clinc_oos/validation"},
            ))
    except Exception:
        # offline fallback: tiny synthetic intent set
        intents = ["weather", "alarm", "music", "calendar"]
        templates = {
            "weather": "what's the weather in {city}",
            "alarm": "set an alarm for {time}",
            "music": "play {song}",
            "calendar": "what's on my calendar {day}",
        }
        cities = ["paris", "tokyo", "berlin"]
        times = ["7am", "noon", "9pm"]
        songs = ["wonderwall", "thunderstruck"]
        days = ["today", "tomorrow"]
        for i in range(n):
            label = rng.choice(intents)
            txt = templates[label].format(city=rng.choice(cities),
                                         time=rng.choice(times),
                                         song=rng.choice(songs),
                                         day=rng.choice(days))
            failed = rng.random() < fail_rate
            wrong = rng.choice([l for l in intents if l != label])
            rows.append(TraceRecord(
                id=str(uuid.uuid4()),
                input=txt, prediction=wrong if failed else label,
                gold=label, verdict="fail" if failed else "pass",
                model_id=deployed_id, task="intent_classification",
                judge_model="exact_match",
                judge_score=0.0 if failed else 1.0,
                metadata={"label": label, "source": "synthetic"},
            ))
    n_inserted = store.insert_many(rows)
    store.record_lineage(deployed_id, base_model, None, None, None)
    return n_inserted


def run(max_iter: int = 5, tier: str = "edge",
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

    deployed_id = f"clinc150-deploy-{stamp()}"
    store = TraceStore(cfg.trace_db_path, backend=cfg.trace_db)
    n = _bootstrap_traces(store, deployed_id, base_model, n=200)
    print(f"[{SCENARIO}] bootstrapped {n} traces under deployed_model_id={deployed_id}")

    result = run_production(
        cfg=cfg,
        deployed_model_id=deployed_id,
        base_model=base_model,
        task="intent_classification",
        max_iterations=max_iter,
        eval_method="exact_match",
        enable_probes=False,            # base model, no adapter to probe
    )
    p = write_result(out_dir, SCENARIO, result, target)
    print(f"\nresult: {p}\nfinal_score: {result.get('best_score')} target: {target}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--tier", default="edge", choices=["edge", "mid", "big"])
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(max_iter=args.max_iter, tier=args.tier, base_model=args.base_model,
        out_dir=args.out, dry_run=args.dry_run, target=args.target)


if __name__ == "__main__":
    main()
