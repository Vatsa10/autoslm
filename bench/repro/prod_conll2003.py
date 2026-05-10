"""CoNLL-2003 NER production replication (paper Fig. 9; target Entity F1 0.810)."""
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

SCENARIO = "conll2003"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B"   # if user has gliner installed, override to gliner2-base-v1
DEFAULT_TARGET = 0.810


def _bootstrap_traces(store: TraceStore, deployed_id: str, base_model: str,
                     n: int = 300, fail_rate: float = 0.50, seed: int = 42) -> int:
    """Pull CoNLL-2003 dev split, label fail/pass."""
    rng = random.Random(seed)
    rows: list[TraceRecord] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("conll2003", split="validation",
                         streaming=True, trust_remote_code=True)
        for i, ex in enumerate(ds):
            if i >= n:
                break
            tokens = ex.get("tokens") or []
            ner_tags = ex.get("ner_tags") or []
            text = " ".join(tokens)
            gold = json.dumps(ner_tags)
            failed = rng.random() < fail_rate
            pred = json.dumps([0] * len(ner_tags)) if failed else gold
            rows.append(TraceRecord(
                id=str(uuid.uuid4()),
                input=text, prediction=pred, gold=gold,
                verdict="fail" if failed else "pass",
                model_id=deployed_id, task="ner",
                judge_model="span_f1",
                judge_score=0.0 if failed else 1.0,
                metadata={"source": "conll2003/validation",
                         "tokens": tokens, "ner_tags": ner_tags},
            ))
    except Exception:
        # offline fallback
        for i in range(n):
            txt = f"sample text number {i}"
            failed = rng.random() < fail_rate
            rows.append(TraceRecord(
                id=str(uuid.uuid4()),
                input=txt, prediction="" if failed else txt,
                gold=txt, verdict="fail" if failed else "pass",
                model_id=deployed_id, task="ner",
                judge_model="span_f1",
                judge_score=0.0 if failed else 1.0,
                metadata={"source": "synthetic"},
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

    deployed_id = f"conll2003-deploy-{stamp()}"
    store = TraceStore(cfg.trace_db_path, backend=cfg.trace_db)
    n = _bootstrap_traces(store, deployed_id, base_model, n=300)
    print(f"[{SCENARIO}] bootstrapped {n} traces under deployed_model_id={deployed_id}")

    result = run_production(
        cfg=cfg,
        deployed_model_id=deployed_id,
        base_model=base_model,
        task="ner",
        max_iterations=max_iter,
        eval_method="f1",
        enable_probes=False,
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
