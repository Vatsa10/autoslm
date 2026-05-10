"""End-to-end smoke harness — exercises the full closed loop on a tiny model.

Stages run in order; each prints `[STAGE] OK|SKIP|FAIL`. Total wall-time
target: < 10 min on CPU, < 2 min on a single consumer GPU.

Default model: HuggingFaceTB/SmolLM2-360M-Instruct.

Usage:
    python -m bench.smoke_e2e            # full
    python -m bench.smoke_e2e --no-train # skip torch/training stages
    autoslm smoke-e2e                    # via CLI
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoslm.config import AutoSLMConfig
from autoslm.search.pipeline import (
    Pipeline, DatasetSpec, HyperParams, LearningStrategy,
)
from autoslm.search.mcgs import MCGS, MCGSConfig, NodeResult
from autoslm.eval.harness import EvalSet, EvalExample, score_pipeline
from autoslm.eval.metrics import exact_match
from autoslm.traces import TraceStore, TraceRecord
from autoslm.audit import AuditLog
from autoslm.telemetry import CostTracker
from autoslm.traces.calibration import (
    update_label_stats, get_label_stats, calibrate,
)


def _stage(label: str):
    print(f"\n=== [{label}] ===", flush=True)


def _ok(label: str, msg: str = ""):
    print(f"[{label}] OK {msg}".rstrip(), flush=True)


def _skip(label: str, why: str):
    print(f"[{label}] SKIP — {why}", flush=True)


def _fail(label: str, why: str):
    print(f"[{label}] FAIL — {why}", flush=True)


def run_smoke(tier: str = "edge",
             base_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
             out_dir: Optional[str] = None,
             skip_train: bool = False) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir or f"runs/smoke/{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AutoSLMConfig(hardware_tier=tier, workdir=out_dir / "autoslm",
                       trace_db_path=out_dir / "autoslm" / "traces.duckdb")
    cfg.ensure_dirs()

    status: dict = {"out_dir": str(out_dir), "tier": tier, "base_model": base_model,
                   "stages": {}}
    audit = AuditLog(out_dir / "data-curation.md", run_id="smoke-e2e", mode="smoke")
    cost = CostTracker(run_id="smoke-e2e", output_dir=out_dir)

    # Stage 1: config + audit log + cost tracker bootstrap
    _stage("init")
    audit.section("Smoke E2E bootstrap",
                 f"- tier: {tier}\n- base_model: `{base_model}`")
    _ok("init")
    status["stages"]["init"] = "ok"

    # Stage 2: trace store + ingest synthetic traces
    _stage("trace_store")
    try:
        store = TraceStore(cfg.trace_db_path)
        deployed_id = f"smoke-deploy-{uuid.uuid4().hex[:6]}"
        rows = []
        for i in range(20):
            verdict = "fail" if i % 2 == 0 else "pass"
            rows.append(TraceRecord(
                id=str(uuid.uuid4()),
                input=f"intent example {i}", prediction="weather",
                gold="alarm" if verdict == "fail" else "weather",
                verdict=verdict, model_id=deployed_id,
                task="intent_classification",
                judge_model="exact_match",
                judge_score=0.0 if verdict == "fail" else 1.0,
                metadata={"label": "alarm" if verdict == "fail" else "weather"},
            ))
        n = store.insert_many(rows)
        store.record_lineage(deployed_id, base_model, None, None, None)
        _ok("trace_store", f"({n} rows)")
        status["stages"]["trace_store"] = "ok"
        status["deployed_model_id"] = deployed_id
    except Exception as e:
        _fail("trace_store", str(e))
        status["stages"]["trace_store"] = f"fail: {e}"
        return status

    # Stage 3: calibration round-trip
    _stage("calibration")
    try:
        conn = store._connect()
        update_label_stats(conn, deployed_id, "weather", correct=True)
        update_label_stats(conn, deployed_id, "weather", correct=False)
        update_label_stats(conn, deployed_id, "alarm", correct=True)
        s = get_label_stats(conn, deployed_id, "weather")
        c = calibrate(raw_conf=0.9, label="weather", stats=s, weight=0.5)
        _ok("calibration", f"weather_acc={s.accuracy:.2f} calibrated_conf={c:.2f}")
        status["stages"]["calibration"] = "ok"
    except Exception as e:
        _fail("calibration", str(e))
        status["stages"]["calibration"] = f"fail: {e}"

    # Stage 4: MCGS engine on a stub evaluator (no torch needed)
    _stage("mcgs_stub")
    try:
        D = DatasetSpec(name="smoke")
        H = HyperParams(base_model=base_model, lora_rank=cfg.tier().lora_rank,
                       quant=cfg.tier().quant)
        S = LearningStrategy(eval_method="exact_match")
        pi = Pipeline(D=D, H=H, S=S, notes="root")

        def stub_eval(p: Pipeline) -> NodeResult:
            return NodeResult(score=0.5 + len(p.notes) * 0.01, regressions=0)

        def stub_expand(parent, m: MCGS) -> Pipeline:
            import copy
            child = copy.deepcopy(parent.pipeline)
            child.notes = parent.pipeline.notes + "x"
            return child

        mcgs = MCGS(MCGSConfig(score_threshold=0.95, max_iterations=5),
                   stub_eval, stub_expand)
        best = mcgs.run(pi)
        _ok("mcgs_stub", f"iters={mcgs.iteration} best={best.result.score:.3f}")
        status["stages"]["mcgs_stub"] = "ok"
    except Exception as e:
        _fail("mcgs_stub", str(e))
        status["stages"]["mcgs_stub"] = f"fail: {e}"

    # Stage 5: ratchet eval (no torch)
    _stage("ratchet")
    try:
        prior = EvalSet(pos=[EvalExample(input="q1", gold="a"),
                            EvalExample(input="q2", gold="b")])
        cur = EvalSet(pos=[EvalExample(input="q3", gold="c")])
        cur.prior = prior

        def predict(prompts):
            ans = {"q1": "a", "q2": "b", "q3": "c"}
            return [ans.get(p, "X") for p in prompts]

        a, r, _ = score_pipeline(cur, [], predict,
                                LearningStrategy(eval_method="exact_match"))
        assert a == 1.0 and r == 0
        _ok("ratchet", f"score={a} regressions={r}")
        status["stages"]["ratchet"] = "ok"
    except Exception as e:
        _fail("ratchet", str(e))
        status["stages"]["ratchet"] = f"fail: {e}"

    # Stage 6: cost telemetry round-trip
    _stage("cost")
    try:
        cost.log_llm_tokens("orchestrator", input_tokens=1000, output_tokens=500)
        cost.log_training_time(2.0)  # minutes
        cost.save()
        d = cost._to_dict()
        _ok("cost", f"total_usd={d.get('total_cost_usd', 0):.4f}")
        status["stages"]["cost"] = "ok"
    except Exception as e:
        _fail("cost", str(e))
        status["stages"]["cost"] = f"fail: {e}"

    # Stage 7: live training (only if torch + ml stack available and not skipped)
    _stage("train")
    if skip_train:
        _skip("train", "--no-train flag")
        status["stages"]["train"] = "skip"
    elif importlib.util.find_spec("torch") is None:
        _skip("train", "torch not installed (run `uv pip install -e .[ml]`)")
        status["stages"]["train"] = "skip"
    else:
        try:
            import torch as _torch
            from autoslm.train.dispatch import train_pipeline
            from autoslm.data.curate import Example
            train_ex = [Example(input=f"x{i}", output=f"y{i}", label="a")
                       for i in range(8)]
            run_dir = out_dir / "lora"
            # Force quant=none on CPU or when bitsandbytes missing.
            import copy
            H_train = copy.deepcopy(H)
            cuda = _torch.cuda.is_available()
            bnb_ok = importlib.util.find_spec("bitsandbytes") is not None
            if not cuda:
                H_train.quant = "none"
                H_train.bf16 = False
            elif H_train.quant in {"4bit", "8bit"} and not bnb_ok:
                H_train.quant = "none"   # bnb not installed; run unquantized
            # tiny smoke loop
            H_train.epochs = 1
            H_train.batch_size = 2
            H_train.grad_accum = 1
            res = train_pipeline(train_ex, H_train, S, run_dir, task="generation")
            if res.error:
                _fail("train", res.error[:200])
                status["stages"]["train"] = f"fail: {res.error[:200]}"
            else:
                _ok("train", f"checkpoint={res.checkpoint_path}")
                status["stages"]["train"] = "ok"
                status["checkpoint_path"] = res.checkpoint_path
        except Exception as e:
            _fail("train", str(e)[:200])
            status["stages"]["train"] = f"fail: {e}"

    # final
    elapsed = time.time() - t0
    status["elapsed_sec"] = round(elapsed, 1)
    print(f"\n=== TOTAL {elapsed:.1f}s ===")
    print(json.dumps({k: v for k, v in status["stages"].items()}, indent=2))
    (out_dir / "smoke_status.json").write_text(
        json.dumps(status, indent=2, default=str), encoding="utf-8")
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="edge", choices=["edge", "mid", "big"])
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-train", action="store_true")
    args = ap.parse_args()
    run_smoke(tier=args.tier, base_model=args.base_model, out_dir=args.out,
             skip_train=args.no_train)


if __name__ == "__main__":
    main()
