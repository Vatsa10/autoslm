"""End-to-end demo: build a tiny AdaptFT-Bench scenario from CLINC150-style data,
ingest as inference logs with seeded failures, then run production mode.

Run:
    python examples/run_production.py --tier edge

This is a smoke test. For real runs use a GPU machine + larger base model.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from autoslm.config import AutoSLMConfig
from autoslm.traces import TraceStore, TraceRecord
from autoslm.modes.production import run_production
from bench.adaptft.build_scenario import build_synthetic_scenario


def _toy_intent_examples(n: int = 600) -> list[dict]:
    """Tiny synthetic intent-classification corpus for smoke-testing the loop."""
    rng = random.Random(0)
    templates = {
        "weather": ["what's the weather in {city}",
                    "is it raining in {city} today",
                    "tell me the forecast for {city}"],
        "alarm":   ["set an alarm for {time}",
                    "wake me up at {time}",
                    "remind me at {time}"],
        "music":   ["play {song}",
                    "put on {song}",
                    "stream {song}"],
        "calendar":["what's on my calendar {day}",
                    "do i have meetings {day}",
                    "schedule {day}"],
    }
    cities = ["paris", "tokyo", "berlin", "lima", "lagos"]
    times  = ["7am", "noon", "9pm", "5:30am"]
    songs  = ["bohemian rhapsody", "thunderstruck", "wonderwall"]
    days   = ["today", "tomorrow", "monday"]

    out: list[dict] = []
    for _ in range(n):
        label = rng.choice(list(templates))
        tpl = rng.choice(templates[label])
        text = tpl.format(
            city=rng.choice(cities), time=rng.choice(times),
            song=rng.choice(songs), day=rng.choice(days),
        )
        out.append({"input": text, "gold": label,
                    "metadata": {"label": label}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="edge", choices=["edge", "mid", "big"])
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--workdir", default="./.autoslm_demo")
    args = ap.parse_args()

    cfg = AutoSLMConfig(hardware_tier=args.tier, workdir=Path(args.workdir),
                       trace_db_path=Path(args.workdir) / "traces.duckdb")
    cfg.ensure_dirs()

    print("[1/4] building synthetic AdaptFT-Bench scenario")
    scenario_dir = Path(args.workdir) / "scenario"
    sc = build_synthetic_scenario(
        name="toy_intent",
        base_examples=_toy_intent_examples(600),
        base_model=args.base_model,
        task="classification",
        out_dir=scenario_dir,
        stage_size=150,
        alt_labels=["weather", "alarm", "music", "calendar"],
    )

    print("[2/4] ingesting stage 1+2 traces as production failures")
    store = TraceStore(cfg.trace_db_path)
    deployed_model_id = "demo-deploy-v0"
    store.record_lineage(deployed_model_id, args.base_model, None, None, None)

    records: list[TraceRecord] = []
    for stage in sc.stages[:3]:
        for ex in stage["test"]:
            is_poison = ex.get("metadata", {}).get("is_poison", False)
            verdict = "fail" if (is_poison or random.random() < 0.4) else "pass"
            records.append(TraceRecord(
                input=ex["input"],
                prediction="<unknown>",
                gold=ex.get("gold"),
                verdict=verdict,
                model_id=deployed_model_id,
                task="classification",
                judge_model="rule_based",
                judge_score=0.0 if verdict == "fail" else 1.0,
                metadata=ex.get("metadata", {}),
                deployment_stage=ex.get("metadata", {}).get("deployment_stage"),
            ))
    store.insert_many(records)
    print(f"    inserted {len(records)} traces")

    print("[3/4] running production mode (MCGS, regression-gated)")
    out = run_production(
        cfg=cfg, deployed_model_id=deployed_model_id,
        base_model=args.base_model,
        task="classification",
        use_mcgs=True,
        max_iterations=args.iters,
        eval_method="exact_match",
    )

    print("[4/4] result:")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
