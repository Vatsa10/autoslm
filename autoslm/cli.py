"""CLI entrypoints. `autoslm <subcommand>`."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import typer

from .config import AutoSLMConfig, TIERS

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("production")
def production(
    deployed_model_id: str = typer.Argument(...),
    base_model: str = typer.Option("meta-llama/Llama-3.2-3B"),
    tier: str = typer.Option("mid", help="edge|mid|big"),
    iters: int = typer.Option(10),
    use_cot: bool = typer.Option(False),
    eval_method: str = typer.Option("exact_match"),
    task: str = typer.Option("classification"),
    workdir: Optional[str] = typer.Option(None),
    orch_model: Optional[str] = typer.Option(None),
):
    """Run production-mode closed-loop adaptation."""
    from .modes.production import run_production
    cfg = AutoSLMConfig(hardware_tier=tier)
    if workdir:
        cfg.workdir = Path(workdir)
    if orch_model:
        cfg.orchestrator_model = orch_model
    cfg.ensure_dirs()
    out = run_production(
        cfg=cfg, deployed_model_id=deployed_model_id, base_model=base_model,
        task=task, max_iterations=iters, use_cot=use_cot, eval_method=eval_method,
    )
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("cold-start")
def cold_start(
    spec: str = typer.Argument(..., help="Natural-language task spec, e.g. 'fine-tune Llama 3.2-3B on ARC-Challenge'"),
    base_model: str = typer.Option("meta-llama/Llama-3.2-3B"),
    dataset_hint: Optional[str] = typer.Option(None, help="HF dataset id, e.g. 'allenai/ai2_arc'"),
    tier: str = typer.Option("mid"),
    iters: int = typer.Option(20),
    tau: Optional[float] = typer.Option(None, help="Override target threshold"),
    workdir: Optional[str] = typer.Option(None),
    orch_model: Optional[str] = typer.Option(None),
    n_train: int = typer.Option(1500),
):
    """Run cold-start mode: build model from a natural-language task spec."""
    from .modes.cold_start import run_cold_start
    cfg = AutoSLMConfig(hardware_tier=tier)
    if workdir:
        cfg.workdir = Path(workdir)
    if orch_model:
        cfg.orchestrator_model = orch_model
    cfg.ensure_dirs()
    out = run_cold_start(
        cfg=cfg, task_spec=spec, base_model=base_model,
        dataset_hint=dataset_hint, target_threshold=tau,
        max_iterations=iters, n_train=n_train,
    )
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("ingest-traces")
def ingest_traces(
    jsonl_path: str = typer.Argument(...),
    db_path: str = typer.Option(".autoslm/traces.duckdb"),
    backend: str = typer.Option("duckdb"),
):
    """Bulk-load JSONL of trace records into the store."""
    from .traces import TraceStore, TraceRecord
    store = TraceStore(db_path, backend=backend)
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        batch = []
        for line in f:
            r = json.loads(line)
            batch.append(TraceRecord(
                input=r["input"],
                prediction=r.get("prediction"),
                gold=r.get("gold"),
                verdict=r.get("verdict") or ("pass" if r.get("score", 0) >= 0.7 else "fail"),
                model_id=r["model_id"],
                task=r.get("task", "generation"),
                judge_model=r.get("judge_model"),
                judge_reason=r.get("judge_reason"),
                judge_score=r.get("judge_score"),
                metadata=r.get("metadata", {}),
                deployment_stage=r.get("metadata", {}).get("deployment_stage"),
            ))
            if len(batch) >= 1000:
                n += store.insert_many(batch)
                batch = []
        if batch:
            n += store.insert_many(batch)
    typer.echo(f"inserted {n} traces into {db_path}")


@app.command("build-scenario")
def build_scenario(
    name: str = typer.Argument(...),
    examples_jsonl: str = typer.Argument(...),
    base_model: str = typer.Option("meta-llama/Llama-3.2-3B"),
    task: str = typer.Option("classification"),
    out_dir: str = typer.Option("./bench_out"),
    stage_size: int = typer.Option(500),
):
    """Build AdaptFT-Bench synthetic scenario from base examples."""
    from bench.adaptft.build_scenario import build_synthetic_scenario
    rows = []
    with open(examples_jsonl, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                rows.append(json.loads(l))
    sc = build_synthetic_scenario(
        name=name, base_examples=rows, base_model=base_model,
        task=task, out_dir=out_dir, stage_size=stage_size,
    )
    typer.echo(f"built scenario {name} with {len(sc.stages)} stages -> {out_dir}")


@app.command("orchestrate")
def orchestrate(
    request: str = typer.Argument(...),
    deployed_model_id: Optional[str] = typer.Option(None),
    base_model: Optional[str] = typer.Option(None),
    tier: str = typer.Option("mid"),
    max_turns: int = typer.Option(50),
):
    """Free-form orchestrator: LLM drives the loop with tool use."""
    from .orchestrator import run_orchestrator
    cfg = AutoSLMConfig(hardware_tier=tier)
    cfg.ensure_dirs()
    convo = run_orchestrator(cfg, request, deployed_model_id, base_model, max_turns)
    typer.echo(json.dumps({"turns": len(convo)}, indent=2))


@app.command("tiers")
def list_tiers():
    """Print available hardware tier presets."""
    for name, t in TIERS.items():
        typer.echo(f"{name}: {t}")


if __name__ == "__main__":
    app()
