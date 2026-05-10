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


@app.command("repro")
def repro(
    scenario: str = typer.Argument(..., help="arc | gsm8k | humaneval | clinc150 | conll2003"),
    max_iter: int = typer.Option(10, "--max-iter"),
    tier: str = typer.Option("edge", "--tier"),
    base_model: Optional[str] = typer.Option(None, "--base-model"),
    out: Optional[str] = typer.Option(None, "--out"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Run a paper-replication scenario from `bench/repro/`."""
    import importlib
    from bench.repro import SCENARIOS, SCENARIO_MODULES
    if scenario not in SCENARIOS:
        typer.echo(f"unknown scenario '{scenario}'. choose from: {SCENARIOS}", err=True)
        raise typer.Exit(2)
    mod = importlib.import_module(SCENARIO_MODULES[scenario])
    out_dict = mod.run(max_iter=max_iter, tier=tier, base_model=base_model,
                      out_dir=out, dry_run=dry_run)
    typer.echo(json.dumps(out_dict, indent=2, default=str))


@app.command("dpo")
def dpo(
    pairs: str = typer.Argument(..., help="JSONL with {input, output, judge_score}"),
    base_model: str = typer.Option(...),
    out: str = typer.Option("./runs/dpo"),
    tier: str = typer.Option("mid"),
):
    """Run DPO preference fine-tuning on a labeled-pairs JSONL."""
    from .train.dpo import train_dpo
    from .search.pipeline import HyperParams, LearningStrategy
    from .data.curate import Example
    cfg = AutoSLMConfig(hardware_tier=tier)
    rows = [json.loads(l) for l in Path(pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    examples = [{"input": r["input"], "output": r["output"],
                "judge_score": r.get("judge_score", 0.5)} for r in rows]
    t = cfg.tier()
    H = HyperParams(base_model=base_model, lora_rank=t.lora_rank,
                    max_seq_len=t.max_seq_len, quant=t.quant, bf16=t.bf16,
                    grad_checkpoint=t.grad_checkpoint, distributed=t.distributed)
    S = LearningStrategy(supervision="direct", eval_method="exact_match", objective="dpo")
    result = train_dpo(examples, H, S, out)
    typer.echo(json.dumps({"model_id": result.model_id, "checkpoint": result.checkpoint_path,
                          "error": result.error}, indent=2, default=str))


@app.command("kto")
def kto(
    pairs: str = typer.Argument(..., help="JSONL with {input, output, judge_score}"),
    base_model: str = typer.Option(...),
    out: str = typer.Option("./runs/kto"),
    tier: str = typer.Option("mid"),
):
    """Run KTO preference fine-tuning on a labeled-pairs JSONL."""
    from .train.dpo import train_kto
    from .search.pipeline import HyperParams, LearningStrategy
    cfg = AutoSLMConfig(hardware_tier=tier)
    rows = [json.loads(l) for l in Path(pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    examples = [{"input": r["input"], "output": r["output"],
                "judge_score": r.get("judge_score", 0.5)} for r in rows]
    t = cfg.tier()
    H = HyperParams(base_model=base_model, lora_rank=t.lora_rank,
                    max_seq_len=t.max_seq_len, quant=t.quant, bf16=t.bf16,
                    grad_checkpoint=t.grad_checkpoint, distributed=t.distributed)
    S = LearningStrategy(supervision="direct", eval_method="exact_match", objective="kto")
    result = train_kto(examples, H, S, out)
    typer.echo(json.dumps({"model_id": result.model_id, "checkpoint": result.checkpoint_path,
                          "error": result.error}, indent=2, default=str))


@app.command("smoke-e2e")
def smoke_e2e(
    tier: str = typer.Option("edge"),
    base_model: str = typer.Option("HuggingFaceTB/SmolLM2-360M-Instruct"),
    out: Optional[str] = typer.Option(None),
):
    """End-to-end smoke: cold-start init -> 1 MCGS branch -> SFT -> probe -> ratchet."""
    from bench.smoke_e2e import run_smoke
    out_dict = run_smoke(tier=tier, base_model=base_model, out_dir=out)
    typer.echo(json.dumps(out_dict, indent=2, default=str))


@app.command("tiers")
def list_tiers():
    """Print available hardware tier presets."""
    for name, t in TIERS.items():
        typer.echo(f"{name}: {t}")


if __name__ == "__main__":
    app()
