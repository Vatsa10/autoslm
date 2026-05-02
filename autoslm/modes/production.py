"""Production mode (paper Section 2.6): primary contribution.

Pipeline:
  1. trace ingestion (T_fail / T_pass)
  2. taxonomy construction (clusters + fixability)
  3. live confirmation (probe deployed model)
  4. parent model awareness (lineage -> D_parent, replay buffer)
  5. curriculum synthesis (D_post = D_gold + D_hard + D_replay)
  6. train on D_post; eval on T_fail and regression set; gate on r(pi) <= eps
  7. cross-checkpoint regression gate (Section 2.7 ratchet)
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..config import AutoSLMConfig
from ..llm import LLMClient
from ..traces import TraceStore
from ..traces.taxonomy import build_taxonomy, Cluster
from ..data.curate import Example, curate_dataset
from ..data.hard_negatives import generate_hard_negatives, nn_label_swap_hard_negatives
from ..data.replay import build_replay_buffer
from ..data.cot_annotate import annotate_cot, teacher_for_task
from ..search.pipeline import Pipeline, DatasetSpec, HyperParams, LearningStrategy
from ..search.mcgs import MCGS, MCGSConfig, NodeResult
from ..search.expand import llm_expander, llm_fuser
from ..train.lora_sft import train_lora_sft
from ..train.inference import load_for_inference, generate_batch
from ..eval.harness import EvalSet, EvalExample, score_pipeline


def _failures_to_eval(records: list[dict]) -> list[EvalExample]:
    """Failure traces -> eval examples (paper: T_fail evaluated with same judgment)."""
    out = []
    for r in records:
        if not r.get("gold"):
            continue
        out.append(EvalExample(
            input=r["input"], gold=r["gold"], slice="pos",
            metadata={"trace_id": r.get("id"), "judge_model": r.get("judge_model")},
        ))
    return out


def _passes_to_regression(records: list[dict], cap: int = 200) -> list[EvalExample]:
    out = []
    for r in records[:cap]:
        if not r.get("gold"):
            continue
        out.append(EvalExample(input=r["input"], gold=r["gold"], slice="pos",
                              metadata={"trace_id": r.get("id"), "regression": True}))
    return out


def _build_corrective_curriculum(
    cfg: AutoSLMConfig,
    teacher: LLMClient,
    fixable_clusters: list[Cluster],
    fail_records: list[dict],
    pass_records: list[dict],
    parent_dataset_path: Optional[str],
    use_cot: bool,
    target_inputs: list[str],
    max_examples: int = 1500,
):
    """Section 2.6 'Curriculum Synthesis' -> D_post = D_gold + D_hard + D_replay."""
    # gold = corrected versions of failed traces in fixable clusters
    fixable_ids = set()
    for c in fixable_clusters:
        fixable_ids.update(c.representative_ids or [])
        fixable_ids.update(r.get("id") for r in c.representative_examples)
    by_id = {r.get("id"): r for r in fail_records}
    gold_records = [by_id[i] for i in fixable_ids if i and i in by_id and by_id[i].get("gold")]
    if not gold_records:
        gold_records = [r for r in fail_records if r.get("gold")][:max_examples // 2]

    gold_examples = [
        Example(input=r["input"], output=r["gold"], label=r.get("metadata", {}).get("label"))
        for r in gold_records
    ]
    if use_cot:
        gold_examples = annotate_cot(teacher, gold_examples)

    # hard negatives: confusion pairs derived from M0 mistakes
    hard_negs = generate_hard_negatives(teacher, gold_examples, max_neg_per_gold=1)
    if not hard_negs:
        hard_negs = nn_label_swap_hard_negatives(gold_examples, k=1)

    # replay buffer if parent fine-tuned exists
    replay = build_replay_buffer(parent_dataset_path, ratio=0.15) if parent_dataset_path else []

    examples, qreport = curate_dataset(
        gold=gold_examples, hard_negs=hard_negs, replay=replay,
        target_inputs=target_inputs, max_examples=max_examples,
    )
    return examples, qreport


def run_production(
    cfg: AutoSLMConfig,
    deployed_model_id: str,
    base_model: str,
    task: str = "classification",
    use_mcgs: bool = True,
    max_iterations: int = 20,
    use_cot: bool = False,
    eval_method: str = "exact_match",
    judge_criteria: Optional[str] = None,
) -> dict:
    """Closed-loop production-mode run. Returns final pipeline + best checkpoint."""
    cfg.ensure_dirs()
    store = TraceStore(cfg.trace_db_path, backend=cfg.trace_db)
    fail_records = store.fail_set(deployed_model_id)
    pass_records = store.pass_set(deployed_model_id)
    if not fail_records:
        return {"status": "no_failures", "model_id": deployed_model_id}

    # parent lineage -> D_parent
    lineage = store.lineage(deployed_model_id)
    parent_dataset_path = (lineage or {}).get("dataset_path")

    # taxonomy
    orchestrator = LLMClient(model=cfg.orchestrator_model,
                            thinking_budget=cfg.thinking_budget)
    taxonomy_client = LLMClient(model=cfg.judge_model)
    clusters = build_taxonomy(taxonomy_client, fail_records)
    fixable = [c for c in clusters if c.fixability == "fixable"]
    poison_count = sum(c.size for c in clusters if c.fixability == "poison")

    # save taxonomy + audit trail
    audit_dir = cfg.workdir / "runs" / deployed_model_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "taxonomy.json").write_text(json.dumps(
        [asdict(c) for c in clusters], indent=2, default=str), encoding="utf-8")

    # eval + regression sets
    eval_examples = _failures_to_eval(fail_records)
    regression_examples = _passes_to_regression(pass_records)
    eval_set = EvalSet(pos=eval_examples)

    # teacher selection per paper Section 2.5
    teacher_model = teacher_for_task(task, cfg)
    teacher = LLMClient(model=teacher_model)
    judge = LLMClient(model=cfg.judge_model) if eval_method == "llm_judge" else None

    # build initial dataset
    target_inputs = [r["input"] for r in fail_records]
    examples, qreport = _build_corrective_curriculum(
        cfg, teacher, fixable, fail_records, pass_records,
        parent_dataset_path, use_cot, target_inputs,
    )
    if not examples:
        return {"status": "no_curriculum", "taxonomy": [asdict(c) for c in clusters]}

    # persist curated train data
    train_path = audit_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"input": ex.input, "output": ex.output,
                                "label": ex.label,
                                "is_hard_negative": ex.is_hard_negative,
                                "is_replay": ex.is_replay,
                                "metadata": ex.metadata}, default=str) + "\n")

    # initial pipeline
    tier = cfg.tier()
    H0 = HyperParams(
        base_model=base_model, lora_rank=tier.lora_rank,
        max_seq_len=tier.max_seq_len, quant=tier.quant, bf16=tier.bf16,
        grad_checkpoint=tier.grad_checkpoint,
    )
    S0 = LearningStrategy(
        supervision="cot" if use_cot else "direct",
        teacher_model=teacher_model,
        eval_method=eval_method,
        judge_model=cfg.judge_model if eval_method == "llm_judge" else None,
        judge_criteria=judge_criteria,
    )
    D0 = DatasetSpec(
        name=f"{deployed_model_id}_v1",
        gold_path=str(train_path),
        gold_ratio=0.55, hard_neg_ratio=0.30,
        replay_ratio=0.15 if parent_dataset_path else 0.0,
    )
    pi0 = Pipeline(D=D0, H=H0, S=S0,
                   notes=f"production_v1 fixable={len(fixable)} poison={poison_count}")

    # evaluator: train -> infer -> score
    def evaluator(pi: Pipeline) -> NodeResult:
        # load examples (dataset spec name maps to file in this minimal impl)
        ex_path = Path(pi.D.gold_path) if pi.D.gold_path else train_path
        with ex_path.open(encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        train_ex = [Example(input=r["input"], output=r["output"], label=r.get("label"),
                            is_hard_negative=r.get("is_hard_negative", False),
                            is_replay=r.get("is_replay", False),
                            metadata=r.get("metadata", {})) for r in rows]
        run_dir = audit_dir / f"iter_{pi.iteration}_{pi.fingerprint()}"
        train_result = train_lora_sft(train_ex, pi.H, pi.S, run_dir)
        if train_result.error:
            return NodeResult(score=0.0, regressions=999, failed=True, error=train_result.error)

        model, tok = load_for_inference(train_result.checkpoint_path,
                                       base_model=pi.H.base_model, quant=pi.H.quant)

        def predict(prompts: list[str]) -> list[str]:
            return generate_batch(model, tok, prompts, system=pi.H.system_prompt)

        a, r, full = score_pipeline(eval_set, regression_examples, predict, pi.S, judge)
        # detach model
        del model, tok
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return NodeResult(
            score=a, regressions=r, metrics=full.per_slice,
            checkpoint_path=train_result.checkpoint_path,
            eval_artifacts_path=str(run_dir / "eval.json"),
        )

    if not use_mcgs:
        n = MCGS(MCGSConfig(max_iterations=0,
                            score_threshold=cfg.score_threshold,
                            regression_epsilon=cfg.regression_epsilon),
                evaluator, expander=lambda *a, **k: pi0).add_node(pi0)
        # single shot
        result = evaluator(pi0)
        return {"status": "single_shot", "score": result.score,
                "regressions": result.regressions,
                "checkpoint": result.checkpoint_path}

    # MCGS run
    mcfg = MCGSConfig(
        score_threshold=cfg.score_threshold,
        regression_epsilon=cfg.regression_epsilon,
        enforce_regression=True,
        max_iterations=max_iterations,
    )

    def failure_summary(node):
        if not node.result:
            return ""
        return json.dumps({
            "score": node.result.score, "regressions": node.result.regressions,
            "metrics": node.result.metrics,
            "fixable_clusters": [
                {"label": c.label, "size": c.size, "description": c.description[:300]}
                for c in fixable
            ],
        }, default=str)[:6000]

    expander = llm_expander(orchestrator, failure_summary_provider=failure_summary)
    fuser = llm_fuser(orchestrator)
    mcgs = MCGS(mcfg, evaluator, expander, fuser, persist_dir=audit_dir / "graph")
    best = mcgs.run(pi0)
    return {
        "status": "ok", "best_node": best.id,
        "best_score": best.result.score if best.result else None,
        "best_regressions": best.result.regressions if best.result else None,
        "checkpoint": best.result.checkpoint_path if best.result else None,
        "iterations": mcgs.iteration,
        "graph_dir": str(audit_dir / "graph"),
        "taxonomy_path": str(audit_dir / "taxonomy.json"),
    }
