"""Cold-start mode (paper Section 2.5).

Stages:
  1. classify_task  — task type, eval method, supervision format
  2. acquire_dataset — HF benchmark or teacher-LLM seed
  3. baseline_survey — set τ
  4. build_holdout   — E_pos ∪ E_neg ∪ E_boundary (held-out, never trained on)
  5. curriculum     — D_gold + D_hard, no replay (D_parent = ∅)
  6. train + iterate via MCGS, unconstrained (no regression gate, paper Eq. 6)

Budget: up to 1500 LangGraph turns / iterations (paper §2.5).
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..config import AutoSLMConfig
from ..llm import LLMClient
from ..data.acquire import (
    classify_task, acquire_dataset, baseline_survey, TaskClassification,
)
from ..data.curate import Example, curate_dataset
from ..data.hard_negatives import generate_hard_negatives, nn_label_swap_hard_negatives
from ..data.cot_annotate import annotate_cot, teacher_for_task
from ..eval.build_holdout import build_holdout
from ..eval.harness import score_pipeline
from ..search.pipeline import Pipeline, DatasetSpec, HyperParams, LearningStrategy
from ..search.mcgs import MCGS, MCGSConfig, NodeResult
from ..search.expand import llm_expander, llm_fuser
from ..train.dispatch import train_pipeline
from ..train.inference import load_for_inference, generate_batch
from ..audit import AuditLog


def _persist_examples(examples: list[Example], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "input": ex.input, "output": ex.output, "label": ex.label,
                "is_hard_negative": ex.is_hard_negative,
                "is_replay": ex.is_replay,
                "metadata": ex.metadata,
            }, default=str) + "\n")


def run_cold_start(
    cfg: AutoSLMConfig,
    task_spec: str,
    base_model: Optional[str] = None,
    dataset_hint: Optional[str] = None,
    target_threshold: Optional[float] = None,
    max_iterations: int = 30,
    use_mcgs: bool = True,
    run_id: Optional[str] = None,
    n_train: int = 1500,
    k_pos: int = 200,
    k_neg: int = 50,
    k_boundary: int = 50,
) -> dict:
    """Run cold-start adaptation per paper §2.5."""
    cfg.ensure_dirs()
    base_model = base_model or cfg.base_model_default
    rid = run_id or f"coldstart-{abs(hash(task_spec)) & 0xffffff:x}"
    audit_dir = cfg.workdir / "runs" / rid
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(audit_dir / "data-curation.md", run_id=rid, mode="cold-start")
    audit.section("Task spec",
                  f"- spec: {task_spec}\n- base_model: `{base_model}`\n"
                  f"- dataset_hint: `{dataset_hint}`")

    orchestrator = LLMClient(model=cfg.orchestrator_model,
                            thinking_budget=cfg.thinking_budget)
    classifier_client = LLMClient(model=cfg.judge_model)

    # 1. classify task
    classification = classify_task(task_spec, classifier_client)
    audit.section("Task classification",
                  "\n".join(f"- **{k}**: `{v}`" for k, v in asdict(classification).items()))

    # 2. acquire dataset
    teacher_model = teacher_for_task(classification.task_type, cfg)
    teacher = LLMClient(model=teacher_model)
    dataset_examples = acquire_dataset(task_spec, teacher, classification, hint=dataset_hint)
    if not dataset_examples:
        return {"status": "no_dataset", "classification": asdict(classification)}
    audit.section("Dataset acquired",
                  f"- n: {len(dataset_examples)}\n"
                  f"- source: `{dataset_examples[0].metadata.get('source')}`")

    # 3. baseline survey -> τ
    survey = baseline_survey(task_spec, base_model, classifier_client)
    tau = float(target_threshold or survey.target_threshold)
    audit.section("Baseline survey",
                  f"- published_sota: {survey.published_sota}\n"
                  f"- target_threshold (τ): {tau}\n- notes: {survey.notes[:300]}")

    # 4. build held-out E (BEFORE training)
    eval_set, train_pool = build_holdout(
        dataset_examples,
        task_type=classification.task_type,
        client=teacher,
        labels=classification.labels,
        k_pos=k_pos, k_neg=k_neg, k_boundary=k_boundary,
    )
    audit.section("Held-out E (paper Eq. 7)",
                  f"- E_pos: {len(eval_set.pos)}\n"
                  f"- E_neg: {len(eval_set.neg)}\n"
                  f"- E_boundary: {len(eval_set.boundary)}")

    # 5. curriculum: D_gold + D_hard (no replay; D_parent = ∅)
    use_cot = classification.supervision == "cot"
    train_pool = train_pool[: max(1, n_train)]
    gold_examples = list(train_pool)
    if use_cot:
        gold_examples = annotate_cot(teacher, gold_examples)
    hard_negs = generate_hard_negatives(teacher, gold_examples, max_neg_per_gold=1)
    if not hard_negs:
        hard_negs = nn_label_swap_hard_negatives(gold_examples, k=1)
    target_inputs = [ex.input for ex in dataset_examples[:200]]
    examples, qreport = curate_dataset(
        gold=gold_examples, hard_negs=hard_negs, replay=None,
        target_inputs=target_inputs, max_examples=n_train,
    )
    audit.section("Curriculum",
                  f"- total: {qreport.final_size}\n"
                  f"- by_label: `{qreport.by_label}`\n"
                  f"- rejected: {qreport.rejected} ({qreport.reasons})")
    train_path = audit_dir / "train.jsonl"
    _persist_examples(examples, train_path)

    # initial pipeline
    tier = cfg.tier()
    H0 = HyperParams(
        base_model=base_model,
        lora_rank=tier.lora_rank,
        max_seq_len=tier.max_seq_len,
        quant=tier.quant, bf16=tier.bf16,
        grad_checkpoint=tier.grad_checkpoint,
        distributed=tier.distributed,
        model_family=classification.model_family,  # type: ignore[arg-type]
        gliner_labels=classification.labels if classification.model_family == "gliner2" else None,
    )
    S0 = LearningStrategy(
        supervision=classification.supervision,  # type: ignore[arg-type]
        teacher_model=teacher_model,
        eval_method=classification.eval_method,  # type: ignore[arg-type]
        judge_model=cfg.judge_model if classification.eval_method == "llm_judge" else None,
    )
    D0 = DatasetSpec(
        name=f"{rid}_v1",
        gold_path=str(train_path),
        gold_ratio=0.65, hard_neg_ratio=0.35, replay_ratio=0.0,
        max_examples=n_train,
    )
    pi0 = Pipeline(D=D0, H=H0, S=S0, notes=f"cold_start_v1 task={classification.task_type}")

    judge = (LLMClient(model=cfg.judge_model)
             if classification.eval_method == "llm_judge" else None)

    def evaluator(pi: Pipeline) -> NodeResult:
        ex_path = Path(pi.D.gold_path) if pi.D.gold_path else train_path
        with ex_path.open(encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        train_ex = [Example(input=r["input"], output=r["output"], label=r.get("label"),
                            is_hard_negative=r.get("is_hard_negative", False),
                            is_replay=r.get("is_replay", False),
                            metadata=r.get("metadata", {})) for r in rows]
        run_dir = audit_dir / f"iter_{pi.iteration}_{pi.fingerprint()}"
        train_result = train_pipeline(train_ex, pi.H, pi.S, run_dir,
                                     task=classification.task_type)
        if train_result.error:
            audit.section(f"Iter {pi.iteration} TRAIN-FAIL",
                          f"`{pi.fingerprint()}` error: `{train_result.error}`")
            return NodeResult(score=0.0, regressions=0, failed=True,
                             error=train_result.error)

        if pi.H.model_family == "gliner2":
            from ..eval.gliner_eval import gliner_predict
            labels = pi.H.gliner_labels or sorted({ex.label for ex in train_ex if ex.label})
            gtask = "ner" if classification.task_type in {"ner", "extraction"} else "classification"

            def predict(prompts: list[str]) -> list[str]:
                preds = gliner_predict(train_result.checkpoint_path, prompts,
                                      list(labels), task=gtask)
                if gtask == "ner":
                    return [json.dumps(p, default=str) for p in preds]
                return [str(p) for p in preds]
            model = tok = None
        else:
            model, tok = load_for_inference(train_result.checkpoint_path,
                                           base_model=pi.H.base_model, quant=pi.H.quant)

            def predict(prompts: list[str]) -> list[str]:
                return generate_batch(model, tok, prompts, system=pi.H.system_prompt)

        a, _r, full = score_pipeline(eval_set, regression_set=None,
                                     predictor=predict, strategy=pi.S,
                                     judge_client=judge)
        if model is not None:
            del model, tok
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        audit.section(f"Iter {pi.iteration}",
                      f"`{pi.fingerprint()}` score={a:.4f} slice={full.per_slice}\n"
                      f"notes: {pi.notes}")
        return NodeResult(score=a, regressions=0, metrics=full.per_slice,
                         checkpoint_path=train_result.checkpoint_path)

    if not use_mcgs:
        result = evaluator(pi0)
        return {"status": "single_shot", "score": result.score,
                "checkpoint": result.checkpoint_path,
                "classification": asdict(classification), "tau": tau,
                "audit_log": str(audit_dir / "data-curation.md")}

    mcfg = MCGSConfig(
        score_threshold=tau,
        regression_epsilon=cfg.regression_epsilon,
        enforce_regression=False,    # paper §2.5 Eq. 6: unconstrained
        max_iterations=max_iterations,
    )

    def failure_summary(node):
        if not node.result:
            return ""
        return json.dumps({
            "score": node.result.score, "regressions": node.result.regressions,
            "metrics": node.result.metrics,
            "tau": tau, "task_type": classification.task_type,
        }, default=str)[:6000]

    expander = llm_expander(orchestrator, failure_summary_provider=failure_summary)
    fuser = llm_fuser(orchestrator)
    mcgs = MCGS(mcfg, evaluator, expander, fuser, persist_dir=audit_dir / "graph")
    best = mcgs.run(pi0)
    audit.section("Final",
                  f"- best: `{best.id if best else None}`\n"
                  f"- score: {best.result.score if best and best.result else None}\n"
                  f"- iterations: {mcgs.iteration}")
    return {
        "status": "ok",
        "best_node": best.id,
        "best_score": best.result.score if best.result else None,
        "checkpoint": best.result.checkpoint_path if best.result else None,
        "classification": asdict(classification),
        "tau": tau,
        "iterations": mcgs.iteration,
        "graph_dir": str(audit_dir / "graph"),
        "audit_log": str(audit_dir / "data-curation.md"),
    }
