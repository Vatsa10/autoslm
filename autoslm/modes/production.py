"""Production mode (paper Section 2.6): primary contribution.

Pipeline:
  1. trace ingestion (T_fail / T_pass)
  2. taxonomy construction (clusters + fixability)
  3. live confirmation (probe deployed model; demote unconfirmed clusters)
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
from ..traces.probes import confirm_all, ProbeResult
from ..data.curate import Example, curate_dataset
from ..data.hard_negatives import generate_hard_negatives, nn_label_swap_hard_negatives
from ..data.replay import build_replay_buffer
from ..data.cot_annotate import annotate_cot, teacher_for_task
from ..search.pipeline import Pipeline, DatasetSpec, HyperParams, LearningStrategy
from ..search.mcgs import MCGS, MCGSConfig, NodeResult
from ..search.expand import llm_expander, llm_fuser
from ..train.lora_sft import train_lora_sft
from ..train.dispatch import train_pipeline
from ..train.inference import load_for_inference, generate_batch
from ..eval.harness import EvalSet, EvalExample, score_pipeline
from ..eval.metrics import exact_match, token_f1
from ..audit import AuditLog


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


def _score_fn_for(eval_method: str):
    if eval_method == "f1":
        return token_f1
    return exact_match


def _make_deployed_predictor(deployed_checkpoint_path: Optional[str],
                            base_model: str, quant: str = "8bit",
                            system_prompt: Optional[str] = None,
                            max_new_tokens: int = 256):
    """Build a predictor against the currently-deployed model. If a fine-tuned
    checkpoint path is provided, load adapter on top of base; else use base only.
    Returns (predictor, teardown) where teardown frees GPU memory."""
    try:
        import torch
        if deployed_checkpoint_path and Path(deployed_checkpoint_path).exists():
            model, tok = load_for_inference(deployed_checkpoint_path,
                                          base_model=base_model, quant=quant)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from ..train.lora_sft import _quant_config
            tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            kwargs = {"trust_remote_code": True}
            bnb = _quant_config(quant)
            if bnb is not None:
                kwargs["quantization_config"] = bnb
            kwargs["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
            model.eval()

        def predict(prompts: list[str]) -> list[str]:
            return generate_batch(model, tok, prompts,
                                 system=system_prompt,
                                 max_new_tokens=max_new_tokens)

        def teardown():
            nonlocal model, tok
            del model, tok
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predict, teardown
    except Exception as e:
        # graceful fallback: predictor returns "" so probing reports cluster unconfirmed
        def predict(prompts: list[str]) -> list[str]:
            return [""] * len(prompts)

        def teardown():
            return None

        predict.__error__ = str(e)  # type: ignore[attr-defined]
        return predict, teardown


def _run_live_confirmation(
    cfg: AutoSLMConfig,
    fixable_clusters: list[Cluster],
    deployed_checkpoint_path: Optional[str],
    base_model: str,
    quant: str,
    eval_method: str,
    audit_dir: Path,
    audit: Optional[AuditLog] = None,
) -> tuple[list[Cluster], list[EvalExample], dict[str, ProbeResult]]:
    """Run probes per fixable cluster. Demote unconfirmed -> external.
    Failing probes become extra gold examples (paper §2.6 step 3)."""
    if not fixable_clusters:
        return [], [], {}
    probe_client = LLMClient(model=cfg.judge_model)
    score_fn = _score_fn_for(eval_method)
    predictor, teardown = _make_deployed_predictor(
        deployed_checkpoint_path, base_model, quant=quant,
    )
    try:
        results = confirm_all(probe_client, fixable_clusters, predictor, score_fn)
    finally:
        teardown()

    confirmed: list[Cluster] = []
    extra_gold: list[EvalExample] = []
    for c in fixable_clusters:
        r = results.get(c.cluster_id)
        if r is None:
            continue
        if r.confirmed:
            confirmed.append(c)
            extra_gold.extend(r.failing_probes)
        else:
            # demote: paper marks as external when weakness not systematic
            c.fixability = "external"

    # persist
    (audit_dir / "probes.json").write_text(json.dumps({
        cid: {"pass_rate": r.pass_rate, "confirmed": r.confirmed,
              "n_probes": len(r.probes), "n_failing": len(r.failing_probes),
              "rationale": r.rationale}
        for cid, r in results.items()
    }, indent=2, default=str), encoding="utf-8")
    if audit:
        audit.section("Live confirmation", "\n".join(
            f"- **{cid}** pass_rate={r.pass_rate:.2f} "
            f"confirmed={r.confirmed} probes={len(r.probes)} failing={len(r.failing_probes)}"
            for cid, r in results.items()
        ) or "_no fixable clusters_")
    return confirmed, extra_gold, results


def _build_corrective_curriculum(
    cfg: AutoSLMConfig,
    teacher: LLMClient,
    fixable_clusters: list[Cluster],
    fail_records: list[dict],
    pass_records: list[dict],
    parent_dataset_path: Optional[str],
    use_cot: bool,
    target_inputs: list[str],
    extra_gold_from_probes: Optional[list[EvalExample]] = None,
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
    # probe-failing inputs become explicit gold (paper §2.6 step 3)
    if extra_gold_from_probes:
        for ex in extra_gold_from_probes:
            gold_examples.append(Example(
                input=ex.input, output=ex.gold, label=None,
                metadata={**(ex.metadata or {}), "from_probe": True},
            ))
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


def _load_prior_eval_set(audit_dir: Path) -> Optional[EvalSet]:
    """Cross-checkpoint ratchet (§2.7): prior accepted iter persists eval_set.json."""
    p = audit_dir / "prior_eval_set.json"
    if not p.exists():
        return None
    try:
        return EvalSet.load(p)
    except Exception:
        return None


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
    deployed_checkpoint_path: Optional[str] = None,
    enable_probes: bool = True,
    enable_ratchet: bool = True,
) -> dict:
    """Closed-loop production-mode run. Returns final pipeline + best checkpoint."""
    cfg.ensure_dirs()
    store = TraceStore(cfg.trace_db_path, backend=cfg.trace_db)
    fail_records = store.fail_set(deployed_model_id)
    pass_records = store.pass_set(deployed_model_id)
    if not fail_records:
        return {"status": "no_failures", "model_id": deployed_model_id}

    audit_dir = cfg.workdir / "runs" / deployed_model_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(audit_dir / "data-curation.md", run_id=deployed_model_id,
                    mode="production")
    audit.section("Inputs",
                  f"- deployed_model_id: `{deployed_model_id}`\n"
                  f"- base_model: `{base_model}`\n"
                  f"- task: `{task}`\n"
                  f"- fail/pass: {len(fail_records)}/{len(pass_records)}")

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
    audit.section("Failure taxonomy",
                  "\n".join(f"- **{c.cluster_id}** [{c.fixability}] {c.label} "
                            f"(n={c.size}) — {c.description[:200]}"
                            for c in clusters) or "_no clusters_")
    (audit_dir / "taxonomy.json").write_text(json.dumps(
        [asdict(c) for c in clusters], indent=2, default=str), encoding="utf-8")

    # live confirmation probes (§2.6 step 3) — demote clusters, harvest gold from probe failures
    extra_gold: list[EvalExample] = []
    probe_results: dict[str, ProbeResult] = {}
    if enable_probes and fixable:
        tier = cfg.tier()
        fixable, extra_gold, probe_results = _run_live_confirmation(
            cfg, fixable, deployed_checkpoint_path, base_model, tier.quant,
            eval_method, audit_dir, audit=audit,
        )

    # eval + regression sets
    eval_examples = _failures_to_eval(fail_records)
    regression_examples = _passes_to_regression(pass_records)
    eval_set = EvalSet(pos=eval_examples)
    if enable_ratchet:
        eval_set.prior = _load_prior_eval_set(audit_dir)

    # teacher selection per paper Section 2.5
    teacher_model = teacher_for_task(task, cfg)
    teacher = LLMClient(model=teacher_model)
    judge = LLMClient(model=cfg.judge_model) if eval_method == "llm_judge" else None

    # build initial dataset
    target_inputs = [r["input"] for r in fail_records]
    examples, qreport = _build_corrective_curriculum(
        cfg, teacher, fixable, fail_records, pass_records,
        parent_dataset_path, use_cot, target_inputs,
        extra_gold_from_probes=extra_gold,
    )
    if not examples:
        return {"status": "no_curriculum", "taxonomy": [asdict(c) for c in clusters]}

    audit.section("Curriculum",
                  f"- total: {qreport.final_size}\n"
                  f"- by_label: `{qreport.by_label}`\n"
                  f"- rejected: {qreport.rejected} ({qreport.reasons})\n"
                  f"- length-match violations: {qreport.length_match_violations}\n"
                  f"- surface-pattern violations: {qreport.surface_pattern_violations}")

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
        distributed=tier.distributed,
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
        ex_path = Path(pi.D.gold_path) if pi.D.gold_path else train_path
        with ex_path.open(encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        train_ex = [Example(input=r["input"], output=r["output"], label=r.get("label"),
                            is_hard_negative=r.get("is_hard_negative", False),
                            is_replay=r.get("is_replay", False),
                            metadata=r.get("metadata", {})) for r in rows]
        run_dir = audit_dir / f"iter_{pi.iteration}_{pi.fingerprint()}"
        train_result = train_pipeline(train_ex, pi.H, pi.S, run_dir, task=task)
        if train_result.error:
            audit.section(f"Iter {pi.iteration} TRAIN-FAIL",
                          f"`{pi.fingerprint()}` error: `{train_result.error}`")
            return NodeResult(score=0.0, regressions=999, failed=True, error=train_result.error)

        if pi.H.model_family == "gliner2":
            from ..eval.gliner_eval import gliner_predict
            labels = pi.H.gliner_labels or sorted({ex.label for ex in train_ex
                                                   if ex.label})
            gtask = "ner" if task in {"ner", "extraction"} else "classification"

            def predict(prompts: list[str]) -> list[str]:
                preds = gliner_predict(train_result.checkpoint_path, prompts,
                                      list(labels), task=gtask)
                if gtask == "ner":
                    # serialize spans to a stable string for metric scoring
                    return [json.dumps(p, default=str) for p in preds]
                return [str(p) for p in preds]

            model = tok = None  # nothing GPU-resident to free
        else:
            model, tok = load_for_inference(train_result.checkpoint_path,
                                           base_model=pi.H.base_model, quant=pi.H.quant)

            def predict(prompts: list[str]) -> list[str]:
                return generate_batch(model, tok, prompts, system=pi.H.system_prompt)

        a, r, full = score_pipeline(eval_set, regression_examples, predict, pi.S, judge)
        # detach model
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
        result = NodeResult(
            score=a, regressions=r, metrics=full.per_slice,
            checkpoint_path=train_result.checkpoint_path,
            eval_artifacts_path=str(run_dir / "eval.json"),
        )
        audit.section(f"Iter {pi.iteration}",
                      f"`{pi.fingerprint()}` score={a:.4f} regressions={r} "
                      f"slice_scores={full.per_slice}\nnotes: {pi.pipeline.notes if hasattr(pi,'pipeline') else pi.notes}")
        return result

    if not use_mcgs:
        result = evaluator(pi0)
        if result and not result.failed:
            eval_set.save(audit_dir / "prior_eval_set.json")
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
            "probes": {cid: {"pass_rate": r.pass_rate, "confirmed": r.confirmed}
                      for cid, r in probe_results.items()},
        }, default=str)[:6000]

    expander = llm_expander(orchestrator, failure_summary_provider=failure_summary)
    fuser = llm_fuser(orchestrator)
    mcgs = MCGS(mcfg, evaluator, expander, fuser, persist_dir=audit_dir / "graph")
    best = mcgs.run(pi0)

    # ratchet: persist accepted eval set for next run's prior
    if best and best.result and not best.result.failed:
        eval_set.save(audit_dir / "prior_eval_set.json")
    audit.section("Final",
                  f"- best: `{best.id if best else None}`\n"
                  f"- score: {best.result.score if best and best.result else None}\n"
                  f"- regressions: {best.result.regressions if best and best.result else None}\n"
                  f"- iterations: {mcgs.iteration}")
    return {
        "status": "ok", "best_node": best.id,
        "best_score": best.result.score if best.result else None,
        "best_regressions": best.result.regressions if best.result else None,
        "checkpoint": best.result.checkpoint_path if best.result else None,
        "iterations": mcgs.iteration,
        "graph_dir": str(audit_dir / "graph"),
        "taxonomy_path": str(audit_dir / "taxonomy.json"),
        "probes_path": str(audit_dir / "probes.json") if probe_results else None,
        "audit_log": str(audit_dir / "data-curation.md"),
    }
