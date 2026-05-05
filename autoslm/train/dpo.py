"""DPO and KTO trainers (paper Section 6.3 limitation addressed).

Wraps trl.DPOTrainer and trl.KTOTrainer for preference-based fine-tuning.
Requires judge_score deltas to construct preference pairs from production traces.

DPO (Direct Preference Optimization):
  - Input: preference pairs (prompt, chosen, rejected)
  - Loss: logistic loss on implicit reward difference

KTO (Kahneman-Tversky Optimization):
  - Input: (prompt, completion, label: "desirable"/"undesirable")
  - Loss: asymmetric Tversky-style loss weighting false positives vs false negatives

LearningStrategy.objective selects: "sft" (default), "dpo", or "kto".
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .lora_sft import TrainResult
from ..search.pipeline import HyperParams, LearningStrategy


# ---------------------------------------------------------------------------
# DPO Dataset Construction
# ---------------------------------------------------------------------------

def _build_dpo_pairs(
    examples: list[dict],
    judge_score_threshold: float = 0.5,
) -> list[dict]:
    """Convert production traces into DPO preference pairs.

    Each pair: {"prompt": input, "chosen": better_output, "rejected": worse_output}

    Strategy: sort by judge_score within same input group; take top as chosen,
    bottom as rejected.  Only keep pairs where scores differ meaningfully.
    """
    pairs: list[dict] = []
    # Group by input/context
    by_input: dict[str, list[dict]] = {}
    for ex in examples:
        key = ex.get("input") or ex.get("prompt", "")
        by_input.setdefault(key, []).append(ex)

    for inp, group in by_input.items():
        if len(group) < 2:
            continue
        scored = sorted(group, key=lambda x: x.get("judge_score", 0.0), reverse=True)
        best = scored[0]
        worst = scored[-1]
        if best.get("judge_score", 0) - worst.get("judge_score", 0) < 0.1:
            continue  # not enough signal
        pairs.append({
            "prompt": inp,
            "chosen": best.get("output") or best.get("prediction", ""),
            "rejected": worst.get("output") or worst.get("prediction", ""),
        })
    return pairs


# ---------------------------------------------------------------------------
# KTO Dataset Construction
# ---------------------------------------------------------------------------

def _build_kto_examples(
    examples: list[dict],
    desirable_threshold: float = 0.7,
) -> list[dict]:
    """Convert production traces into KTO examples.

    Each example: {"prompt": input, "completion": output, "label": "desirable"/"undesirable"}
    """
    out: list[dict] = []
    for ex in examples:
        score = ex.get("judge_score", 0.0)
        label = "desirable" if score >= desirable_threshold else "undesirable"
        out.append({
            "prompt": ex.get("input") or ex.get("prompt", ""),
            "completion": ex.get("output") or ex.get("prediction", ""),
            "label": label,
        })
    return out


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------

def train_dpo(
    examples: list[dict],
    H: HyperParams,
    S: LearningStrategy,
    output_dir: str | Path,
    eval_examples: Optional[list[dict]] = None,
    log_callback=None,
) -> TrainResult:
    """Run DPO fine-tuning using trl.DPOTrainer.

    Args:
        examples: production traces with judge_score fields
        H: HyperParams (base_model, lora_rank, etc.)
        S: LearningStrategy (eval_method, etc.)
        output_dir: where to save the checkpoint
        eval_examples: optional held-out evaluation set
    """
    import time
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_id = f"dpo-{uuid.uuid4().hex[:8]}"
    ckpt_dir = out / model_id
    ckpt_dir.mkdir(exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from trl import DPOTrainer, DPOConfig

        pairs = _build_dpo_pairs(examples)
        if not pairs:
            raise ValueError("No DPO pairs generated; check judge_score fields in examples")

        tok = AutoTokenizer.from_pretrained(H.base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            H.base_model,
            torch_dtype=torch.bfloat16 if H.bf16 else torch.float16,
            trust_remote_code=True,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            H.base_model,
            torch_dtype=torch.bfloat16 if H.bf16 else torch.float16,
            trust_remote_code=True,
        )

        if not H.full_finetune:
            lora_cfg = LoraConfig(
                r=H.lora_rank,
                lora_alpha=H.lora_alpha,
                lora_dropout=H.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            ref_model = get_peft_model(ref_model, lora_cfg)

        dpo_cfg = DPOConfig(
            output_dir=str(ckpt_dir),
            per_device_train_batch_size=H.batch_size,
            gradient_accumulation_steps=H.grad_accum,
            learning_rate=H.learning_rate,
            num_train_epochs=H.epochs,
            warmup_ratio=H.warmup_ratio,
            bf16=H.bf16,
            fp16=not H.bf16,
            gradient_checkpointing=H.grad_checkpoint,
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=10,
            report_to="none",
            seed=H.seed,
            beta=0.1,  # KL penalty weight
            max_length=H.max_seq_len,
            max_prompt_length=H.max_seq_len // 2,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_cfg,
            tokenizer=tok,
            train_dataset=pairs,
        )
        history = trainer.train()
        trainer.save_model(str(ckpt_dir))
        tok.save_pretrained(str(ckpt_dir))

        final_loss = float(history.training_loss) if history and getattr(history, "training_loss", None) else None
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=final_loss,
            train_examples=len(pairs),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S), "method": "dpo"},
        )
    except Exception as e:
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=None,
            train_examples=len(examples),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S), "method": "dpo"},
            error=str(e),
        )


# ---------------------------------------------------------------------------
# KTO Trainer
# ---------------------------------------------------------------------------

def train_kto(
    examples: list[dict],
    H: HyperParams,
    S: LearningStrategy,
    output_dir: str | Path,
    eval_examples: Optional[list[dict]] = None,
    log_callback=None,
) -> TrainResult:
    """Run KTO fine-tuning using trl.KTOTrainer.

    KTO uses a Kahneman-Tversky loss that is asymmetric between desirable and
    undesirable examples, better matching human preference psychology.
    """
    import time
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_id = f"kto-{uuid.uuid4().hex[:8]}"
    ckpt_dir = out / model_id
    ckpt_dir.mkdir(exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from trl import KTOTrainer, KTOConfig

        kto_examples = _build_kto_examples(examples)
        if not kto_examples:
            raise ValueError("No KTO examples generated")

        tok = AutoTokenizer.from_pretrained(H.base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            H.base_model,
            torch_dtype=torch.bfloat16 if H.bf16 else torch.float16,
            trust_remote_code=True,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            H.base_model,
            torch_dtype=torch.bfloat16 if H.bf16 else torch.float16,
            trust_remote_code=True,
        )

        if not H.full_finetune:
            lora_cfg = LoraConfig(
                r=H.lora_rank,
                lora_alpha=H.lora_alpha,
                lora_dropout=H.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            ref_model = get_peft_model(ref_model, lora_cfg)

        kto_cfg = KTOConfig(
            output_dir=str(ckpt_dir),
            per_device_train_batch_size=H.batch_size,
            gradient_accumulation_steps=H.grad_accum,
            learning_rate=H.learning_rate,
            num_train_epochs=H.epochs,
            warmup_ratio=H.warmup_ratio,
            bf16=H.bf16,
            fp16=not H.bf16,
            gradient_checkpointing=H.grad_checkpoint,
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=10,
            report_to="none",
            seed=H.seed,
            beta=0.1,
            max_length=H.max_seq_len,
        )

        trainer = KTOTrainer(
            model=model,
            ref_model=ref_model,
            args=kto_cfg,
            tokenizer=tok,
            train_dataset=kto_examples,
        )
        history = trainer.train()
        trainer.save_model(str(ckpt_dir))
        tok.save_pretrained(str(ckpt_dir))

        final_loss = float(history.training_loss) if history and getattr(history, "training_loss", None) else None
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=final_loss,
            train_examples=len(kto_examples),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S), "method": "kto"},
        )
    except Exception as e:
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=None,
            train_examples=len(examples),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S), "method": "kto"},
            error=str(e),
        )
