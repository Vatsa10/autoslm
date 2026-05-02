"""LoRA SFT trainer wrapping HuggingFace transformers + peft + trl.

Paper uses Tinker SDK (proprietary). We swap for open-source equivalents:
- peft: LoRA implementation
- trl: SFTTrainer
- bitsandbytes: 4/8-bit quant for edge tier
- accelerate: distributed (FSDP) for big tier

Hardware tiers (configs.py):
- edge: 4-bit + LoRA r=8, grad_checkpoint, 1024 ctx
- mid:  8-bit + LoRA r=32, grad_checkpoint, 2048 ctx
- big:  bf16 + LoRA r=64, 4096 ctx, optional FSDP
"""
from __future__ import annotations
import json
import os
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from ..search.pipeline import HyperParams, LearningStrategy
from ..data.curate import Example


@dataclass
class TrainResult:
    model_id: str
    checkpoint_path: str
    final_loss: Optional[float]
    train_examples: int
    runtime_sec: float
    config_used: dict
    error: Optional[str] = None


def _format_example(ex: Example, sys_prompt: Optional[str], cot: bool) -> dict:
    user = ex.input
    assistant = ex.output
    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def _materialize_dataset(examples: list[Example], H: HyperParams, S: LearningStrategy):
    from datasets import Dataset
    rows = [_format_example(ex, H.system_prompt, S.supervision == "cot") for ex in examples]
    return Dataset.from_list(rows)


def _quant_config(quant: str):
    if quant == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
        import torch
        if quant == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if quant == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        return None
    return None


def train_lora_sft(
    examples: list[Example],
    H: HyperParams,
    S: LearningStrategy,
    output_dir: str | Path,
    eval_examples: Optional[list[Example]] = None,
    log_callback=None,
) -> TrainResult:
    """Run one SFT job. Returns checkpoint path + metrics."""
    import time
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_id = f"ft-{uuid.uuid4().hex[:8]}"
    ckpt_dir = out / model_id
    ckpt_dir.mkdir(exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer

        tok = AutoTokenizer.from_pretrained(H.base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        bnb = _quant_config(H.quant)
        model_kwargs: dict = {"trust_remote_code": True}
        if bnb is not None:
            model_kwargs["quantization_config"] = bnb
        model_kwargs["torch_dtype"] = torch.bfloat16 if H.bf16 else torch.float16

        model = AutoModelForCausalLM.from_pretrained(H.base_model, **model_kwargs)
        if bnb is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=H.grad_checkpoint)

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

        train_ds = _materialize_dataset(examples, H, S)
        eval_ds = _materialize_dataset(eval_examples, H, S) if eval_examples else None

        sft_cfg = SFTConfig(
            output_dir=str(ckpt_dir),
            per_device_train_batch_size=H.batch_size,
            gradient_accumulation_steps=H.grad_accum,
            learning_rate=H.learning_rate,
            num_train_epochs=H.epochs,
            warmup_ratio=H.warmup_ratio,
            max_seq_length=H.max_seq_len,
            bf16=H.bf16,
            fp16=not H.bf16,
            gradient_checkpointing=H.grad_checkpoint,
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=10,
            report_to="none",
            seed=H.seed,
            packing=False,
        )
        trainer = SFTTrainer(
            model=model, args=sft_cfg, tokenizer=tok,
            train_dataset=train_ds, eval_dataset=eval_ds,
        )
        history = trainer.train()
        trainer.save_model(str(ckpt_dir))
        tok.save_pretrained(str(ckpt_dir))
        final_loss = float(history.training_loss) if history and getattr(history, "training_loss", None) else None
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=final_loss,
            train_examples=len(examples),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S)},
        )
    except Exception as e:
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=None,
            train_examples=len(examples),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S)},
            error=str(e),
        )
