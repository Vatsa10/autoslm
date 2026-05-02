"""GLiNER2 encoder trainer (paper §2.1).

Encoder family for NER + classification with shared representations.
Supports both full fine-tuning (paper notes practical at GLiNER2 scale,
2-5 min training) and LoRA via the `gliner` library when installed.

This module is dependency-soft: if `gliner` isn't installed it raises a
descriptive error rather than at import time, so the rest of autoslm stays
importable on minimal environments.
"""
from __future__ import annotations
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..search.pipeline import HyperParams, LearningStrategy
from ..data.curate import Example
from .lora_sft import TrainResult


def _require_gliner():
    try:
        import gliner  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "GLiNER2 path requires the optional `gliner` package. "
            "Install with `pip install gliner` or via the [ml] extra."
        ) from e


def _examples_to_ner_format(examples: list[Example]) -> list[dict]:
    """GLiNER training format: {tokenized_text, ner: [[start, end, type]]}.

    Inputs in autoslm Example carry annotations as bracketed `[surface](TYPE)`
    spans inside `output` or `metadata['entities']`. We accept either.
    """
    import re
    out: list[dict] = []
    ent_re = re.compile(r"\[([^\[\]]{1,200})\]\(([A-Za-z_][A-Za-z0-9_]*)\)")
    for ex in examples:
        if ex.metadata and "tokenized_text" in ex.metadata:
            out.append({
                "tokenized_text": ex.metadata["tokenized_text"],
                "ner": ex.metadata.get("ner", []),
            })
            continue
        text = ex.input
        toks = text.split()
        ner: list[list] = []
        # parse spans from output if formatted, else assume label-as-class
        annot = ex.output or ""
        for m in ent_re.finditer(annot):
            surface, ent_type = m.group(1), m.group(2)
            for i in range(len(toks) - len(surface.split()) + 1):
                window = " ".join(toks[i:i + len(surface.split())])
                if window.strip(",.;:") == surface:
                    ner.append([i, i + len(surface.split()) - 1, ent_type])
                    break
        out.append({"tokenized_text": toks, "ner": ner})
    return out


def _examples_to_cls_format(examples: list[Example]) -> list[dict]:
    return [{"text": ex.input, "label": ex.label or ex.output}
            for ex in examples if (ex.label or ex.output)]


def train_gliner(
    examples: list[Example],
    H: HyperParams,
    S: LearningStrategy,
    output_dir: str | Path,
    eval_examples: Optional[list[Example]] = None,
    task: str = "ner",     # 'ner' | 'classification' | 'multitask'
) -> TrainResult:
    """Run one GLiNER2 train job. Returns a TrainResult mirroring lora_sft."""
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_id = f"gliner-{uuid.uuid4().hex[:8]}"
    ckpt_dir = out / model_id
    ckpt_dir.mkdir(exist_ok=True)

    try:
        _require_gliner()
        from gliner import GLiNER
        # Trainer interface in gliner is evolving; we use the `Trainer` if
        # available, otherwise fall back to the model's own `.train()` helper.
        try:
            from gliner.training import Trainer as GLinerTrainer  # type: ignore
        except ImportError:
            GLinerTrainer = None  # type: ignore

        model = GLiNER.from_pretrained(H.base_model)

        if task == "ner":
            data = _examples_to_ner_format(examples)
            eval_data = _examples_to_ner_format(eval_examples) if eval_examples else None
        elif task == "classification":
            data = _examples_to_cls_format(examples)
            eval_data = _examples_to_cls_format(eval_examples) if eval_examples else None
        else:
            data = _examples_to_ner_format(examples)
            eval_data = _examples_to_ner_format(eval_examples) if eval_examples else None

        if GLinerTrainer is not None:
            trainer = GLinerTrainer(
                model=model,
                train_data=data,
                eval_data=eval_data,
                num_epochs=H.epochs,
                batch_size=H.batch_size,
                lr_encoder=H.learning_rate,
                lr_others=H.learning_rate,
                save_directory=str(ckpt_dir),
                full_finetune=H.full_finetune,
            )
            trainer.train()
        else:
            # generic fallback: use HF Trainer pattern via gliner's exposed pipeline
            model.train_model(  # type: ignore[attr-defined]
                data, epochs=H.epochs, batch_size=H.batch_size,
                lr=H.learning_rate, save_dir=str(ckpt_dir),
            )

        # save final
        try:
            model.save_pretrained(str(ckpt_dir))
        except Exception:
            pass
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=None,
            train_examples=len(examples),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S), "task": task,
                         "family": "gliner2"},
        )
    except Exception as e:
        return TrainResult(
            model_id=model_id,
            checkpoint_path=str(ckpt_dir),
            final_loss=None,
            train_examples=len(examples),
            runtime_sec=time.time() - t0,
            config_used={"H": asdict(H), "S": asdict(S), "task": task,
                         "family": "gliner2"},
            error=str(e),
        )
