"""Batch inference on a fine-tuned LoRA checkpoint. Used by eval harness."""
from __future__ import annotations
from pathlib import Path
from typing import Optional


def load_for_inference(checkpoint_path: str, base_model: Optional[str] = None,
                       quant: str = "8bit"):
    """Load LoRA adapter on top of base model."""
    import importlib.util
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from .lora_sft import _quant_config

    # Graceful bnb fallback (CPU build / not installed)
    if quant in {"4bit", "8bit"} and importlib.util.find_spec("bitsandbytes") is None:
        quant = "none"

    ck = Path(checkpoint_path)
    base = base_model
    if base is None:
        # try infer from adapter_config.json
        cfg_path = ck / "adapter_config.json"
        if cfg_path.exists():
            import json
            base = json.loads(cfg_path.read_text())["base_model_name_or_path"]
    bnb = _quant_config(quant)
    kwargs: dict = {"trust_remote_code": True}
    if bnb is not None:
        kwargs["quantization_config"] = bnb
    kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    try:
        model = PeftModel.from_pretrained(model, str(ck))
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(str(ck), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def generate_batch(model, tok, prompts: list[str], system: Optional[str] = None,
                   max_new_tokens: int = 512, temperature: float = 0.0,
                   batch_size: int = 8) -> list[str]:
    import torch
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        formatted = []
        for p in chunk:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = (system + "\n\n" if system else "") + p
            formatted.append(text)
        enc = tok(formatted, return_tensors="pt", padding=True, truncation=True,
                  max_length=tok.model_max_length).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tok.pad_token_id,
            )
        for j, ids in enumerate(gen):
            input_len = enc["input_ids"][j].shape[0]
            decoded = tok.decode(ids[input_len:], skip_special_tokens=True)
            out.append(decoded.strip())
    return out
