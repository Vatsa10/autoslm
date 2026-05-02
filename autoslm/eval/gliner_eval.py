"""GLiNER2 inference + span-F1 / classification metrics (paper §2.1, §4.4.2)."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Optional


def gliner_predict(checkpoint_path: str | Path, inputs: list[str],
                   labels: list[str], threshold: float = 0.5,
                   task: str = "ner") -> list:
    """Returns per-input predictions. NER -> list[list[dict]]; classification -> list[str]."""
    try:
        from gliner import GLiNER
    except ImportError as e:
        raise ImportError("gliner not installed; install with `pip install gliner`") from e
    model = GLiNER.from_pretrained(str(checkpoint_path))
    if task == "ner":
        out = []
        for txt in inputs:
            preds = model.predict_entities(txt, labels, threshold=threshold)
            out.append([{"start": p["start"], "end": p["end"],
                         "text": p["text"], "label": p["label"],
                         "score": float(p.get("score", 0.0))} for p in preds])
        return out
    if task == "classification":
        out = []
        for txt in inputs:
            try:
                preds = model.classify(txt, labels)  # type: ignore[attr-defined]
                out.append(max(preds.items(), key=lambda kv: kv[1])[0]
                          if isinstance(preds, dict) else str(preds))
            except Exception:
                ents = model.predict_entities(txt, labels, threshold=threshold)
                out.append(ents[0]["label"] if ents else "")
        return out
    raise ValueError(f"unknown task {task}")


def span_f1(pred_spans: list[list[dict]], gold_spans: list[list[dict]]) -> dict:
    """Micro-averaged entity F1 (paper §4.4.2)."""
    tp = fp = fn = 0
    for preds, golds in zip(pred_spans, gold_spans):
        pset = {(p["start"], p["end"], p["label"]) for p in preds}
        gset = {(g["start"], g["end"], g["label"]) for g in golds}
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    if tp == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": fp, "fn": fn}
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}
