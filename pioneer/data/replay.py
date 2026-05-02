"""Replay buffer (paper Eq. 15): D_replay subset of D_parent, ratio 10-20%.

Used in production mode when improving an already fine-tuned model.
Skipped in cold-start (replay allocation reallocated to gold).
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from .curate import Example


def build_replay_buffer(parent_dataset_path: str | Path, ratio: float = 0.15,
                       seed: int = 42) -> list[Example]:
    """Sample `ratio` fraction of parent dataset stratified by label."""
    p = Path(parent_dataset_path)
    if not p.exists():
        return []
    items: list[Example] = []
    if p.suffix in {".jsonl", ".json"}:
        if p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.loads(p.read_text(encoding="utf-8"))
        for r in rows:
            items.append(Example(
                input=str(r.get("input") or r.get("prompt") or ""),
                output=str(r.get("output") or r.get("completion") or ""),
                label=r.get("label"),
                metadata=r.get("metadata", {}),
                is_replay=True,
            ))
    if not items:
        return []

    # stratified sample
    from collections import defaultdict
    by_lab: dict[str, list[Example]] = defaultdict(list)
    for ex in items:
        by_lab[ex.label or "_unlabeled"].append(ex)
    rng = random.Random(seed)
    out: list[Example] = []
    target = int(len(items) * ratio)
    if target <= 0:
        return []
    per_lab = max(1, target // max(1, len(by_lab)))
    for lab, group in by_lab.items():
        rng.shuffle(group)
        for ex in group[:per_lab]:
            ex.is_replay = True
            out.append(ex)
    return out[:target]
