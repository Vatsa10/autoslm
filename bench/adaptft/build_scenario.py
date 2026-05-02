"""Build AdaptFT-Bench scenario (paper Section 3.1, Appendix B).

Stage protocol: 3 stages with poison rates [15%, 25%, 40%].
Each stage ~500 inference logs. 70/30 train/test split.
Held-out eval = union of all stage test splits.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .perturbations import build_stage


STAGE_POISON_RATES = [0.0, 0.15, 0.25, 0.40]
STAGE_TRAIN_FRAC = 0.70
DEFAULT_STAGE_SIZE = 500


@dataclass
class Scenario:
    name: str
    base_model: str
    task: str
    stages: list[dict] = field(default_factory=list)        # per-stage inference logs
    held_out_test: list[dict] = field(default_factory=list) # union of test splits
    train_per_stage: list[list[dict]] = field(default_factory=list)


def build_synthetic_scenario(
    name: str,
    base_examples: list[dict],
    base_model: str,
    task: str,
    out_dir: str | Path,
    stage_size: int = DEFAULT_STAGE_SIZE,
    poison_rates: Optional[list[float]] = None,
    alt_labels: Optional[list[str]] = None,
    seed: int = 42,
) -> Scenario:
    rates = poison_rates or STAGE_POISON_RATES
    rng = random.Random(seed)
    pool = list(base_examples)
    rng.shuffle(pool)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc = Scenario(name=name, base_model=base_model, task=task)

    for s_i, rate in enumerate(rates):
        if not pool:
            break
        chunk = pool[:stage_size]
        pool = pool[stage_size:]
        stage = build_stage(chunk, poison_rate=rate, alt_labels=alt_labels,
                            seed=seed + s_i)
        for r in stage:
            r["metadata"] = {**r.get("metadata", {}), "deployment_stage": s_i}
        n_train = int(len(stage) * STAGE_TRAIN_FRAC)
        train, test = stage[:n_train], stage[n_train:]
        sc.stages.append({"stage": s_i, "rate": rate, "train": train, "test": test})
        sc.train_per_stage.append(train)
        sc.held_out_test.extend(test)

    (out / "scenario.json").write_text(json.dumps({
        "name": name, "base_model": base_model, "task": task,
        "rates": rates,
        "n_stages": len(sc.stages),
        "n_train_per_stage": [len(t) for t in sc.train_per_stage],
        "n_held_out": len(sc.held_out_test),
    }, indent=2), encoding="utf-8")

    for i, s in enumerate(sc.stages):
        with (out / f"stage_{i}_train.jsonl").open("w", encoding="utf-8") as f:
            for r in s["train"]:
                f.write(json.dumps(r, default=str) + "\n")
        with (out / f"stage_{i}_test.jsonl").open("w", encoding="utf-8") as f:
            for r in s["test"]:
                f.write(json.dumps(r, default=str) + "\n")

    with (out / "held_out_test.jsonl").open("w", encoding="utf-8") as f:
        for r in sc.held_out_test:
            f.write(json.dumps(r, default=str) + "\n")
    return sc
