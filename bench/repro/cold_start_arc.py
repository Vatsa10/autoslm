"""ARC-Challenge cold-start replication (paper Table 2).

Target: ~72.6% on Llama-3.2-3B (paper reports 72.6% for ARC-Challenge).

Usage:
    python bench/repro/cold_start_arc.py --base-model meta-llama/Llama-3.2-3B --max-iter 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autoslm.modes.cold_start import run_cold_start
from autoslm.search.pipeline import DatasetSpec, HyperParams, LearningStrategy, Pipeline
from autoslm.eval.metrics import exact_match


def build_arc_pipeline(base_model: str) -> Pipeline:
    D = DatasetSpec(
        name="arc-challenge",
        gold_ratio=0.7,
        hard_neg_ratio=0.3,
        replay_ratio=0.0,
        max_examples=2000,
    )
    H = HyperParams(
        base_model=base_model,
        lora_rank=32,
        lora_alpha=64,
        learning_rate=2e-4,
        batch_size=4,
        grad_accum=4,
        epochs=3,
        max_seq_len=2048,
        quant="8bit",
        bf16=True,
        grad_checkpoint=True,
    )
    S = LearningStrategy(
        supervision="direct",
        eval_method="exact_match",
    )
    return Pipeline(D=D, H=H, S=S, notes="ARC-Challenge cold-start")


def main():
    parser = argparse.ArgumentParser(description="ARC-Challenge cold-start replication")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--target", type=float, default=0.726, help="Target accuracy from paper")
    parser.add_argument("--output", type=str, default="bench/repro/results/arc_result.json")
    args = parser.parse_args()

    print(f"Running ARC-Challenge cold-start with {args.base_model}")
    print(f"Target accuracy: {args.target:.1%}")

    pipeline = build_arc_pipeline(args.base_model)
    result = run_cold_start(
        cfg=pipeline,
        task_spec="ARC-Challenge: multiple choice question answering",
        base_model=args.base_model,
        dataset_hint="arc-challenge",
        target_threshold=args.target,
        max_iterations=args.max_iter,
    )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResult saved to {out_path}")

    final_score = result.get("final_score", 0.0)
    print(f"Final score: {final_score:.3f} (target: {args.target:.3f})")
    if final_score >= args.target * 0.95:  # 5% tolerance
        print("SUCCESS: Reached target performance!")
    else:
        print("Note: Full budget may be needed to reach paper target.")


if __name__ == "__main__":
    main()
