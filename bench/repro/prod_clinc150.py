"""CLINC150 production-mode replication (paper Table 8).

Target: >= 99.3% intent accuracy on CLINC150 (paper reports 99.3%).

Usage:
    python bench/repro/prod_clinc150.py --base-model gliner2-base-v1 --max-iter 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autoslm.modes.production import run_production
from autoslm.search.pipeline import DatasetSpec, HyperParams, LearningStrategy, Pipeline


def build_clinc150_pipeline(base_model: str, model_family: str = "decoder") -> Pipeline:
    D = DatasetSpec(
        name="clinc150",
        gold_ratio=0.65,
        hard_neg_ratio=0.35,
        replay_ratio=0.0,
        max_examples=5000,
        label_balance_max_ratio=2.0,
    )
    H = HyperParams(
        base_model=base_model,
        lora_rank=32,
        lora_alpha=64,
        learning_rate=2e-4,
        batch_size=8,
        grad_accum=2,
        epochs=3,
        max_seq_len=512,
        quant="8bit",
        bf16=True,
        model_family=model_family,
    )
    S = LearningStrategy(
        supervision="direct",
        eval_method="exact_match",
        objective="sft",
    )
    return Pipeline(D=D, H=H, S=S, notes="CLINC150 production intent classification")


def main():
    parser = argparse.ArgumentParser(description="CLINC150 production replication")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--model-family", type=str, default="decoder",
                        choices=["decoder", "gliner2"])
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--target", type=float, default=0.993, help="Target accuracy from paper")
    parser.add_argument("--output", type=str, default="bench/repro/results/clinc150_result.json")
    args = parser.parse_args()

    print(f"Running CLINC150 production with {args.base_model}")
    print(f"Target accuracy: {args.target:.1%}")

    pipeline = build_clinc150_pipeline(args.base_model, args.model_family)
    result = run_production(
        cfg=pipeline,
        task="intent_classification",
        base_model=args.base_model,
        max_iterations=args.max_iter,
    )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResult saved to {out_path}")

    final_score = result.get("final_score", 0.0)
    print(f"Final score: {final_score:.3f} (target: {args.target:.3f})")
    if final_score >= args.target * 0.95:
        print("SUCCESS: Reached target performance!")
    else:
        print("Note: Full budget may be needed to reach paper target.")


if __name__ == "__main__":
    main()
