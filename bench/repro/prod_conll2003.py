"""CoNLL-2003 NER production replication (paper Fig. 9).

Target: Entity F1 0.810 per paper Figure 9.

Usage:
    python bench/repro/prod_conll2003.py --base-model gliner2-base-v1 --max-iter 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autoslm.modes.production import run_production
from autoslm.search.pipeline import DatasetSpec, HyperParams, LearningStrategy, Pipeline


def build_conll_pipeline(base_model: str, model_family: str = "gliner2") -> Pipeline:
    labels = ["PER", "ORG", "LOC", "MISC"]
    D = DatasetSpec(
        name="conll2003",
        gold_ratio=0.65,
        hard_neg_ratio=0.35,
        replay_ratio=0.0,
        max_examples=3000,
        label_balance_max_ratio=1.5,
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
        gliner_labels=labels,
    )
    S = LearningStrategy(
        supervision="direct",
        eval_method="f1",  # span-F1 for NER
        objective="sft",
    )
    return Pipeline(D=D, H=H, S=S, notes="CoNLL-2003 NER production")


def main():
    parser = argparse.ArgumentParser(description="CoNLL-2003 NER production replication")
    parser.add_argument("--base-model", type=str, default="gliner2-base-v1")
    parser.add_argument("--model-family", type=str, default="gliner2",
                        choices=["decoder", "gliner2"])
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--target", type=float, default=0.810, help="Target Entity F1 from paper")
    parser.add_argument("--output", type=str, default="bench/repro/results/conll2003_result.json")
    args = parser.parse_args()

    print(f"Running CoNLL-2003 NER production with {args.base_model}")
    print(f"Target Entity F1: {args.target:.3f}")

    pipeline = build_conll_pipeline(args.base_model, args.model_family)
    result = run_production(
        cfg=pipeline,
        task="ner",
        base_model=args.base_model,
        max_iterations=args.max_iter,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResult saved to {out_path}")

    final_score = result.get("final_score", 0.0)
    print(f"Final Entity F1: {final_score:.3f} (target: {args.target:.3f})")
    if final_score >= args.target * 0.95:
        print("SUCCESS: Reached target performance!")
    else:
        print("Note: Full budget may be needed to reach paper target.")


if __name__ == "__main__":
    main()
