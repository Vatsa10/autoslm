"""autoslm paper-replication benchmark scripts.

Each scenario module (cold_start_*, prod_*) exposes:
    run(max_iter: int, tier: str, base_model: str | None,
        out_dir: str | Path, dry_run: bool = False) -> dict

Scenarios:
    arc        — ARC-Challenge cold-start (paper Table 2, target 72.6%)
    gsm8k      — GSM8K math cold-start (paper Table 2, target 43.7%)
    humaneval  — HumanEval code cold-start (paper Table 2, target 92.7% pass@1)
    clinc150   — CLINC150 production (paper Table 8, target 99.3%)
    conll2003  — CoNLL-2003 NER production (paper Fig. 9, target Entity F1 0.810)
"""
from __future__ import annotations

SCENARIOS = ["arc", "gsm8k", "humaneval", "clinc150", "conll2003"]

SCENARIO_MODULES = {
    "arc": "bench.repro.cold_start_arc",
    "gsm8k": "bench.repro.cold_start_gsm8k",
    "humaneval": "bench.repro.cold_start_humaneval",
    "clinc150": "bench.repro.prod_clinc150",
    "conll2003": "bench.repro.prod_conll2003",
}

SCENARIO_DEFAULTS = {
    "arc":       {"base_model": "meta-llama/Llama-3.2-3B", "target": 0.726},
    "gsm8k":     {"base_model": "meta-llama/Llama-3.2-3B", "target": 0.437},
    "humaneval": {"base_model": "Qwen/Qwen3-8B",          "target": 0.927},
    "clinc150":  {"base_model": "meta-llama/Llama-3.2-3B", "target": 0.993},
    "conll2003": {"base_model": "meta-llama/Llama-3.2-3B", "target": 0.810},
}
