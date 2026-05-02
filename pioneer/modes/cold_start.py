"""Cold-start mode (paper Section 2.5). Stub for v1.

Pipeline:
  1. task classification + base model selection
  2. data acquisition (web/HF benchmark)
  3. baseline survey (calibrate target tau)
  4. evaluation set construction (E_pos U E_neg U E_boundary, BEFORE training)
  5. curriculum synthesis (D_gold + D_hard, no replay)
  6. parallel training of >=2 configs; iterate via MCGS

Production-first scope: cold_start currently routes to a minimal one-shot run;
expand to full agent loop once production loop is validated.
"""
from __future__ import annotations
from typing import Optional


def run_cold_start(*args, **kwargs):  # noqa: D401
    raise NotImplementedError(
        "cold_start mode pending. Production-first per scope; cold_start lands in v0.2."
    )
