"""Sandbox runners for distributed training (paper Section 2.5 / 2.6).

Supports:
  - Local process pool (default)
  - Modal cloud sandbox (--sandbox modal)
"""

from .modal_runner import evaluate_batch, is_modal_available  # noqa: F401
