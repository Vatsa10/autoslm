"""Modal sandbox runner for distributed training (paper Section 2.5 / 2.6).

Wraps train_lora_sft as a Modal function that runs in the cloud with GPU.
Returns same TrainResult as local training.

Usage:
    modal deploy autoslm/sandbox/modal_runner.py
    # Then from production.py or mcgs.py:
    from autoslm.sandbox import evaluate_batch
    results = evaluate_batch([pipeline1, pipeline2])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


def is_modal_available() -> bool:
    return MODAL_AVAILABLE


# ---------------------------------------------------------------------------
# Modal image: includes all training dependencies
# ---------------------------------------------------------------------------

if MODAL_AVAILABLE:
    # Base image with Python + PyTorch
    train_image = (
        modal.Image.debian_slim()
        .pip_install(
            "torch", "torchvision", "torchaudio",
            "transformers", "accelerate", "peft",
            "trl", "bitsandbytes", "datasets",
            "scikit-learn", "sentence-transformers",
            "networkx", "duckdb",  # for traces
        )
        .copy_local_file(
            "autoslm/train/lora_sft.py",
            "/root/autoslm/train/lora_sft.py",
        )
        .copy_local_file(
            "autoslm/search/pipeline.py",
            "/root/autoslm/search/pipeline.py",
        )
        .copy_local_file(
            "autoslm/data/curate.py",
            "/root/autoslm/data/curate.py",
        )
    )

    app = modal.App("autoslm-trainer", image=train_image)

    @app.function(
        gpu="A10G",
        timeout=86400,  # 24h max per training job
        volumes={
            "/outputs": modal.Volume.from_name("autoslm-outputs", create_if_missing=True),
        },
    )
    def train_remote(
        pipeline_json: str,
        examples_json: str,
        output_dir: str,
    ) -> str:
        """Run one training job remotely. Returns TrainResult as JSON."""
        import json
        from autoslm.search.pipeline import Pipeline
        from autoslm.train.lora_sft import train_lora_sft
        from autoslm.data.curate import Example

        pipeline = Pipeline.from_dict(json.loads(pipeline_json))
        examples = [Example(**e) for e in json.loads(examples_json)]
        result = train_lora_sft(
            examples, pipeline.H, pipeline.S, output_dir,
        )
        return json.dumps(result.__dict__, default=str)

else:
    # Stub for when modal is not installed
    def train_remote(*args, **kwargs):
        raise ImportError(
            "Modal is not installed. Install with: pip install modal\n"
            "Then configure with: modal setup"
        )


# ---------------------------------------------------------------------------
# Batch evaluation interface
# ---------------------------------------------------------------------------

def evaluate_batch(
    pipelines: list,
    examples_map: Optional[list] = None,
    output_base: str = "/outputs",
) -> list:
    """Evaluate multiple pipelines in parallel via Modal.

    Args:
        pipelines: list of Pipeline objects
        examples_map: list of example lists (one per pipeline), or None to use pipeline.D
        output_base: base output directory in Modal volume

    Returns:
        list of TrainResult-like dicts
    """
    if not MODAL_AVAILABLE:
        raise ImportError(
            "Modal is not installed. Install with: pip install modal\n"
            "Then configure with: modal setup"
        )

    results = []
    with modal.running_app(app):
        for i, pipeline in enumerate(pipelines):
            output_dir = str(Path(output_base) / f"run_{i}_{pipeline.fingerprint()}")
            pipeline_json = json.dumps(pipeline.to_dict(), default=str)
            # Use pipeline's dataset or provided examples
            from autoslm.data.curate import Example
            if examples_map and i < len(examples_map):
                examples = examples_map[i]
            else:
                # Load from dataset path in pipeline.D
                examples = []  # placeholder - would load from pipeline.D.gold_path
            examples_json = json.dumps([e.__dict__ for e in examples], default=str)

            result_json = train_remote.remote(
                pipeline_json, examples_json, output_dir,
            )
            results.append(json.loads(result_json))

    return results
