"""Smoke test: every autoslm + bench module imports clean."""
import importlib
import pytest


MODULES = [
    "autoslm", "autoslm.config", "autoslm.cli", "autoslm.orchestrator",
    "autoslm.audit",
    "autoslm.llm.client",
    "autoslm.search.pipeline", "autoslm.search.mcgs", "autoslm.search.expand",
    "autoslm.data.curate", "autoslm.data.hard_negatives", "autoslm.data.replay",
    "autoslm.data.cot_annotate", "autoslm.data.acquire",
    "autoslm.eval.harness", "autoslm.eval.metrics", "autoslm.eval.llm_judge",
    "autoslm.eval.build_holdout", "autoslm.eval.gliner_eval",
    "autoslm.traces.ingest", "autoslm.traces.taxonomy", "autoslm.traces.probes",
    "autoslm.train.lora_sft", "autoslm.train.inference",
    "autoslm.train.gliner_train", "autoslm.train.dispatch",
    "autoslm.modes.production", "autoslm.modes.cold_start",
    "autoslm.tools.registry",
    "autoslm.agents.trace_analyzer",
    "bench.adaptft.perturbations", "bench.adaptft.build_scenario",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)
