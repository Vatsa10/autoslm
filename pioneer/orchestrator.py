"""Orchestrator: top-level state machine. Drives the loop with tool use.

The MCGS evaluator already automates the heavy lifting. The orchestrator wraps
the agent-tool surface so an LLM can:
  - inspect traces (query_traces)
  - run shell (bash)
  - read/edit curated artifacts (read_file/edit_file)
  - delegate to sub-agents (delegate_task)
  - kick off MCGS via tool_calls (run_search)

This is intentionally thin: most heavy work is in pioneer.modes.production.
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .config import PioneerConfig
from .llm import LLMClient, chat_with_tools
from .traces import TraceStore
from .tools import build_tool_registry
from .modes.production import run_production


SYSTEM = """You are Pioneer Agent (paper arXiv:2604.09791), driving closed-loop SLM adaptation.

You have tools to query inference traces, run shell, read/edit files, delegate sub-agents.
Workflow when asked to improve a deployed model:
  1. query failures (query_traces)
  2. cluster + label fixability (delegate_task: trace_analyzer)
  3. inspect parent lineage
  4. call run_search with desired iteration budget
  5. report best checkpoint + score + regressions

Never load >50 raw rows into context: filter, summarize, persist via bash_pipeline.
"""


def run_orchestrator(
    cfg: PioneerConfig,
    user_request: str,
    deployed_model_id: Optional[str] = None,
    base_model: Optional[str] = None,
    max_turns: int = 50,
) -> list[dict]:
    cfg.ensure_dirs()
    store = TraceStore(cfg.trace_db_path, backend=cfg.trace_db)
    tools, handlers = build_tool_registry(cfg, store)

    # add a meta-tool: run_search (delegates to production mode)
    tools.append({
        "type": "function",
        "function": {
            "name": "run_search",
            "description": "Launch MCGS production-mode search. Returns best pipeline + checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deployed_model_id": {"type": "string"},
                    "base_model": {"type": "string"},
                    "max_iterations": {"type": "integer", "default": 10},
                    "use_cot": {"type": "boolean", "default": False},
                    "eval_method": {"type": "string", "default": "exact_match"},
                    "task": {"type": "string", "default": "classification"},
                },
                "required": ["deployed_model_id", "base_model"],
            },
        },
    })

    def _run_search(args: dict) -> dict:
        return run_production(
            cfg=cfg,
            deployed_model_id=args["deployed_model_id"],
            base_model=args["base_model"],
            task=args.get("task", "classification"),
            use_mcgs=True,
            max_iterations=int(args.get("max_iterations", 10)),
            use_cot=bool(args.get("use_cot", False)),
            eval_method=args.get("eval_method", "exact_match"),
        )
    handlers["run_search"] = _run_search

    client = LLMClient(model=cfg.orchestrator_model, thinking_budget=cfg.thinking_budget)
    user_seed = {
        "request": user_request,
        "deployed_model_id": deployed_model_id,
        "base_model": base_model or cfg.base_model_default,
    }
    convo = chat_with_tools(
        client,
        messages=[{"role": "user", "content": json.dumps(user_seed, default=str)}],
        tools=tools,
        tool_handlers=handlers,
        system=SYSTEM,
        max_iters=max_turns,
    )
    return convo
