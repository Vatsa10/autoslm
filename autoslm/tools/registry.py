"""Agent tools (paper Section 2.5/2.6).

Production tools:
- query_traces (SQL + optional bash post-process)
- bash (sandboxed shell)
- read_file / edit_file
- delegate_task (sub-agent spawn)
- web_search (cold-start)

Schemas follow OpenAI/Anthropic tool-use spec via LiteLLM.
"""
from __future__ import annotations
import json
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..config import AutoSLMConfig
from ..traces import TraceStore


def tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_traces",
                "description": "SQL query against production inference store. "
                               "Optional bash_pipeline post-processes JSON rows; only stdout summary "
                               "returns to context (mirrors paper Listing 1).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {"type": "string"},
                        "bash_pipeline": {"type": "string"},
                    },
                    "required": ["sql_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run shell command in sandboxed working directory. "
                               "Output truncated to 8KB.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"},
                                   "timeout": {"type": "integer", "default": 120}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file. Returns up to 8KB of content.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"},
                                   "offset": {"type": "integer", "default": 0},
                                   "max_bytes": {"type": "integer", "default": 8192}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Write or replace contents of a file. Creates parent dirs.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "Spawn a sub-agent (Trace Analyzer style) for data-heavy work. "
                               "Sub-agent runs with 10x token limit and writes summary to disk.",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"},
                                   "subagent": {"type": "string",
                                                "enum": ["trace_analyzer", "data_synth", "eval_runner"]}},
                    "required": ["task", "subagent"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web (Exa or Tavily). Returns top results as JSON.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"},
                                   "k": {"type": "integer", "default": 5}},
                    "required": ["query"],
                },
            },
        },
    ]


def default_handlers(cfg: AutoSLMConfig, store: TraceStore,
                    sandbox_dir: Optional[Path] = None,
                    delegate: Optional[Callable[[str, str], dict]] = None,
                    web_search: Optional[Callable[[str, int], list]] = None,
                    ) -> dict[str, Callable]:
    sandbox = Path(sandbox_dir or cfg.workdir / "sandbox")
    sandbox.mkdir(parents=True, exist_ok=True)

    def _query_traces(args: dict) -> dict:
        return store.query_with_pipeline(
            args["sql_query"], args.get("bash_pipeline"), workdir=sandbox,
        )

    def _bash(args: dict) -> dict:
        cmd = args["command"]
        timeout = int(args.get("timeout", 120))
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(sandbox),
            )
            return {
                "stdout": proc.stdout[-8192:],
                "stderr": proc.stderr[-2048:],
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s"}

    def _read_file(args: dict) -> dict:
        p = Path(args["path"])
        if not p.exists():
            return {"error": "not_found"}
        offset = int(args.get("offset", 0))
        n = int(args.get("max_bytes", 8192))
        try:
            data = p.read_bytes()[offset:offset + n]
            return {"path": str(p), "size": p.stat().st_size,
                    "content": data.decode("utf-8", errors="replace")}
        except Exception as e:
            return {"error": str(e)}

    def _edit_file(args: dict) -> dict:
        p = Path(args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return {"path": str(p), "bytes_written": len(args["content"])}

    def _delegate(args: dict) -> dict:
        if delegate is None:
            return {"error": "delegate not configured"}
        return delegate(args["task"], args["subagent"])

    def _web(args: dict) -> dict:
        if web_search is None:
            return {"error": "web_search not configured"}
        return {"results": web_search(args["query"], int(args.get("k", 5)))}

    return {
        "query_traces": _query_traces,
        "bash": _bash,
        "read_file": _read_file,
        "edit_file": _edit_file,
        "delegate_task": _delegate,
        "web_search": _web,
    }


def build_tool_registry(cfg: AutoSLMConfig, store: TraceStore, **kw):
    return tool_specs(), default_handlers(cfg, store, **kw)
