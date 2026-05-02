"""Persistent data-curation.md audit log (paper Section 2.1).

Survives context compaction: agent can re-read at any point. Records
dataset versions, composition ratios, quality-control decisions, and
per-iteration evaluation results.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class AuditLog:
    """Append-only markdown log. Each call to `section()` writes a header + body."""

    def __init__(self, path: str | Path, run_id: str = "", mode: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(self._header(run_id, mode), encoding="utf-8")

    @staticmethod
    def _header(run_id: str, mode: str) -> str:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "")
        return (
            f"# autoslm — data curation log\n\n"
            f"- **run_id:** `{run_id}`\n"
            f"- **mode:** `{mode}`\n"
            f"- **started:** {ts}Z\n\n"
            f"---\n\n"
        )

    def section(self, title: str, body: str = "") -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "")
        block = f"## {title}\n_{ts}Z_\n\n{body}\n\n---\n\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(block)

    def append_raw(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)
