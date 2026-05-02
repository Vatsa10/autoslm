"""Trace store: DuckDB by default; SQLite fallback. Implements query_traces tool semantics."""
from __future__ import annotations
import json
import subprocess
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class TraceRecord:
    input: str
    prediction: Optional[str]
    gold: Optional[str]
    verdict: str                     # 'pass' | 'fail'
    model_id: str
    task: str
    judge_model: Optional[str] = None
    judge_reason: Optional[str] = None
    judge_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    deployment_stage: Optional[int] = None
    user_id: Optional[str] = None
    id: Optional[str] = None

    def ensure_id(self) -> str:
        if not self.id:
            self.id = str(uuid.uuid4())
        return self.id


SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


class TraceStore:
    def __init__(self, db_path: str | Path, backend: str = "duckdb"):
        self.db_path = str(db_path)
        self.backend = backend
        self._conn = None
        self._init_db()

    def _connect(self):
        if self._conn is not None:
            return self._conn
        if self.backend == "duckdb":
            import duckdb
            self._conn = duckdb.connect(self.db_path)
        else:
            import sqlite3
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL;")
        return self._conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        for stmt in [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]:
            try:
                conn.execute(stmt)
            except Exception:
                pass

    # ---- writes ----

    def insert_many(self, records: Iterable[TraceRecord]) -> int:
        conn = self._connect()
        n = 0
        for r in records:
            r.ensure_id()
            md = json.dumps(r.metadata or {}, default=str)
            conn.execute(
                """INSERT OR REPLACE INTO inferences
                   (id,user_id,model_id,task,input,prediction,gold,verdict,
                    judge_model,judge_reason,judge_score,metadata,deployment_stage)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""" if self.backend == "sqlite" else
                """INSERT OR REPLACE INTO inferences
                   (id,user_id,model_id,task,input,prediction,gold,verdict,
                    judge_model,judge_reason,judge_score,metadata,deployment_stage)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?::JSON,?)""",
                (r.id, r.user_id, r.model_id, r.task, r.input, r.prediction, r.gold,
                 r.verdict, r.judge_model, r.judge_reason, r.judge_score, md, r.deployment_stage),
            )
            n += 1
        return n

    def record_lineage(self, model_id: str, base_model: str,
                       parent_model_id: Optional[str], dataset_path: Optional[str],
                       pipeline_json: Optional[dict]) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO model_lineage VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (model_id, base_model, parent_model_id, dataset_path,
             json.dumps(pipeline_json or {}, default=str)),
        )

    # ---- reads ----

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._connect()
        cur = conn.execute(sql, params)
        if self.backend == "duckdb":
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fail_set(self, model_id: str, limit: int = 10_000) -> list[dict]:
        return self.query(
            "SELECT * FROM inferences WHERE verdict='fail' AND model_id=? LIMIT ?",
            (model_id, limit),
        )

    def pass_set(self, model_id: str, limit: int = 10_000) -> list[dict]:
        return self.query(
            "SELECT * FROM inferences WHERE verdict='pass' AND model_id=? LIMIT ?",
            (model_id, limit),
        )

    def lineage(self, model_id: str) -> Optional[dict]:
        rows = self.query("SELECT * FROM model_lineage WHERE model_id=?", (model_id,))
        return rows[0] if rows else None

    # ---- query_traces tool semantics: SQL + optional bash post-processing ----

    def query_with_pipeline(self, sql: str, bash_pipeline: Optional[str] = None,
                            workdir: str | Path = "/tmp") -> dict:
        """Mirrors paper Listing 1: SQL result piped to bash, only stdout returns to agent.

        Returns {'summary': stdout_tail, 'rows_returned': n, 'pipeline_used': bool}.
        Full result NOT loaded into context; agent reads disk side-effects.
        """
        rows = self.query(sql)
        result = {"rows_returned": len(rows), "pipeline_used": bool(bash_pipeline)}
        if not bash_pipeline:
            result["sample"] = rows[:25]
            return result
        try:
            payload = json.dumps(rows, default=str)
            proc = subprocess.run(
                bash_pipeline, input=payload, shell=True, capture_output=True,
                text=True, timeout=120, cwd=str(workdir),
            )
            tail = proc.stdout[-8000:]
            result["summary"] = tail
            result["stderr"] = proc.stderr[-2000:] if proc.stderr else ""
            result["returncode"] = proc.returncode
        except Exception as e:
            result["error"] = str(e)
        return result
