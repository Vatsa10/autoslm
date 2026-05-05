"""Confidence calibration + TF-IDF correction propagation (paper Section 2.7).

Tracks per-label historical accuracy to adjust model confidence scores.
When human overrides arrive, propagates corrections to TF-IDF-similar pending items.

Tables:
  - LabelStats: (model_id, label, n_total, n_correct, last_updated)
  - eval_history: (model_id, eval_set_path, score, regressions, timestamp)
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Schema additions (merge into traces/schema.sql via migration)
# ---------------------------------------------------------------------------

LABEL_STATS_DDL = """
CREATE TABLE IF NOT EXISTS label_stats (
    model_id      TEXT NOT NULL,
    label         TEXT NOT NULL,
    n_total       INTEGER NOT NULL DEFAULT 0,
    n_correct     INTEGER NOT NULL DEFAULT 0,
    last_updated  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, label)
);
"""

EVAL_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS eval_history (
    id            TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL,
    eval_set_path TEXT,
    score         DOUBLE,
    regressions   INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LabelStats:
    model_id: str
    label: str
    n_total: int
    n_correct: int
    last_updated: Optional[str] = None

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0


@dataclass
class Correction:
    """A human override / correction to propagate."""
    record_id: str
    model_id: str
    label: str
    old_prediction: str
    new_gold: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(raw_conf: float, label: str, stats: Optional[LabelStats],
              weight: float = 0.5) -> float:
    """Adjust raw model confidence using per-label historical accuracy.

    paper Section 2.7: calibrated = weight * raw_conf + (1-weight) * label_accuracy.

    Args:
        raw_conf: model's raw confidence/probability for this prediction
        label: the predicted label (for classification) or task type
        stats: LabelStats row for this label (may be None if no history)
        weight: interpolation weight toward raw confidence

    Returns:
        Calibrated confidence in [0, 1]
    """
    label_acc = stats.accuracy if stats else 0.5  # neutral prior if unknown
    return weight * max(0.0, min(1.0, raw_conf)) + (1.0 - weight) * label_acc


# ---------------------------------------------------------------------------
# TF-IDF correction propagation
# ---------------------------------------------------------------------------

def _build_tfidf(corpus: list[str], max_features: int = 2048):
    """Build TF-IDF vectorizer and matrix from a corpus of texts."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             min_df=1, max_features=max_features)
        X = v.fit_transform(corpus)
        return v, X
    except ImportError:
        return None, None


def _cosine_top_k(query_vec, matrix, k: int = 10) -> list[int]:
    """Return indices of top-k rows in matrix most similar to query_vec."""
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(query_vec, matrix).flatten()
        # exclude self (highest similarity = 1.0) and get top k
        top_k_idx = np.argsort(sims)[::-1]
        top_k_idx = [i for i in top_k_idx if sims[i] < 0.999][:k]
        return list(top_k_idx)
    except ImportError:
        return []


def propagate_correction(
    correction: Correction,
    candidate_pool: list[dict],
    k: int = 10,
) -> list[str]:
    """Propagate a human correction to TF-IDF-similar pending items.

    Returns list of record IDs that should receive the same correction.

    Args:
        correction: the human override / correction to propagate
        candidate_pool: list of dicts with keys 'id', 'input', 'prediction', etc.
        k: number of nearest neighbors to return
    """
    if not candidate_pool:
        return []
    texts = [c.get("input") or "" for c in candidate_pool]
    query_text = correction.new_gold + " " + (correction.old_prediction or "")
    v, X = _build_tfidf([query_text] + texts)
    if v is None or X is None:
        return []
    query_vec = X[0:1]
    doc_matrix = X[1:]
    indices = _cosine_top_k(query_vec, doc_matrix, k=k)
    return [candidate_pool[i]["id"] for i in indices if i < len(candidate_pool)]


# ---------------------------------------------------------------------------
# DB helpers for LabelStats
# ---------------------------------------------------------------------------

def _init_calibration_tables(conn) -> None:
    """Create label_stats and eval_history tables if not present."""
    for ddl in (LABEL_STATS_DDL, EVAL_HISTORY_DDL):
        try:
            conn.execute(ddl)
        except Exception:
            pass


def update_label_stats(
    conn, model_id: str, label: str, correct: bool
) -> LabelStats:
    """Upsert label stats row, incrementing counters."""
    _init_calibration_tables(conn)
    cur = conn.execute(
        "SELECT n_total, n_correct FROM label_stats WHERE model_id=? AND label=?",
        (model_id, label),
    )
    row = cur.fetchone()
    if row:
        n_total, n_correct = row[0] + 1, row[1] + (1 if correct else 0)
        conn.execute(
            "UPDATE label_stats SET n_total=?, n_correct=?, last_updated=CURRENT_TIMESTAMP "
            "WHERE model_id=? AND label=?",
            (n_total, n_correct, model_id, label),
        )
    else:
        n_total, n_correct = 1, (1 if correct else 0)
        conn.execute(
            "INSERT INTO label_stats (model_id, label, n_total, n_correct) VALUES (?, ?, ?, ?)",
            (model_id, label, n_total, n_correct),
        )
    return LabelStats(model_id=model_id, label=label,
                      n_total=n_total, n_correct=n_correct)


def get_label_stats(conn, model_id: str, label: str) -> Optional[LabelStats]:
    _init_calibration_tables(conn)
    cur = conn.execute(
        "SELECT n_total, n_correct, last_updated FROM label_stats "
        "WHERE model_id=? AND label=?",
        (model_id, label),
    )
    row = cur.fetchone()
    if not row:
        return None
    return LabelStats(model_id=model_id, label=label,
                      n_total=row[0], n_correct=row[1], last_updated=row[2])


def log_eval_history(conn, model_id: str, eval_set_path: Optional[str],
                    score: float, regressions: int = 0) -> str:
    _init_calibration_tables(conn)
    rid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO eval_history (id, model_id, eval_set_path, score, regressions) "
        "VALUES (?, ?, ?, ?, ?)",
        (rid, model_id, eval_set_path, score, regressions),
    )
    return rid


# ---------------------------------------------------------------------------
# Hook for TraceStore.insert_many
# ---------------------------------------------------------------------------

def on_human_override(
    store,  # TraceStore
    record_id: str,
    model_id: str,
    label: str,
    old_prediction: str,
    new_gold: str,
    pending_pool: Optional[list[dict]] = None,
) -> list[str]:
    """Call this when a human override arrives.

    Updates label stats and optionally propagates correction to pending items.

    Returns list of propagated record IDs (empty if no propagation).
    """
    conn = store._connect()
    update_label_stats(conn, model_id, label, correct=True)
    if pending_pool:
        corr = Correction(record_id=record_id, model_id=model_id, label=label,
                          old_prediction=old_prediction, new_gold=new_gold)
        return propagate_correction(corr, pending_pool, k=10)
    return []
