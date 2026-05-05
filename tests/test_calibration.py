"""Tests for confidence calibration + TF-IDF propagation (paper Section 2.7)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from autoslm.traces.calibration import (
    calibrate, LabelStats, Correction,
    update_label_stats, get_label_stats, log_eval_history,
    propagate_correction, on_human_override,
)


def _make_conn():
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    conn = sqlite3.connect(db.name)
    conn.execute("""CREATE TABLE IF NOT EXISTS label_stats (
        model_id TEXT NOT NULL, label TEXT NOT NULL,
        n_total INTEGER DEFAULT 0, n_correct INTEGER DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (model_id, label))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_history (
        id TEXT PRIMARY KEY, model_id TEXT, eval_set_path TEXT,
        score DOUBLE, regressions INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    return conn, db.name


def test_calibrate_no_history():
    """Calibration with no history returns weight-blended value."""
    result = calibrate(0.9, "label_a", None, weight=0.5)
    # 0.5 * 0.9 + 0.5 * 0.5 = 0.7
    assert abs(result - 0.7) < 1e-6


def test_calibrate_with_history():
    stats = LabelStats(model_id="m1", label="l1", n_total=10, n_correct=8)
    # accuracy = 0.8; calibrated = 0.5*0.9 + 0.5*0.8 = 0.85
    result = calibrate(0.9, "l1", stats, weight=0.5)
    assert abs(result - 0.85) < 1e-6


def test_update_label_stats():
    conn, path = _make_conn()
    try:
        stats = update_label_stats(conn, "model1", "intent_a", correct=True)
        assert stats.n_total == 1
        assert stats.n_correct == 1
        assert stats.accuracy == 1.0

        stats2 = update_label_stats(conn, "model1", "intent_a", correct=False)
        assert stats2.n_total == 2
        assert stats2.n_correct == 1
        assert stats2.accuracy == 0.5
    finally:
        conn.close()
        Path(path).unlink(missing_ok=True)


def test_get_label_stats():
    conn, path = _make_conn()
    try:
        update_label_stats(conn, "model1", "intent_a", correct=True)
        stats = get_label_stats(conn, "model1", "intent_a")
        assert stats is not None
        assert stats.n_total == 1
        assert stats.n_correct == 1

        missing = get_label_stats(conn, "model1", "nonexistent")
        assert missing is None
    finally:
        conn.close()
        Path(path).unlink(missing_ok=True)


def test_log_eval_history():
    conn, path = _make_conn()
    try:
        rid = log_eval_history(conn, "model1", "eval_set.json", 0.956, 1)
        assert rid is not None
        cur = conn.execute("SELECT score, regressions FROM eval_history WHERE id=?", (rid,))
        row = cur.fetchone()
        assert row[0] == 0.956
        assert row[1] == 1
    finally:
        conn.close()
        Path(path).unlink(missing_ok=True)


def test_propagate_correction():
    corr = Correction(
        record_id="r1", model_id="m1", label="intent_a",
        old_prediction="wrong", new_gold="correct",
    )
    pool = [
        {"id": "r2", "input": "similar to correct answer"},
        {"id": "r3", "input": "completely different question"},
        {"id": "r4", "input": "correct answer here"},
    ]
    ids = propagate_correction(corr, pool, k=2)
    assert isinstance(ids, list)
    # TF-IDF should find similar inputs
    assert len(ids) <= 2


def test_on_human_override():
    """Test the hook for TraceStore.insert_many."""
    conn, path = _make_conn()
    try:
        # Mock TraceStore
        class MockStore:
            def _connect(self):
                return conn
        store = MockStore()
        propagated = on_human_override(
            store, "r1", "m1", "intent_a", "wrong", "correct",
        )
        # Label stats should be updated
        stats = get_label_stats(conn, "m1", "intent_a")
        assert stats is not None
        assert stats.n_correct == 1
    finally:
        conn.close()
        Path(path).unlink(missing_ok=True)
