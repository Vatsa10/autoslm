-- Production inference store. Mirrors paper Section 2.6 Eq.10/12.
-- Backend: DuckDB or SQLite (same DDL works on both).

CREATE TABLE IF NOT EXISTS inferences (
    id              TEXT PRIMARY KEY,
    user_id         TEXT,
    model_id        TEXT NOT NULL,            -- base ckpt id or fine-tune UUID
    task            TEXT NOT NULL,            -- intent | ner | summarize | code | qa | reasoning
    input           TEXT NOT NULL,            -- x_i
    prediction      TEXT,                     -- y_hat_i (raw model output)
    gold            TEXT,                     -- y*_i (judge-corrected or human)
    verdict         TEXT NOT NULL,            -- 'pass' | 'fail'
    judge_model     TEXT,                     -- e.g. deepseek-reasoner / token_f1
    judge_reason    TEXT,                     -- r_i
    judge_score     DOUBLE,
    metadata        JSON,                     -- m_i: prompt template, criteria, perturbations_applied
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployment_stage INTEGER                  -- AdaptFT-Bench stage 0..3
);

CREATE INDEX IF NOT EXISTS idx_inferences_verdict   ON inferences(verdict);
CREATE INDEX IF NOT EXISTS idx_inferences_model     ON inferences(model_id);
CREATE INDEX IF NOT EXISTS idx_inferences_task      ON inferences(task);
CREATE INDEX IF NOT EXISTS idx_inferences_stage     ON inferences(deployment_stage);

-- Lineage of fine-tuned models. Lets the agent know parent dataset (Section 2.6 step 4).
CREATE TABLE IF NOT EXISTS model_lineage (
    model_id        TEXT PRIMARY KEY,
    base_model      TEXT NOT NULL,
    parent_model_id TEXT,                     -- previous fine-tune (null = base)
    dataset_path    TEXT,                     -- D_parent
    pipeline_json   JSON,                     -- pi = (D, H, S) snapshot
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Failure clusters from taxonomy step (cached so subsequent iterations reuse).
CREATE TABLE IF NOT EXISTS failure_clusters (
    cluster_id      TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    label           TEXT NOT NULL,            -- human-readable name
    fixability      TEXT NOT NULL,            -- 'fixable' | 'external' | 'poison'
    size            INTEGER,
    description     TEXT,
    representative_ids JSON,                  -- list of inference ids
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-label historical accuracy (paper Section 2.7 confidence calibration).
CREATE TABLE IF NOT EXISTS label_stats (
    model_id      TEXT NOT NULL,
    label         TEXT NOT NULL,
    n_total       INTEGER NOT NULL DEFAULT 0,
    n_correct     INTEGER NOT NULL DEFAULT 0,
    last_updated  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, label)
);

-- Eval history for cross-checkpoint tracking.
CREATE TABLE IF NOT EXISTS eval_history (
    id            TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL,
    eval_set_path TEXT,
    score         DOUBLE,
    regressions   INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
