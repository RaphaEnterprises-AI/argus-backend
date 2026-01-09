-- Argus D1 Schema for Test History

CREATE TABLE IF NOT EXISTS test_runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    steps_total INTEGER DEFAULT 0,
    steps_passed INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    result_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_test_runs_project ON test_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_test_runs_created ON test_runs(created_at DESC);

-- Healing patterns tracking
CREATE TABLE IF NOT EXISTS healing_events (
    id TEXT PRIMARY KEY,
    test_run_id TEXT,
    original_selector TEXT,
    healed_selector TEXT,
    success INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_healing_events_test ON healing_events(test_run_id);
