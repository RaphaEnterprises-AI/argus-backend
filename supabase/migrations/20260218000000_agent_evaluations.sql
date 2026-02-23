-- Agent Reliability Evaluation & Benchmarking System
-- Stores per-execution metrics and aggregated benchmarks for all AI agents.
-- Based on the CLEAR framework (Cost, Latency, Efficacy, Assurance, Reliability).

-- =============================================================================
-- Table: agent_evaluations
-- Stores individual agent execution results with structured metrics.
-- =============================================================================
CREATE TABLE IF NOT EXISTS agent_evaluations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id UUID NOT NULL,
  project_id UUID,
  agent_type TEXT NOT NULL,
  task_type TEXT NOT NULL,

  -- CLEAR Framework Dimensions
  task_completed BOOLEAN NOT NULL,
  efficacy_score DOUBLE PRECISION,
  latency_ms INTEGER,
  cost_usd DOUBLE PRECISION,
  assurance_score DOUBLE PRECISION,

  -- Detailed Metrics
  input_tokens INTEGER,
  output_tokens INTEGER,
  tool_calls_count INTEGER,
  self_corrections INTEGER DEFAULT 0,
  human_escalated BOOLEAN DEFAULT FALSE,
  error_type TEXT,
  error_message TEXT,

  -- Context
  trigger_source TEXT,
  execution_trace JSONB,
  metadata JSONB,

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_eval_org_agent
  ON agent_evaluations(organization_id, agent_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_task_type
  ON agent_evaluations(task_type, task_completed, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_created
  ON agent_evaluations(created_at DESC);

-- =============================================================================
-- Table: agent_benchmarks
-- Stores aggregated daily/weekly benchmark results per agent type.
-- =============================================================================
CREATE TABLE IF NOT EXISTS agent_benchmarks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id UUID,
  agent_type TEXT NOT NULL,
  period_start DATE NOT NULL,
  period_end DATE NOT NULL,

  -- Pass@k metrics
  total_executions INTEGER NOT NULL,
  pass_at_1 DOUBLE PRECISION,
  pass_at_3 DOUBLE PRECISION,
  pass_at_8 DOUBLE PRECISION,

  -- CLEAR aggregates
  avg_efficacy DOUBLE PRECISION,
  avg_latency_ms DOUBLE PRECISION,
  p50_latency_ms INTEGER,
  p95_latency_ms INTEGER,
  p99_latency_ms INTEGER,
  avg_cost_usd DOUBLE PRECISION,
  cost_normalized_accuracy DOUBLE PRECISION,
  avg_assurance DOUBLE PRECISION,

  -- Trust metrics
  self_correction_rate DOUBLE PRECISION,
  human_escalation_rate DOUBLE PRECISION,
  hallucination_rate DOUBLE PRECISION,
  false_positive_rate DOUBLE PRECISION,

  -- SRE-specific
  mttr_seconds DOUBLE PRECISION,
  mttd_seconds DOUBLE PRECISION,
  automated_resolution_rate DOUBLE PRECISION,

  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(organization_id, agent_type, period_start)
);

CREATE INDEX IF NOT EXISTS idx_bench_org_agent
  ON agent_benchmarks(organization_id, agent_type, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_bench_global
  ON agent_benchmarks(agent_type, period_start DESC)
  WHERE organization_id IS NULL;

-- Enable RLS
ALTER TABLE agent_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_benchmarks ENABLE ROW LEVEL SECURITY;

-- RLS policies: users can only see their own org's data
CREATE POLICY "Users can view own org evaluations"
  ON agent_evaluations FOR SELECT
  USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id'
         OR current_setting('role', true) = 'service_role');

CREATE POLICY "Service role can insert evaluations"
  ON agent_evaluations FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Users can view own org benchmarks"
  ON agent_benchmarks FOR SELECT
  USING (organization_id IS NULL
         OR organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id'
         OR current_setting('role', true) = 'service_role');

CREATE POLICY "Service role can manage benchmarks"
  ON agent_benchmarks FOR ALL
  USING (current_setting('role', true) = 'service_role');
