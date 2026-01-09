-- LangGraph Checkpoints for Durable Execution
-- This enables test runs to survive server restarts

-- Checkpoints table (LangGraph PostgresSaver schema)
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT NOT NULL,
    checkpoint JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON langgraph_checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON langgraph_checkpoints(created_at DESC);

-- Checkpoint writes (for pending writes)
CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB,
    PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
);

-- Enable Row Level Security
ALTER TABLE langgraph_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE langgraph_checkpoint_writes ENABLE ROW LEVEL SECURITY;

-- Policy for service role access
CREATE POLICY "Service role full access to checkpoints" ON langgraph_checkpoints
    FOR ALL USING (true);
CREATE POLICY "Service role full access to checkpoint_writes" ON langgraph_checkpoint_writes
    FOR ALL USING (true);
