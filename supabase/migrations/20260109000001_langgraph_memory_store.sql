-- LangGraph Memory Store for Long-Term Learning
-- Enables semantic search on failure patterns and healing solutions
-- Migration: 20260109000001_langgraph_memory_store.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Memory store table for general cross-session memory
CREATE TABLE IF NOT EXISTS langgraph_memory_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT[] NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    embedding vector(1536),  -- OpenAI embedding dimension
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(namespace, key)
);

-- HNSW index for fast semantic search on memory store
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON langgraph_memory_store
USING hnsw (embedding vector_cosine_ops);

-- Index for namespace queries
CREATE INDEX IF NOT EXISTS idx_memory_namespace ON langgraph_memory_store USING GIN (namespace);

-- Index for key lookups
CREATE INDEX IF NOT EXISTS idx_memory_key ON langgraph_memory_store(key);

-- Test failure patterns table (specialized memory for self-healing)
CREATE TABLE IF NOT EXISTS test_failure_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID,
    error_message TEXT NOT NULL,
    error_type TEXT,
    selector TEXT,
    healed_selector TEXT,
    healing_method TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for semantic search on failure patterns
CREATE INDEX IF NOT EXISTS idx_failure_patterns_embedding ON test_failure_patterns
USING hnsw (embedding vector_cosine_ops);

-- Index for error type filtering
CREATE INDEX IF NOT EXISTS idx_failure_patterns_error_type ON test_failure_patterns(error_type);

-- Index for selector lookups
CREATE INDEX IF NOT EXISTS idx_failure_patterns_selector ON test_failure_patterns(selector);

-- Function to search similar failures using cosine similarity
CREATE OR REPLACE FUNCTION search_similar_failures(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    error_message TEXT,
    healed_selector TEXT,
    healing_method TEXT,
    success_rate FLOAT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tfp.id,
        tfp.error_message,
        tfp.healed_selector,
        tfp.healing_method,
        CASE
            WHEN (tfp.success_count + tfp.failure_count) > 0
            THEN tfp.success_count::FLOAT / (tfp.success_count + tfp.failure_count)
            ELSE 0
        END AS success_rate,
        1 - (tfp.embedding <=> query_embedding) AS similarity
    FROM test_failure_patterns tfp
    WHERE tfp.embedding IS NOT NULL
      AND 1 - (tfp.embedding <=> query_embedding) > match_threshold
    ORDER BY tfp.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function to search memory store by semantic similarity
CREATE OR REPLACE FUNCTION search_memory_store(
    query_embedding vector(1536),
    search_namespace TEXT[],
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    key TEXT,
    value JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        lms.id,
        lms.key,
        lms.value,
        1 - (lms.embedding <=> query_embedding) AS similarity
    FROM langgraph_memory_store lms
    WHERE lms.embedding IS NOT NULL
      AND lms.namespace = search_namespace
      AND 1 - (lms.embedding <=> query_embedding) > match_threshold
    ORDER BY lms.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Trigger to update updated_at on changes
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_langgraph_memory_store_updated_at
    BEFORE UPDATE ON langgraph_memory_store
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_test_failure_patterns_updated_at
    BEFORE UPDATE ON test_failure_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable RLS for security
ALTER TABLE langgraph_memory_store ENABLE ROW LEVEL SECURITY;
ALTER TABLE test_failure_patterns ENABLE ROW LEVEL SECURITY;

-- Policies for service role access (full access for backend)
CREATE POLICY "Service role full access to memory_store" ON langgraph_memory_store
    FOR ALL USING (true);

CREATE POLICY "Service role full access to failure_patterns" ON test_failure_patterns
    FOR ALL USING (true);

-- Add comments for documentation
COMMENT ON TABLE langgraph_memory_store IS 'General-purpose memory store for LangGraph cross-session learning with semantic search';
COMMENT ON TABLE test_failure_patterns IS 'Specialized memory for test failure patterns and healing solutions';
COMMENT ON FUNCTION search_similar_failures IS 'Search for similar failure patterns using vector similarity';
COMMENT ON FUNCTION search_memory_store IS 'Search memory store by namespace using vector similarity';
