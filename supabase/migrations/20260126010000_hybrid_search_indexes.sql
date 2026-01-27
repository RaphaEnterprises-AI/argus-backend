-- ============================================================================
-- Hybrid Search: BM25 (Full-Text Search) + Vector Search Indexes
-- Migration: 20260126000000_hybrid_search_indexes.sql
-- ============================================================================
--
-- This migration adds PostgreSQL Full-Text Search (BM25) capabilities
-- alongside existing pgvector semantic search for hybrid retrieval.
--
-- Why Hybrid Search?
-- - BM25 excels at keyword/term matching (exact error messages)
-- - Vector search excels at semantic similarity (similar concepts)
-- - Combining both outperforms either alone (industry standard)
--
-- References:
-- - Google Vertex AI: Hybrid search is the default
-- - Microsoft GraphRAG: Combines vector + semantic ranking + graph
-- - Weaviate/Qdrant: Native hybrid search is a key feature
-- ============================================================================

-- ============================================================================
-- 1. Add tsvector columns for Full-Text Search
-- ============================================================================

-- Add tsvector column to test_failure_patterns (error_message)
ALTER TABLE test_failure_patterns
    ADD COLUMN IF NOT EXISTS error_message_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(error_message, ''))
    ) STORED;

-- Add tsvector column for selector (original selector)
ALTER TABLE test_failure_patterns
    ADD COLUMN IF NOT EXISTS selector_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(selector, ''))
    ) STORED;

-- Add tsvector column to healing_patterns
ALTER TABLE healing_patterns
    ADD COLUMN IF NOT EXISTS original_selector_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(original_selector, ''))
    ) STORED;

ALTER TABLE healing_patterns
    ADD COLUMN IF NOT EXISTS healed_selector_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(healed_selector, ''))
    ) STORED;

-- Add tsvector column to langgraph_memory_store
-- Extract text from JSONB value for full-text indexing
ALTER TABLE langgraph_memory_store
    ADD COLUMN IF NOT EXISTS value_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english',
            COALESCE(value->>'text', '') || ' ' ||
            COALESCE(value->>'error_message', '') || ' ' ||
            COALESCE(value->>'selector', '') || ' ' ||
            COALESCE(value->>'description', '')
        )
    ) STORED;

-- ============================================================================
-- 2. Create GIN indexes for fast Full-Text Search (BM25-like ranking)
-- ============================================================================

-- Index for test_failure_patterns.error_message
CREATE INDEX IF NOT EXISTS idx_failure_patterns_error_tsv
    ON test_failure_patterns
    USING GIN (error_message_tsv);

-- Index for test_failure_patterns.selector
CREATE INDEX IF NOT EXISTS idx_failure_patterns_selector_tsv
    ON test_failure_patterns
    USING GIN (selector_tsv);

-- Index for healing_patterns.original_selector
CREATE INDEX IF NOT EXISTS idx_healing_patterns_original_selector_tsv
    ON healing_patterns
    USING GIN (original_selector_tsv);

-- Index for healing_patterns.healed_selector
CREATE INDEX IF NOT EXISTS idx_healing_patterns_healed_selector_tsv
    ON healing_patterns
    USING GIN (healed_selector_tsv);

-- Index for langgraph_memory_store.value
CREATE INDEX IF NOT EXISTS idx_memory_store_value_tsv
    ON langgraph_memory_store
    USING GIN (value_tsv);

-- ============================================================================
-- 3. Create BM25 search functions with ranking
-- ============================================================================

-- BM25 search for test_failure_patterns
CREATE OR REPLACE FUNCTION search_failure_patterns_bm25(
    query_text TEXT,
    match_threshold FLOAT DEFAULT 0.0,
    match_count INT DEFAULT 5,
    error_type_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    error_message TEXT,
    error_type TEXT,
    selector TEXT,
    healed_selector TEXT,
    healing_method TEXT,
    success_count INTEGER,
    failure_count INTEGER,
    success_rate FLOAT,
    bm25_rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tfp.id,
        tfp.error_message,
        tfp.error_type,
        tfp.selector,
        tfp.healed_selector,
        tfp.healing_method,
        tfp.success_count,
        tfp.failure_count,
        CASE
            WHEN (tfp.success_count + tfp.failure_count) > 0
            THEN tfp.success_count::FLOAT / (tfp.success_count + tfp.failure_count)
            ELSE 0
        END AS success_rate,
        ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text)) AS bm25_rank
    FROM test_failure_patterns tfp
    WHERE tfp.error_message_tsv @@ websearch_to_tsquery('english', query_text)
        AND (error_type_filter IS NULL OR tfp.error_type = error_type_filter)
        AND ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text)) > match_threshold
    ORDER BY bm25_rank DESC
    LIMIT match_count;
END;
$$;

-- BM25 search for healing_patterns
CREATE OR REPLACE FUNCTION search_healing_patterns_bm25(
    query_text TEXT,
    match_threshold FLOAT DEFAULT 0.0,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    fingerprint TEXT,
    original_selector TEXT,
    healed_selector TEXT,
    error_type TEXT,
    success_count INTEGER,
    failure_count INTEGER,
    confidence NUMERIC,
    bm25_rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        hp.id,
        hp.fingerprint,
        hp.original_selector,
        hp.healed_selector,
        hp.error_type,
        hp.success_count,
        hp.failure_count,
        hp.confidence,
        ts_rank_cd(hp.original_selector_tsv, websearch_to_tsquery('english', query_text)) AS bm25_rank
    FROM healing_patterns hp
    WHERE hp.original_selector_tsv @@ websearch_to_tsquery('english', query_text)
        AND ts_rank_cd(hp.original_selector_tsv, websearch_to_tsquery('english', query_text)) > match_threshold
    ORDER BY bm25_rank DESC
    LIMIT match_count;
END;
$$;

-- BM25 search for langgraph_memory_store
CREATE OR REPLACE FUNCTION search_memory_store_bm25(
    query_text TEXT,
    search_namespace TEXT[],
    match_threshold FLOAT DEFAULT 0.0,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    namespace TEXT[],
    key TEXT,
    value JSONB,
    bm25_rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        lms.id,
        lms.namespace,
        lms.key,
        lms.value,
        ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text)) AS bm25_rank
    FROM langgraph_memory_store lms
    WHERE lms.value_tsv @@ websearch_to_tsquery('english', query_text)
        AND (search_namespace IS NULL OR lms.namespace = search_namespace)
        AND ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text)) > match_threshold
    ORDER BY bm25_rank DESC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 4. Create hybrid search functions (BM25 + Vector with RRF)
-- ============================================================================

-- Hybrid search for test_failure_patterns
-- Combines BM25 (keyword) + pgvector (semantic) using Reciprocal Rank Fusion
CREATE OR REPLACE FUNCTION search_failure_patterns_hybrid(
    query_text TEXT,
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5,
    error_type_filter TEXT DEFAULT NULL,
    bm25_weight FLOAT DEFAULT 0.5,
    vector_weight FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    id UUID,
    error_message TEXT,
    error_type TEXT,
    selector TEXT,
    healed_selector TEXT,
    healing_method TEXT,
    success_count INTEGER,
    failure_count INTEGER,
    success_rate FLOAT,
    bm25_score FLOAT,
    vector_score FLOAT,
    hybrid_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH bm25_results AS (
        SELECT
            tfp.id,
            ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text)) AS bm25_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text)) DESC) AS bm25_rank
        FROM test_failure_patterns tfp
        WHERE tfp.error_message_tsv @@ websearch_to_tsquery('english', query_text)
            AND (error_type_filter IS NULL OR tfp.error_type = error_type_filter)
    ),
    vector_results AS (
        SELECT
            tfp.id,
            1 - (tfp.embedding <=> query_embedding) AS vector_score,
            ROW_NUMBER() OVER (ORDER BY tfp.embedding <=> query_embedding) AS vector_rank
        FROM test_failure_patterns tfp
        WHERE tfp.embedding IS NOT NULL
            AND (error_type_filter IS NULL OR tfp.error_type = error_type_filter)
    ),
    -- Reciprocal Rank Fusion (RRF)
    -- Score = 1 / (k + rank) where k is typically 60
    rrf_scores AS (
        SELECT
            COALESCE(bm25.id, vec.id) AS id,
            COALESCE(bm25.bm25_score, 0.0) AS bm25_score,
            COALESCE(vec.vector_score, 0.0) AS vector_score,
            -- RRF formula with configurable weights
            (bm25_weight * (1.0 / (60 + COALESCE(bm25.bm25_rank, 999999)))) +
            (vector_weight * (1.0 / (60 + COALESCE(vec.vector_rank, 999999)))) AS hybrid_score
        FROM bm25_results bm25
        FULL OUTER JOIN vector_results vec ON bm25.id = vec.id
    )
    SELECT
        tfp.id,
        tfp.error_message,
        tfp.error_type,
        tfp.selector,
        tfp.healed_selector,
        tfp.healing_method,
        tfp.success_count,
        tfp.failure_count,
        CASE
            WHEN (tfp.success_count + tfp.failure_count) > 0
            THEN tfp.success_count::FLOAT / (tfp.success_count + tfp.failure_count)
            ELSE 0
        END AS success_rate,
        rrf.bm25_score,
        rrf.vector_score,
        rrf.hybrid_score
    FROM rrf_scores rrf
    JOIN test_failure_patterns tfp ON rrf.id = tfp.id
    WHERE rrf.hybrid_score > match_threshold
    ORDER BY rrf.hybrid_score DESC
    LIMIT match_count;
END;
$$;

-- Hybrid search for langgraph_memory_store
CREATE OR REPLACE FUNCTION search_memory_store_hybrid(
    query_text TEXT,
    query_embedding vector(1536),
    search_namespace TEXT[],
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5,
    bm25_weight FLOAT DEFAULT 0.5,
    vector_weight FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    id UUID,
    namespace TEXT[],
    key TEXT,
    value JSONB,
    bm25_score FLOAT,
    vector_score FLOAT,
    hybrid_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH bm25_results AS (
        SELECT
            lms.id,
            ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text)) AS bm25_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text)) DESC) AS bm25_rank
        FROM langgraph_memory_store lms
        WHERE lms.value_tsv @@ websearch_to_tsquery('english', query_text)
            AND (search_namespace IS NULL OR lms.namespace = search_namespace)
    ),
    vector_results AS (
        SELECT
            lms.id,
            1 - (lms.embedding <=> query_embedding) AS vector_score,
            ROW_NUMBER() OVER (ORDER BY lms.embedding <=> query_embedding) AS vector_rank
        FROM langgraph_memory_store lms
        WHERE lms.embedding IS NOT NULL
            AND (search_namespace IS NULL OR lms.namespace = search_namespace)
    ),
    rrf_scores AS (
        SELECT
            COALESCE(bm25.id, vec.id) AS id,
            COALESCE(bm25.bm25_score, 0.0) AS bm25_score,
            COALESCE(vec.vector_score, 0.0) AS vector_score,
            (bm25_weight * (1.0 / (60 + COALESCE(bm25.bm25_rank, 999999)))) +
            (vector_weight * (1.0 / (60 + COALESCE(vec.vector_rank, 999999)))) AS hybrid_score
        FROM bm25_results bm25
        FULL OUTER JOIN vector_results vec ON bm25.id = vec.id
    )
    SELECT
        lms.id,
        lms.namespace,
        lms.key,
        lms.value,
        rrf.bm25_score,
        rrf.vector_score,
        rrf.hybrid_score
    FROM rrf_scores rrf
    JOIN langgraph_memory_store lms ON rrf.id = lms.id
    WHERE rrf.hybrid_score > match_threshold
    ORDER BY rrf.hybrid_score DESC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 5. Add comments for documentation
-- ============================================================================

COMMENT ON FUNCTION search_failure_patterns_bm25 IS 'BM25 keyword search on test failure patterns';
COMMENT ON FUNCTION search_healing_patterns_bm25 IS 'BM25 keyword search on healing patterns';
COMMENT ON FUNCTION search_memory_store_bm25 IS 'BM25 keyword search on memory store';
COMMENT ON FUNCTION search_failure_patterns_hybrid IS 'Hybrid search (BM25 + Vector + RRF) on test failure patterns';
COMMENT ON FUNCTION search_memory_store_hybrid IS 'Hybrid search (BM25 + Vector + RRF) on memory store';

COMMENT ON COLUMN test_failure_patterns.error_message_tsv IS 'Full-text search vector for error_message (BM25)';
COMMENT ON COLUMN test_failure_patterns.selector_tsv IS 'Full-text search vector for selector (BM25)';
COMMENT ON COLUMN healing_patterns.original_selector_tsv IS 'Full-text search vector for original_selector (BM25)';
COMMENT ON COLUMN healing_patterns.healed_selector_tsv IS 'Full-text search vector for healed_selector (BM25)';
COMMENT ON COLUMN langgraph_memory_store.value_tsv IS 'Full-text search vector for JSONB value (BM25)';
