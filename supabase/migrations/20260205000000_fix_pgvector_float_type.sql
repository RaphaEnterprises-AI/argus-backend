-- ============================================================================
-- Fix pgvector type mismatch: FLOAT (real) -> DOUBLE PRECISION
-- Migration: 20260205000000_fix_pgvector_float_type.sql
-- ============================================================================
--
-- CRITICAL FIX: pgvector's <=> operator returns double precision, but our
-- functions were declaring FLOAT (which PostgreSQL maps to 'real').
-- This caused: "Returned type real does not match expected type double precision"
--
-- Solution: Change all similarity/score columns from FLOAT to DOUBLE PRECISION
-- ============================================================================

-- ============================================================================
-- 1. Fix search_similar_failures function
-- ============================================================================
DROP FUNCTION IF EXISTS search_similar_failures(vector(1536), FLOAT, INT);

CREATE OR REPLACE FUNCTION search_similar_failures(
    query_embedding vector(1536),
    match_threshold DOUBLE PRECISION DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    error_message TEXT,
    healed_selector TEXT,
    healing_method TEXT,
    success_rate DOUBLE PRECISION,
    similarity DOUBLE PRECISION
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
            THEN tfp.success_count::DOUBLE PRECISION / (tfp.success_count + tfp.failure_count)
            ELSE 0::DOUBLE PRECISION
        END AS success_rate,
        (1 - (tfp.embedding <=> query_embedding))::DOUBLE PRECISION AS similarity
    FROM test_failure_patterns tfp
    WHERE tfp.embedding IS NOT NULL
      AND 1 - (tfp.embedding <=> query_embedding) > match_threshold
    ORDER BY tfp.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 2. Fix search_memory_store function
-- ============================================================================
DROP FUNCTION IF EXISTS search_memory_store(vector(1536), TEXT[], FLOAT, INT);

CREATE OR REPLACE FUNCTION search_memory_store(
    query_embedding vector(1536),
    search_namespace TEXT[],
    match_threshold DOUBLE PRECISION DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    key TEXT,
    value JSONB,
    similarity DOUBLE PRECISION
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        lms.id,
        lms.key,
        lms.value,
        (1 - (lms.embedding <=> query_embedding))::DOUBLE PRECISION AS similarity
    FROM langgraph_memory_store lms
    WHERE lms.embedding IS NOT NULL
      AND lms.namespace = search_namespace
      AND 1 - (lms.embedding <=> query_embedding) > match_threshold
    ORDER BY lms.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 3. Fix search_failure_patterns_bm25 function
-- ============================================================================
DROP FUNCTION IF EXISTS search_failure_patterns_bm25(TEXT, FLOAT, INT, TEXT);

CREATE OR REPLACE FUNCTION search_failure_patterns_bm25(
    query_text TEXT,
    match_threshold DOUBLE PRECISION DEFAULT 0.0,
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
    success_rate DOUBLE PRECISION,
    bm25_rank DOUBLE PRECISION
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
            THEN tfp.success_count::DOUBLE PRECISION / (tfp.success_count + tfp.failure_count)
            ELSE 0::DOUBLE PRECISION
        END AS success_rate,
        ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text))::DOUBLE PRECISION AS bm25_rank
    FROM test_failure_patterns tfp
    WHERE tfp.error_message_tsv @@ websearch_to_tsquery('english', query_text)
        AND (error_type_filter IS NULL OR tfp.error_type = error_type_filter)
        AND ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text)) > match_threshold
    ORDER BY bm25_rank DESC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 4. Fix search_healing_patterns_bm25 function
-- ============================================================================
DROP FUNCTION IF EXISTS search_healing_patterns_bm25(TEXT, FLOAT, INT);

CREATE OR REPLACE FUNCTION search_healing_patterns_bm25(
    query_text TEXT,
    match_threshold DOUBLE PRECISION DEFAULT 0.0,
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
    bm25_rank DOUBLE PRECISION
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
        ts_rank_cd(hp.original_selector_tsv, websearch_to_tsquery('english', query_text))::DOUBLE PRECISION AS bm25_rank
    FROM healing_patterns hp
    WHERE hp.original_selector_tsv @@ websearch_to_tsquery('english', query_text)
        AND ts_rank_cd(hp.original_selector_tsv, websearch_to_tsquery('english', query_text)) > match_threshold
    ORDER BY bm25_rank DESC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 5. Fix search_memory_store_bm25 function
-- ============================================================================
DROP FUNCTION IF EXISTS search_memory_store_bm25(TEXT, TEXT[], FLOAT, INT);

CREATE OR REPLACE FUNCTION search_memory_store_bm25(
    query_text TEXT,
    search_namespace TEXT[],
    match_threshold DOUBLE PRECISION DEFAULT 0.0,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    namespace TEXT[],
    key TEXT,
    value JSONB,
    bm25_rank DOUBLE PRECISION
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
        ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text))::DOUBLE PRECISION AS bm25_rank
    FROM langgraph_memory_store lms
    WHERE lms.value_tsv @@ websearch_to_tsquery('english', query_text)
        AND (search_namespace IS NULL OR lms.namespace = search_namespace)
        AND ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text)) > match_threshold
    ORDER BY bm25_rank DESC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 6. Fix search_failure_patterns_hybrid function
-- ============================================================================
DROP FUNCTION IF EXISTS search_failure_patterns_hybrid(TEXT, vector(1536), FLOAT, INT, TEXT, FLOAT, FLOAT);

CREATE OR REPLACE FUNCTION search_failure_patterns_hybrid(
    query_text TEXT,
    query_embedding vector(1536),
    match_threshold DOUBLE PRECISION DEFAULT 0.5,
    match_count INT DEFAULT 5,
    error_type_filter TEXT DEFAULT NULL,
    bm25_weight DOUBLE PRECISION DEFAULT 0.5,
    vector_weight DOUBLE PRECISION DEFAULT 0.5
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
    success_rate DOUBLE PRECISION,
    bm25_score DOUBLE PRECISION,
    vector_score DOUBLE PRECISION,
    hybrid_score DOUBLE PRECISION
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH bm25_results AS (
        SELECT
            tfp.id,
            ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text))::DOUBLE PRECISION AS bm25_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tfp.error_message_tsv, websearch_to_tsquery('english', query_text)) DESC) AS bm25_rank
        FROM test_failure_patterns tfp
        WHERE tfp.error_message_tsv @@ websearch_to_tsquery('english', query_text)
            AND (error_type_filter IS NULL OR tfp.error_type = error_type_filter)
    ),
    vector_results AS (
        SELECT
            tfp.id,
            (1 - (tfp.embedding <=> query_embedding))::DOUBLE PRECISION AS vector_score,
            ROW_NUMBER() OVER (ORDER BY tfp.embedding <=> query_embedding) AS vector_rank
        FROM test_failure_patterns tfp
        WHERE tfp.embedding IS NOT NULL
            AND (error_type_filter IS NULL OR tfp.error_type = error_type_filter)
    ),
    -- Reciprocal Rank Fusion (RRF) with k=60
    rrf_scores AS (
        SELECT
            COALESCE(bm25.id, vec.id) AS id,
            COALESCE(bm25.bm25_score, 0.0::DOUBLE PRECISION) AS bm25_score,
            COALESCE(vec.vector_score, 0.0::DOUBLE PRECISION) AS vector_score,
            (bm25_weight * (1.0 / (60 + COALESCE(bm25.bm25_rank, 999999)::DOUBLE PRECISION))) +
            (vector_weight * (1.0 / (60 + COALESCE(vec.vector_rank, 999999)::DOUBLE PRECISION))) AS hybrid_score
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
            THEN tfp.success_count::DOUBLE PRECISION / (tfp.success_count + tfp.failure_count)
            ELSE 0::DOUBLE PRECISION
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

-- ============================================================================
-- 7. Fix search_memory_store_hybrid function
-- ============================================================================
DROP FUNCTION IF EXISTS search_memory_store_hybrid(TEXT, vector(1536), TEXT[], FLOAT, INT, FLOAT, FLOAT);

CREATE OR REPLACE FUNCTION search_memory_store_hybrid(
    query_text TEXT,
    query_embedding vector(1536),
    search_namespace TEXT[],
    match_threshold DOUBLE PRECISION DEFAULT 0.5,
    match_count INT DEFAULT 5,
    bm25_weight DOUBLE PRECISION DEFAULT 0.5,
    vector_weight DOUBLE PRECISION DEFAULT 0.5
)
RETURNS TABLE (
    id UUID,
    namespace TEXT[],
    key TEXT,
    value JSONB,
    bm25_score DOUBLE PRECISION,
    vector_score DOUBLE PRECISION,
    hybrid_score DOUBLE PRECISION
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH bm25_results AS (
        SELECT
            lms.id,
            ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text))::DOUBLE PRECISION AS bm25_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(lms.value_tsv, websearch_to_tsquery('english', query_text)) DESC) AS bm25_rank
        FROM langgraph_memory_store lms
        WHERE lms.value_tsv @@ websearch_to_tsquery('english', query_text)
            AND (search_namespace IS NULL OR lms.namespace = search_namespace)
    ),
    vector_results AS (
        SELECT
            lms.id,
            (1 - (lms.embedding <=> query_embedding))::DOUBLE PRECISION AS vector_score,
            ROW_NUMBER() OVER (ORDER BY lms.embedding <=> query_embedding) AS vector_rank
        FROM langgraph_memory_store lms
        WHERE lms.embedding IS NOT NULL
            AND (search_namespace IS NULL OR lms.namespace = search_namespace)
    ),
    rrf_scores AS (
        SELECT
            COALESCE(bm25.id, vec.id) AS id,
            COALESCE(bm25.bm25_score, 0.0::DOUBLE PRECISION) AS bm25_score,
            COALESCE(vec.vector_score, 0.0::DOUBLE PRECISION) AS vector_score,
            (bm25_weight * (1.0 / (60 + COALESCE(bm25.bm25_rank, 999999)::DOUBLE PRECISION))) +
            (vector_weight * (1.0 / (60 + COALESCE(vec.vector_rank, 999999)::DOUBLE PRECISION))) AS hybrid_score
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
-- 8. Update comments
-- ============================================================================
COMMENT ON FUNCTION search_similar_failures IS 'Search for similar failure patterns using vector similarity (DOUBLE PRECISION)';
COMMENT ON FUNCTION search_memory_store IS 'Search memory store by namespace using vector similarity (DOUBLE PRECISION)';
COMMENT ON FUNCTION search_failure_patterns_bm25 IS 'BM25 keyword search on test failure patterns (DOUBLE PRECISION)';
COMMENT ON FUNCTION search_healing_patterns_bm25 IS 'BM25 keyword search on healing patterns (DOUBLE PRECISION)';
COMMENT ON FUNCTION search_memory_store_bm25 IS 'BM25 keyword search on memory store (DOUBLE PRECISION)';
COMMENT ON FUNCTION search_failure_patterns_hybrid IS 'Hybrid search (BM25 + Vector + RRF) on test failure patterns (DOUBLE PRECISION)';
COMMENT ON FUNCTION search_memory_store_hybrid IS 'Hybrid search (BM25 + Vector + RRF) on memory store (DOUBLE PRECISION)';
