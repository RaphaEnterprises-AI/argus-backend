-- Migration: Add atomic counter functions to prevent race conditions
-- These functions use PostgreSQL's built-in atomic increment to avoid
-- read-modify-write race conditions in concurrent operations

-- =============================================================================
-- Atomic counter increment for discovery_patterns
-- =============================================================================

CREATE OR REPLACE FUNCTION increment_pattern_times_seen(pattern_id UUID)
RETURNS TABLE(id UUID, times_seen INTEGER, updated_at TIMESTAMPTZ)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    UPDATE discovery_patterns
    SET times_seen = discovery_patterns.times_seen + 1,
        updated_at = NOW()
    WHERE discovery_patterns.id = pattern_id
    RETURNING discovery_patterns.id, discovery_patterns.times_seen, discovery_patterns.updated_at;
END;
$$;

-- =============================================================================
-- Atomic counter increment for healing_patterns success_count
-- =============================================================================

CREATE OR REPLACE FUNCTION increment_healing_success(pattern_id UUID)
RETURNS TABLE(id UUID, success_count INTEGER, failure_count INTEGER, updated_at TIMESTAMPTZ)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    UPDATE healing_patterns
    SET success_count = healing_patterns.success_count + 1,
        updated_at = NOW()
    WHERE healing_patterns.id = pattern_id
    RETURNING healing_patterns.id, healing_patterns.success_count, healing_patterns.failure_count, healing_patterns.updated_at;
END;
$$;

-- =============================================================================
-- Atomic counter increment for healing_patterns failure_count
-- =============================================================================

CREATE OR REPLACE FUNCTION increment_healing_failure(pattern_id UUID)
RETURNS TABLE(id UUID, success_count INTEGER, failure_count INTEGER, updated_at TIMESTAMPTZ)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    UPDATE healing_patterns
    SET failure_count = healing_patterns.failure_count + 1,
        updated_at = NOW()
    WHERE healing_patterns.id = pattern_id
    RETURNING healing_patterns.id, healing_patterns.success_count, healing_patterns.failure_count, healing_patterns.updated_at;
END;
$$;

-- =============================================================================
-- Atomic counter increment for test_failure_patterns success_count
-- =============================================================================

CREATE OR REPLACE FUNCTION increment_failure_pattern_success(pattern_id UUID)
RETURNS TABLE(id UUID, success_count INTEGER, updated_at TIMESTAMPTZ)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    UPDATE test_failure_patterns
    SET success_count = test_failure_patterns.success_count + 1,
        updated_at = NOW()
    WHERE test_failure_patterns.id = pattern_id
    RETURNING test_failure_patterns.id, test_failure_patterns.success_count, test_failure_patterns.updated_at;
END;
$$;

-- =============================================================================
-- Grant execute permissions to authenticated users
-- =============================================================================

GRANT EXECUTE ON FUNCTION increment_pattern_times_seen(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_pattern_times_seen(UUID) TO service_role;

GRANT EXECUTE ON FUNCTION increment_healing_success(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_healing_success(UUID) TO service_role;

GRANT EXECUTE ON FUNCTION increment_healing_failure(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_healing_failure(UUID) TO service_role;

GRANT EXECUTE ON FUNCTION increment_failure_pattern_success(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_failure_pattern_success(UUID) TO service_role;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON FUNCTION increment_pattern_times_seen(UUID) IS 
'Atomically increment times_seen for a discovery pattern. Prevents race conditions.';

COMMENT ON FUNCTION increment_healing_success(UUID) IS 
'Atomically increment success_count for a healing pattern. Prevents race conditions.';

COMMENT ON FUNCTION increment_healing_failure(UUID) IS 
'Atomically increment failure_count for a healing pattern. Prevents race conditions.';

COMMENT ON FUNCTION increment_failure_pattern_success(UUID) IS 
'Atomically increment success_count for a test failure pattern. Prevents race conditions.';
