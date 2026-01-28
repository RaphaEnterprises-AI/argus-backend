-- =============================================================================
-- Frontend Missing Tables Migration
-- =============================================================================
-- Creates tables that the frontend expects but don't exist in the database:
-- - global_tests: Global/edge location performance testing
-- - global_test_results: Results per region for global tests
-- - live_sessions: Real-time session tracking for UI operations
-- - activity_logs: Real-time activity events for live sessions
-- - chat_conversations: Chat conversation metadata
-- - chat_messages: Individual chat messages
-- =============================================================================

-- =============================================================================
-- 1. GLOBAL TESTS TABLE
-- =============================================================================
-- Stores metadata for global performance tests that measure latency from
-- multiple edge locations around the world.

CREATE TABLE IF NOT EXISTS global_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Test configuration
    url TEXT NOT NULL,
    triggered_by TEXT,  -- User ID or 'scheduled' or 'api'

    -- Execution state
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Aggregated results
    avg_latency_ms INTEGER,
    avg_ttfb_ms INTEGER,  -- Time to first byte
    success_rate DECIMAL(5,2),  -- Percentage 0-100
    slow_regions INTEGER DEFAULT 0,
    failed_regions INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_global_tests_project ON global_tests(project_id);
CREATE INDEX IF NOT EXISTS idx_global_tests_created ON global_tests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_global_tests_status ON global_tests(status);

-- =============================================================================
-- 2. GLOBAL TEST RESULTS TABLE
-- =============================================================================
-- Stores per-region results for each global test.

CREATE TABLE IF NOT EXISTS global_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    global_test_id UUID NOT NULL REFERENCES global_tests(id) ON DELETE CASCADE,

    -- Region info
    region_code TEXT NOT NULL,  -- e.g., 'US-EAST', 'EU-WEST', 'APAC-EAST'
    city TEXT NOT NULL,  -- Human-readable location

    -- Results
    status TEXT NOT NULL CHECK (status IN ('success', 'error', 'slow', 'timeout')),
    latency_ms INTEGER,
    ttfb_ms INTEGER,  -- Time to first byte
    page_load_ms INTEGER,  -- Full page load time

    -- Error details (if status is error)
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_global_test_results_test ON global_test_results(global_test_id);
CREATE INDEX IF NOT EXISTS idx_global_test_results_region ON global_test_results(region_code);
CREATE INDEX IF NOT EXISTS idx_global_test_results_status ON global_test_results(status);

-- =============================================================================
-- 3. ACTIVITY LOGS TABLE (Real-time events)
-- =============================================================================
-- Stores granular activity events for live session tracking.
-- Enables real-time visibility into test execution via Supabase Realtime.

CREATE TABLE IF NOT EXISTS activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,  -- Groups related activities (references live_sessions)

    -- Activity classification
    activity_type TEXT NOT NULL CHECK (activity_type IN (
        'discovery', 'visual_test', 'test_run', 'quality_audit', 'global_test'
    )),
    event_type TEXT NOT NULL CHECK (event_type IN (
        'started', 'step', 'screenshot', 'thinking', 'action',
        'error', 'completed', 'cancelled'
    )),

    -- Event details
    title TEXT NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    screenshot_url TEXT,  -- Base64 or URL for screenshots
    duration_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_activity_logs_session ON activity_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_project_time ON activity_logs(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_type ON activity_logs(activity_type);

-- =============================================================================
-- 4. LIVE SESSIONS TABLE
-- =============================================================================
-- Tracks active operations for real-time UI updates.
-- Links to activity_logs for detailed event streaming.

CREATE TABLE IF NOT EXISTS live_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Session type
    session_type TEXT NOT NULL CHECK (session_type IN (
        'discovery', 'visual_test', 'test_run', 'quality_audit', 'global_test'
    )),

    -- Status tracking
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed', 'cancelled')),
    current_step TEXT,
    total_steps INTEGER DEFAULT 0,
    completed_steps INTEGER DEFAULT 0,

    -- Visual state
    last_screenshot_url TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Additional metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_live_sessions_project ON live_sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_live_sessions_status ON live_sessions(status);
CREATE INDEX IF NOT EXISTS idx_live_sessions_active ON live_sessions(project_id, status) WHERE status = 'active';

-- =============================================================================
-- 5. CHAT CONVERSATIONS TABLE
-- =============================================================================
-- Stores chat conversation metadata for the AI assistant feature.

CREATE TABLE IF NOT EXISTS chat_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,  -- Clerk user ID
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,

    -- Conversation metadata
    title TEXT NOT NULL DEFAULT 'New Conversation',
    preview TEXT,  -- Preview of the last message
    message_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chat_conversations_user ON chat_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_project ON chat_conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_updated ON chat_conversations(updated_at DESC);

-- =============================================================================
-- 6. CHAT MESSAGES TABLE
-- =============================================================================
-- Stores individual chat messages within conversations.

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,

    -- Message content
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,

    -- Tool usage (for AI assistant messages)
    tool_calls JSONB,  -- Tool calls made by assistant
    tool_results JSONB,  -- Results from tool executions
    tool_invocations JSONB,  -- Detailed tool invocation info

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created ON chat_messages(conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON chat_messages(role);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE global_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE global_test_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE live_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- RLS POLICIES - Global Tests (Project-scoped)
-- =============================================================================

CREATE POLICY "global_tests_policy" ON global_tests
    FOR ALL USING (
        public.is_service_role() OR
        public.has_project_access(project_id)
    );

CREATE POLICY "global_test_results_policy" ON global_test_results
    FOR ALL USING (
        public.is_service_role() OR
        EXISTS (
            SELECT 1 FROM global_tests gt
            WHERE gt.id = global_test_results.global_test_id
            AND public.has_project_access(gt.project_id)
        )
    );

-- =============================================================================
-- RLS POLICIES - Activity System (Project-scoped)
-- =============================================================================

CREATE POLICY "activity_logs_policy" ON activity_logs
    FOR ALL USING (
        public.is_service_role() OR
        public.has_project_access(project_id)
    );

CREATE POLICY "live_sessions_policy" ON live_sessions
    FOR ALL USING (
        public.is_service_role() OR
        public.has_project_access(project_id)
    );

-- =============================================================================
-- RLS POLICIES - Chat (User-scoped)
-- =============================================================================

-- Conversations: Users can only access their own conversations
CREATE POLICY "chat_conversations_select" ON chat_conversations
    FOR SELECT USING (
        public.is_service_role() OR
        user_id = public.current_user_id()
    );

CREATE POLICY "chat_conversations_insert" ON chat_conversations
    FOR INSERT WITH CHECK (
        public.is_service_role() OR
        user_id = public.current_user_id()
    );

CREATE POLICY "chat_conversations_update" ON chat_conversations
    FOR UPDATE USING (
        public.is_service_role() OR
        user_id = public.current_user_id()
    );

CREATE POLICY "chat_conversations_delete" ON chat_conversations
    FOR DELETE USING (
        public.is_service_role() OR
        user_id = public.current_user_id()
    );

-- Messages: Users can access messages in their conversations
CREATE POLICY "chat_messages_select" ON chat_messages
    FOR SELECT USING (
        public.is_service_role() OR
        EXISTS (
            SELECT 1 FROM chat_conversations cc
            WHERE cc.id = chat_messages.conversation_id
            AND cc.user_id = public.current_user_id()
        )
    );

CREATE POLICY "chat_messages_insert" ON chat_messages
    FOR INSERT WITH CHECK (
        public.is_service_role() OR
        EXISTS (
            SELECT 1 FROM chat_conversations cc
            WHERE cc.id = chat_messages.conversation_id
            AND cc.user_id = public.current_user_id()
        )
    );

CREATE POLICY "chat_messages_update" ON chat_messages
    FOR UPDATE USING (
        public.is_service_role() OR
        EXISTS (
            SELECT 1 FROM chat_conversations cc
            WHERE cc.id = chat_messages.conversation_id
            AND cc.user_id = public.current_user_id()
        )
    );

CREATE POLICY "chat_messages_delete" ON chat_messages
    FOR DELETE USING (
        public.is_service_role() OR
        EXISTS (
            SELECT 1 FROM chat_conversations cc
            WHERE cc.id = chat_messages.conversation_id
            AND cc.user_id = public.current_user_id()
        )
    );

-- =============================================================================
-- TRIGGERS - Auto-update timestamps
-- =============================================================================

-- Trigger function for chat_conversations updated_at
CREATE OR REPLACE FUNCTION update_chat_conversations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chat_conversations_updated ON chat_conversations;
CREATE TRIGGER trg_chat_conversations_updated
    BEFORE UPDATE ON chat_conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_chat_conversations_updated_at();

-- Trigger to update conversation metadata when messages are added
CREATE OR REPLACE FUNCTION update_conversation_on_message()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_conversations
    SET
        message_count = (
            SELECT COUNT(*) FROM chat_messages
            WHERE conversation_id = NEW.conversation_id
        ),
        preview = SUBSTRING(NEW.content, 1, 100),
        updated_at = NOW()
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_conversation_on_message ON chat_messages;
CREATE TRIGGER trg_update_conversation_on_message
    AFTER INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_on_message();

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to clean up old activity logs (keep last 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_activity_logs()
RETURNS void AS $$
BEGIN
    DELETE FROM activity_logs
    WHERE created_at < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Function to mark stale sessions as failed (older than 10 minutes and still active)
CREATE OR REPLACE FUNCTION cleanup_stale_sessions()
RETURNS void AS $$
BEGIN
    UPDATE live_sessions
    SET status = 'failed',
        completed_at = NOW()
    WHERE status = 'active'
    AND started_at < NOW() - INTERVAL '10 minutes';
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ENABLE REALTIME (for live session updates)
-- =============================================================================
-- Note: This may fail if the tables are already in the publication,
-- which is fine - we wrap in a DO block to handle errors gracefully.

DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE activity_logs;
EXCEPTION WHEN duplicate_object THEN
    NULL; -- Already added
END $$;

DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE live_sessions;
EXCEPTION WHEN duplicate_object THEN
    NULL; -- Already added
END $$;

-- =============================================================================
-- GRANTS
-- =============================================================================

-- Grant service_role full access to all tables
GRANT ALL ON global_tests TO service_role;
GRANT ALL ON global_test_results TO service_role;
GRANT ALL ON activity_logs TO service_role;
GRANT ALL ON live_sessions TO service_role;
GRANT ALL ON chat_conversations TO service_role;
GRANT ALL ON chat_messages TO service_role;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE global_tests IS 'Global performance tests measuring latency from multiple edge locations';
COMMENT ON TABLE global_test_results IS 'Per-region results for global performance tests';
COMMENT ON TABLE activity_logs IS 'Real-time activity events for live session tracking';
COMMENT ON TABLE live_sessions IS 'Active operation tracking for real-time UI updates';
COMMENT ON TABLE chat_conversations IS 'Chat conversation metadata for AI assistant feature';
COMMENT ON TABLE chat_messages IS 'Individual messages within chat conversations';

-- =============================================================================
-- Migration Complete
-- =============================================================================
SELECT 'Frontend missing tables created successfully!' as message;
