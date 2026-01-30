-- Collaboration tables for real-time collaborative editing
-- Provides persistence for CRDT documents, sessions, and presence

-- ============================================================================
-- Collaboration Sessions
-- Tracks active real-time collaboration sessions
-- ============================================================================
CREATE TABLE IF NOT EXISTS collaboration_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    workspace_id TEXT,
    test_id TEXT,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    connected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'idle', 'disconnected')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for collaboration_sessions
CREATE INDEX IF NOT EXISTS idx_collab_sessions_user_id ON collaboration_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_collab_sessions_workspace ON collaboration_sessions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_collab_sessions_test ON collaboration_sessions(test_id);
CREATE INDEX IF NOT EXISTS idx_collab_sessions_org ON collaboration_sessions(organization_id);
CREATE INDEX IF NOT EXISTS idx_collab_sessions_status ON collaboration_sessions(status);
CREATE INDEX IF NOT EXISTS idx_collab_sessions_last_activity ON collaboration_sessions(last_activity);

-- ============================================================================
-- CRDT Documents
-- Stores CRDT document state for collaborative test editing
-- ============================================================================
CREATE TABLE IF NOT EXISTS crdt_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id TEXT UNIQUE NOT NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    document_state JSONB NOT NULL DEFAULT '{}',
    vector_clock JSONB NOT NULL DEFAULT '{}',
    version INTEGER DEFAULT 1,
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_modified_by TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for crdt_documents
CREATE INDEX IF NOT EXISTS idx_crdt_docs_test ON crdt_documents(test_id);
CREATE INDEX IF NOT EXISTS idx_crdt_docs_org ON crdt_documents(organization_id);
CREATE INDEX IF NOT EXISTS idx_crdt_docs_modified ON crdt_documents(last_modified);

-- ============================================================================
-- CRDT Operations History
-- Stores operation history for sync and conflict resolution
-- ============================================================================
CREATE TABLE IF NOT EXISTS crdt_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES crdt_documents(id) ON DELETE CASCADE,
    operation_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    operation_type VARCHAR(20) NOT NULL CHECK (operation_type IN ('set', 'delete', 'insert', 'move')),
    path JSONB NOT NULL DEFAULT '[]',
    value JSONB,
    previous_value JSONB,
    vector_clock JSONB NOT NULL DEFAULT '{}',
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id TEXT,
    UNIQUE(document_id, operation_id)
);

-- Indexes for crdt_operations
CREATE INDEX IF NOT EXISTS idx_crdt_ops_document ON crdt_operations(document_id);
CREATE INDEX IF NOT EXISTS idx_crdt_ops_applied ON crdt_operations(applied_at);
CREATE INDEX IF NOT EXISTS idx_crdt_ops_node ON crdt_operations(node_id);

-- ============================================================================
-- User Presence (optional persistence for audit/analytics)
-- Ephemeral data but persisted for historical tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_presence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    user_name TEXT,
    user_email TEXT,
    workspace_id TEXT,
    test_id TEXT,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'online' CHECK (status IN ('online', 'idle', 'offline', 'busy')),
    cursor_position JSONB,
    selection_range JSONB,
    color VARCHAR(20),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    connected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, workspace_id, COALESCE(test_id, ''))
);

-- Indexes for user_presence
CREATE INDEX IF NOT EXISTS idx_presence_user ON user_presence(user_id);
CREATE INDEX IF NOT EXISTS idx_presence_workspace ON user_presence(workspace_id);
CREATE INDEX IF NOT EXISTS idx_presence_test ON user_presence(test_id);
CREATE INDEX IF NOT EXISTS idx_presence_org ON user_presence(organization_id);
CREATE INDEX IF NOT EXISTS idx_presence_status ON user_presence(status);
CREATE INDEX IF NOT EXISTS idx_presence_last_seen ON user_presence(last_seen);

-- ============================================================================
-- Collaborative Comments
-- Comments attached to tests or test steps
-- ============================================================================
CREATE TABLE IF NOT EXISTS collaborative_comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id TEXT NOT NULL,
    step_index INTEGER,
    author_id TEXT NOT NULL,
    author_name TEXT,
    author_avatar TEXT,
    content TEXT NOT NULL,
    mentions TEXT[] DEFAULT '{}',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,
    parent_id UUID REFERENCES collaborative_comments(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for collaborative_comments
CREATE INDEX IF NOT EXISTS idx_collab_comments_test ON collaborative_comments(test_id);
CREATE INDEX IF NOT EXISTS idx_collab_comments_author ON collaborative_comments(author_id);
CREATE INDEX IF NOT EXISTS idx_collab_comments_parent ON collaborative_comments(parent_id);
CREATE INDEX IF NOT EXISTS idx_collab_comments_org ON collaborative_comments(organization_id);
CREATE INDEX IF NOT EXISTS idx_collab_comments_resolved ON collaborative_comments(resolved);

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================
ALTER TABLE collaboration_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE crdt_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE crdt_operations ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_presence ENABLE ROW LEVEL SECURITY;
ALTER TABLE collaborative_comments ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "collaboration_sessions_org_isolation" ON collaboration_sessions;
DROP POLICY IF EXISTS "crdt_documents_org_isolation" ON crdt_documents;
DROP POLICY IF EXISTS "crdt_operations_org_isolation" ON crdt_operations;
DROP POLICY IF EXISTS "user_presence_org_isolation" ON user_presence;
DROP POLICY IF EXISTS "collaborative_comments_org_isolation" ON collaborative_comments;

-- RLS policies for organization isolation
CREATE POLICY "collaboration_sessions_org_isolation" ON collaboration_sessions
    FOR ALL USING (
        organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()::text
        )
        OR organization_id IS NULL
    );

CREATE POLICY "crdt_documents_org_isolation" ON crdt_documents
    FOR ALL USING (
        organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()::text
        )
        OR organization_id IS NULL
    );

CREATE POLICY "crdt_operations_org_isolation" ON crdt_operations
    FOR ALL USING (
        document_id IN (
            SELECT id FROM crdt_documents WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = auth.uid()::text
            )
        )
    );

CREATE POLICY "user_presence_org_isolation" ON user_presence
    FOR ALL USING (
        organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()::text
        )
        OR organization_id IS NULL
    );

CREATE POLICY "collaborative_comments_org_isolation" ON collaborative_comments
    FOR ALL USING (
        organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()::text
        )
        OR organization_id IS NULL
    );

-- ============================================================================
-- Service Role Bypass Policies
-- ============================================================================
DROP POLICY IF EXISTS "collaboration_sessions_service_role" ON collaboration_sessions;
DROP POLICY IF EXISTS "crdt_documents_service_role" ON crdt_documents;
DROP POLICY IF EXISTS "crdt_operations_service_role" ON crdt_operations;
DROP POLICY IF EXISTS "user_presence_service_role" ON user_presence;
DROP POLICY IF EXISTS "collaborative_comments_service_role" ON collaborative_comments;

CREATE POLICY "collaboration_sessions_service_role" ON collaboration_sessions
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "crdt_documents_service_role" ON crdt_documents
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "crdt_operations_service_role" ON crdt_operations
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "user_presence_service_role" ON user_presence
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "collaborative_comments_service_role" ON collaborative_comments
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ============================================================================
-- Triggers for updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION update_collaboration_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_collaboration_sessions_updated_at ON collaboration_sessions;
CREATE TRIGGER update_collaboration_sessions_updated_at
    BEFORE UPDATE ON collaboration_sessions
    FOR EACH ROW EXECUTE FUNCTION update_collaboration_updated_at();

DROP TRIGGER IF EXISTS update_crdt_documents_updated_at ON crdt_documents;
CREATE TRIGGER update_crdt_documents_updated_at
    BEFORE UPDATE ON crdt_documents
    FOR EACH ROW EXECUTE FUNCTION update_collaboration_updated_at();

DROP TRIGGER IF EXISTS update_user_presence_updated_at ON user_presence;
CREATE TRIGGER update_user_presence_updated_at
    BEFORE UPDATE ON user_presence
    FOR EACH ROW EXECUTE FUNCTION update_collaboration_updated_at();

DROP TRIGGER IF EXISTS update_collaborative_comments_updated_at ON collaborative_comments;
CREATE TRIGGER update_collaborative_comments_updated_at
    BEFORE UPDATE ON collaborative_comments
    FOR EACH ROW EXECUTE FUNCTION update_collaboration_updated_at();

-- ============================================================================
-- Cleanup function for stale sessions and presence
-- ============================================================================
CREATE OR REPLACE FUNCTION cleanup_stale_collaboration_data()
RETURNS void AS $$
BEGIN
    -- Mark sessions as disconnected if no activity for 10 minutes
    UPDATE collaboration_sessions
    SET status = 'disconnected'
    WHERE status = 'active'
      AND last_activity < NOW() - INTERVAL '10 minutes';

    -- Delete disconnected sessions older than 1 hour
    DELETE FROM collaboration_sessions
    WHERE status = 'disconnected'
      AND last_activity < NOW() - INTERVAL '1 hour';

    -- Mark presence as offline if not seen for 5 minutes
    UPDATE user_presence
    SET status = 'offline'
    WHERE status != 'offline'
      AND last_seen < NOW() - INTERVAL '5 minutes';

    -- Delete offline presence records older than 24 hours
    DELETE FROM user_presence
    WHERE status = 'offline'
      AND last_seen < NOW() - INTERVAL '24 hours';

    -- Delete old CRDT operations (keep last 7 days)
    DELETE FROM crdt_operations
    WHERE applied_at < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION cleanup_stale_collaboration_data() TO service_role;
