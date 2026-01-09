# User Management Hierarchy - Implementation Strategy

## Executive Summary

This document outlines a comprehensive strategy to implement an industry-standard user management system for Argus, supporting Individual, Team, and Enterprise tiers following best practices from Vercel, BrowserStack, GitHub, Stripe, and LambdaTest.

---

## Part 1: Current State Analysis

### What EXISTS

| Component | Status | Location |
|-----------|--------|----------|
| Organizations table | Complete | `supabase/migrations/20260106200000_organizations_and_ai_tracking.sql` |
| Organization members | Complete | Same migration file |
| API keys management | Complete | `src/api/api_keys.py` + Dashboard `/api-keys` |
| Team management API | Complete | `src/api/teams.py` |
| Team management UI | Complete | `dashboard/app/team/page.tsx` |
| Settings with org section | Partial | `dashboard/app/settings/page.tsx` |
| Projects table | Complete | Links to organization_id |
| Clerk JWT authentication | Complete | `src/api/security/auth.py` |
| Row-Level Security | Partial | RLS policies exist but need refinement |
| AI usage tracking | Complete | Per-organization budgets and tracking |

### What's MISSING

| Component | Priority | Impact |
|-----------|----------|--------|
| User profiles table | High | No persistent user preferences |
| Organization creation UI | Critical | Users can't create orgs |
| Organization switching UI | Critical | No multi-org support in UI |
| Project assignment to org | High | Projects not scoped properly |
| Invitation email sending | High | Invites stored but not sent |
| Billing/subscription integration | Medium | Stripe IDs exist but no integration |
| Workspace/Team sub-groups | Low | Flat org structure only |
| SAML/OIDC SSO | Low | Enterprise feature |
| SCIM provisioning | Low | Enterprise feature |

---

## Part 2: Industry Standard Hierarchy

### Recommended Three-Tier Structure

```
ORGANIZATION (Billing Entity)
├── Settings (Name, Billing, Plan)
├── Members (Users with org-level roles)
├── API Keys (Org-scoped)
│
├── TEAM (Optional Logical Grouping)
│   ├── Members subset
│   └── Team-specific settings
│
└── PROJECTS (Test Resources)
    ├── Tests
    ├── Discoveries
    ├── Visual Baselines
    └── Schedules
```

### Role Hierarchy

| Level | Role | Permissions |
|-------|------|-------------|
| Organization | **Owner** | Full control, billing, delete org |
| Organization | **Admin** | Manage members, settings, all projects |
| Organization | **Billing Admin** | View/manage billing only |
| Organization | **Member** | Access projects, run tests |
| Organization | **Viewer** | Read-only access |
| Project | **Maintainer** | Full project control |
| Project | **Developer** | Run tests, view results |
| Project | **Viewer** | Read-only |

### Plan Tiers

| Tier | Organizations | Members | Projects | Features |
|------|---------------|---------|----------|----------|
| **Free** | 1 | 1 | 3 | Basic testing |
| **Team** | 1 | 10 | Unlimited | + Advanced analytics, Integrations |
| **Enterprise** | Unlimited | Unlimited | Unlimited | + SSO, Audit logs, Custom |

---

## Part 3: Database Schema Updates

### 3.1 New Migration: User Profiles

```sql
-- supabase/migrations/20260110000000_user_profiles.sql

CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL UNIQUE,  -- Clerk user ID
    email TEXT NOT NULL,

    -- Profile info
    display_name TEXT,
    avatar_url TEXT,
    bio TEXT,

    -- Preferences
    timezone TEXT DEFAULT 'UTC',
    language TEXT DEFAULT 'en',
    theme TEXT DEFAULT 'system' CHECK (theme IN ('light', 'dark', 'system')),

    -- Notification preferences
    notification_preferences JSONB DEFAULT jsonb_build_object(
        'email_test_failures', true,
        'email_daily_digest', false,
        'slack_enabled', false,
        'in_app_enabled', true
    ),

    -- Default settings
    default_organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    default_project_id UUID REFERENCES projects(id) ON DELETE SET NULL,

    -- Onboarding
    onboarding_completed BOOLEAN DEFAULT false,
    onboarding_step INTEGER DEFAULT 0,

    -- Activity
    last_login_at TIMESTAMPTZ,
    last_active_at TIMESTAMPTZ,
    login_count INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_profiles_email ON user_profiles(email);
CREATE INDEX idx_user_profiles_default_org ON user_profiles(default_organization_id);
```

### 3.2 New Migration: Invitations System

```sql
-- supabase/migrations/20260110000001_invitations.sql

CREATE TABLE IF NOT EXISTS invitations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,

    -- Invitation details
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),

    -- Token for email link
    token TEXT NOT NULL UNIQUE,
    token_expires_at TIMESTAMPTZ NOT NULL,

    -- Invitation message
    message TEXT,

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'expired', 'revoked')),

    -- Tracking
    invited_by UUID REFERENCES organization_members(id),
    accepted_by UUID REFERENCES organization_members(id),
    accepted_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate pending invites
    UNIQUE(organization_id, email, status) WHERE status = 'pending'
);

CREATE INDEX idx_invitations_org ON invitations(organization_id);
CREATE INDEX idx_invitations_email ON invitations(email);
CREATE INDEX idx_invitations_token ON invitations(token);
CREATE INDEX idx_invitations_status ON invitations(status) WHERE status = 'pending';
```

### 3.3 New Migration: Project Members (Granular Access)

```sql
-- supabase/migrations/20260110000002_project_members.sql

-- Project-level access control (optional granular permissions)
CREATE TABLE IF NOT EXISTS project_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    organization_member_id UUID NOT NULL REFERENCES organization_members(id) ON DELETE CASCADE,

    -- Project-specific role (can differ from org role)
    role TEXT NOT NULL DEFAULT 'developer' CHECK (role IN ('maintainer', 'developer', 'viewer')),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, organization_member_id)
);

CREATE INDEX idx_project_members_project ON project_members(project_id);
CREATE INDEX idx_project_members_member ON project_members(organization_member_id);

-- Add organization_id to projects if not exists (ensure proper FK)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'projects_organization_id_fkey'
    ) THEN
        ALTER TABLE projects
        ADD CONSTRAINT projects_organization_id_fkey
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL;
    END IF;
END $$;
```

### 3.4 Update Organization Table

```sql
-- supabase/migrations/20260110000003_org_enhancements.sql

-- Add missing columns to organizations
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    logo_url TEXT;

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    domain TEXT;  -- For SSO domain verification

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    sso_enabled BOOLEAN DEFAULT false;

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    sso_config JSONB;  -- SAML/OIDC configuration

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    allowed_email_domains TEXT[];  -- Restrict signups to these domains

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    require_2fa BOOLEAN DEFAULT false;

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    data_retention_days INTEGER DEFAULT 90;

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    created_by TEXT;  -- Clerk user ID of creator

-- Index for domain-based lookups (SSO)
CREATE INDEX IF NOT EXISTS idx_organizations_domain ON organizations(domain) WHERE domain IS NOT NULL;
```

---

## Part 4: API Endpoints

### 4.1 User Profile Endpoints

```python
# src/api/users.py

@router.get("/api/v1/users/me")
async def get_current_user_profile(request: Request):
    """Get or create user profile for authenticated user."""

@router.put("/api/v1/users/me")
async def update_user_profile(request: Request, body: UpdateProfileRequest):
    """Update user profile settings."""

@router.put("/api/v1/users/me/preferences")
async def update_user_preferences(request: Request, body: PreferencesRequest):
    """Update notification and default preferences."""

@router.post("/api/v1/users/me/default-organization")
async def set_default_organization(request: Request, body: SetDefaultOrgRequest):
    """Set the user's default organization."""
```

### 4.2 Organization Management Endpoints

```python
# src/api/organizations.py (new file)

@router.post("/api/v1/organizations")
async def create_organization(request: Request, body: CreateOrgRequest):
    """Create a new organization (user becomes owner)."""

@router.get("/api/v1/organizations")
async def list_user_organizations(request: Request):
    """List all organizations user has access to."""

@router.get("/api/v1/organizations/{org_id}")
async def get_organization(org_id: str, request: Request):
    """Get organization details."""

@router.put("/api/v1/organizations/{org_id}")
async def update_organization(org_id: str, request: Request, body: UpdateOrgRequest):
    """Update organization settings (admin/owner only)."""

@router.delete("/api/v1/organizations/{org_id}")
async def delete_organization(org_id: str, request: Request):
    """Delete organization (owner only)."""

@router.post("/api/v1/organizations/{org_id}/transfer")
async def transfer_ownership(org_id: str, request: Request, body: TransferRequest):
    """Transfer ownership to another member (owner only)."""
```

### 4.3 Invitation Endpoints

```python
# src/api/invitations.py (new file)

@router.post("/api/v1/organizations/{org_id}/invitations")
async def send_invitation(org_id: str, request: Request, body: InviteRequest):
    """Send invitation email to join organization."""

@router.get("/api/v1/organizations/{org_id}/invitations")
async def list_invitations(org_id: str, request: Request):
    """List pending invitations."""

@router.delete("/api/v1/organizations/{org_id}/invitations/{invite_id}")
async def revoke_invitation(org_id: str, invite_id: str, request: Request):
    """Revoke a pending invitation."""

@router.post("/api/v1/invitations/accept/{token}")
async def accept_invitation(token: str, request: Request):
    """Accept invitation via email link."""

@router.get("/api/v1/invitations/validate/{token}")
async def validate_invitation(token: str):
    """Validate invitation token (public endpoint)."""
```

### 4.4 Project Access Endpoints

```python
# Add to src/api/projects.py

@router.get("/api/v1/organizations/{org_id}/projects")
async def list_org_projects(org_id: str, request: Request):
    """List all projects in organization."""

@router.post("/api/v1/organizations/{org_id}/projects")
async def create_project_in_org(org_id: str, request: Request, body: CreateProjectRequest):
    """Create project within organization."""

@router.put("/api/v1/projects/{project_id}/members")
async def update_project_members(project_id: str, request: Request, body: UpdateMembersRequest):
    """Update project-level member access."""
```

---

## Part 5: UI Components

### 5.1 New Pages Required

| Page | Route | Description |
|------|-------|-------------|
| Organization Selector | `/organizations` | Switch between orgs |
| Create Organization | `/organizations/new` | Create new org |
| Organization Settings | `/organizations/[id]/settings` | Full org settings |
| Invitations | `/invitations/[token]` | Accept invitation page |
| User Profile | `/profile` | User preferences |
| Onboarding | `/onboarding` | New user flow |

### 5.2 Component Updates

#### Sidebar - Add Organization Switcher

```tsx
// dashboard/components/layout/org-switcher.tsx

export function OrganizationSwitcher() {
  // Shows current org
  // Dropdown to switch orgs
  // "Create new organization" button
  // Settings link for current org
}
```

#### Header - Show Current Context

```tsx
// Show: Organization > Project context breadcrumb
// Quick access to org/project settings
```

### 5.3 Settings Page Enhancements

```
/settings
├── Profile (User-level)
│   ├── Name, Avatar, Bio
│   ├── Timezone, Language
│   └── Notification preferences
│
├── Organization (Org-level, if admin)
│   ├── Organization name, logo
│   ├── Default settings
│   ├── Allowed domains
│   └── Danger zone (delete)
│
├── Team Members
│   ├── Member list with roles
│   ├── Invite modal
│   └── Pending invitations
│
├── Billing (if owner)
│   ├── Current plan
│   ├── Usage stats
│   └── Upgrade/manage subscription
│
├── Security
│   ├── 2FA settings
│   ├── SSO configuration (enterprise)
│   └── Session management
│
└── API Keys (existing)
```

---

## Part 6: Implementation Phases

### Phase 1: Foundation (Priority: Critical)

**Goal**: Enable basic multi-organization support

1. Create `user_profiles` migration and apply
2. Create `invitations` migration and apply
3. Build `/api/v1/users/me` endpoints
4. Build `/api/v1/organizations` endpoints
5. Create organization switcher component
6. Update sidebar with org switcher
7. Create "Create Organization" page

**Estimated Effort**: 8-10 hours

### Phase 2: Invitations (Priority: High)

**Goal**: Enable team collaboration

1. Build invitation API endpoints
2. Set up email service (SendGrid/Resend)
3. Create invitation email template
4. Build invitation acceptance page
5. Add pending invitations UI to team page
6. Create resend/revoke functionality

**Estimated Effort**: 6-8 hours

### Phase 3: User Profile (Priority: High)

**Goal**: Persistent user preferences

1. Build user profile page
2. Implement preference saving
3. Add onboarding flow for new users
4. Create default org/project selection
5. Sync with Clerk user metadata

**Estimated Effort**: 4-6 hours

### Phase 4: Project Scoping (Priority: High)

**Goal**: Proper project isolation

1. Ensure all projects have organization_id
2. Update project list to filter by org
3. Add project creation within org context
4. Implement project-level permissions (optional)
5. Update RLS policies

**Estimated Effort**: 4-6 hours

### Phase 5: Billing Integration (Priority: Medium)

**Goal**: Subscription management

1. Set up Stripe integration
2. Create pricing page
3. Build checkout flow
4. Implement plan limits enforcement
5. Add usage-based billing support
6. Create billing portal link

**Estimated Effort**: 8-12 hours

### Phase 6: Enterprise Features (Priority: Low)

**Goal**: Enterprise readiness

1. SAML 2.0 SSO integration
2. OIDC support
3. SCIM provisioning
4. Domain verification
5. Enforced 2FA
6. Advanced audit logging

**Estimated Effort**: 15-20 hours

---

## Part 7: Migration Path

### For Existing Users

1. Create default organization for users without one
2. Associate orphan projects with user's default org
3. Set existing users as "owner" of their default org
4. Send notification about new organization features

### Data Migration Script

```python
# scripts/migrate_to_orgs.py

async def migrate_existing_users():
    """
    1. For each user without an org, create one
    2. For each project without org, assign to creator's org
    3. Create owner membership record
    """
    pass
```

---

## Part 8: Security Considerations

### RLS Policy Updates

```sql
-- Ensure all project queries filter by organization
CREATE POLICY "projects_org_isolation" ON projects
    FOR ALL USING (
        organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = current_setting('app.user_id', true)
        )
    );

-- Ensure test data is org-scoped
CREATE POLICY "tests_org_isolation" ON tests
    FOR ALL USING (
        project_id IN (
            SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );
```

### API Security

- All org endpoints require authentication
- Validate user has appropriate role for action
- Rate limit invitation sending
- Token expiration for invitations (7 days)
- Log all administrative actions

---

## Part 9: Success Metrics

| Metric | Target |
|--------|--------|
| Organization creation | Users can create orgs within 30s |
| Invitation acceptance | < 2 min from email to access |
| Org switching | < 1s response time |
| Plan enforcement | 100% compliance with limits |
| SSO setup (Enterprise) | < 30 min configuration |

---

## Part 10: File Checklist

### New Files to Create

- [ ] `supabase/migrations/20260110000000_user_profiles.sql`
- [ ] `supabase/migrations/20260110000001_invitations.sql`
- [ ] `supabase/migrations/20260110000002_project_members.sql`
- [ ] `supabase/migrations/20260110000003_org_enhancements.sql`
- [ ] `src/api/users.py`
- [ ] `src/api/organizations.py`
- [ ] `src/api/invitations.py`
- [ ] `src/services/email_service.py`
- [ ] `dashboard/app/organizations/page.tsx`
- [ ] `dashboard/app/organizations/new/page.tsx`
- [ ] `dashboard/app/organizations/[id]/settings/page.tsx`
- [ ] `dashboard/app/profile/page.tsx`
- [ ] `dashboard/app/invitations/[token]/page.tsx`
- [ ] `dashboard/app/onboarding/page.tsx`
- [ ] `dashboard/components/layout/org-switcher.tsx`

### Files to Modify

- [ ] `src/main.py` - Register new routers
- [ ] `src/api/projects.py` - Add org scoping
- [ ] `dashboard/components/layout/sidebar.tsx` - Add org switcher
- [ ] `dashboard/app/settings/page.tsx` - Enhance organization section
- [ ] `dashboard/app/team/page.tsx` - Add invitations section
- [ ] `dashboard/app/projects/page.tsx` - Filter by current org

---

## Conclusion

This strategy provides a clear roadmap to transform Argus from a single-user tool to a full multi-tenant SaaS platform with:

- **Industry-standard hierarchy**: Organization > Team > Project
- **Flexible roles**: Owner, Admin, Member, Viewer at org level
- **Scalable plans**: Free, Team, Enterprise tiers
- **Enterprise readiness**: SSO, SCIM, audit logging foundation

Start with **Phase 1** to unblock core functionality, then iterate through subsequent phases based on user feedback and business priorities.
