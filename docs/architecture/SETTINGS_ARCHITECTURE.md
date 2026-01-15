# Settings Architecture Documentation

> **Version:** 2.2.0
> **Last Updated:** 2026-01-12
> **Document Status:** Production Ready

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [User Profile Management](#3-user-profile-management)
4. [Organization Settings](#4-organization-settings)
5. [API Key Management](#5-api-key-management)
6. [Notification Preferences](#6-notification-preferences)
7. [Team Management](#7-team-management)
8. [Data Flow](#8-data-flow)
9. [API Endpoints Reference](#9-api-endpoints-reference)
10. [Frontend Hooks Reference](#10-frontend-hooks-reference)
11. [Database Schema](#11-database-schema)
12. [Security Considerations](#12-security-considerations)

---

## 1. Overview

The Argus settings architecture provides a comprehensive system for managing user profiles, organization settings, team members, API keys, and notification preferences. The architecture follows a multi-tenant design where:

- **Users** have personal profiles with preferences and defaults
- **Organizations** serve as the primary billing and access control entity
- **Teams** are optional logical groupings within organizations
- **Projects** belong to organizations and can have project-level permissions

### Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Multi-Tenancy** | Complete data isolation between organizations using RLS |
| **Role-Based Access** | Hierarchical roles (Owner > Admin > Member > Viewer) |
| **API-First** | All settings accessible via REST API |
| **Real-Time Sync** | Changes propagate immediately across sessions |
| **Audit Logging** | All administrative actions are logged |

---

## 2. Architecture Diagram

### 2.1 High-Level Settings Architecture

```
+------------------------------------------------------------------+
|                        FRONTEND LAYER                             |
|  +-------------------+  +-------------------+  +----------------+ |
|  | Settings Page     |  | Team Management   |  | API Keys Page  | |
|  | /settings         |  | /team             |  | /api-keys      | |
|  +-------------------+  +-------------------+  +----------------+ |
|           |                     |                     |           |
|           v                     v                     v           |
|  +-----------------------------------------------------------+   |
|  |              React Hooks Layer                             |   |
|  | useAuthApi | usePermissions | useNotifications | etc.      |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        API LAYER                                  |
|  +-------------------+  +-------------------+  +----------------+ |
|  | /api/v1/users     |  | /api/v1/teams     |  | /api/v1/       | |
|  |   /me             |  |   /organizations  |  |   api-keys     | |
|  |   /me/preferences |  |   /{id}/members   |  |                | |
|  +-------------------+  +-------------------+  +----------------+ |
|           |                     |                     |           |
|           v                     v                     v           |
|  +-----------------------------------------------------------+   |
|  |              Authentication Middleware                     |   |
|  |  Clerk JWT Validation | API Key Authentication             |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                       DATA LAYER                                  |
|  +-------------------+  +-------------------+  +----------------+ |
|  | user_profiles     |  | organizations     |  | api_keys       | |
|  |                   |  | org_members       |  |                | |
|  |                   |  +-------------------+  +----------------+ |
|  +-------------------+                                            |
|           |                     |                     |           |
|           v                     v                     v           |
|  +-----------------------------------------------------------+   |
|  |              Row-Level Security (RLS)                      |   |
|  |  Policies ensure data isolation per organization           |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
```

### 2.2 Settings Data Flow

```
User Action                    API Call                      Database
    |                             |                              |
    |  Update Profile             |                              |
    +------------------------->   |                              |
    |                          PUT /api/v1/users/me              |
    |                             +-------------------------->   |
    |                             |                    UPDATE user_profiles
    |                             |                              |
    |                             |    <--------------------------+
    |                             |         Updated record       |
    |  <--------------------------+                              |
    |     Success + new data      |                              |
    |                             |                              |
    |  Set Default Org            |                              |
    +------------------------->   |                              |
    |              POST /api/v1/users/me/default-organization    |
    |                             +-------------------------->   |
    |                             |      1. Verify membership    |
    |                             |      2. Update profile       |
    |                             |                              |
    |                             |    <--------------------------+
    |  <--------------------------+                              |
    |     Updated profile         |                              |
```

---

## 3. User Profile Management

### 3.1 Profile Data Model

The `user_profiles` table stores individual user settings and preferences:

```sql
user_profiles
├── id                          UUID (PK)
├── user_id                     TEXT (Clerk user ID, unique)
├── email                       TEXT
├── display_name                TEXT
├── avatar_url                  TEXT
├── bio                         TEXT
├── timezone                    TEXT (default: 'UTC')
├── language                    TEXT (default: 'en')
├── theme                       TEXT (light|dark|system)
├── notification_preferences    JSONB
├── default_organization_id     UUID (FK -> organizations)
├── default_project_id          UUID (FK -> projects)
├── onboarding_completed        BOOLEAN
├── onboarding_step             INTEGER
├── last_login_at               TIMESTAMPTZ
├── last_active_at              TIMESTAMPTZ
├── login_count                 INTEGER
├── created_at                  TIMESTAMPTZ
└── updated_at                  TIMESTAMPTZ
```

### 3.2 Profile Auto-Creation

User profiles are automatically created on first API access:

```python
async def get_or_create_profile(user_id: str, email: Optional[str] = None) -> dict:
    """Get user profile, creating it if it doesn't exist."""
    # Try to get existing profile
    result = await supabase.request(f"/user_profiles?user_id=eq.{user_id}")

    if result.get("data") and len(result["data"]) > 0:
        return result["data"][0]

    # Profile doesn't exist, create with defaults
    new_profile = {
        "user_id": user_id,
        "email": email,
        "notification_preferences": get_default_notification_preferences(),
        "onboarding_completed": False,
        "theme": "system",
        "language": "en",
    }

    return await supabase.insert("user_profiles", new_profile)
```

### 3.3 User Preferences Schema

```json
{
  "notification_preferences": {
    "email_test_failures": true,
    "email_test_completions": false,
    "email_weekly_digest": true,
    "slack_test_failures": false,
    "slack_test_completions": false,
    "in_app_test_failures": true,
    "in_app_test_completions": true
  }
}
```

---

## 4. Organization Settings

### 4.1 Organization Data Model

Organizations are the primary billing and access control entity:

```sql
organizations
├── id                          UUID (PK)
├── name                        TEXT
├── slug                        TEXT (unique, URL-friendly)
├── plan                        TEXT (free|team|enterprise)
├── ai_budget_daily             NUMERIC (USD limit per day)
├── ai_budget_monthly           NUMERIC (USD limit per month)
├── ai_spend_today              NUMERIC (tracking)
├── ai_spend_this_month         NUMERIC (tracking)
├── settings                    JSONB
├── features                    JSONB
├── stripe_customer_id          TEXT
├── stripe_subscription_id      TEXT
├── logo_url                    TEXT
├── domain                      TEXT (for SSO)
├── sso_enabled                 BOOLEAN
├── sso_config                  JSONB
├── allowed_email_domains       TEXT[]
├── require_2fa                 BOOLEAN
├── data_retention_days         INTEGER
├── created_at                  TIMESTAMPTZ
└── updated_at                  TIMESTAMPTZ
```

### 4.2 Organization Settings Structure

```json
{
  "settings": {
    "default_test_timeout": 300,
    "max_parallel_tests": 5,
    "screenshot_quality": "high",
    "video_recording": true,
    "self_healing_enabled": true,
    "auto_retry_failed_tests": true
  },
  "features": {
    "ai_test_generation": true,
    "visual_regression": true,
    "api_testing": true,
    "custom_integrations": false,
    "sso": false,
    "audit_logs": true
  }
}
```

### 4.3 Plan Tiers and Limits

| Feature | Free | Team | Enterprise |
|---------|------|------|------------|
| Organizations | 1 | 1 | Unlimited |
| Members | 1 | 10 | Unlimited |
| Projects | 3 | Unlimited | Unlimited |
| AI Budget (Daily) | $1 | $10 | Custom |
| AI Budget (Monthly) | $25 | $250 | Custom |
| Self-Healing | Basic | Advanced | Advanced |
| SSO | No | No | Yes |
| Audit Logs | 7 days | 30 days | 1 year |

---

## 5. API Key Management

### 5.1 API Key Data Model

```sql
api_keys
├── id                          UUID (PK)
├── organization_id             UUID (FK -> organizations)
├── name                        TEXT
├── key_hash                    TEXT (SHA-256 hash, indexed)
├── key_prefix                  TEXT (first 16 chars for display)
├── scopes                      TEXT[] (read, write, admin, tests, webhooks)
├── expires_at                  TIMESTAMPTZ
├── revoked_at                  TIMESTAMPTZ
├── last_used_at                TIMESTAMPTZ
├── request_count               INTEGER
├── created_by                  UUID (FK -> organization_members)
├── created_at                  TIMESTAMPTZ
└── updated_at                  TIMESTAMPTZ
```

### 5.2 Key Generation and Security

API keys follow the format: `argus_sk_<64_hex_chars>`

```python
KEY_PREFIX = "argus_sk_"
KEY_LENGTH = 32  # 32 bytes = 64 hex chars

def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its hash."""
    random_bytes = secrets.token_hex(KEY_LENGTH)
    full_key = f"{KEY_PREFIX}{random_bytes}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, key_hash
```

### 5.3 Key Scopes

| Scope | Description |
|-------|-------------|
| `read` | Read access to tests, results, projects |
| `write` | Create/update tests, run executions |
| `admin` | Manage team members, settings |
| `tests` | Execute tests (subset of write) |
| `webhooks` | Configure webhooks and integrations |

### 5.4 Key Lifecycle

```
     +-------------+
     |   Create    |
     +------+------+
            |
            v
     +-------------+
     |   Active    |<---+
     +------+------+    |
            |           |
            v           |
     +-------------+    |
     | Last Used   |----+
     +------+------+
            |
    +-------+-------+
    |               |
    v               v
+-------+     +----------+
| Expire|     |  Revoke  |
+-------+     +----------+
    |               |
    v               v
     +-------------+
     |  Inactive   |
     +-------------+
```

---

## 6. Notification Preferences

### 6.1 Notification Channels

```sql
notification_channels
├── id                          UUID (PK)
├── organization_id             UUID (FK)
├── project_id                  UUID (FK, optional)
├── name                        TEXT
├── channel_type                TEXT (slack|email|webhook|discord|teams)
├── config                      JSONB
├── enabled                     BOOLEAN
├── verified                    BOOLEAN
├── verification_token          TEXT
├── rate_limit_per_hour         INTEGER
├── last_sent_at                TIMESTAMPTZ
├── sent_today                  INTEGER
├── created_at                  TIMESTAMPTZ
└── updated_at                  TIMESTAMPTZ
```

### 6.2 Channel Configuration Examples

**Slack Channel:**
```json
{
  "channel_type": "slack",
  "config": {
    "webhook_url": "https://hooks.slack.com/services/...",
    "channel": "#testing-alerts",
    "username": "Argus Bot",
    "icon_emoji": ":robot_face:"
  }
}
```

**Email Channel:**
```json
{
  "channel_type": "email",
  "config": {
    "recipients": ["team@example.com"],
    "from_name": "Argus Testing",
    "reply_to": "noreply@argus.io"
  }
}
```

**Webhook Channel:**
```json
{
  "channel_type": "webhook",
  "config": {
    "url": "https://api.example.com/webhook",
    "method": "POST",
    "headers": {
      "Authorization": "Bearer ${SECRET_TOKEN}"
    },
    "payload_template": "{ \"event\": \"{{event_type}}\", \"data\": {{payload}} }"
  }
}
```

### 6.3 Notification Rules

```sql
notification_rules
├── id                          UUID (PK)
├── channel_id                  UUID (FK)
├── name                        TEXT
├── description                 TEXT
├── event_type                  TEXT
├── conditions                  JSONB
├── message_template            TEXT
├── priority                    TEXT (low|normal|high|urgent)
├── cooldown_minutes            INTEGER
├── last_triggered_at           TIMESTAMPTZ
├── enabled                     BOOLEAN
├── created_at                  TIMESTAMPTZ
└── updated_at                  TIMESTAMPTZ
```

### 6.4 Event Types

| Event Type | Description |
|------------|-------------|
| `test.passed` | Test execution passed |
| `test.failed` | Test execution failed |
| `test.healed` | Test auto-healed by AI |
| `run.started` | Test run started |
| `run.completed` | Test run completed |
| `run.failed` | Test run failed |
| `schedule.triggered` | Scheduled test triggered |
| `api_key.created` | New API key created |
| `member.invited` | Team member invited |
| `member.joined` | Team member joined |

---

## 7. Team Management

### 7.1 Organization Members

```sql
organization_members
├── id                          UUID (PK)
├── organization_id             UUID (FK)
├── user_id                     TEXT (Clerk user ID)
├── email                       TEXT
├── role                        TEXT (owner|admin|member|viewer)
├── status                      TEXT (active|pending|suspended)
├── invited_by                  UUID (FK -> self)
├── invited_at                  TIMESTAMPTZ
├── accepted_at                 TIMESTAMPTZ
├── created_at                  TIMESTAMPTZ
└── updated_at                  TIMESTAMPTZ
```

### 7.2 Role Permissions

```
+------------------------------------------------------------------+
|                        ROLE HIERARCHY                             |
+------------------------------------------------------------------+
|                                                                   |
|  OWNER                                                           |
|  +-- All permissions                                             |
|  +-- Delete organization                                         |
|  +-- Transfer ownership                                          |
|  +-- Manage billing                                              |
|                                                                   |
|  ADMIN                                                           |
|  +-- Manage members (except owner)                               |
|  +-- Manage settings                                             |
|  +-- Manage API keys                                             |
|  +-- All project operations                                      |
|                                                                   |
|  MEMBER                                                          |
|  +-- Read/write tests                                            |
|  +-- Execute test runs                                           |
|  +-- View projects                                               |
|  +-- View reports                                                |
|                                                                   |
|  VIEWER                                                          |
|  +-- Read-only access                                            |
|  +-- View tests                                                  |
|  +-- View projects                                               |
|  +-- View reports                                                |
|                                                                   |
+------------------------------------------------------------------+
```

### 7.3 Invitation Flow

```
     +---------------+
     | Admin Invites |
     | via Email     |
     +-------+-------+
             |
             v
     +---------------+
     | Pending Member|
     | Created       |
     +-------+-------+
             |
             v
     +---------------+
     | Email Sent    |
     | with Token    |
     +-------+-------+
             |
    +--------+--------+
    |                 |
    v                 v
+-------+       +---------+
| Accept|       | Expires |
+---+---+       +---------+
    |
    v
+---------------+
| Member Active |
| user_id set   |
+---------------+
```

---

## 8. Data Flow

### 8.1 Profile Update Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Client    │      │   Backend   │      │  Database   │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       │ PUT /users/me      │                    │
       │ {display_name}     │                    │
       ├───────────────────>│                    │
       │                    │                    │
       │                    │ Auth Middleware    │
       │                    │ (Verify JWT)       │
       │                    │                    │
       │                    │ get_or_create      │
       │                    │ profile            │
       │                    ├───────────────────>│
       │                    │                    │
       │                    │<───────────────────│
       │                    │ (profile exists)   │
       │                    │                    │
       │                    │ UPDATE             │
       │                    │ user_profiles      │
       │                    ├───────────────────>│
       │                    │                    │
       │                    │<───────────────────│
       │                    │ (updated row)      │
       │                    │                    │
       │ 200 OK             │                    │
       │ {updated profile}  │                    │
       │<───────────────────│                    │
       │                    │                    │
```

### 8.2 Organization Access Verification

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Request   │      │  Middleware │      │   Handler   │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       │ Authorization:     │                    │
       │ Bearer <jwt>       │                    │
       ├───────────────────>│                    │
       │                    │                    │
       │                    │ 1. Validate JWT    │
       │                    │    with Clerk      │
       │                    │                    │
       │                    │ 2. Extract:        │
       │                    │    - user_id       │
       │                    │    - email         │
       │                    │    - org_id        │
       │                    │                    │
       │                    │ request.state.user │
       │                    ├───────────────────>│
       │                    │                    │
       │                    │                    │ 3. verify_org_access()
       │                    │                    │    - Check membership
       │                    │                    │    - Verify role
       │                    │                    │
       │                    │                    │ 4. Execute handler
       │                    │                    │
       │<───────────────────┼────────────────────│
       │     Response       │                    │
       │                    │                    │
```

---

## 9. API Endpoints Reference

### 9.1 User Profile Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/v1/users/me` | Get current user profile | Yes |
| PUT | `/api/v1/users/me` | Update profile settings | Yes |
| PUT | `/api/v1/users/me/preferences` | Update notification preferences | Yes |
| POST | `/api/v1/users/me/default-organization` | Set default organization | Yes |
| GET | `/api/v1/users/me/organizations` | List user's organizations | Yes |

#### GET /api/v1/users/me

**Response:**
```json
{
  "id": "uuid",
  "user_id": "clerk_user_id",
  "email": "user@example.com",
  "display_name": "John Doe",
  "avatar_url": "https://...",
  "timezone": "America/New_York",
  "language": "en",
  "theme": "dark",
  "notification_preferences": {
    "email_test_failures": true,
    "slack_test_failures": false
  },
  "default_organization_id": "uuid",
  "default_project_id": "uuid",
  "onboarding_completed": true,
  "created_at": "2026-01-01T00:00:00Z"
}
```

#### PUT /api/v1/users/me

**Request:**
```json
{
  "display_name": "Jane Doe",
  "timezone": "Europe/London",
  "theme": "light"
}
```

### 9.2 Organization Endpoints

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| GET | `/api/v1/organizations` | List organizations | Yes | Any |
| POST | `/api/v1/organizations` | Create organization | Yes | - |
| GET | `/api/v1/organizations/{id}` | Get organization | Yes | Member+ |
| PUT | `/api/v1/organizations/{id}` | Update organization | Yes | Admin+ |
| DELETE | `/api/v1/organizations/{id}` | Delete organization | Yes | Owner |
| POST | `/api/v1/organizations/{id}/transfer` | Transfer ownership | Yes | Owner |

#### POST /api/v1/organizations

**Request:**
```json
{
  "name": "My Company"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "My Company",
  "slug": "my-company",
  "plan": "free",
  "ai_budget_daily": 1.0,
  "ai_budget_monthly": 25.0,
  "member_count": 1,
  "created_at": "2026-01-01T00:00:00Z"
}
```

### 9.3 Team Member Endpoints

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| GET | `/api/v1/teams/organizations/{id}/members` | List members | Yes | Member+ |
| POST | `/api/v1/teams/organizations/{id}/members/invite` | Invite member | Yes | Admin+ |
| PATCH | `/api/v1/teams/organizations/{id}/members/{mid}/role` | Update role | Yes | Owner |
| DELETE | `/api/v1/teams/organizations/{id}/members/{mid}` | Remove member | Yes | Admin+ |

#### POST /api/v1/teams/organizations/{id}/members/invite

**Request:**
```json
{
  "email": "newmember@example.com",
  "role": "member"
}
```

### 9.4 API Key Endpoints

| Method | Endpoint | Description | Auth Required | Role |
|--------|----------|-------------|---------------|------|
| GET | `/api/v1/api-keys/organizations/{id}/keys` | List API keys | Yes | Admin+ |
| POST | `/api/v1/api-keys/organizations/{id}/keys` | Create API key | Yes | Admin+ |
| POST | `/api/v1/api-keys/organizations/{id}/keys/{kid}/rotate` | Rotate key | Yes | Admin+ |
| DELETE | `/api/v1/api-keys/organizations/{id}/keys/{kid}` | Revoke key | Yes | Admin+ |

#### POST /api/v1/api-keys/organizations/{id}/keys

**Request:**
```json
{
  "name": "CI/CD Pipeline",
  "scopes": ["read", "write", "tests"],
  "expires_in_days": 365
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "CI/CD Pipeline",
  "key_prefix": "argus_sk_abc123",
  "key": "argus_sk_abc123def456...",
  "scopes": ["read", "write", "tests"],
  "expires_at": "2027-01-01T00:00:00Z",
  "is_active": true,
  "created_at": "2026-01-01T00:00:00Z"
}
```

---

## 10. Frontend Hooks Reference

### 10.1 useAuthApi Hook

The primary hook for making authenticated API calls:

```typescript
import { useAuthApi } from '@/lib/hooks/use-auth-api';

function MyComponent() {
  const { api, fetchJson, fetchStream, isSignedIn, userId, orgId } = useAuthApi();

  // Simple JSON request
  const loadProfile = async () => {
    const { data, error } = await fetchJson<UserProfile>('/api/v1/users/me');
    if (error) console.error(error);
    return data;
  };

  // Stream request for SSE
  const streamTests = async () => {
    await fetchStream('/api/v1/stream/test', { projectId: 'xxx' }, (event, data) => {
      console.log('Event:', event, data);
    });
  };

  return <div>...</div>;
}
```

**Hook Return Values:**

| Property | Type | Description |
|----------|------|-------------|
| `api` | AuthenticatedClient | Low-level fetch client |
| `fetchJson` | Function | Helper for JSON responses |
| `fetchStream` | Function | Helper for SSE streams |
| `isLoaded` | boolean | Clerk auth loaded |
| `isSignedIn` | boolean | User is authenticated |
| `userId` | string | Clerk user ID |
| `orgId` | string | Current organization ID |
| `getToken` | Function | Get JWT token |
| `backendUrl` | string | Backend API URL |

### 10.2 usePermissions Hook

Check user permissions based on role:

```typescript
import { usePermissions } from '@/lib/hooks/use-auth-api';

function AdminPanel() {
  const {
    permissions,
    hasPermission,
    hasAllPermissions,
    hasAnyPermission,
    role
  } = usePermissions(['tests:write', 'admin:read']);

  if (!hasPermission('admin:read')) {
    return <AccessDenied />;
  }

  return <div>Admin Panel</div>;
}
```

**Permission Strings:**

| Permission | Description |
|------------|-------------|
| `tests:read` | Read test data |
| `tests:write` | Create/update tests |
| `tests:execute` | Run test executions |
| `projects:read` | View projects |
| `projects:write` | Manage projects |
| `reports:read` | View reports |
| `admin:*` | Full admin access |
| `*` | Wildcard (owner only) |

### 10.3 useNotificationChannels Hook

Manage notification channels:

```typescript
import {
  useNotificationChannels,
  useCreateNotificationChannel,
  useDeleteNotificationChannel,
  useTestNotificationChannel
} from '@/lib/hooks/use-notifications';

function NotificationSettings() {
  const { data: channels, isLoading } = useNotificationChannels();
  const createChannel = useCreateNotificationChannel();
  const deleteChannel = useDeleteNotificationChannel();
  const testChannel = useTestNotificationChannel();

  const handleCreate = () => {
    createChannel.mutate({
      name: 'Slack Alerts',
      channel_type: 'slack',
      config: { webhook_url: '...' },
      enabled: true,
      rate_limit_per_hour: 60,
      rules: [{ event_type: 'test.failed', priority: 'high' }]
    });
  };

  return <div>...</div>;
}
```

### 10.4 useNotificationStats Hook

Get notification statistics:

```typescript
import { useNotificationStats } from '@/lib/hooks/use-notifications';

function NotificationDashboard() {
  const stats = useNotificationStats();

  return (
    <div>
      <p>Total Channels: {stats.totalChannels}</p>
      <p>Enabled: {stats.enabledChannels}</p>
      <p>Sent Today: {stats.notificationsSentToday}</p>
      <p>Success Rate: {stats.successRate.toFixed(1)}%</p>
    </div>
  );
}
```

---

## 11. Database Schema

### 11.1 Entity Relationship Diagram

```
+----------------+       +----------------------+       +-------------+
| user_profiles  |       | organization_members |       | api_keys    |
+----------------+       +----------------------+       +-------------+
| id (PK)        |       | id (PK)              |       | id (PK)     |
| user_id (UK)   |       | organization_id (FK) |<------| org_id (FK) |
| email          |       | user_id              |       | name        |
| display_name   |       | email                |       | key_hash    |
| timezone       |       | role                 |       | scopes      |
| theme          |       | status               |       | expires_at  |
| notification_  |       | invited_by           |       | revoked_at  |
|   preferences  |       +----------+-----------+       | created_by  |
| default_org_id |                  |                   +-------------+
+-------+--------+                  |
        |                           |
        |    +----------------------+
        |    |
        v    v
+----------------+       +----------------------+
| organizations  |       | notification_channels|
+----------------+       +----------------------+
| id (PK)        |<------| organization_id (FK) |
| name           |       | name                 |
| slug (UK)      |       | channel_type         |
| plan           |       | config               |
| ai_budget_*    |       | enabled              |
| settings       |       +----------------------+
| features       |                  |
| stripe_*       |                  v
+----------------+       +----------------------+
                         | notification_rules   |
                         +----------------------+
                         | channel_id (FK)      |
                         | event_type           |
                         | conditions           |
                         | priority             |
                         +----------------------+
```

### 11.2 Indexes

```sql
-- User Profiles
CREATE UNIQUE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_profiles_email ON user_profiles(email);
CREATE INDEX idx_user_profiles_default_org ON user_profiles(default_organization_id);

-- Organizations
CREATE UNIQUE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_domain ON organizations(domain) WHERE domain IS NOT NULL;

-- Organization Members
CREATE UNIQUE INDEX idx_org_members_org_user ON organization_members(organization_id, user_id);
CREATE INDEX idx_org_members_email ON organization_members(email);
CREATE INDEX idx_org_members_status ON organization_members(status);

-- API Keys
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_org ON api_keys(organization_id);
CREATE INDEX idx_api_keys_expires ON api_keys(expires_at) WHERE revoked_at IS NULL;
```

---

## 12. Security Considerations

### 12.1 Authentication Methods

| Method | Use Case | Security Level |
|--------|----------|----------------|
| Clerk JWT | Dashboard/Web UI | High |
| API Key | Programmatic access | High |
| Service Token | Internal services | High |

### 12.2 Row-Level Security

All tables with organization data have RLS policies:

```sql
-- Organizations accessible only to members
CREATE POLICY "org_member_access" ON organizations
    FOR ALL USING (
        id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = current_setting('app.user_id', true)
        )
    );

-- API keys scoped to organization
CREATE POLICY "api_keys_org_access" ON api_keys
    FOR ALL USING (
        organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = current_setting('app.user_id', true)
            AND role IN ('owner', 'admin')
        )
    );
```

### 12.3 Audit Logging

All administrative actions are logged:

```sql
audit_logs
├── id                          UUID (PK)
├── organization_id             UUID (FK)
├── user_id                     TEXT
├── user_email                  TEXT
├── action                      TEXT
├── resource_type               TEXT
├── resource_id                 TEXT
├── description                 TEXT
├── metadata                    JSONB
├── ip_address                  TEXT
├── user_agent                  TEXT
├── status                      TEXT
└── created_at                  TIMESTAMPTZ
```

### 12.4 API Key Security Best Practices

1. **Never log full keys** - Only store hashes, display prefixes
2. **Rotate regularly** - Use the rotate endpoint for seamless key rotation
3. **Minimal scopes** - Grant only necessary permissions
4. **Expiration** - Set reasonable expiration (90-365 days)
5. **Monitor usage** - Track `last_used_at` and `request_count`
6. **Revoke promptly** - Remove unused or compromised keys immediately

---

*Document generated: 2026-01-12*
*Architecture Version: 2.2.0*
*Argus E2E Testing Agent*
