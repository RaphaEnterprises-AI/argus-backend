# Multi-Tenant Architecture

This document describes Argus's multi-tenant architecture, enabling complete data isolation between organizations while sharing infrastructure efficiently.

## Overview

Argus implements **organization-based multi-tenancy** where:
- Each organization has completely isolated data
- Users can belong to multiple organizations
- All API requests are scoped to an organization context
- Knowledge graphs (Cognee/Neo4j) use tenant-prefixed datasets

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MULTI-TENANT ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Org: Acme  │    │  Org: Beta   │    │  Org: Gamma  │              │
│  │   (UUID-1)   │    │   (UUID-2)   │    │   (UUID-3)   │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    TenantMiddleware                          │       │
│  │              (X-Organization-ID Header)                      │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         ▼                    ▼                    ▼                    │
│  ┌─────────────┐     ┌─────────────┐      ┌─────────────┐             │
│  │  Supabase   │     │   Neo4j     │      │   Kafka     │             │
│  │    (RLS)    │     │   Aura      │      │  (Redpanda) │             │
│  └─────────────┘     └─────────────┘      └─────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Isolation Strategy

### 1. Supabase (PostgreSQL) - Row Level Security

All tables use Row Level Security (RLS) policies that filter data by organization:

```sql
-- Helper function to check org access
CREATE FUNCTION auth.has_org_access(check_org_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    IF check_org_id IS NULL THEN RETURN FALSE; END IF;
    IF auth.is_service_role() THEN RETURN TRUE; END IF;
    RETURN check_org_id = ANY(auth.user_org_ids());
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- Example RLS policy
CREATE POLICY "org_select" ON projects
    FOR SELECT USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );
```

**Key tables with RLS:**
- `organizations` - Organization details
- `organization_members` - User-to-org membership
- `projects` - Projects scoped to organizations
- `tests`, `test_runs`, `test_results` - Test data
- `audit_logs` - Audit trail per organization
- All other feature tables (visual AI, discovery, etc.)

### 2. Neo4j Aura - Dataset Prefixing

Cognee knowledge graphs use **tenant-prefixed dataset names**:

```
Format: org_{org_id}_project_{project_id}_{type}

Examples:
- org_abc123_project_def456_codebase
- org_abc123_project_def456_tests
- org_abc123_project_def456_failures
```

**Neo4j Schema Constraints:**
```cypher
-- All nodes require org_id (NOT NULL constraint)
CREATE CONSTRAINT project_org_id_exists
FOR (n:Project) REQUIRE n.org_id IS NOT NULL;

-- Composite indexes lead with org_id for query performance
CREATE INDEX project_org_id_idx
FOR (n:Project) ON (n.org_id);

CREATE INDEX project_org_id_name_idx
FOR (n:Project) ON (n.org_id, n.name);
```

### 3. Kafka (Redpanda) - Message Key Partitioning

Events use `org_id:project_id` as the message key for partition-ordered processing:

```python
# Message key ensures all events for an org go to same partition
message_key = f"{org_id}:{project_id}"

# Event payload includes tenant context
{
    "event_type": "TEST_EXECUTED",
    "tenant": {
        "org_id": "abc123",
        "project_id": "def456",
        "user_id": "user789"
    },
    "payload": { ... }
}
```

## API Design

### Tenant Context

The `TenantContext` dataclass carries organization scope through the request lifecycle:

```python
@dataclass(frozen=True)
class TenantContext:
    org_id: str                          # Required - organization UUID
    project_id: Optional[str] = None     # Optional - project scope
    user_id: Optional[str] = None        # Optional - acting user
    user_email: Optional[str] = None     # Optional - for audit
    plan: str = "free"                   # Organization plan tier
    request_id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def cognee_dataset_prefix(self) -> str:
        """Generate Cognee dataset name prefix."""
        if self.project_id:
            return f"org_{self.org_id}_project_{self.project_id}"
        return f"org_{self.org_id}"
```

### Request Flow

```
1. Request arrives with X-Organization-ID header
                    ↓
2. TenantMiddleware extracts org_id, validates access
                    ↓
3. TenantContext stored in contextvars
                    ↓
4. Route handlers use TenantDep dependency
                    ↓
5. Database queries automatically scoped via RLS
                    ↓
6. Events emitted with tenant context
```

### API Endpoints

**New org-scoped endpoints (`/api/v1/orgs/{org_id}/...`):**

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/orgs` | List user's organizations |
| `GET /api/v1/orgs/{org_id}` | Get organization details |
| `GET /api/v1/orgs/{org_id}/projects` | List org's projects |
| `POST /api/v1/orgs/{org_id}/projects` | Create project in org |
| `GET /api/v1/orgs/{org_id}/projects/{id}` | Get project details |

**Legacy endpoints (still supported):**

Existing endpoints like `/api/v1/projects` continue to work using the `X-Organization-ID` header for scoping.

### FastAPI Dependencies

```python
from src.api.middleware.tenant import TenantDep, ProjectTenantDep

@router.get("/{org_id}/projects")
async def list_projects(
    org_id: str,
    tenant: TenantDep,  # Validates org access, provides context
):
    # tenant.org_id is guaranteed to match org_id
    # All queries automatically scoped
    ...

@router.post("/{org_id}/projects/{project_id}/tests")
async def create_test(
    org_id: str,
    project_id: str,
    tenant: ProjectTenantDep,  # Requires both org AND project context
):
    ...
```

## Dashboard Integration

### Organization Context Provider

The dashboard uses React Context to manage organization state:

```typescript
// lib/contexts/organization-context.tsx
export function OrganizationProvider({ children }) {
  const [currentOrg, setCurrentOrg] = useState<Organization | null>(null);
  const [organizations, setOrganizations] = useState<Organization[]>([]);

  // Fetch from /api/v1/orgs on mount
  // Store selection in localStorage
  // Inject org ID into global API client
}

// Usage in components
const { currentOrg, switchOrganization } = useCurrentOrg();
```

### API Client Integration

All API requests automatically include the organization ID:

```typescript
// lib/api-client.ts
export async function authenticatedFetch(url, options) {
  const orgId = getCurrentOrgId(); // From OrganizationContext

  const headers = {
    'Authorization': `Bearer ${token}`,
    'X-Organization-ID': orgId,  // Automatic injection
    ...options.headers,
  };

  return fetch(url, { ...options, headers });
}
```

### Organization Switcher

The sidebar includes an organization switcher component:

```typescript
// components/layout/org-switcher.tsx
export function OrganizationSwitcher() {
  const { currentOrg, organizations, switchOrganization } = useCurrentOrg();

  const handleSwitch = (org: Organization) => {
    switchOrganization(org.id);
    queryClient.invalidateQueries(); // Refetch all data
    router.refresh();
  };

  return (
    <Dropdown>
      <CurrentOrgDisplay org={currentOrg} />
      <OrgList orgs={organizations} onSelect={handleSwitch} />
      <CreateOrgButton />
    </Dropdown>
  );
}
```

## Event System

### Event Schemas

All events include tenant context for proper routing:

```python
class TenantInfo(BaseModel):
    org_id: str = Field(..., description="Organization ID (required)")
    project_id: Optional[str] = Field(None, description="Project ID")
    user_id: Optional[str] = Field(None, description="Acting user")

class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    tenant: TenantInfo
    metadata: EventMetadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

### Kafka Topics

| Topic | Description | Key Format |
|-------|-------------|------------|
| `argus.codebase.ingested` | Codebase analysis complete | `org:project` |
| `argus.test.created` | New test created | `org:project` |
| `argus.test.executed` | Test execution complete | `org:project` |
| `argus.test.failed` | Test failure detected | `org:project` |
| `argus.healing.requested` | Self-healing requested | `org:project` |
| `argus.dlq` | Dead letter queue | `org:project` |

### Cognee Worker Processing

The Cognee worker processes events with tenant isolation:

```python
async def process_event(self, event: dict):
    # Extract tenant context
    org_id = event["tenant"]["org_id"]
    project_id = event["tenant"]["project_id"]

    # Build tenant-specific dataset name
    dataset_name = f"org_{org_id}_project_{project_id}_codebase"

    # Configure Cognee with isolated dataset
    cognee.config.set_llm_config({...})

    # Process into tenant-specific knowledge graph
    await cognee.add(data, dataset_name=dataset_name)
    await cognee.cognify()
```

## Deployment Considerations

### Neo4j Aura Free Tier

**Limitations:**
- 200,000 nodes maximum
- Auto-pauses after 3 days of inactivity
- 30-60 second cold start wake-up time

**Mitigations:**
- Keep-alive CronJob runs every 2 days
- Extended probe timeouts (90s initial, 30s timeout)
- Retry logic with exponential backoff

```yaml
# kubernetes/neo4j-keepalive.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: neo4j-keepalive
spec:
  schedule: "0 3 */2 * *"  # Every 2 days at 3am UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: keepalive
              command: ["python", "-c", "...RETURN 1 query..."]
```

### Environment Variables

```bash
# Neo4j Aura
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<secret>

# Kafka/Redpanda
REDPANDA_BROKERS=redpanda.argus-data.svc:9092
REDPANDA_SASL_USERNAME=argus-service
REDPANDA_SASL_PASSWORD=<secret>

# Feature flags
MULTI_TENANT_ENABLED=true
COGNEE_GRAPH_PROVIDER=neo4j
```

## Security Considerations

### Data Isolation Guarantees

1. **Database Level**: RLS policies enforce org_id filtering on every query
2. **API Level**: TenantMiddleware validates org access before processing
3. **Event Level**: All events tagged with tenant context, validated on consumption
4. **Graph Level**: Dataset names include org_id, preventing cross-tenant queries

### Audit Trail

All organization actions are logged:

```python
await log_audit(
    organization_id=tenant.org_id,
    user_id=tenant.user_id,
    action="project.create",
    resource_type="project",
    resource_id=project_id,
    description=f"Created project '{name}'",
    metadata={"name": name},
    request=request,
)
```

### Service Role Bypass

Backend services use service role for cross-org operations:

```sql
CREATE FUNCTION auth.is_service_role()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN COALESCE(
        current_setting('request.jwt.claims', true)::jsonb ->> 'role' = 'service_role',
        FALSE
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;
```

## Migration Guide

### For Existing Single-Tenant Deployments

1. **Run RLS migration**: Apply `20260127000000_rls_multitenant_standardization.sql`
2. **Create default org**: Assign existing data to a default organization
3. **Update API clients**: Include `X-Organization-ID` header
4. **Deploy Cognee worker**: With Neo4j configuration
5. **Verify isolation**: Test that users only see their org's data

### For New Deployments

Multi-tenancy is enabled by default. Simply:
1. Create organizations via `/api/v1/organizations`
2. Invite users via `/api/v1/invitations`
3. Create projects scoped to organizations
4. All data automatically isolated

## Troubleshooting

### Common Issues

**Q: User sees "Organization not found" error**
- Check `X-Organization-ID` header is being sent
- Verify user is a member of the organization
- Check `organization_members` table for active membership

**Q: Neo4j queries return empty results**
- Verify `org_id` is included in Cypher MATCH clauses
- Check dataset name format matches `org_{id}_project_{id}_{type}`
- Ensure Neo4j Aura instance is awake (not paused)

**Q: Events not being processed**
- Check Kafka topic exists with correct partitions
- Verify message key format is `org_id:project_id`
- Check Cognee worker logs for connection errors

### Debug Queries

```sql
-- Check user's organizations
SELECT o.*, om.role
FROM organizations o
JOIN organization_members om ON o.id = om.organization_id
WHERE om.user_id = 'user_xxx' AND om.status = 'active';

-- Check RLS is working
SET app.user_id = 'user_xxx';
SELECT * FROM projects;  -- Should only show user's org projects
```

```cypher
// Check Neo4j node counts by org
MATCH (n)
WHERE n.org_id IS NOT NULL
RETURN n.org_id, labels(n)[0] as type, count(*) as count
ORDER BY n.org_id, type;
```
