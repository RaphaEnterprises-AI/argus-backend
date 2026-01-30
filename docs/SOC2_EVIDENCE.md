# SOC 2 Type II Evidence Documentation

**Argus E2E Testing Platform**
**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Classification:** Internal - Auditor Access

---

## Executive Summary

This document provides comprehensive evidence of security controls implemented in the Argus platform, aligned with SOC 2 Trust Service Criteria (TSC). The evidence demonstrates our commitment to Security, Availability, Processing Integrity, Confidentiality, and Privacy.

---

## 1. Security (Common Criteria)

### 1.1 Access Control (CC6.1)

#### Authentication Mechanisms

| Control | Implementation | Evidence Location |
|---------|---------------|-------------------|
| JWT Authentication | Supabase Auth with JWT tokens | `src/api/security/auth.py` |
| API Key Authentication | Secure API key validation with hashing | `src/api/security/auth.py:185-250` |
| Device Authorization | OAuth 2.0 device flow for CLI | `src/api/auth.py` |
| Session Management | Configurable session expiry (default 7 days) | `src/config.py` |

**Code Reference:**
```python
# src/api/security/auth.py
async def get_current_user(request: Request) -> UserContext:
    """Validates JWT token or API key and returns user context."""
    # Multiple authentication methods supported
    # - Bearer JWT token
    # - X-API-Key header
    # - Device token
```

#### Authorization Controls

| Control | Implementation | Evidence |
|---------|---------------|----------|
| Row Level Security (RLS) | 541 RLS policies across 48 tables | `supabase/migrations/` |
| Tenant Isolation | TenantMiddleware enforces org-level access | `src/api/middleware/tenant.py` |
| IDOR Prevention | Project ownership validation | `src/api/middleware/tenant.py:338-411` |
| Role-Based Access | User, Admin, Owner roles per organization | Database schema |

**RLS Policy Evidence:**
```sql
-- Example: tests table RLS policy (from migrations)
CREATE POLICY "tests_tenant_isolation" ON tests
  USING (organization_id = auth.jwt() ->> 'organization_id');
```

### 1.2 Input Validation (CC6.6)

#### UUID Validation

All API endpoints validate UUID format for ID parameters to prevent injection attacks.

| Endpoint Category | Files Modified | Validation Function |
|-------------------|----------------|---------------------|
| API Testing | `src/api/api_testing.py` | `validate_uuid()` |
| Approvals | `src/api/approvals.py` | `validate_uuid()` |
| Audit | `src/api/audit.py` | `validate_uuid()` |
| Chat | `src/api/chat.py` | `validate_uuid()` |
| Correlations | `src/api/correlations.py` | `validate_uuid()` |
| Discovery | `src/api/discovery.py` | `validate_uuid()` |
| Healing | `src/api/healing.py` | `validate_uuid()` |
| Insights | `src/api/insights.py` | `validate_uuid()` |
| Integrations | `src/api/integrations.py` | `validate_uuid()` |
| Invitations | `src/api/invitations.py` | `validate_uuid()` |
| Notifications | `src/api/notifications.py` | `validate_uuid()` |
| Organizations | `src/api/organizations.py` | `validate_uuid()` |
| Parameterized | `src/api/parameterized.py` | `validate_uuid()` |
| Projects | `src/api/projects.py` | `validate_uuid()` |
| Quality | `src/api/quality.py` | `validate_uuid()` |
| Recording | `src/api/recording.py` | `validate_uuid()` |
| Reports | `src/api/reports.py` | `validate_uuid()` |
| SAST Analysis | `src/api/sast_analysis.py` | `validate_uuid()` |
| Teams | `src/api/teams.py` | `validate_uuid()` |
| Tests | `src/api/tests.py` | `validate_uuid()` |
| Webhooks | `src/api/webhooks.py` | `validate_uuid()` |

**Code Reference:**
```python
# src/api/middleware/tenant.py:414-448
def validate_uuid(value: str, field_name: str = "id") -> UUID:
    """Validate that a string is a valid UUID format.

    RAP-292: UUID Validation - Prevents invalid/malicious ID
    inputs from reaching database queries.
    """
    if not value:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: value cannot be empty")
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: not a valid UUID format")
```

#### SQL Injection Prevention

- All database queries use parameterized statements via Supabase client
- No string interpolation in SQL queries
- ORM-based queries where applicable

### 1.3 Rate Limiting (CC6.7)

Endpoint-specific rate limits protect against abuse and DDoS attacks.

#### LLM-Heavy Endpoints (10-15 req/min)
| Endpoint | Limit | Rationale |
|----------|-------|-----------|
| `/api/v1/chat/stream` | 10/min | High cost per request |
| `/api/v1/chat/message` | 15/min | LLM token consumption |
| `/api/v1/quality/generate-test` | 10/min | Test generation cost |
| `/api/v1/visual-ai/analyze` | 10/min | Vision API cost |

#### Authentication Endpoints (Brute Force Protection)
| Endpoint | Limit | Rationale |
|----------|-------|-----------|
| `/api/v1/auth/device/authorize` | 10/min | Prevent enumeration |
| `/api/v1/api-keys` | 10/min | Sensitive operations |

#### Data Modification Endpoints (5-30 req/min)
| Endpoint | Limit | Rationale |
|----------|-------|-----------|
| `/api/v1/organizations` | 5/min | Critical operations |
| `/api/v1/projects` | 20/min | Common operations |
| `/api/v1/tests` | 30/min | Bulk operations |

**Code Reference:**
```python
# src/api/security/middleware.py
ENDPOINT_LIMITS = {
    "/api/v1/chat/stream": {"requests": 10, "window": 60},
    "/api/v1/auth/device/authorize": {"requests": 10, "window": 60},
    # ... 35+ endpoint-specific limits
}
```

### 1.4 Security Headers (CC6.8)

| Header | Value | Purpose |
|--------|-------|---------|
| Content-Security-Policy | `default-src 'self'; script-src 'self' 'unsafe-inline'; ...` | XSS prevention |
| X-Frame-Options | `DENY` | Clickjacking protection |
| X-Content-Type-Options | `nosniff` | MIME sniffing prevention |
| Strict-Transport-Security | `max-age=31536000; includeSubDomains` | HTTPS enforcement |
| Referrer-Policy | `strict-origin-when-cross-origin` | Referrer leakage prevention |

---

## 2. Availability (A1)

### 2.1 Infrastructure

| Component | Provider | SLA | Monitoring |
|-----------|----------|-----|------------|
| API Backend | Railway | 99.99% | Health checks |
| Database | Supabase | 99.95% | Metrics dashboard |
| Data Layer | Vultr VKE | 99.99% | Prometheus/Grafana |
| CDN/Cache | Cloudflare | 99.99% | Real-time analytics |

### 2.2 Health Endpoints

```
GET /health          - Basic liveness check
GET /ready           - Readiness with dependency checks
GET /api/v1/health   - Detailed component status
```

### 2.3 Auto-Scaling

- Railway auto-scaling based on CPU/memory
- Kubernetes HPA for data layer components
- Cloudflare caching reduces origin load

---

## 3. Processing Integrity (PI1)

### 3.1 Audit Logging

All critical operations are logged for compliance and forensics.

**Audit Action Types:**
```python
# src/services/audit_logger.py
class AuditAction(str, Enum):
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"

    # Data Operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Test Operations
    TEST_RUN = "test_run"
    TEST_PASS = "test_pass"
    TEST_FAIL = "test_fail"

    # Security Events
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
```

### 3.2 Data Validation

- Pydantic models validate all request/response data
- Field validators enforce business rules
- Type checking with Python type hints

---

## 4. Confidentiality (C1)

### 4.1 Encryption at Rest

| Data Type | Encryption | Key Management |
|-----------|------------|----------------|
| Database | AES-256 (Supabase) | Supabase managed |
| API Keys | AES-256-GCM | Cloudflare Key Vault |
| Secrets | AES-256 | Environment variables |
| Backups | AES-256 | Provider managed |

### 4.2 Encryption in Transit

| Connection | Protocol | Certificate |
|------------|----------|-------------|
| API Traffic | TLS 1.3 | Let's Encrypt (auto-renewed) |
| Database | TLS 1.2+ | Supabase managed |
| Internal Services | mTLS | Kubernetes certificates |

### 4.3 Secret Management

```python
# .gitignore - Secrets never committed
.env
.env.local
.env.*.local
*.env
secrets/
*.key
*.pem
credentials.json
*-secrets.yaml
```

---

## 5. Privacy (P1)

### 5.1 Data Minimization

- Only necessary data collected for functionality
- No unnecessary PII storage
- Clear data retention policies

### 5.2 Access Controls

- Role-based access to user data
- Audit trails for data access
- Tenant isolation prevents cross-org access

---

## 6. Change Management

### 6.1 Version Control

- All code changes tracked in Git
- Pull request reviews required
- Automated CI/CD pipeline

### 6.2 Deployment Process

1. Code review approval required
2. Automated tests must pass
3. Security scan (SAST) clean
4. Staged rollout with monitoring

---

## 7. Monitoring and Alerting

### 7.1 Monitoring Stack

| Component | Purpose | Location |
|-----------|---------|----------|
| Prometheus | Metrics collection | Kubernetes |
| Grafana | Dashboards | monitoring.heyargus.ai |
| Alertmanager | Alert routing | Kubernetes |
| Langfuse | LLM observability | Cloud service |
| Sentry | Error tracking | Cloud service |

### 7.2 Alert Categories

- **Critical**: Security incidents, service down
- **High**: Performance degradation, error spikes
- **Medium**: Unusual patterns, capacity warnings
- **Low**: Informational, scheduled maintenance

---

## 8. Evidence Repository

### 8.1 Code Evidence

| Control | File Path | Commit |
|---------|-----------|--------|
| UUID Validation | `src/api/middleware/tenant.py:414-468` | `4304362` |
| Rate Limiting | `src/api/security/middleware.py` | `00273f2` |
| IDOR Prevention | `src/api/middleware/tenant.py:338-411` | `89bcc8c` |
| Audit Logging | `src/services/audit_logger.py` | Various |

### 8.2 Configuration Evidence

| Control | Location | Description |
|---------|----------|-------------|
| RLS Policies | `supabase/migrations/` | 541 policies |
| Security Headers | `src/api/security/middleware.py` | CSP, HSTS, etc. |
| CORS Config | `src/main.py` | Origin restrictions |

---

## 9. Attestation

This document accurately represents the security controls implemented in the Argus platform as of the document date. All code references can be verified in the source repository.

**Prepared by:** Engineering Team
**Reviewed by:** Security Team
**Date:** 2026-01-30

---

## Appendix A: Related Documents

1. `docs/architecture/` - System architecture diagrams
2. `docs/deployment/` - Deployment procedures
3. `CLAUDE.md` - Development guidelines
4. `supabase/migrations/` - Database schema and RLS policies

## Appendix B: Commit History (Security Fixes)

| Date | Commit | Description |
|------|--------|-------------|
| 2026-01-30 | `d3625c7` | Fix Sentry bugs (empty messages, invalid model IDs) |
| 2026-01-30 | `00273f2` | Harden rate limiting for enterprise compliance |
| 2026-01-30 | `4304362` | Add UUID validation to 122 API endpoints |
| 2026-01-30 | `89bcc8c` | Add UUID validation to high-priority endpoints |
| 2026-01-30 | `5a7a940` | Prevent SQL injection and IDOR vulnerabilities |
| 2026-01-30 | `ce07561` | Expand .gitignore for sensitive files |
