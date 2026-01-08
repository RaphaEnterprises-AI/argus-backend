# Enterprise Security Architecture

This document describes the security features built into the E2E Testing Agent for enterprise deployments.

## Overview

The E2E Testing Agent is designed with **security-first principles** to ensure:

1. **No secrets leak to AI** - Automatic detection and redaction
2. **User consent** - Explicit approval before sending data
3. **Audit trails** - SOC2/ISO27001 compliant logging
4. **Data classification** - Multi-level sensitivity handling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SECURITY DATA FLOW                               │
└─────────────────────────────────────────────────────────────────────────┘

  Your Codebase                                              Anthropic AI
       │                                                          │
       ▼                                                          │
┌──────────────┐                                                  │
│ 1. CLASSIFY  │  Identify sensitivity level                     │
│    FILES     │  (public/internal/confidential/restricted)       │
└──────┬───────┘                                                  │
       ▼                                                          │
┌──────────────┐                                                  │
│ 2. CHECK     │  User approval required                         │
│    CONSENT   │  before any data leaves                         │
└──────┬───────┘                                                  │
       ▼                                                          │
┌──────────────┐                                                  │
│ 3. SANITIZE  │  Remove secrets, API keys,                      │
│    CONTENT   │  passwords, tokens, PII                         │
└──────┬───────┘                                                  │
       ▼                                                          │
┌──────────────┐                                                  │
│ 4. AUDIT     │  Log all access for                             │
│    LOG       │  compliance reporting                           │
└──────┬───────┘                                                  │
       ▼                                                          ▼
       └─────────────────────────────────────────────────────────►│
                        SANITIZED DATA ONLY                       │
                                                                  ▼
                                                        ┌──────────────┐
                                                        │   CLAUDE     │
                                                        │   ANALYSIS   │
                                                        └──────────────┘
```

## Components

### 1. Data Classification (`classifier.py`)

Classifies every file by sensitivity level:

| Level | Description | AI Access |
|-------|-------------|-----------|
| `PUBLIC` | Open source, licenses | Yes |
| `INTERNAL` | Source code, tests | Yes (with consent) |
| `CONFIDENTIAL` | Config files | Yes (after sanitization) |
| `RESTRICTED` | Secrets, credentials | **NEVER** |

```python
from src.security import DataClassifier, SensitivityLevel

classifier = DataClassifier()
result = classifier.classify_file("/path/to/file.py")

if result.sensitivity == SensitivityLevel.RESTRICTED:
    print("This file will NEVER be sent to AI")
elif result.pii_detected:
    print("PII detected - special handling required")
```

#### Automatic Detection

The classifier automatically detects:

- **Credentials**: `.env`, `credentials.json`, private keys
- **Secrets**: API keys, tokens, passwords
- **PII**: Email addresses, phone numbers, SSN, credit cards
- **Sensitive Config**: Database URLs, connection strings

### 2. Code Sanitization (`sanitizer.py`)

Removes secrets from code before sending to AI:

```python
from src.security import CodeSanitizer

sanitizer = CodeSanitizer()
result = sanitizer.sanitize_file("/path/to/config.py")

# Original content:
# API_KEY = "sk-ant-1234567890abcdef"
# DATABASE_URL = "postgres://user:password@host/db"

# Sanitized content:
# API_KEY = "[REDACTED]:api_key"
# DATABASE_URL = "[REDACTED]:connection_string"
```

#### Secret Patterns Detected

| Type | Pattern Examples |
|------|------------------|
| API Keys | `sk-ant-*`, `AIza*`, `xox*-*` |
| Tokens | `ghp_*`, `glpat-*`, Bearer tokens |
| Passwords | password=, pwd=, passwd= |
| AWS | `AKIA*`, aws_secret_access_key |
| Private Keys | `-----BEGIN PRIVATE KEY-----` |
| Connection Strings | `postgres://`, `mongodb://` |
| JWT | `eyJ*.*.*` |

#### Forbidden Files

These files are **never read**, even if requested:

- `.env`, `.env.local`, `.env.production`
- `credentials.json`, `secrets.yaml`
- `id_rsa`, `*.pem`, `*.key`, `*.p12`
- `.npmrc`, `.pypirc`, `.netrc`
- `.aws/credentials`

### 3. User Consent (`consent.py`)

Explicit consent required before sending data to external services:

```python
from src.security import ConsentManager, ConsentScope

consent = ConsentManager(session_id="user-session-123")

# Check if consent exists
if not consent.has_consent(ConsentScope.SEND_TO_ANTHROPIC):
    # Prompt user for consent
    consent.prompt_for_consent([
        ConsentScope.SOURCE_CODE,
        ConsentScope.SEND_TO_ANTHROPIC,
    ])

# Now safe to proceed
consent.require_consent(ConsentScope.SEND_TO_ANTHROPIC)
```

#### Consent Scopes

| Scope | Description |
|-------|-------------|
| `SOURCE_CODE` | Read source code files |
| `TEST_FILES` | Read existing test files |
| `CONFIG_FILES` | Read configuration (sanitized) |
| `SCREENSHOTS` | Capture browser screenshots |
| `BROWSER_ACTIONS` | Execute browser automation |
| `API_RESPONSES` | Capture API responses |
| `SEND_TO_ANTHROPIC` | Send data to Claude API |
| `SEND_TO_GITHUB` | Post PR comments |
| `SEND_TO_SLACK` | Send notifications |
| `STORE_LOCALLY` | Save results to disk |
| `STORE_AUDIT_LOGS` | Maintain audit trail |

#### Consent Modes

For CLI usage, you can use auto-consent modes:

```bash
# Minimal - only code analysis
CONSENT_MODE=minimal e2e-agent ...

# Standard - typical testing workflow
CONSENT_MODE=standard e2e-agent ...

# Full - all features enabled
CONSENT_MODE=full e2e-agent ...
```

### 4. Audit Logging (`audit.py`)

SOC2/ISO27001 compliant audit trail:

```python
from src.security import get_audit_logger, AuditEventType

audit = get_audit_logger()

# Log AI request
audit.log_ai_request(
    user_id="user-123",
    model="claude-sonnet-4-5",
    action="analyze_code",
    prompt_hash="abc123...",  # Never log actual prompts
    input_tokens=1500,
)

# Log file access
audit.log_file_read(
    user_id="user-123",
    file_path="/app/src/main.py",
    classification="internal",
    was_sanitized=True,
    secrets_redacted=3,
)

# Generate compliance report
report = audit.generate_compliance_report(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
)
```

#### Audit Events

| Event Type | When Logged |
|------------|-------------|
| `AI_REQUEST` | Every API call to Claude |
| `AI_RESPONSE` | Every response received |
| `FILE_READ` | Every file accessed |
| `SECRET_DETECTED` | When secrets are redacted |
| `TEST_COMPLETED` | Test execution results |
| `BROWSER_ACTION` | Browser automation actions |
| `INTEGRATION_CONNECTED` | External service connections |

#### Audit Log Format

Logs are stored in JSONL format with automatic rotation:

```json
{
  "id": "uuid",
  "timestamp": "2024-01-15T10:30:00Z",
  "event_type": "file_read",
  "user_id": "user-123",
  "session_id": "session-456",
  "action": "read",
  "resource": "/app/src/config.py",
  "data_classification": "confidential",
  "metadata": {
    "was_sanitized": true,
    "secrets_redacted": 2
  },
  "content_hash": "abc123..."
}
```

## Secure Code Reader (`secure_reader.py`)

The `SecureCodeReader` integrates all security components:

```python
from src.security import create_secure_reader

# Create reader with security features
reader = create_secure_reader(
    user_id="user-123",
    auto_consent_mode="standard",  # Or prompt interactively
)

# Read codebase - automatically:
# 1. Checks consent
# 2. Classifies files
# 3. Skips restricted files
# 4. Sanitizes secrets
# 5. Logs all access
results = reader.read_codebase("/path/to/app")

# Get safe content for AI
context = reader.get_context_for_ai(results)
# This content is SAFE to send to Claude
```

## Enterprise Configuration

### Environment Variables

```bash
# Audit logging
AUDIT_LOG_DIR=./audit-logs
AUDIT_RETENTION_DAYS=90

# Consent
CONSENT_MODE=standard  # minimal, standard, full
REQUIRE_EXPLICIT_CONSENT=true

# Classification
STRICT_CLASSIFICATION=true  # Unknown files = confidential
SCAN_FOR_PII=true
```

### Custom Secret Patterns

Add organization-specific patterns:

```python
from src.security import CodeSanitizer, SecretType

sanitizer = CodeSanitizer(
    additional_patterns={
        SecretType.API_KEY: [
            r"MYORG_API_KEY_[a-zA-Z0-9]{32}",
        ],
    },
    additional_forbidden_files={
        "internal-secrets.yaml",
        "*.myorg-key",
    },
)
```

### Custom Forbidden Directories

```python
sanitizer = CodeSanitizer(
    additional_forbidden_dirs={
        "internal-tools",
        "legacy-secrets",
    },
)
```

## Compliance Reports

Generate compliance reports for auditors:

```python
from datetime import datetime, timedelta
from src.security import get_audit_logger

audit = get_audit_logger()

# Last 30 days
report = audit.generate_compliance_report(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
)

print(f"Total AI requests: {report['summary']['ai_requests']}")
print(f"Files accessed: {report['summary']['files_accessed']}")
print(f"Secrets detected: {report['summary']['secrets_detected']}")
print(f"Total cost: ${report['summary']['total_cost_usd']:.2f}")
```

### Sample Report

```json
{
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "summary": {
    "total_events": 15420,
    "ai_requests": 342,
    "files_accessed": 1256,
    "secrets_detected": 47,
    "tests_run": 890,
    "total_cost_usd": 45.67
  },
  "by_user": {
    "user-123": {"events": 8200, "cost_usd": 25.00},
    "user-456": {"events": 7220, "cost_usd": 20.67}
  },
  "secrets_by_type": {
    "api_key": 23,
    "password": 12,
    "token": 8,
    "connection_string": 4
  },
  "errors": []
}
```

## Security Best Practices

### 1. Never Skip Sanitization

```python
# WRONG - direct file read
content = Path("config.py").read_text()
send_to_ai(content)  # May contain secrets!

# CORRECT - use secure reader
reader = create_secure_reader(user_id="me")
result = reader.read_file("config.py")
send_to_ai(result.content)  # Safe, sanitized
```

### 2. Always Check Consent

```python
# WRONG - assume consent
send_data_to_anthropic(data)

# CORRECT - verify consent
consent = get_consent_manager()
consent.require_consent(ConsentScope.SEND_TO_ANTHROPIC)
send_data_to_anthropic(data)
```

### 3. Log Everything

```python
# WRONG - silent operation
response = call_ai_api(prompt)

# CORRECT - audit trail
audit.log_ai_request(...)
response = call_ai_api(prompt)
audit.log_ai_response(...)
```

### 4. Use Content Hashes

```python
# WRONG - log actual content
audit.log(content=sensitive_data)

# CORRECT - log hash only
from src.security import hash_content
audit.log(content_hash=hash_content(sensitive_data))
```

## Deployment Considerations

### On-Premise Deployment

For maximum security, deploy the agent on-premise:

1. **No data leaves your network** except to Anthropic API
2. **Audit logs stay local** on your infrastructure
3. **Custom secret patterns** for your organization
4. **Integration with your IAM** for user authentication

### Air-Gapped Environments

For air-gapped or regulated environments:

1. Use a local LLM instead of Anthropic API
2. Disable all external integrations
3. Store all audit logs locally
4. Manual export for compliance reports

### Cloud Deployment

For cloud deployments:

1. Use secrets manager (AWS Secrets Manager, GCP Secret Manager)
2. Enable encryption at rest for audit logs
3. Set up log shipping to SIEM
4. Configure VPC endpoints for Anthropic API

## Regulatory Compliance

### GDPR

- PII detection and redaction
- Explicit consent management
- Data subject access requests via audit logs
- Right to erasure via log retention policies

### SOC2

- Comprehensive audit trails
- Access logging
- Change management tracking
- Security event monitoring

### HIPAA

- PHI detection (when enabled)
- Access controls
- Audit requirements
- Encryption support

### PCI-DSS

- Credit card number detection
- Access logging
- Secure data handling
- Audit requirements

## Troubleshooting

### "Consent not granted" Error

```
PermissionError: Consent required for 'send_to_anthropic'
```

**Solution**: Grant consent explicitly or use auto-consent mode:

```python
consent = ConsentManager(auto_grant_mode="standard")
```

### Files Being Skipped

Check if files are classified as RESTRICTED:

```python
classifier = DataClassifier()
result = classifier.classify_file("path/to/file")
print(f"Sensitivity: {result.sensitivity}")
print(f"Reasons: {result.reasons}")
```

### Secrets Not Being Detected

Add custom patterns for your organization:

```python
sanitizer = CodeSanitizer(
    additional_patterns={
        SecretType.API_KEY: [r"MY_CUSTOM_PATTERN"],
    }
)
```

## Support

For security concerns or vulnerabilities, please contact:
- Security Team: security@yourcompany.com
- Create a private security advisory on GitHub
