# Secret Rotation Runbook

This document provides step-by-step procedures for rotating secrets in the Argus platform.

## Table of Contents

- [Overview](#overview)
- [Rotation Schedule](#rotation-schedule)
- [Critical Secrets Inventory](#critical-secrets-inventory)
- [Rotation Procedures](#rotation-procedures)
  - [Supabase Secrets](#supabase-secrets)
  - [Anthropic API Key](#anthropic-api-key)
  - [OpenRouter API Key](#openrouter-api-key)
  - [Langfuse Secrets](#langfuse-secrets)
  - [Confluent/Kafka SASL Credentials](#confluentkafka-sasl-credentials)
  - [Railway Deployment Secrets](#railway-deployment-secrets)
  - [Cloudflare Secrets](#cloudflare-secrets)
  - [Vercel Secrets](#vercel-secrets)
- [Post-Rotation Verification](#post-rotation-verification)
- [Emergency Rotation](#emergency-rotation)
- [Rotation Log](#rotation-log)

---

## Overview

### When to Rotate

- **Immediately**: If a secret is exposed in git history, logs, or public channels
- **Scheduled**: Every 90 days for production secrets
- **On-demand**: When team members with access leave

### Before You Begin

1. Ensure you have admin access to all relevant services
2. Schedule a maintenance window if needed
3. Notify the team via Slack #argus-ops
4. Have rollback procedures ready

---

## Rotation Schedule

| Secret Type | Rotation Frequency | Last Rotated | Next Due |
|-------------|-------------------|--------------|----------|
| Supabase Service Key | 90 days | - | - |
| Anthropic API Key | 90 days | - | - |
| OpenRouter API Key | 90 days | - | - |
| Langfuse Secrets | 90 days | - | - |
| Confluent SASL | 90 days | - | - |
| Railway Secrets | 90 days | - | - |
| Cloudflare API Token | 90 days | - | - |

---

## Critical Secrets Inventory

### Identified from Gitleaks Scan (2026-01-30)

The following secrets were found in git history and **MUST be rotated**:

#### 1. Langfuse Secrets (data-layer/kubernetes/monitoring/)

| Secret | File | Action Required |
|--------|------|-----------------|
| PostgreSQL password | langfuse-secrets.yaml:40 | Rotate in Kubernetes + Langfuse |
| Root password | langfuse-secrets.yaml:54 | Rotate in Kubernetes |
| NextAuth secret | langfuse-secrets.yaml:67 | Rotate in Kubernetes + Langfuse |
| Project secret | langfuse-secrets.yaml:70 | Rotate in Langfuse dashboard |
| S3 secret access key | langfuse-secrets.yaml:86 | Rotate in MinIO/S3 |

#### 2. Confluent/Kafka SASL

| Secret | File | Action Required |
|--------|------|-----------------|
| SASL password | scripts/create_confluent_topics.py:12 | Rotate in Confluent Cloud |
| SASL password | scripts/validate_confluent.py:12 | Same as above |

#### 3. Flink SASL

| Secret | File | Action Required |
|--------|------|-----------------|
| SASL password | data-layer/kubernetes/flink-cluster.yaml:89 | Rotate in Redpanda |
| SASL password | data-layer/kubernetes/flink-platform/keda-autoscaler.yaml:79 | Same as above |

#### 4. Supabase JWT

| Secret | File | Action Required |
|--------|------|-----------------|
| Service role JWT | dashboard/scripts/test-chat-api.ts:9 | Rotate in Supabase dashboard |

---

## Rotation Procedures

### Supabase Secrets

**Secrets to rotate:**
- `SUPABASE_URL` (if compromised)
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_KEY`

**Steps:**

1. **Generate new keys in Supabase Dashboard**
   ```
   1. Go to https://supabase.com/dashboard/project/YOUR_PROJECT/settings/api
   2. Click "Regenerate" for the compromised key
   3. Copy the new key immediately
   ```

2. **Update Railway environment variables**
   ```bash
   railway variables set SUPABASE_SERVICE_KEY="new_key_here"
   railway variables set SUPABASE_ANON_KEY="new_key_here"
   ```

3. **Update Vercel environment variables (Dashboard)**
   ```
   1. Go to Vercel project settings
   2. Update NEXT_PUBLIC_SUPABASE_ANON_KEY
   3. Redeploy
   ```

4. **Update Kubernetes secrets (if applicable)**
   ```bash
   kubectl create secret generic supabase-secrets \
     --from-literal=service-key="new_key" \
     --from-literal=anon-key="new_key" \
     -n argus-data --dry-run=client -o yaml | kubectl apply -f -
   ```

5. **Verify**
   ```bash
   curl -H "apikey: NEW_ANON_KEY" https://YOUR_PROJECT.supabase.co/rest/v1/
   ```

---

### Anthropic API Key

**Steps:**

1. **Generate new key at Anthropic Console**
   ```
   1. Go to https://console.anthropic.com/settings/keys
   2. Create a new API key with same permissions
   3. Copy immediately (shown only once)
   ```

2. **Update Railway**
   ```bash
   railway variables set ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. **Verify**
   ```bash
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: NEW_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     -d '{"model": "claude-3-5-haiku-20241022", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}'
   ```

4. **Delete old key in Anthropic Console**

---

### OpenRouter API Key

**Steps:**

1. **Generate new key**
   ```
   1. Go to https://openrouter.ai/keys
   2. Create new key
   3. Copy immediately
   ```

2. **Update Railway**
   ```bash
   railway variables set OPENROUTER_API_KEY="sk-or-..."
   ```

3. **Verify**
   ```bash
   curl https://openrouter.ai/api/v1/models \
     -H "Authorization: Bearer NEW_KEY"
   ```

4. **Delete old key in OpenRouter dashboard**

---

### Langfuse Secrets

**Secrets to rotate:**
- PostgreSQL password
- NextAuth secret
- Project public/secret keys
- S3/MinIO credentials

**Steps:**

1. **PostgreSQL Password**
   ```bash
   # Generate new password
   NEW_PG_PASS=$(openssl rand -hex 32)

   # Update in PostgreSQL
   kubectl exec -n argus-data postgresql-0 -- \
     psql -U postgres -c "ALTER USER langfuse PASSWORD '$NEW_PG_PASS';"

   # Update Kubernetes secret
   kubectl create secret generic langfuse-db-secret \
     --from-literal=password="$NEW_PG_PASS" \
     -n monitoring --dry-run=client -o yaml | kubectl apply -f -

   # Restart Langfuse pods
   kubectl rollout restart deployment/langfuse -n monitoring
   ```

2. **NextAuth Secret**
   ```bash
   NEW_NEXTAUTH=$(openssl rand -base64 32)

   kubectl create secret generic langfuse-secrets \
     --from-literal=nextauth-secret="$NEW_NEXTAUTH" \
     -n monitoring --dry-run=client -o yaml | kubectl apply -f -
   ```

3. **Langfuse Project Keys**
   ```
   1. Go to Langfuse dashboard > Settings > API Keys
   2. Rotate public and secret keys
   3. Update in Railway and Kubernetes
   ```

---

### Confluent/Kafka SASL Credentials

**Steps:**

1. **Create new API key in Confluent Cloud**
   ```
   1. Go to https://confluent.cloud
   2. Navigate to Cluster > API Keys
   3. Create new key with same permissions
   4. Download credentials
   ```

2. **Update Kubernetes secrets**
   ```bash
   kubectl create secret generic kafka-sasl-credentials \
     --from-literal=username="NEW_API_KEY" \
     --from-literal=password="NEW_API_SECRET" \
     -n argus-data --dry-run=client -o yaml | kubectl apply -f -
   ```

3. **Restart consumers**
   ```bash
   kubectl rollout restart deployment/cognee-worker -n argus-data
   kubectl rollout restart deployment/flink-jobmanager -n argus-data
   ```

4. **Delete old API key in Confluent Cloud**

---

### Railway Deployment Secrets

All Railway secrets are managed via the Railway CLI or dashboard.

```bash
# List current variables
railway variables

# Set new variable
railway variables set KEY_NAME="new_value"

# Trigger redeploy
railway up
```

---

### Cloudflare Secrets

**Steps:**

1. **Rotate API Token**
   ```
   1. Go to https://dash.cloudflare.com/profile/api-tokens
   2. Create new token with same permissions
   3. Update in Railway: CLOUDFLARE_API_TOKEN
   ```

2. **Rotate R2 Access Keys**
   ```
   1. Go to R2 > Manage R2 API Tokens
   2. Create new token
   3. Update CLOUDFLARE_R2_ACCESS_KEY_ID and CLOUDFLARE_R2_SECRET_ACCESS_KEY
   ```

---

### Vercel Secrets

Managed via Vercel dashboard or CLI:

```bash
# List environment variables
vercel env ls

# Add/update variable
vercel env add VARIABLE_NAME

# Remove old variable
vercel env rm VARIABLE_NAME
```

---

## Post-Rotation Verification

After rotating any secret, verify the following:

### 1. Health Checks
```bash
# Backend API
curl https://argus-brain-production.up.railway.app/health

# Data layer
curl https://argus-brain-production.up.railway.app/api/v1/health/data-layer \
  -H "X-API-Key: YOUR_API_KEY"
```

### 2. Authentication Test
```bash
# Test with new API key
curl https://argus-brain-production.up.railway.app/api/v1/projects \
  -H "X-API-Key: YOUR_NEW_KEY"
```

### 3. Integration Tests
```bash
# Run integration tests
pytest tests/integration/ -v
```

### 4. Monitor Logs
```bash
# Railway logs
railway logs --follow

# Kubernetes logs
kubectl logs -n argus-data -l app=cognee-worker --tail=100 -f
```

---

## Emergency Rotation

If a secret is actively being exploited:

### Immediate Actions (< 5 minutes)

1. **Disable the compromised key immediately** in the service dashboard
2. **Generate a new key**
3. **Update Railway** (auto-deploys)
   ```bash
   railway variables set COMPROMISED_KEY="new_value"
   ```
4. **Monitor for unauthorized access** in service logs

### Post-Incident (< 24 hours)

1. Document the incident in the security log
2. Audit access logs for the compromised credential
3. Notify affected parties if data was accessed
4. Update rotation schedule
5. Review how the secret was exposed

---

## Rotation Log

| Date | Secret | Reason | Rotated By | Verified |
|------|--------|--------|------------|----------|
| 2026-01-30 | Initial audit | Gitleaks scan | - | Pending |
| | | | | |

---

## Best Practices

### DO

- ✅ Use environment variables, never hardcode secrets
- ✅ Use Kubernetes Secrets or external secret managers
- ✅ Rotate secrets on a regular schedule
- ✅ Use different secrets per environment
- ✅ Log all secret rotations

### DON'T

- ❌ Commit secrets to git (even in private repos)
- ❌ Share secrets via Slack, email, or chat
- ❌ Use the same secret across environments
- ❌ Keep old secrets active after rotation
- ❌ Store secrets in plain text files

---

## External Secret Management (Recommended)

For production, consider using:

1. **HashiCorp Vault** - Industry standard
2. **AWS Secrets Manager** - If using AWS
3. **Google Secret Manager** - If using GCP
4. **Azure Key Vault** - If using Azure
5. **Kubernetes External Secrets** - For K8s native integration

Example with External Secrets Operator:
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: supabase-secrets
spec:
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: supabase-secrets
  data:
    - secretKey: service-key
      remoteRef:
        key: argus/supabase
        property: service_key
```

---

## References

- [Gitleaks Documentation](https://github.com/gitleaks/gitleaks)
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Supabase Key Management](https://supabase.com/docs/guides/api/api-keys)
- [Railway Environment Variables](https://docs.railway.app/develop/variables)
