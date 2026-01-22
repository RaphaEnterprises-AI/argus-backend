# Secrets Management & Rotation Guide

**Last Updated**: 2026-01-22
**Audit Status**: Compliant

This document outlines the secrets management practices and rotation procedures for the Argus E2E Testing Agent.

---

## Secrets Inventory

### Critical Secrets (Rotate Quarterly)

| Secret | Location | Purpose | Rotation Impact |
|--------|----------|---------|-----------------|
| `SUPABASE_SERVICE_KEY` | Backend, Workers | Database admin access | Zero downtime if rotated via dashboard |
| `DATABASE_URL` | Backend | Direct PostgreSQL connection | Requires backend restart |
| `JWT_SECRET_KEY` | Backend | Session signing | Invalidates all sessions |
| `BROWSER_POOL_JWT_SECRET` | Workers, K8s | Browser pool auth | Requires coordinated update |
| `KEK_SECRET` | Cloudflare Worker | BYOK encryption master key | **CRITICAL** - Cannot be rotated without re-encrypting all DEKs |

### Standard Secrets (Rotate Annually)

| Secret | Location | Purpose | Rotation Impact |
|--------|----------|---------|-----------------|
| `OPENROUTER_API_KEY` | Backend, Workers | LLM access | Immediate, no session impact |
| `ANTHROPIC_API_KEY` | Backend | Direct Claude access | Immediate, no session impact |
| `CLOUDFLARE_API_TOKEN` | Backend | R2/KV/Vectorize access | Immediate |
| `CLOUDFLARE_R2_SECRET_ACCESS_KEY` | Backend | Presigned URL generation | Immediate |
| `GITHUB_TOKEN` | CI/CD | PR creation | Immediate |
| `SENTRY_DSN` | Backend | Error tracking | Can be changed anytime |

### Environment-Specific

| Environment | Configuration |
|-------------|---------------|
| Development | `.env.local` (git-ignored) |
| CI/CD | GitHub Actions Secrets |
| Staging | Railway/Vercel environment variables |
| Production | Railway + Cloudflare Worker secrets |

---

## Rotation Procedures

### 1. Supabase Keys

**Service Role Key:**
```bash
# 1. Generate new key in Supabase Dashboard > Project Settings > API
# 2. Update in all locations:

# Backend (.env or Railway)
SUPABASE_SERVICE_KEY=new-jwt-key-here

# Cloudflare Worker
wrangler secret put SUPABASE_SERVICE_KEY --env production

# 3. Verify connectivity
curl -H "apikey: $NEW_KEY" https://your-project.supabase.co/rest/v1/

# 4. Revoke old key in Supabase Dashboard
```

**Anon Key (Public - No rotation needed unless compromised):**
- Can be rotated in Supabase Dashboard
- Update `NEXT_PUBLIC_SUPABASE_ANON_KEY` in dashboard deploy

### 2. Cloudflare Worker Secrets

```bash
# List current secrets
wrangler secret list

# Rotate a secret
wrangler secret put SECRET_NAME
# Enter new value when prompted

# For production
wrangler secret put SECRET_NAME --env production
```

### 3. Browser Pool JWT Secret

**Coordinated rotation required:**

```bash
# 1. Generate new secret
NEW_SECRET=$(openssl rand -hex 64)

# 2. Update Kubernetes secret
kubectl create secret generic browser-pool-secrets \
  --from-literal=JWT_SECRET=$NEW_SECRET \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Restart browser-pool pods
kubectl rollout restart deployment/browser-pool

# 4. Update Cloudflare Worker
wrangler secret put BROWSER_POOL_JWT_SECRET
# Paste $NEW_SECRET

# 5. Verify
curl -H "Authorization: Bearer $(jwt encode --secret $NEW_SECRET ...)" \
  https://browser-pool.heyargus.ai/health
```

### 4. KEK (Key Encryption Key) - CRITICAL

**WARNING**: KEK cannot be rotated without re-encrypting all stored DEKs. This is by design for envelope encryption security.

If KEK rotation is required:
1. Create new KEK
2. Decrypt all DEKs with old KEK
3. Re-encrypt all DEKs with new KEK
4. Update KEK in Cloudflare Worker
5. Verify all BYOK keys are accessible

### 5. GitHub Tokens

```bash
# 1. Go to GitHub > Settings > Developer settings > Personal access tokens
# 2. Generate new token with required scopes:
#    - repo (for PR creation)
#    - workflow (for CI triggers)

# 3. Update in GitHub Actions Secrets
#    Repository > Settings > Secrets and variables > Actions

# 4. Revoke old token after verification
```

### 6. Database Password

```bash
# 1. In Supabase Dashboard > Project Settings > Database
# 2. Reset database password

# 3. Update DATABASE_URL in all locations
#    Format: postgresql://postgres:[NEW_PASSWORD]@db.xxx.supabase.co:5432/postgres

# 4. Restart backend services
railway up  # or docker-compose restart
```

---

## Secret Storage Best Practices

### Git-Ignored Files
Ensure `.gitignore` includes:
```
.env
.env.local
.env.*.local
*.env
secrets/
*.key
*.pem
credentials.json
secrets.yaml
```

### GitHub Actions
- Use `${{ secrets.SECRET_NAME }}` syntax
- Never echo secrets in logs
- Use environment-specific secrets when possible

### Cloudflare Workers
- Use `wrangler secret put` for all sensitive values
- Never put secrets in `wrangler.toml` `[vars]` section
- Secrets are encrypted at rest

### Kubernetes
- Use Kubernetes Secrets objects
- Enable encryption at rest: `--encryption-provider-config`
- Use external secrets operators for production (Vault, AWS Secrets Manager)

---

## Audit Checklist

Run this checklist monthly:

- [ ] Scan git history: `gitleaks detect --source . --verbose`
- [ ] Verify `.gitignore` is correct
- [ ] Check for hardcoded secrets: `grep -rn "sk-" src/`
- [ ] Review GitHub Actions secrets list
- [ ] Verify Cloudflare Worker secrets: `wrangler secret list`
- [ ] Check Supabase key usage in dashboard
- [ ] Review API key access logs

---

## Emergency Procedures

### If a Secret is Leaked

1. **Immediate**: Revoke/rotate the compromised secret
2. **Assess**: Check access logs for unauthorized usage
3. **Notify**: Alert the team via secure channel
4. **Document**: Record the incident for SOC 2 compliance
5. **Prevent**: Add the pattern to gitleaks configuration

### Gitleaks Configuration

Add false positives to `.gitleaks.toml`:
```toml
[allowlist]
paths = [
    '''tests/fixtures/.*''',
    '''.*_test\.py''',
]
regexes = [
    '''sk-ant-test-.*''',  # Test keys
    '''EXAMPLE_.*''',       # Documentation examples
]
```

---

## Compliance Notes

### SOC 2 Requirements
- Secrets must be rotated at least annually
- All rotations must be documented
- Access to secrets must be audited
- Emergency rotation procedures must be documented

### GDPR Considerations
- Encryption keys protecting PII must be managed securely
- Key rotation should not cause data loss
- Document key management for data protection audits
