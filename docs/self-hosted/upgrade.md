# Upgrade Guide

Procedures for upgrading Argus Enterprise to new versions.

## Before You Upgrade

### 1. Review Release Notes

Check the changelog for:
- Breaking changes
- Database migrations
- New required configuration
- Deprecated features

```bash
# View available versions
helm search repo argus/argus-enterprise --versions
```

### 2. Backup Current State

```bash
# Export current values
helm get values argus -n argus -o yaml > values-pre-upgrade.yaml

# Backup database
kubectl exec -n argus argus-postgresql-0 -- \
  pg_dump -U argus -d argus -Fc > backup-pre-upgrade.dump

# Backup secrets
kubectl get secret argus-secrets -n argus -o yaml > secrets-pre-upgrade.yaml
```

### 3. Check Current Version

```bash
helm list -n argus
# NAME   NAMESPACE  REVISION  STATUS    CHART                    APP VERSION
# argus  argus      3         deployed  argus-enterprise-1.2.0   1.2.0
```

## Standard Upgrade

### Helm Upgrade

```bash
# Update repository
helm repo update

# Dry-run to preview changes
helm upgrade argus argus/argus-enterprise \
  --namespace argus \
  -f values.yaml \
  --dry-run

# Apply upgrade
helm upgrade argus argus/argus-enterprise \
  --namespace argus \
  -f values.yaml
```

### Monitor Upgrade

```bash
# Watch pods
kubectl get pods -n argus -w

# Check rollout status
kubectl rollout status deployment/argus-brain -n argus
kubectl rollout status deployment/argus-mcp -n argus

# Verify health
kubectl exec deploy/argus-brain -n argus -- curl -s localhost:8000/health
```

## Database Migrations

### Automatic Migrations

Migrations run automatically via init container:

```yaml
brain:
  migrations:
    enabled: true
    runOnUpgrade: true
```

### Manual Migrations

If needed, run migrations manually:

```bash
# Run migrations
kubectl exec deploy/argus-brain -n argus -- \
  alembic upgrade head

# Check current revision
kubectl exec deploy/argus-brain -n argus -- \
  alembic current
```

## Rolling Updates

### Zero-Downtime Upgrade

Ensure proper configuration for zero-downtime:

```yaml
brain:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  podDisruptionBudget:
    minAvailable: 2
```

### Canary Deployment

For major upgrades, use canary deployment:

```bash
# Deploy canary (10% traffic)
helm upgrade argus-canary argus/argus-enterprise \
  --namespace argus \
  -f values.yaml \
  --set brain.replicas=1 \
  --set brain.nameOverride=argus-canary

# Monitor canary
kubectl logs -f deploy/argus-canary-brain -n argus

# If successful, upgrade main deployment
helm upgrade argus argus/argus-enterprise \
  --namespace argus \
  -f values.yaml

# Remove canary
helm uninstall argus-canary -n argus
```

## Rollback

### Immediate Rollback

```bash
# View history
helm history argus -n argus

# Rollback to previous revision
helm rollback argus 2 -n argus

# Rollback to specific revision
helm rollback argus 1 -n argus
```

### Database Rollback

If migrations need rollback:

```bash
# Rollback last migration
kubectl exec deploy/argus-brain -n argus -- \
  alembic downgrade -1

# Rollback to specific revision
kubectl exec deploy/argus-brain -n argus -- \
  alembic downgrade abc123
```

## Major Version Upgrades

### 1.x to 2.x Example

```bash
# 1. Read migration guide for 2.x
# 2. Backup everything
helm get values argus -n argus -o yaml > values-1x.yaml
kubectl exec -n argus argus-postgresql-0 -- pg_dump -U argus -d argus -Fc > backup-1x.dump

# 3. Update values for 2.x compatibility
# Review breaking changes and update values.yaml

# 4. Scale down to prevent writes during migration
kubectl scale deploy argus-brain -n argus --replicas=0

# 5. Run pre-upgrade migrations if required
kubectl apply -f pre-upgrade-job.yaml

# 6. Upgrade
helm upgrade argus argus/argus-enterprise \
  --namespace argus \
  -f values-2x.yaml \
  --version 2.0.0

# 7. Verify
kubectl exec deploy/argus-brain -n argus -- curl -s localhost:8000/health
```

## Component-Specific Upgrades

### PostgreSQL Upgrade

```bash
# 1. Backup
kubectl exec -n argus argus-postgresql-0 -- \
  pg_dumpall -U postgres > full-backup.sql

# 2. Update chart values
postgresql:
  image:
    tag: "16"  # New version

# 3. Delete StatefulSet (keeps PVC)
kubectl delete statefulset argus-postgresql -n argus --cascade=orphan

# 4. Upgrade
helm upgrade argus argus/argus-enterprise -n argus -f values.yaml

# 5. Run pg_upgrade if needed
kubectl exec -n argus argus-postgresql-0 -- pg_upgrade ...
```

### Redis Upgrade

Redis upgrade is simpler (data can be rebuilt):

```yaml
redis:
  image:
    tag: "7.2"  # New version
```

```bash
helm upgrade argus argus/argus-enterprise -n argus -f values.yaml
```

### MinIO Upgrade

```yaml
minio:
  image:
    tag: "RELEASE.2026-01-15"  # New version
```

```bash
helm upgrade argus argus/argus-enterprise -n argus -f values.yaml
```

## Upgrade Checklist

### Pre-Upgrade

- [ ] Review release notes and breaking changes
- [ ] Backup database (pg_dump)
- [ ] Backup MinIO artifacts
- [ ] Export current Helm values
- [ ] Export secrets
- [ ] Test backup restoration in separate namespace
- [ ] Schedule maintenance window
- [ ] Notify stakeholders

### During Upgrade

- [ ] Run helm upgrade with --dry-run first
- [ ] Apply upgrade
- [ ] Monitor pod status
- [ ] Watch for migration errors
- [ ] Verify health endpoints

### Post-Upgrade

- [ ] Verify all pods running
- [ ] Test API endpoints
- [ ] Test MCP connection
- [ ] Run smoke tests
- [ ] Check logs for errors
- [ ] Verify metrics are flowing
- [ ] Update documentation
- [ ] Remove old backups after verification period

## Troubleshooting Upgrades

### Pods Stuck in Pending

```bash
# Check events
kubectl describe pod argus-brain-xxx -n argus

# Check resources
kubectl top nodes
```

### Migration Failed

```bash
# Check migration logs
kubectl logs job/argus-migration -n argus

# Manual recovery
kubectl exec deploy/argus-brain -n argus -- alembic current
kubectl exec deploy/argus-brain -n argus -- alembic stamp head  # If needed
```

### Rollback Failed

```bash
# Force rollback
helm rollback argus 1 -n argus --force

# Manual cleanup if needed
kubectl delete pod -l app.kubernetes.io/instance=argus -n argus
```
