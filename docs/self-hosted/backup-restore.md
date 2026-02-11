# Backup and Restore

Comprehensive backup and disaster recovery procedures for Skopaq Enterprise.

## What to Backup

| Component | Data Type | Priority | RTO |
|-----------|-----------|----------|-----|
| PostgreSQL | Test data, configurations | Critical | 1 hour |
| MinIO | Screenshots, artifacts | High | 4 hours |
| Redis | Sessions, cache | Low | No backup needed |
| Secrets | API keys, credentials | Critical | 15 min |
| Configuration | Helm values, ConfigMaps | Critical | 15 min |

## PostgreSQL Backup

### Manual Backup

```bash
# Create logical backup
kubectl exec -n argus argus-postgresql-0 -- \
  pg_dump -U argus -d argus -Fc > argus-backup-$(date +%Y%m%d).dump

# Verify backup
pg_restore --list argus-backup-$(date +%Y%m%d).dump
```

### Automated Backup with CronJob

```yaml
# postgres-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: argus
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:16
              command:
                - /bin/bash
                - -c
                - |
                  pg_dump -h argus-postgresql -U argus -d argus -Fc \
                    > /backup/argus-$(date +%Y%m%d-%H%M%S).dump
                  # Keep last 7 days
                  find /backup -name "*.dump" -mtime +7 -delete
              env:
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: argus-secrets
                      key: postgresql-password
              volumeMounts:
                - name: backup-storage
                  mountPath: /backup
          restartPolicy: OnFailure
          volumes:
            - name: backup-storage
              persistentVolumeClaim:
                claimName: postgres-backup-pvc
```

### Point-in-Time Recovery (PITR)

Enable WAL archiving for PITR:

```yaml
# values.yaml
postgresql:
  primary:
    configuration: |
      wal_level = replica
      archive_mode = on
      archive_command = 'cp %p /archive/%f'
    persistence:
      size: 100Gi
  volumePermissions:
    enabled: true
```

## MinIO Backup

### Using MinIO Client (mc)

```bash
# Configure mc client
kubectl exec -n argus deploy/argus-minio -- \
  mc alias set local http://localhost:9000 argus-admin argus-secret

# Mirror to backup bucket
kubectl exec -n argus deploy/argus-minio -- \
  mc mirror local/argus-artifacts /backup/argus-artifacts-$(date +%Y%m%d)

# Mirror to external S3
kubectl exec -n argus deploy/argus-minio -- \
  mc alias set backup https://s3.us-east-1.amazonaws.com ACCESS_KEY SECRET_KEY
kubectl exec -n argus deploy/argus-minio -- \
  mc mirror local/argus-artifacts backup/argus-backup/
```

### Automated MinIO Backup

```yaml
# minio-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: minio-backup
  namespace: argus
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: minio/mc:latest
              command:
                - /bin/sh
                - -c
                - |
                  mc alias set local http://argus-minio:9000 $MINIO_ACCESS_KEY $MINIO_SECRET_KEY
                  mc alias set backup $BACKUP_ENDPOINT $BACKUP_ACCESS_KEY $BACKUP_SECRET_KEY
                  mc mirror --overwrite local/argus-artifacts backup/argus-backup/$(date +%Y%m%d)/
              envFrom:
                - secretRef:
                    name: minio-backup-credentials
          restartPolicy: OnFailure
```

## Secrets Backup

### Export Secrets

```bash
# Export secrets (encrypted)
kubectl get secret argus-secrets -n argus -o yaml | \
  kubeseal --format yaml > argus-secrets-sealed.yaml

# Or use SOPS
kubectl get secret argus-secrets -n argus -o json | \
  sops --encrypt --kms arn:aws:kms:... /dev/stdin > argus-secrets.enc.json
```

### Backup with Velero

```bash
# Install Velero
velero install \
  --provider aws \
  --bucket argus-backups \
  --secret-file ./credentials-velero

# Create backup
velero backup create argus-full-backup \
  --include-namespaces argus \
  --include-cluster-resources

# Schedule daily backups
velero schedule create argus-daily \
  --schedule="0 2 * * *" \
  --include-namespaces argus \
  --ttl 168h  # Keep for 7 days
```

## Configuration Backup

### Helm Values

```bash
# Export current values
helm get values argus -n argus -o yaml > values-backup-$(date +%Y%m%d).yaml

# Export all manifests
helm get manifest argus -n argus > manifests-backup-$(date +%Y%m%d).yaml
```

### GitOps Approach

Store configuration in Git:

```bash
# Directory structure
argus-config/
├── base/
│   ├── kustomization.yaml
│   └── values.yaml
├── overlays/
│   ├── production/
│   │   ├── kustomization.yaml
│   │   └── values-prod.yaml
│   └── staging/
│       ├── kustomization.yaml
│       └── values-staging.yaml
└── secrets/
    └── secrets.enc.yaml  # SOPS encrypted
```

## Restore Procedures

### PostgreSQL Restore

```bash
# Stop Brain pods to prevent writes
kubectl scale deploy argus-brain -n argus --replicas=0

# Restore from backup
kubectl exec -i argus-postgresql-0 -n argus -- \
  pg_restore -U argus -d argus -c < argus-backup-20260130.dump

# Start Brain pods
kubectl scale deploy argus-brain -n argus --replicas=2

# Verify restoration
kubectl exec -n argus argus-postgresql-0 -- \
  psql -U argus -d argus -c "SELECT count(*) FROM tests;"
```

### MinIO Restore

```bash
# Restore from external backup
kubectl exec -n argus deploy/argus-minio -- \
  mc mirror backup/argus-backup/20260130/ local/argus-artifacts

# Verify restoration
kubectl exec -n argus deploy/argus-minio -- \
  mc ls local/argus-artifacts
```

### Full Cluster Restore with Velero

```bash
# List available backups
velero backup get

# Restore specific backup
velero restore create --from-backup argus-full-backup-20260130

# Monitor restore progress
velero restore describe argus-full-backup-20260130-xxxxx
```

### Secrets Restore

```bash
# Restore sealed secret
kubectl apply -f argus-secrets-sealed.yaml

# Or restore SOPS-encrypted secret
sops --decrypt argus-secrets.enc.json | kubectl apply -f -
```

## Disaster Recovery Plan

### RPO and RTO Targets

| Component | RPO | RTO | Backup Frequency |
|-----------|-----|-----|------------------|
| PostgreSQL | 1 hour | 1 hour | Hourly + WAL |
| MinIO | 24 hours | 4 hours | Daily |
| Configuration | 0 | 15 min | On change (GitOps) |
| Secrets | 0 | 15 min | On change |

### DR Runbook

1. **Assess Damage**
   ```bash
   kubectl get pods -n argus
   kubectl get pvc -n argus
   ```

2. **Restore Secrets First**
   ```bash
   kubectl apply -f argus-secrets-sealed.yaml
   ```

3. **Restore PostgreSQL**
   ```bash
   # Deploy fresh PostgreSQL
   helm upgrade argus ./skopaq-enterprise -n argus \
     --set postgresql.enabled=true

   # Wait for pod
   kubectl wait --for=condition=ready pod/argus-postgresql-0 -n argus

   # Restore data
   kubectl exec -i argus-postgresql-0 -n argus -- \
     pg_restore -U argus -d argus -c < latest-backup.dump
   ```

4. **Restore MinIO**
   ```bash
   # Restore from external backup
   mc mirror backup/argus-backup/latest/ local/argus-artifacts
   ```

5. **Deploy Application**
   ```bash
   helm upgrade argus ./skopaq-enterprise -n argus -f values.yaml
   ```

6. **Verify**
   ```bash
   curl http://argus-brain:8000/health
   kubectl exec deploy/argus-brain -n argus -- \
     psql "$DATABASE_URL" -c "SELECT count(*) FROM tests;"
   ```

## Testing Backups

### Monthly Backup Verification

```bash
#!/bin/bash
# backup-verification.sh

# 1. Create test namespace
kubectl create namespace argus-backup-test

# 2. Restore to test namespace
velero restore create backup-test \
  --from-backup argus-daily-$(date +%Y%m%d) \
  --namespace-mappings argus:argus-backup-test

# 3. Wait for restore
sleep 120

# 4. Verify pods running
kubectl get pods -n argus-backup-test

# 5. Verify data integrity
kubectl exec -n argus-backup-test argus-postgresql-0 -- \
  psql -U argus -d argus -c "SELECT count(*) FROM tests;"

# 6. Cleanup
kubectl delete namespace argus-backup-test
```

### Backup Monitoring

```yaml
# Alert if backup fails
- alert: SkopaqBackupFailed
  expr: |
    time() - backup_last_success_timestamp{job="argus-postgres-backup"} > 90000
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "PostgreSQL backup hasn't succeeded in 25 hours"
```
