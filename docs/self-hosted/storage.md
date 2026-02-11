# Storage Configuration

Configure persistent storage for Skopaq Enterprise.

## Storage Overview

| Component | Storage Type | Purpose | Default Size |
|-----------|--------------|---------|--------------|
| PostgreSQL | PVC | Relational data, embeddings | 50Gi |
| Redis | PVC | Cache, sessions | 8Gi |
| MinIO | PVC | Screenshots, reports, artifacts | 50Gi |
| Ollama | PVC | LLM models | 200Gi |

## Storage Classes

### Configure Storage Class

```yaml
# values.yaml
global:
  storageClass: "standard"  # Use your cluster's storage class
```

### Common Storage Classes

| Cloud Provider | Storage Class | Description |
|----------------|---------------|-------------|
| AWS EKS | `gp3` | General purpose SSD |
| GCP GKE | `standard-rwo` | Regional SSD |
| Azure AKS | `managed-premium` | Premium SSD |
| On-prem | `local-storage` | Local volumes |

### Create Custom Storage Class

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: argus-fast
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "10000"
  throughput: "500"
reclaimPolicy: Retain
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

## MinIO Configuration

### Embedded MinIO

```yaml
# values.yaml
minio:
  enabled: true
  auth:
    rootUser: argus
    rootPassword: "secure-password"
  persistence:
    enabled: true
    size: 100Gi
    storageClass: "argus-fast"
  defaultBuckets: "argus-artifacts,argus-screenshots,argus-reports"
```

### External S3/MinIO

```yaml
minio:
  enabled: false
  external:
    endpoint: "s3.us-east-1.amazonaws.com"
    accessKey: "AKIAIOSFODNN7EXAMPLE"
    secretKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    bucket: "argus-artifacts"
    region: "us-east-1"
    useSSL: true
```

### S3 Bucket Policy

If using AWS S3, create an appropriate bucket policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::123456789:role/argus-service-role"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::argus-artifacts",
                "arn:aws:s3:::argus-artifacts/*"
            ]
        }
    ]
}
```

### IAM Role for S3 (EKS)

```yaml
# ServiceAccount with IAM role
serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::123456789:role/argus-s3-access"
```

## PostgreSQL Storage

### Embedded PostgreSQL

```yaml
postgresql:
  enabled: true
  primary:
    persistence:
      enabled: true
      size: 100Gi
      storageClass: "argus-fast"
```

### External PostgreSQL

```yaml
postgresql:
  enabled: false
  external:
    host: "postgres.company.com"
    port: 5432
    database: "argus"
    username: "argus"
    existingSecret: "postgres-credentials"
```

### Database Sizing

| Test Volume | Recommended Size | IOPS |
|-------------|------------------|------|
| < 10K tests/month | 50Gi | 3000 |
| 10K-100K tests/month | 100Gi | 6000 |
| > 100K tests/month | 500Gi | 16000 |

## Redis Storage

```yaml
redis:
  enabled: true
  master:
    persistence:
      enabled: true
      size: 8Gi
      storageClass: "argus-fast"
```

For Redis, persistence is optional if you can tolerate cache loss on restart:

```yaml
redis:
  master:
    persistence:
      enabled: false  # Use memory-only
```

## Ollama Model Storage

Large models require substantial storage:

```yaml
ollama:
  enabled: true
  persistence:
    enabled: true
    size: 200Gi  # For llama3.1:70b + codellama:34b
    storageClass: "argus-fast"
  models:
    - "llama3.1:70b"    # ~40GB
    - "codellama:34b"   # ~20GB
    - "mistral:7b"      # ~5GB
```

### Model Size Reference

| Model | Size |
|-------|------|
| llama3.1:8b | 4.7GB |
| llama3.1:70b | 40GB |
| codellama:7b | 4GB |
| codellama:34b | 20GB |
| mistral:7b | 4GB |
| mixtral:8x7b | 26GB |

## Volume Expansion

### Enable Volume Expansion

Ensure your storage class supports expansion:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: argus-fast
# ...
allowVolumeExpansion: true
```

### Expand a PVC

```bash
# Edit PVC to increase size
kubectl patch pvc data-argus-postgresql-0 -n argus \
  -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'

# Monitor expansion
kubectl get pvc -n argus -w
```

## Backup Strategies

### PostgreSQL Backup

```bash
# Create backup
kubectl exec -n argus argus-postgresql-0 -- \
  pg_dump -U argus argus > backup.sql

# Using pg_basebackup for larger databases
kubectl exec -n argus argus-postgresql-0 -- \
  pg_basebackup -D /backup -Ft -z -P
```

### MinIO Backup

```bash
# Using MinIO Client (mc)
kubectl exec -n argus deploy/argus-minio -- \
  mc mirror local/argus-artifacts /backup/
```

### Velero Backup

```bash
# Install Velero
velero install --provider aws --bucket argus-backups ...

# Create backup
velero backup create argus-backup --include-namespaces argus

# Restore
velero restore create --from-backup argus-backup
```

## High Availability Storage

### Distributed MinIO

For HA MinIO, use distributed mode:

```yaml
minio:
  mode: distributed
  replicas: 4
  persistence:
    size: 100Gi
```

### PostgreSQL Replication

```yaml
postgresql:
  architecture: replication
  primary:
    persistence:
      size: 100Gi
  readReplicas:
    replicaCount: 2
    persistence:
      size: 100Gi
```

## Troubleshooting

### Check PVC Status

```bash
# List PVCs
kubectl get pvc -n argus

# Describe PVC
kubectl describe pvc data-argus-postgresql-0 -n argus
```

### Storage Capacity Issues

```bash
# Check node storage
kubectl describe nodes | grep -A5 "Allocated resources"

# Check PV usage
kubectl exec -n argus argus-postgresql-0 -- df -h
```

### Permission Issues

```bash
# Check pod security context
kubectl get pod -n argus -o yaml | grep -A10 securityContext

# Fix permission (if needed)
kubectl exec -n argus argus-minio-xxx -- chown -R 1000:1000 /data
```
