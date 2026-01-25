# Argus Data Layer - Kubernetes Deployment

Kubernetes manifests for deploying the Argus data layer infrastructure on Vultr Kubernetes Engine (VKE).

## Components

| Component | Description | Replicas | Storage | Memory |
|-----------|-------------|----------|---------|--------|
| **Redpanda** | Kafka-compatible streaming platform with Schema Registry | 1 | 40GB | 2GB |
| **FalkorDB** | Graph database for knowledge graphs and relationships | 1 | 40GB | 1GB |
| **Valkey** | Redis-compatible cache for sessions and caching | 1 | 40GB | 512MB |
| **Cognee Worker** | Knowledge graph builder (consumes from Redpanda) | 1-4 (HPA) | - | 1GB |

## Prerequisites

1. **Vultr Kubernetes Cluster** (VKE) with at least 4GB RAM available
2. **kubectl** configured with cluster credentials
3. **Helm v3** for Redpanda deployment
4. **Storage Class**: `vultr-block-storage-hdd` (Vultr's default HDD storage)

```bash
# Verify cluster connection
export KUBECONFIG=/path/to/kubeconfig.yaml
kubectl cluster-info
kubectl get nodes

# Verify storage class exists
kubectl get storageclass vultr-block-storage-hdd
```

## Quick Start (Automated)

The easiest way to deploy is using the automated script:

```bash
cd data-layer/kubernetes

# Generate random passwords and deploy everything
./deploy.sh --generate-secrets

# Or if you've already configured secrets.yaml manually:
./deploy.sh
```

## Manual Deployment

### Step 1: Add Helm Repository

```bash
helm repo add redpanda https://charts.redpanda.com/
helm repo update
```

### Step 2: Create Namespace

```bash
kubectl apply -f namespace.yaml
```

### Step 3: Configure Secrets

**IMPORTANT**: Edit `secrets.yaml` and replace ALL placeholder values before applying.

```bash
# Generate secure passwords
export FALKORDB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
export VALKEY_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
export REDPANDA_ADMIN_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
export REDPANDA_SERVICE_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)

echo "FalkorDB Password: $FALKORDB_PASSWORD"
echo "Valkey Password: $VALKEY_PASSWORD"
echo "Redpanda Admin: $REDPANDA_ADMIN_PASS"
echo "Redpanda Service: $REDPANDA_SERVICE_PASS"

# Edit secrets.yaml with your actual values, then apply:
kubectl apply -f secrets.yaml
```

### Step 4: Deploy Storage Layer

```bash
kubectl apply -f falkordb.yaml
kubectl apply -f valkey.yaml

# Wait for PVCs to be bound
kubectl get pvc -n argus-data -w
```

### Step 5: Deploy Redpanda

```bash
helm install redpanda redpanda/redpanda \
  -n argus-data \
  -f redpanda-values.yaml \
  --wait \
  --timeout 15m
```

### Step 6: Apply Network Policies and Services

```bash
kubectl apply -f network-policies.yaml
kubectl apply -f services.yaml
```

### Step 7: Create Kafka Topics

```bash
# Get admin password from secret
ADMIN_PASS=$(kubectl get secret redpanda-superusers -n argus-data \
  -o jsonpath='{.data.users\.txt}' | base64 -d | grep admin | cut -d: -f2)

# Create topics
kubectl exec -n argus-data redpanda-0 -- rpk topic create \
  argus.codebase.ingested \
  argus.codebase.analyzed \
  argus.test.created \
  argus.test.executed \
  argus.test.failed \
  argus.healing.requested \
  argus.healing.completed \
  argus.dlq \
  --partitions 3 \
  --replication-factor 1 \
  -X user=admin \
  -X pass="$ADMIN_PASS" \
  -X sasl.mechanism=SCRAM-SHA-512
```

### Step 8: Deploy Cognee Worker

```bash
kubectl apply -f cognee-worker.yaml
```

## Verification

### Check All Resources

```bash
kubectl get all -n argus-data
```

### Verify Pods Are Running

```bash
kubectl get pods -n argus-data -w
```

Expected output:
```
NAME                             READY   STATUS    RESTARTS   AGE
cognee-worker-xxx                1/1     Running   0          2m
falkordb-0                       2/2     Running   0          10m
redpanda-0                       1/1     Running   0          8m
valkey-0                         2/2     Running   0          10m
```

### Test Connectivity

#### FalkorDB

```bash
# Get password
FALKOR_PASS=$(kubectl get secret falkordb-auth -n argus-data \
  -o jsonpath='{.data.password}' | base64 -d)

# Port-forward and test
kubectl port-forward -n argus-data svc/falkordb 6379:6379 &
redis-cli -p 6379 -a "$FALKOR_PASS" PING
```

#### Valkey

```bash
VALKEY_PASS=$(kubectl get secret valkey-auth -n argus-data \
  -o jsonpath='{.data.password}' | base64 -d)

kubectl port-forward -n argus-data svc/valkey 6380:6379 &
redis-cli -p 6380 -a "$VALKEY_PASS" PING
```

#### Redpanda

```bash
# Check cluster health
kubectl exec -n argus-data redpanda-0 -- rpk cluster health

# List topics
kubectl exec -n argus-data redpanda-0 -- rpk topic list \
  -X user=admin \
  -X pass="$ADMIN_PASS" \
  -X sasl.mechanism=SCRAM-SHA-512
```

## Connection Strings

Use these connection strings from other pods in the `argus-data` namespace:

| Component | Connection String |
|-----------|------------------|
| FalkorDB | `redis://:${FALKORDB_PASSWORD}@falkordb-headless.argus-data.svc.cluster.local:6379` |
| Valkey | `redis://:${VALKEY_PASSWORD}@valkey-headless.argus-data.svc.cluster.local:6379` |
| Redpanda Kafka | `redpanda.argus-data.svc.cluster.local:9092` (SASL auth required) |

## Configuration Reference

### Redpanda SASL Authentication

The Redpanda cluster uses SCRAM-SHA-512 authentication. Clients must provide:

```yaml
# For aiokafka/kafka-python clients:
security_protocol: SASL_PLAINTEXT
sasl_mechanism: SCRAM-SHA-512
sasl_plain_username: argus-service
sasl_plain_password: <from-secret>
```

### Environment Variables for Cognee Worker

The Cognee worker expects these environment variables:

```yaml
# Kafka/Redpanda
KAFKA_BOOTSTRAP_SERVERS: redpanda.argus-data.svc.cluster.local:9092
KAFKA_SECURITY_PROTOCOL: SASL_PLAINTEXT
KAFKA_SASL_MECHANISM: SCRAM-SHA-512
KAFKA_SASL_USERNAME: argus-service
KAFKA_SASL_PASSWORD: <from-secret>

# FalkorDB
FALKORDB_HOST: falkordb-headless.argus-data.svc.cluster.local
FALKORDB_PORT: 6379
FALKORDB_PASSWORD: <from-secret>

# Valkey
VALKEY_HOST: valkey-headless.argus-data.svc.cluster.local
VALKEY_PORT: 6379
VALKEY_PASSWORD: <from-secret>
```

## Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod <pod-name> -n argus-data
kubectl logs <pod-name> -n argus-data
```

### PVC Pending

```bash
# Check PVC status
kubectl describe pvc -n argus-data

# Vultr minimum storage is 40GB
# Ensure storageClass is vultr-block-storage-hdd
```

### Authentication Errors

**FalkorDB "WRONGPASS" error:**
- Ensure `FALKORDB_PASSWORD` env var is defined BEFORE `REDIS_ARGS` in the manifest
- Kubernetes evaluates env vars in order

**Redpanda "SASL authentication failed":**
- Verify the secret `redpanda-superusers` contains `users.txt` in format: `username:password:SCRAM-SHA-512`
- The secret MUST exist BEFORE helm install

### Redpanda Configuration Job Timeout

```bash
# Increase timeout
helm install redpanda redpanda/redpanda \
  -n argus-data \
  -f redpanda-values.yaml \
  --wait \
  --timeout 20m
```

### Network Policy Issues

```bash
# Temporarily disable for debugging
kubectl delete networkpolicy --all -n argus-data

# Test connectivity, then re-apply
kubectl apply -f network-policies.yaml
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f cognee-worker.yaml
helm uninstall redpanda -n argus-data
kubectl delete -f valkey.yaml
kubectl delete -f falkordb.yaml
kubectl delete -f services.yaml
kubectl delete -f network-policies.yaml
kubectl delete -f secrets.yaml
kubectl delete -f namespace.yaml

# Delete PVCs (data will be lost!)
kubectl delete pvc -n argus-data --all
```

## File Structure

```
kubernetes/
├── deploy.sh              # Automated deployment script
├── namespace.yaml         # Namespace + ResourceQuota + LimitRange
├── secrets.yaml           # All secrets (edit before applying!)
├── falkordb.yaml          # FalkorDB StatefulSet + Service
├── valkey.yaml            # Valkey StatefulSet + Service
├── redpanda-values.yaml   # Helm values for Redpanda
├── cognee-worker.yaml     # Cognee worker Deployment + HPA
├── network-policies.yaml  # Network isolation policies
├── services.yaml          # ClusterIP services (FalkorDB, Valkey)
└── README.md              # This file
```

## Security Considerations

1. **Secrets Management**: For production, consider using:
   - [External Secrets Operator](https://external-secrets.io/)
   - [Sealed Secrets](https://sealed-secrets.netlify.app/)
   - [Vault](https://www.vaultproject.io/)

2. **Network Policies**: Default-deny policies are in place. All traffic is restricted to namespace-internal only.

3. **TLS**: Currently disabled for simplicity. For production, enable TLS with cert-manager.

4. **RBAC**: Consider adding RBAC policies for service accounts.

## Sources

- [Redpanda Helm Chart Documentation](https://docs.redpanda.com/current/reference/k-redpanda-helm-spec/)
- [Redpanda SASL Authentication](https://docs.redpanda.com/current/manage/kubernetes/security/authentication/k-authentication/)
- [FalkorDB Docker Documentation](https://docs.falkordb.com/operations/docker.html)
- [Valkey Security Documentation](https://valkey.io/topics/security/)
