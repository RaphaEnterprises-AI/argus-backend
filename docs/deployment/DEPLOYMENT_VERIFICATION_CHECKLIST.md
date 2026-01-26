# Multi-Tenant Deployment Verification Checklist

This checklist verifies that the Argus multi-tenant data layer is properly deployed and functioning.

## Prerequisites

- [ ] Vultr Kubernetes cluster is accessible (`kubectl get nodes`)
- [ ] Helm is installed (`helm version`)
- [ ] `gh` CLI is authenticated for GitHub access
- [ ] Required secrets are available (Neo4j, Redpanda, Supabase credentials)

---

## Phase 1: Namespace and Secrets

### 1.1 Create Namespace
```bash
kubectl apply -f data-layer/kubernetes/namespace.yaml
kubectl get namespace argus-data
```
- [ ] Namespace `argus-data` exists
- [ ] Labels applied: `app.kubernetes.io/part-of: argus`

### 1.2 Deploy Secrets
```bash
kubectl apply -f data-layer/kubernetes/secrets.yaml
kubectl get secrets -n argus-data
```
- [ ] Secret `argus-data-secrets` exists
- [ ] Contains keys: `REDPANDA_SASL_PASSWORD`, `FALKORDB_PASSWORD`, `NEO4J_PASSWORD`
- [ ] Contains keys: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `OPENROUTER_API_KEY`

### 1.3 Verify Secrets (non-destructive)
```bash
kubectl get secret argus-data-secrets -n argus-data -o jsonpath='{.data}' | jq 'keys'
```
- [ ] All expected keys present

---

## Phase 2: Storage Layer

### 2.1 Deploy FalkorDB
```bash
kubectl apply -f data-layer/kubernetes/falkordb.yaml
kubectl get pods -n argus-data -l app=falkordb
```
- [ ] FalkorDB StatefulSet running (2 replicas)
- [ ] PersistentVolumeClaims bound

### 2.2 Verify FalkorDB
```bash
kubectl exec -n argus-data falkordb-0 -- redis-cli PING
```
- [ ] Returns `PONG`

### 2.3 Deploy Valkey
```bash
kubectl apply -f data-layer/kubernetes/valkey.yaml
kubectl get pods -n argus-data -l app=valkey
```
- [ ] Valkey Deployment running
- [ ] PersistentVolumeClaim bound

### 2.4 Verify Valkey
```bash
kubectl exec -n argus-data -l app=valkey -- redis-cli PING
```
- [ ] Returns `PONG`

---

## Phase 3: Redpanda (Kafka)

### 3.1 Deploy Redpanda via Helm
```bash
helm repo add redpanda https://charts.redpanda.com
helm repo update
helm install redpanda redpanda/redpanda \
  -n argus-data \
  -f data-layer/kubernetes/redpanda-values.yaml \
  --wait --timeout 10m
```
- [ ] Helm release deployed successfully
- [ ] All 3 Redpanda pods running

### 3.2 Verify Cluster Health
```bash
kubectl exec -n argus-data redpanda-0 -- rpk cluster health
```
- [ ] All nodes healthy
- [ ] All partitions have leaders

### 3.3 Create Kafka Topics
```bash
kubectl exec -n argus-data redpanda-0 -- rpk topic create \
  argus.codebase.ingested \
  argus.codebase.analyzed \
  argus.test.created \
  argus.test.executed \
  argus.test.failed \
  argus.healing.requested \
  argus.healing.completed \
  argus.dlq \
  --partitions 6 --replicas 2
```
- [ ] All 8 topics created
- [ ] Topics have correct partition and replica counts

### 3.4 Verify Topics
```bash
kubectl exec -n argus-data redpanda-0 -- rpk topic list
```
- [ ] All topics listed with correct configuration

---

## Phase 4: Neo4j Aura (External)

### 4.1 Verify Connection
```bash
# Using Neo4j browser or cypher-shell
RETURN 1 AS connectivity_test;
```
- [ ] Query returns successfully
- [ ] Instance is not paused

### 4.2 Verify Constraints
```cypher
SHOW CONSTRAINTS;
```
- [ ] `project_org_id_exists` constraint exists
- [ ] `test_org_id_exists` constraint exists
- [ ] `failure_org_id_exists` constraint exists

### 4.3 Verify Indexes
```cypher
SHOW INDEXES;
```
- [ ] `project_org_id_idx` index exists
- [ ] `test_org_id_idx` index exists
- [ ] Composite indexes exist for common query patterns

### 4.4 Verify Keep-Alive CronJob
```bash
kubectl get cronjob -n argus-data neo4j-keepalive
kubectl get jobs -n argus-data | grep neo4j-keepalive
```
- [ ] CronJob exists with schedule `0 3 */2 * *`
- [ ] Recent job completed successfully (if applicable)

---

## Phase 5: Network Policies

### 5.1 Apply Network Policies
```bash
kubectl apply -f data-layer/kubernetes/network-policies.yaml
kubectl get networkpolicies -n argus-data
```
- [ ] `falkordb-network-policy` exists
- [ ] `valkey-network-policy` exists
- [ ] `redpanda-network-policy` exists
- [ ] `cognee-worker-network-policy` exists

---

## Phase 6: Cognee Worker

### 6.1 Deploy Cognee Worker
```bash
kubectl apply -f data-layer/kubernetes/cognee-worker.yaml
kubectl get pods -n argus-data -l app=cognee-worker
```
- [ ] Deployment running (2 replicas)
- [ ] All pods healthy

### 6.2 Verify Worker Logs
```bash
kubectl logs -n argus-data -l app=cognee-worker --tail=50
```
- [ ] No startup errors
- [ ] Connected to Redpanda
- [ ] Connected to Neo4j Aura

### 6.3 Verify Worker Environment
```bash
kubectl exec -n argus-data -l app=cognee-worker -- env | grep -E "REDPANDA|NEO4J|COGNEE"
```
- [ ] `REDPANDA_BROKERS` set correctly
- [ ] `NEO4J_URI` set correctly
- [ ] `COGNEE_GRAPH_PROVIDER` = `neo4j`

---

## Phase 7: Services

### 7.1 Apply Services
```bash
kubectl apply -f data-layer/kubernetes/services.yaml
kubectl get services -n argus-data
```
- [ ] `falkordb` service exists (ClusterIP)
- [ ] `valkey` service exists (ClusterIP)
- [ ] `redpanda-kafka` service exists (ClusterIP:9092)

### 7.2 Verify DNS Resolution
```bash
kubectl run test-dns --rm -it --restart=Never --image=busybox -- \
  nslookup redpanda-kafka.argus-data.svc.cluster.local
```
- [ ] DNS resolves to service IP

---

## Phase 8: Backend Integration

### 8.1 Verify Backend Environment Variables
Check Railway/deployment environment:
- [ ] `REDPANDA_BROKERS` = `redpanda-kafka.argus-data.svc.cluster.local:9092`
- [ ] `NEO4J_URI` = Neo4j Aura URI
- [ ] `NEO4J_USERNAME` = `neo4j`
- [ ] `NEO4J_PASSWORD` = (secret)
- [ ] `MULTI_TENANT_ENABLED` = `true`
- [ ] `COGNEE_GRAPH_PROVIDER` = `neo4j`

### 8.2 Verify API Health
```bash
curl https://argus-brain-production.up.railway.app/api/v1/health
```
- [ ] Returns `200 OK`
- [ ] Response includes data layer status

### 8.3 Verify Event Gateway
```bash
curl https://argus-brain-production.up.railway.app/api/v1/health/kafka
```
- [ ] Kafka connection healthy
- [ ] Producer initialized

---

## Phase 9: Multi-Tenant Verification

### 9.1 Create Test Organization (via API)
```bash
# Use dashboard or API to create test org
curl -X POST https://argus-brain-production.up.railway.app/api/v1/orgs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Org", "slug": "test-org"}'
```
- [ ] Organization created successfully
- [ ] Returns org ID (UUID format)

### 9.2 Verify RLS Policies
```sql
-- In Supabase SQL Editor
SET app.user_id = 'test-user-id';
SELECT * FROM projects WHERE organization_id = 'test-org-id';
```
- [ ] Only returns projects for specified org
- [ ] No cross-tenant data leakage

### 9.3 Verify Cognee Dataset Isolation
```cypher
// In Neo4j Browser
MATCH (n)
WHERE n.org_id IS NOT NULL
RETURN n.org_id, labels(n)[0] as type, count(*) as count
ORDER BY n.org_id, type;
```
- [ ] Each org_id has separate data
- [ ] No nodes without org_id (except system nodes)

---

## Phase 10: End-to-End Event Flow

### 10.1 Produce Test Event
```bash
kubectl exec -n argus-data redpanda-0 -- rpk topic produce argus.codebase.ingested <<EOF
{"event_id":"test-123","event_type":"codebase.ingested","tenant":{"org_id":"test-org","project_id":"test-project"},"metadata":{"source":"verification","request_id":"verify-001"},"repository_id":"repo-1","repository_url":"https://github.com/test/repo","branch":"main","commit_sha":"abc123","file_count":10,"total_size_bytes":1000}
EOF
```
- [ ] Event produced successfully

### 10.2 Verify Event Processing
```bash
# Check Cognee worker processed the event
kubectl logs -n argus-data -l app=cognee-worker --tail=20 | grep "test-123"
```
- [ ] Event received by worker
- [ ] Processing started

### 10.3 Verify Output Event
```bash
kubectl exec -n argus-data redpanda-0 -- rpk topic consume argus.codebase.analyzed --num 1 --from-start
```
- [ ] Analyzed event produced
- [ ] Contains same tenant context

### 10.4 Verify Neo4j Data
```cypher
MATCH (n {org_id: 'test-org'})
RETURN labels(n), count(*);
```
- [ ] Nodes created with correct org_id
- [ ] Relationships created

---

## Phase 11: Dashboard Verification

### 11.1 Verify Organization Context
- [ ] Login to dashboard
- [ ] Organization switcher visible
- [ ] Can switch between organizations

### 11.2 Verify API Calls Include Org Header
- [ ] Open browser DevTools > Network
- [ ] Make any API call
- [ ] `X-Organization-ID` header present in request

### 11.3 Verify Data Isolation
- [ ] Switch to Org A
- [ ] Note projects/tests visible
- [ ] Switch to Org B
- [ ] Different projects/tests visible
- [ ] No data from Org A visible

---

## Phase 12: Monitoring and Alerts

### 12.1 Verify Metrics
```bash
kubectl port-forward -n argus-data svc/redpanda-kafka 9644:9644
curl http://localhost:9644/metrics | grep kafka_
```
- [ ] Kafka metrics available
- [ ] Consumer lag metrics present

### 12.2 Verify Logs
```bash
kubectl logs -n argus-data -l app=cognee-worker --tail=100 | grep -i error
```
- [ ] No persistent errors
- [ ] Log level appropriate

---

## Troubleshooting

### Common Issues

1. **Neo4j Aura paused**: Run keep-alive query manually
   ```cypher
   RETURN datetime() AS keepalive;
   ```

2. **Kafka connection refused**: Check network policies allow traffic
   ```bash
   kubectl describe networkpolicy cognee-worker-network-policy -n argus-data
   ```

3. **Events not processing**: Check consumer group lag
   ```bash
   kubectl exec -n argus-data redpanda-0 -- rpk group describe argus-workers
   ```

4. **RLS blocking queries**: Verify user has org membership
   ```sql
   SELECT * FROM organization_members WHERE user_id = 'xxx';
   ```

---

## Rollback Procedure

If deployment fails, rollback in reverse order:

```bash
# 1. Remove Cognee worker
kubectl delete -f data-layer/kubernetes/cognee-worker.yaml

# 2. Remove network policies
kubectl delete -f data-layer/kubernetes/network-policies.yaml

# 3. Uninstall Redpanda
helm uninstall redpanda -n argus-data

# 4. Remove storage
kubectl delete -f data-layer/kubernetes/valkey.yaml
kubectl delete -f data-layer/kubernetes/falkordb.yaml

# 5. Remove secrets and namespace (DESTRUCTIVE)
kubectl delete -f data-layer/kubernetes/secrets.yaml
kubectl delete namespace argus-data
```

---

## Sign-off

| Phase | Verified By | Date | Notes |
|-------|-------------|------|-------|
| Phase 1: Namespace/Secrets | | | |
| Phase 2: Storage Layer | | | |
| Phase 3: Redpanda | | | |
| Phase 4: Neo4j Aura | | | |
| Phase 5: Network Policies | | | |
| Phase 6: Cognee Worker | | | |
| Phase 7: Services | | | |
| Phase 8: Backend Integration | | | |
| Phase 9: Multi-Tenant | | | |
| Phase 10: Event Flow | | | |
| Phase 11: Dashboard | | | |
| Phase 12: Monitoring | | | |

**Deployment Verified By**: ___________________ **Date**: ___________
