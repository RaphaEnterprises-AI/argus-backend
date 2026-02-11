# Air-Gap Deployment Validation Checklist

This document provides a comprehensive checklist for validating Skopaq Enterprise
deployments in air-gapped (fully offline) environments.

## Pre-Deployment Checklist

### 1. Infrastructure Preparation

- [ ] **GPU Hardware Available**
  - Minimum: NVIDIA GPU with 8GB VRAM (for llama3.1:8b)
  - Recommended: NVIDIA A100 80GB or 2x A10 (for llama3.1:70b)
  - Verify: `nvidia-smi` shows expected GPU(s)

- [ ] **Storage Provisioned**
  - Ollama models: 200GB+ (for multiple models)
  - MinIO artifacts: 100GB+ (depends on usage)
  - PostgreSQL: 50GB+
  - Verify: `df -h` shows sufficient space

- [ ] **Network Isolated**
  - No outbound internet connectivity
  - Internal DNS configured
  - Load balancer/ingress configured

### 2. Container Images Transferred

Transfer all required images to your air-gapped registry:

```bash
# Required images
- [ ] ollama/ollama:latest
- [ ] minio/minio:latest
- [ ] postgres:15
- [ ] redis:7-alpine (or valkey/valkey:7)
- [ ] ghcr.io/argus/argus-brain:latest
- [ ] ghcr.io/argus/argus-dashboard:latest
```

Verification:
```bash
# List images in local registry
curl -s http://registry.local:5000/v2/_catalog | jq .
```

### 3. Ollama Models Pre-Downloaded

- [ ] **Primary Model** (required)
  ```bash
  ollama list | grep llama3.1
  # Expected: llama3.1:70b or llama3.1:8b
  ```

- [ ] **Code Analysis Model** (recommended)
  ```bash
  ollama list | grep codellama
  # Expected: codellama:34b
  ```

- [ ] **Embedding Model** (required for semantic search)
  ```bash
  ollama list | grep nomic-embed-text
  # Expected: nomic-embed-text
  ```

### 4. Helm Charts Available

- [ ] Skopaq Enterprise Helm chart downloaded
- [ ] Values file configured for air-gap mode

```bash
ls -la helm-charts/
# Expected: skopaq-enterprise-1.0.0.tgz
```

---

## Deployment Validation Checklist

### 5. Core Services Running

Verify all pods are running:

```bash
kubectl get pods -n argus
```

- [ ] `argus-brain-*` - Running
- [ ] `argus-ollama-*` - Running
- [ ] `argus-minio-*` - Running
- [ ] `argus-postgresql-*` - Running
- [ ] `argus-valkey-*` - Running

### 6. Ollama Health Check

```bash
# Test Ollama API
kubectl exec deploy/argus-ollama -n argus -- \
  curl -s http://localhost:11434/api/tags | jq '.models[].name'
```

- [ ] Returns list of available models
- [ ] Required models are present

```bash
# Test model inference
kubectl exec deploy/argus-ollama -n argus -- \
  curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"Say OK","stream":false}' | jq '.response'
```

- [ ] Returns valid response

### 7. MinIO Health Check

```bash
# Test MinIO health
kubectl exec deploy/argus-brain -n argus -- \
  curl -s http://argus-minio:9000/minio/health/live
```

- [ ] Returns HTTP 200

```bash
# Test bucket access (from brain pod)
kubectl exec deploy/argus-brain -n argus -- \
  python -c "
from minio import Minio
client = Minio('argus-minio:9000', access_key='minioadmin', secret_key='minioadmin', secure=False)
print('Buckets:', [b.name for b in client.list_buckets()])
"
```

- [ ] Lists buckets successfully

### 8. Brain Service Health Check

```bash
# Test Brain API health
kubectl exec deploy/argus-brain -n argus -- \
  curl -s http://localhost:8000/health | jq .
```

- [ ] Returns `{"status": "healthy"}`

```bash
# Test LLM connectivity from Brain
kubectl exec deploy/argus-brain -n argus -- \
  curl -s http://argus-ollama:11434/api/tags | jq '.models | length'
```

- [ ] Returns number > 0

### 9. Database Connectivity

```bash
# Test PostgreSQL connection
kubectl exec deploy/argus-brain -n argus -- \
  python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://argus:password@argus-postgresql:5432/argus')
    result = await conn.fetchval('SELECT 1')
    print('PostgreSQL OK:', result == 1)
asyncio.run(test())
"
```

- [ ] Prints "PostgreSQL OK: True"

---

## Functional Validation Checklist

### 10. LLM Inference Test

```bash
# Run LLM inference test
kubectl exec deploy/argus-brain -n argus -- \
  python -c "
import asyncio
from src.core.providers.ollama_provider import OllamaProvider, ChatMessage

async def test():
    provider = OllamaProvider(base_url='http://argus-ollama:11434')
    response = await provider.chat(
        messages=[ChatMessage(role='user', content='What is 2+2? Answer with just the number.')],
        model='llama3.1:8b',
        max_tokens=10
    )
    print('Response:', response.content)
    print('Tokens:', response.output_tokens)
    await provider.close()

asyncio.run(test())
"
```

- [ ] Returns valid response (e.g., "4")
- [ ] No external network errors

### 11. Storage Test

```bash
# Run storage test
kubectl exec deploy/argus-brain -n argus -- \
  python -c "
import asyncio
import base64
from src.services.storage.minio_provider import MinIOStorageProvider, MinIOConfig

async def test():
    config = MinIOConfig(
        endpoint='argus-minio:9000',
        access_key='minioadmin',
        secret_key='minioadmin',
        bucket='argus-test',
        secure=False
    )
    provider = MinIOStorageProvider(config)
    await provider.ensure_bucket()

    # Store test artifact
    test_data = base64.b64encode(b'test content').decode()
    ref = await provider.store_screenshot(test_data, {'test': 'airgap'})
    print('Stored:', ref.artifact_id)

    # Retrieve
    data = await provider.get_artifact(ref.artifact_id)
    print('Retrieved:', len(data), 'bytes')

    # Cleanup
    await provider.delete_artifact(ref.artifact_id)
    print('Deleted OK')

asyncio.run(test())
"
```

- [ ] Stores artifact successfully
- [ ] Retrieves artifact successfully
- [ ] Deletes artifact successfully

### 12. Integration Tests

Run the automated air-gap validation tests:

```bash
# Copy test file to pod (if not already present)
kubectl cp tests/integration/test_airgap.py \
  argus/argus-brain-xxx:/app/tests/integration/test_airgap.py

# Run tests
kubectl exec deploy/argus-brain -n argus -- \
  pytest tests/integration/test_airgap.py -v -m airgap \
  --tb=short 2>&1 | tee airgap-test-results.txt
```

- [ ] `test_ollama_connectivity` - PASSED
- [ ] `test_ollama_no_external_calls` - PASSED
- [ ] `test_minio_connectivity` - PASSED
- [ ] `test_minio_local_storage` - PASSED
- [ ] `test_full_test_workflow_offline` - PASSED
- [ ] `test_airgap_validation_report` - PASSED

### 13. Network Isolation Verification

Verify no external network calls are being made:

```bash
# Check DNS queries (should be empty or local only)
kubectl exec deploy/argus-brain -n argus -- \
  cat /etc/resolv.conf

# Test that external APIs are unreachable (expected to fail)
kubectl exec deploy/argus-brain -n argus -- \
  curl -s --connect-timeout 5 https://api.anthropic.com/v1/health || echo "BLOCKED (expected)"

kubectl exec deploy/argus-brain -n argus -- \
  curl -s --connect-timeout 5 https://api.openai.com/v1/models || echo "BLOCKED (expected)"
```

- [ ] External API calls are blocked or timeout

### 14. Generate Validation Report

```bash
kubectl exec deploy/argus-brain -n argus -- \
  pytest tests/integration/test_airgap.py::test_airgap_validation_report -v \
  2>&1 | tee airgap-validation-report.txt
```

- [ ] Report shows "air_gap_ready": true
- [ ] No external calls detected

---

## Post-Deployment Checklist

### 15. Dashboard Access

- [ ] Dashboard accessible via internal URL
- [ ] Login works with local authentication
- [ ] Can view test runs and results

### 16. End-to-End Test

Create and run a test to verify the full workflow:

1. [ ] Create a project in the dashboard
2. [ ] Add a test target (internal application URL)
3. [ ] Generate test suggestions using AI
4. [ ] Execute a test run
5. [ ] View results and screenshots

### 17. Monitoring Setup

- [ ] Prometheus scraping Skopaq metrics
- [ ] Grafana dashboards configured
- [ ] Alerts configured for service health

### 18. Backup Configuration

- [ ] PostgreSQL backup scheduled
- [ ] MinIO backup configured
- [ ] Ollama model volume backed up

---

## Troubleshooting

### Ollama Not Responding

```bash
# Check Ollama logs
kubectl logs deploy/argus-ollama -n argus --tail=100

# Check GPU access
kubectl exec deploy/argus-ollama -n argus -- nvidia-smi
```

### MinIO Connection Errors

```bash
# Check MinIO logs
kubectl logs deploy/argus-minio -n argus --tail=100

# Verify credentials
kubectl get secret argus-minio-credentials -n argus -o jsonpath='{.data.accesskey}' | base64 -d
```

### Brain Service Errors

```bash
# Check Brain logs
kubectl logs deploy/argus-brain -n argus --tail=100

# Check environment variables
kubectl exec deploy/argus-brain -n argus -- env | grep -E 'LLM_|OLLAMA_|MINIO_|STORAGE_'
```

### Tests Failing with External Call Detection

If tests report external calls when they shouldn't:

1. Check DNS configuration in pods
2. Verify no proxy environment variables are set
3. Check for webhook configurations pointing externally

```bash
# Check for proxy settings
kubectl exec deploy/argus-brain -n argus -- env | grep -i proxy

# Check for external URLs in config
kubectl get configmap argus-config -n argus -o yaml | grep -E 'http[s]?://'
```

---

## Certification Sign-Off

### Air-Gap Deployment Certification

| Check | Status | Verified By | Date |
|-------|--------|-------------|------|
| All pre-deployment items complete | | | |
| All deployment validation items pass | | | |
| All functional validation items pass | | | |
| Network isolation verified | | | |
| Validation report generated | | | |
| End-to-end test successful | | | |

**Certification Statement:**

I certify that this Skopaq Enterprise deployment has been validated for air-gap
operation and does not require external network connectivity for core functionality.

Signed: _________________________ Date: _____________

---

## Appendix: Quick Validation Script

Save this script and run it for quick validation:

```bash
#!/bin/bash
# air-gap-quick-check.sh

echo "=== Skopaq Air-Gap Quick Validation ==="

# Check pods
echo -e "\n[1/5] Checking pods..."
kubectl get pods -n argus --no-headers | grep -v Running && echo "WARN: Not all pods running" || echo "OK: All pods running"

# Check Ollama
echo -e "\n[2/5] Checking Ollama..."
kubectl exec deploy/argus-ollama -n argus -- curl -s http://localhost:11434/api/tags | jq -e '.models | length > 0' > /dev/null && echo "OK: Ollama has models" || echo "FAIL: No models"

# Check MinIO
echo -e "\n[3/5] Checking MinIO..."
kubectl exec deploy/argus-brain -n argus -- curl -s http://argus-minio:9000/minio/health/live > /dev/null && echo "OK: MinIO healthy" || echo "FAIL: MinIO unhealthy"

# Check Brain
echo -e "\n[4/5] Checking Brain..."
kubectl exec deploy/argus-brain -n argus -- curl -s http://localhost:8000/health | jq -e '.status == "healthy"' > /dev/null && echo "OK: Brain healthy" || echo "FAIL: Brain unhealthy"

# Check external access blocked
echo -e "\n[5/5] Checking network isolation..."
kubectl exec deploy/argus-brain -n argus -- curl -s --connect-timeout 3 https://api.anthropic.com 2>&1 | grep -q "timed out\|refused\|unreachable" && echo "OK: External blocked" || echo "WARN: External may be accessible"

echo -e "\n=== Validation Complete ==="
```

Make executable and run:
```bash
chmod +x air-gap-quick-check.sh
./air-gap-quick-check.sh
```
