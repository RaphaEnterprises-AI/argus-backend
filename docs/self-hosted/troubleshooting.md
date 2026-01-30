# Troubleshooting Guide

Common issues and solutions for Argus Enterprise self-hosted deployments.

## Quick Diagnostics

### Health Check Script

```bash
#!/bin/bash
# argus-health-check.sh

echo "=== Argus Health Check ==="

echo -e "\n--- Pod Status ---"
kubectl get pods -n argus

echo -e "\n--- Services ---"
kubectl get svc -n argus

echo -e "\n--- PVCs ---"
kubectl get pvc -n argus

echo -e "\n--- Brain Health ---"
kubectl exec deploy/argus-brain -n argus -- curl -s localhost:8000/health 2>/dev/null || echo "FAILED"

echo -e "\n--- Database Connection ---"
kubectl exec deploy/argus-brain -n argus -- psql "$DATABASE_URL" -c "SELECT 1" 2>/dev/null || echo "FAILED"

echo -e "\n--- Redis Connection ---"
kubectl exec deploy/argus-brain -n argus -- redis-cli -u "$REDIS_URL" ping 2>/dev/null || echo "FAILED"

echo -e "\n--- Recent Errors ---"
kubectl logs deploy/argus-brain -n argus --tail=20 | grep -i error
```

## Installation Issues

### Pods Not Starting

**Symptom:** Pods stuck in `Pending` or `ContainerCreating`

**Diagnosis:**
```bash
kubectl describe pod argus-brain-xxx -n argus
kubectl get events -n argus --sort-by='.lastTimestamp'
```

**Common Causes:**

1. **Insufficient Resources**
   ```
   Events:
     Warning  FailedScheduling  pod/argus-brain-xxx  0/3 nodes are available: 3 Insufficient cpu
   ```

   **Solution:** Reduce resource requests or add nodes
   ```yaml
   brain:
     resources:
       requests:
         cpu: "250m"    # Reduce from 500m
         memory: "512Mi"  # Reduce from 1Gi
   ```

2. **PVC Not Bound**
   ```
   Events:
     Warning  FailedMount  pod/argus-postgresql-0  persistentvolumeclaim "data-argus-postgresql-0" not found
   ```

   **Solution:** Check storage class exists
   ```bash
   kubectl get storageclass
   kubectl get pvc -n argus
   ```

3. **Image Pull Error**
   ```
   Events:
     Warning  Failed  pod/argus-brain-xxx  Failed to pull image "ghcr.io/...": unauthorized
   ```

   **Solution:** Configure image pull secret
   ```yaml
   global:
     imagePullSecrets:
       - name: ghcr-credentials
   ```

### Helm Install Fails

**Dependency Issues:**
```bash
# Update dependencies first
helm dependency update ./helm/argus-enterprise

# Check dependencies
helm dependency list ./helm/argus-enterprise
```

**Template Errors:**
```bash
# Validate templates
helm template argus ./helm/argus-enterprise -f values.yaml

# Debug specific issue
helm install argus ./helm/argus-enterprise -f values.yaml --debug --dry-run
```

## Database Issues

### Connection Failed

**Symptom:** `could not connect to server: Connection refused`

**Diagnosis:**
```bash
# Check PostgreSQL pod
kubectl get pod -l app.kubernetes.io/component=primary -n argus

# Check PostgreSQL logs
kubectl logs argus-postgresql-0 -n argus

# Test connection from Brain
kubectl exec deploy/argus-brain -n argus -- \
  psql -h argus-postgresql -U argus -d argus -c "SELECT 1"
```

**Solutions:**

1. **Pod Not Ready**
   ```bash
   # Wait for PostgreSQL to be ready
   kubectl wait --for=condition=ready pod/argus-postgresql-0 -n argus --timeout=300s
   ```

2. **Wrong Password**
   ```bash
   # Check secret
   kubectl get secret argus-secrets -n argus -o jsonpath='{.data.postgresql-password}' | base64 -d

   # Update secret
   kubectl create secret generic argus-secrets \
     --from-literal=postgresql-password=NEW_PASSWORD \
     -n argus --dry-run=client -o yaml | kubectl apply -f -
   ```

3. **Network Policy Blocking**
   ```bash
   # Check network policies
   kubectl get networkpolicy -n argus

   # Temporarily disable for debugging
   kubectl delete networkpolicy --all -n argus
   ```

### Migration Failures

**Symptom:** `alembic.util.exc.CommandError: Target database is not up to date`

**Diagnosis:**
```bash
kubectl exec deploy/argus-brain -n argus -- alembic current
kubectl exec deploy/argus-brain -n argus -- alembic history
```

**Solutions:**
```bash
# Stamp current revision (if database is actually up to date)
kubectl exec deploy/argus-brain -n argus -- alembic stamp head

# Run missing migrations
kubectl exec deploy/argus-brain -n argus -- alembic upgrade head
```

## Redis Issues

### Connection Timeout

**Symptom:** `Error: Connection timed out`

**Diagnosis:**
```bash
# Check Redis pod
kubectl get pod -l app.kubernetes.io/component=master -n argus

# Test connection
kubectl exec deploy/argus-brain -n argus -- \
  redis-cli -h argus-redis-master -a $REDIS_PASSWORD ping
```

**Solutions:**

1. **Check Redis is running**
   ```bash
   kubectl logs argus-redis-master-0 -n argus
   ```

2. **Memory issues**
   ```bash
   kubectl exec argus-redis-master-0 -n argus -- redis-cli info memory
   ```

## MinIO/Storage Issues

### Bucket Not Found

**Symptom:** `The specified bucket does not exist`

**Diagnosis:**
```bash
kubectl exec deploy/argus-minio -n argus -- \
  mc ls local/
```

**Solution:**
```bash
# Create bucket
kubectl exec deploy/argus-minio -n argus -- \
  mc mb local/argus-artifacts

# Set policy
kubectl exec deploy/argus-minio -n argus -- \
  mc anonymous set download local/argus-artifacts
```

### Upload Failures

**Symptom:** `Error uploading file: connection reset`

**Diagnosis:**
```bash
# Check MinIO health
kubectl exec deploy/argus-minio -n argus -- \
  curl -s http://localhost:9000/minio/health/live

# Check disk space
kubectl exec deploy/argus-minio -n argus -- df -h /data
```

**Solutions:**

1. **Increase storage**
   ```yaml
   minio:
     persistence:
       size: 100Gi  # Increase from 50Gi
   ```

2. **Check network timeout**
   ```yaml
   brain:
     env:
       MINIO_TIMEOUT: "120"
   ```

## MCP Server Issues

### SSE Connection Drops

**Symptom:** AI assistant loses connection frequently

**Diagnosis:**
```bash
# Check MCP logs
kubectl logs deploy/argus-mcp -n argus

# Test SSE endpoint
curl -N http://argus-mcp:3000/sse
```

**Solutions:**

1. **Increase timeouts**
   ```yaml
   ingress:
     annotations:
       nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
       nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
   ```

2. **Enable sticky sessions**
   ```yaml
   ingress:
     annotations:
       nginx.ingress.kubernetes.io/affinity: "cookie"
   ```

### Tools Not Available

**Symptom:** AI assistant can't see Argus tools

**Diagnosis:**
```bash
# Check MCP tools registration
kubectl exec deploy/argus-mcp -n argus -- \
  curl -s http://localhost:3000/tools | jq .
```

**Solution:**
```bash
# Restart MCP server
kubectl rollout restart deployment/argus-mcp -n argus
```

## LLM Issues

### API Key Invalid

**Symptom:** `Invalid API key` or `Authentication failed`

**Diagnosis:**
```bash
# Check secret
kubectl get secret argus-secrets -n argus -o jsonpath='{.data.anthropic-api-key}' | base64 -d

# Test API directly
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $(kubectl get secret argus-secrets -n argus -o jsonpath='{.data.anthropic-api-key}' | base64 -d)" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-5","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

**Solution:**
```bash
# Update API key
kubectl create secret generic argus-secrets \
  --from-literal=anthropic-api-key=sk-ant-NEW_KEY \
  -n argus --dry-run=client -o yaml | kubectl apply -f -

# Restart Brain to pick up new secret
kubectl rollout restart deployment/argus-brain -n argus
```

### Rate Limiting

**Symptom:** `Rate limit exceeded`

**Solutions:**

1. **Add retry logic**
   ```yaml
   brain:
     env:
       LLM_RETRY_MAX_ATTEMPTS: "5"
       LLM_RETRY_DELAY_MS: "2000"
   ```

2. **Use OpenRouter for failover**
   ```yaml
   brain:
     env:
       LLM_PROVIDER: "openrouter"
       LLM_FALLBACK_PROVIDER: "anthropic"
   ```

### Ollama Not Responding

**Symptom:** `Connection refused to Ollama`

**Diagnosis:**
```bash
# Check Ollama pod
kubectl get pod -l app.kubernetes.io/component=ollama -n argus

# Check logs
kubectl logs deploy/argus-ollama -n argus

# Test endpoint
kubectl exec deploy/argus-ollama -n argus -- \
  curl -s http://localhost:11434/api/tags
```

**Solutions:**

1. **Model not loaded**
   ```bash
   # Pull model manually
   kubectl exec deploy/argus-ollama -n argus -- \
     ollama pull llama3.1:70b
   ```

2. **GPU not available**
   ```bash
   # Check GPU
   kubectl exec deploy/argus-ollama -n argus -- nvidia-smi
   ```

## Selenium Grid Issues

### No Available Nodes

**Symptom:** `No available node to run the test`

**Diagnosis:**
```bash
# Check Selenium status
kubectl exec deploy/argus-selenium-hub -n argus -- \
  curl -s http://localhost:4444/status | jq .

# Check Chrome nodes
kubectl get pods -l app.kubernetes.io/component=chrome-node -n argus
```

**Solutions:**

1. **Scale up nodes**
   ```yaml
   seleniumGrid:
     chrome:
       replicas: 5  # Increase from 3
   ```

2. **Check node resources**
   ```bash
   kubectl describe pod argus-selenium-chrome-node-xxx -n argus
   ```

3. **Increase shared memory**
   ```yaml
   seleniumGrid:
     chrome:
       extraEnvs:
         - name: SE_NODE_MAX_SESSIONS
           value: "1"  # Reduce concurrent sessions
   ```

## Performance Issues

### High Latency

**Diagnosis:**
```bash
# Check response times
kubectl exec deploy/argus-brain -n argus -- \
  curl -w "@/dev/stdin" -o /dev/null -s http://localhost:8000/health <<'EOF'
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
          time_total:  %{time_total}\n
EOF
```

**Solutions:**

1. **Scale Brain replicas**
   ```yaml
   brain:
     replicas: 3
     autoscaling:
       enabled: true
   ```

2. **Increase resources**
   ```yaml
   brain:
     resources:
       limits:
         cpu: "4"
         memory: "8Gi"
   ```

3. **Add caching**
   ```yaml
   brain:
     env:
       CACHE_ENABLED: "true"
       CACHE_TTL_SECONDS: "300"
   ```

### High Memory Usage

**Diagnosis:**
```bash
kubectl top pods -n argus
kubectl describe pod argus-brain-xxx -n argus | grep -A5 "Limits:"
```

**Solutions:**
```yaml
brain:
  resources:
    limits:
      memory: "4Gi"  # Increase limit
  env:
    PYTHON_GC_THRESHOLD: "700,10,10"  # Tune garbage collection
```

## Getting Help

### Collect Debug Information

```bash
# Create support bundle
mkdir argus-debug
kubectl get pods -n argus -o yaml > argus-debug/pods.yaml
kubectl get events -n argus > argus-debug/events.txt
kubectl logs deploy/argus-brain -n argus > argus-debug/brain.log
kubectl logs deploy/argus-mcp -n argus > argus-debug/mcp.log
helm get values argus -n argus > argus-debug/values.yaml
tar czf argus-debug.tar.gz argus-debug/
```

### Support Channels

- Documentation: https://docs.heyargus.ai/self-hosted
- GitHub Issues: https://github.com/raphaenterprises-ai/argus-e2e-testing-agent/issues
- Email: support@heyargus.ai
- Enterprise Support: enterprise@heyargus.ai
