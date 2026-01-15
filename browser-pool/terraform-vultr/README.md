# Vultr Browser Pool Deployment

Deploy the Argus Browser Pool on Vultr Kubernetes Engine (VKE) in Mumbai for lowest latency from India.

## Prerequisites

1. Vultr account with API access
2. Terraform installed (`brew install terraform`)
3. kubectl installed (`brew install kubectl`)

## Quick Start

```bash
# 1. Get your Vultr API key from:
#    https://my.vultr.com/settings/#settingsapi

# 2. Export your API key
export VULTR_API_KEY=your-api-key-here

# 3. Deploy (choose preset: dev, small, medium)
cd browser-pool
./deploy-vultr.sh dev
```

## Cost Estimates (Mumbai Region)

| Preset | Nodes | Spec | Monthly Cost | With $300 Credit |
|--------|-------|------|--------------|------------------|
| **dev** | 2 | 2 vCPU, 4GB | ~$50/mo | **6 months free** |
| small | 3 | 2 vCPU, 4GB | ~$70/mo | ~4 months free |
| medium | 5 | 4 vCPU, 8GB | ~$210/mo | ~1.5 months free |

## Manual Deployment

```bash
cd terraform-vultr

# Copy and edit config
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your API key

# Deploy
terraform init
terraform apply

# Get kubeconfig
export KUBECONFIG=$(pwd)/kubeconfig.yaml

# Deploy browser pool
cd ../kubernetes
kubectl apply -f namespace.yaml
kubectl apply -f configmaps.yaml
kubectl apply -f browser-deployment.yaml
kubectl apply -f services.yaml
kubectl apply -f hpa.yaml

# Get LoadBalancer IP
kubectl get svc -n browser-pool browser-manager
```

## Security Setup

```bash
# Generate API key
export API_KEY=$(openssl rand -hex 32)

# Create secret
kubectl create secret generic browser-pool-auth \
  --from-literal=api-key=$API_KEY \
  -n browser-pool

# Restart manager
kubectl rollout restart deployment/browser-manager -n browser-pool

# Test with auth
curl -X POST http://YOUR_LB_IP/observe \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com"}'
```

## Cleanup (Stop Billing)

```bash
cd terraform-vultr
terraform destroy
```

## Troubleshooting

### Pods not starting
```bash
kubectl describe pod -n browser-pool -l app=browser-worker
kubectl logs -n browser-pool -l app=browser-worker
```

### LoadBalancer pending
```bash
# Check LB status
kubectl describe svc browser-manager -n browser-pool
```

### Connection refused
```bash
# Check if manager is running
kubectl get pods -n browser-pool
kubectl logs -n browser-pool -l app=browser-manager
```
