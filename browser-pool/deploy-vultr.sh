#!/bin/bash
# Argus Browser Pool - Vultr Deployment Script
# Usage: ./deploy-vultr.sh [dev|small|medium]

set -e

SCALE_PRESET=${1:-dev}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Argus Browser Pool - Vultr Deployment              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Scale preset: $SCALE_PRESET"
echo "║  Region: Mumbai (bom) - fastest for India"
echo "╚══════════════════════════════════════════════════════════════╝"

# Check prerequisites
command -v terraform >/dev/null 2>&1 || { echo "Error: terraform is not installed"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "Error: kubectl is not installed"; exit 1; }

# Check for Vultr API key
if [ -z "$VULTR_API_KEY" ]; then
  if [ ! -f "$SCRIPT_DIR/terraform-vultr/terraform.tfvars" ]; then
    echo "Error: VULTR_API_KEY not set and terraform.tfvars not found"
    echo ""
    echo "Get your API key from: https://my.vultr.com/settings/#settingsapi"
    echo ""
    echo "Then either:"
    echo "  export VULTR_API_KEY=your-api-key"
    echo "  OR"
    echo "  cp terraform-vultr/terraform.tfvars.example terraform-vultr/terraform.tfvars"
    echo "  (and edit the file)"
    exit 1
  fi
fi

# Set scale preset variables
case $SCALE_PRESET in
  dev)
    NODE_COUNT=2
    NODE_PLAN="vc2-2c-4gb"
    NODE_MIN=1
    NODE_MAX=3
    echo "║  Config: 2 nodes, 2vCPU/4GB each (~\$50/mo)"
    ;;
  small)
    NODE_COUNT=3
    NODE_PLAN="vc2-2c-4gb"
    NODE_MIN=2
    NODE_MAX=5
    echo "║  Config: 3 nodes, 2vCPU/4GB each (~\$70/mo)"
    ;;
  medium)
    NODE_COUNT=5
    NODE_PLAN="vc2-4c-8gb"
    NODE_MIN=3
    NODE_MAX=10
    echo "║  Config: 5 nodes, 4vCPU/8GB each (~\$210/mo)"
    ;;
  *)
    echo "Unknown preset: $SCALE_PRESET"
    echo "Usage: ./deploy-vultr.sh [dev|small|medium]"
    exit 1
    ;;
esac

echo ""
echo "Step 1/4: Deploying VKE cluster with Terraform..."
echo "────────────────────────────────────────────────────────"

cd "$SCRIPT_DIR/terraform-vultr"

# Set Terraform variables
if [ -n "$VULTR_API_KEY" ]; then
  export TF_VAR_vultr_api_key="$VULTR_API_KEY"
fi
export TF_VAR_node_pool_count="$NODE_COUNT"
export TF_VAR_node_pool_plan="$NODE_PLAN"
export TF_VAR_node_pool_min="$NODE_MIN"
export TF_VAR_node_pool_max="$NODE_MAX"

terraform init
terraform apply -auto-approve

echo ""
echo "Step 2/4: Waiting for cluster to be ready..."
echo "────────────────────────────────────────────────────────"

# Export kubeconfig
export KUBECONFIG="$SCRIPT_DIR/terraform-vultr/kubeconfig.yaml"

# Wait for nodes to be ready
echo "Waiting for nodes..."
for i in {1..60}; do
  if kubectl get nodes 2>/dev/null | grep -q "Ready"; then
    echo "Nodes are ready!"
    kubectl get nodes
    break
  fi
  echo "Waiting for nodes to be ready... ($i/60)"
  sleep 10
done

echo ""
echo "Step 3/4: Deploying browser pool to Kubernetes..."
echo "────────────────────────────────────────────────────────"

cd "$SCRIPT_DIR/kubernetes"

# Apply manifests in order
kubectl apply -f namespace.yaml
sleep 2
kubectl apply -f configmaps.yaml
sleep 2
kubectl apply -f browser-deployment.yaml
kubectl apply -f services.yaml
kubectl apply -f hpa.yaml

echo ""
echo "Step 4/4: Setting up LoadBalancer and verifying..."
echo "────────────────────────────────────────────────────────"

# Wait for LoadBalancer IP
echo "Waiting for LoadBalancer IP..."
for i in {1..30}; do
  LB_IP=$(kubectl get svc -n browser-pool browser-manager -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
  if [ -n "$LB_IP" ]; then
    echo "LoadBalancer IP: $LB_IP"
    break
  fi
  echo "Waiting for LoadBalancer... ($i/30)"
  sleep 10
done

# If no LB IP, try hostname
if [ -z "$LB_IP" ]; then
  LB_IP=$(kubectl get svc -n browser-pool browser-manager -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")
fi

# Wait for pods
echo "Waiting for browser pods..."
kubectl wait --for=condition=ready pod -l app=browser-worker -n browser-pool --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=browser-manager -n browser-pool --timeout=300s || true

echo ""
echo "Deployment Status:"
kubectl get pods -n browser-pool
kubectl get svc -n browser-pool
kubectl get hpa -n browser-pool

# Generate API key
API_KEY=$(openssl rand -hex 32)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Deployment Complete!                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║"
echo "║  Browser Pool URL: http://$LB_IP"
echo "║  Kubeconfig: $SCRIPT_DIR/terraform-vultr/kubeconfig.yaml"
echo "║"
echo "║  Generated API Key (save this!):"
echo "║  $API_KEY"
echo "║"
echo "║  Apply security (run these commands):"
echo "║"
echo "║  # Set the API key secret"
echo "║  kubectl create secret generic browser-pool-auth \\"
echo "║    --from-literal=api-key=$API_KEY \\"
echo "║    -n browser-pool"
echo "║"
echo "║  # Restart manager to pick up the secret"
echo "║  kubectl rollout restart deployment/browser-manager -n browser-pool"
echo "║"
echo "║  Test endpoints:"
echo "║    curl http://$LB_IP/health"
echo "║    curl -X POST http://$LB_IP/observe \\"
echo "║      -H 'Authorization: Bearer $API_KEY' \\"
echo "║      -H 'Content-Type: application/json' \\"
echo "║      -d '{\"url\":\"https://example.com\"}'"
echo "║"
echo "║  Add to your .env:"
echo "║    BROWSER_POOL_URL=http://$LB_IP"
echo "║    BROWSER_POOL_API_KEY=$API_KEY"
echo "║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Save config to file for reference
cat > "$SCRIPT_DIR/deployment-config.env" << EOF
# Browser Pool Deployment Config
# Generated: $(date)
BROWSER_POOL_URL=http://$LB_IP
BROWSER_POOL_API_KEY=$API_KEY
KUBECONFIG=$SCRIPT_DIR/terraform-vultr/kubeconfig.yaml
EOF

echo ""
echo "Config saved to: $SCRIPT_DIR/deployment-config.env"
