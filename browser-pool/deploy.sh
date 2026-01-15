#!/bin/bash
# Argus Browser Pool - Deployment Script
# Usage: ./deploy.sh [dev|small|medium|large|enterprise]

set -e

SCALE_PRESET=${1:-dev}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Argus Browser Pool - Hetzner Deployment            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Scale preset: $SCALE_PRESET"
echo "╚══════════════════════════════════════════════════════════════╝"

# Check prerequisites
command -v terraform >/dev/null 2>&1 || { echo "Error: terraform is not installed"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "Error: kubectl is not installed"; exit 1; }

# Check for Hetzner token
if [ -z "$HCLOUD_TOKEN" ]; then
  if [ ! -f "$SCRIPT_DIR/terraform/terraform.tfvars" ]; then
    echo "Error: HCLOUD_TOKEN not set and terraform.tfvars not found"
    echo "Please set HCLOUD_TOKEN or copy terraform.tfvars.example to terraform.tfvars"
    exit 1
  fi
fi

echo ""
echo "Step 1/4: Deploying Kubernetes cluster with Terraform..."
echo "────────────────────────────────────────────────────────"

cd "$SCRIPT_DIR/terraform"

# Set scale preset
if [ -n "$HCLOUD_TOKEN" ]; then
  export TF_VAR_hcloud_token="$HCLOUD_TOKEN"
fi
export TF_VAR_scale_preset="$SCALE_PRESET"

terraform init
terraform apply -auto-approve

# Get outputs
KUBECONFIG_PATH=$(terraform output -raw kubeconfig_path)
LB_IP=$(terraform output -raw load_balancer_ip)

echo ""
echo "Step 2/4: Waiting for cluster to be ready..."
echo "────────────────────────────────────────────────────────"

export KUBECONFIG="$KUBECONFIG_PATH"

# Wait for nodes to be ready
echo "Waiting for nodes..."
for i in {1..30}; do
  if kubectl get nodes 2>/dev/null | grep -q "Ready"; then
    echo "Nodes are ready!"
    kubectl get nodes
    break
  fi
  echo "Waiting for nodes to be ready... ($i/30)"
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
echo "Step 4/4: Verifying deployment..."
echo "────────────────────────────────────────────────────────"

# Wait for pods to be ready
echo "Waiting for browser pods..."
kubectl wait --for=condition=ready pod -l app=browser-worker -n browser-pool --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=browser-manager -n browser-pool --timeout=300s || true

echo ""
echo "Deployment Status:"
kubectl get pods -n browser-pool
kubectl get svc -n browser-pool
kubectl get hpa -n browser-pool

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Deployment Complete!                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Browser Pool URL: http://$LB_IP"
echo "║  Kubeconfig: $KUBECONFIG_PATH"
echo "║"
echo "║  Test endpoints:"
echo "║    curl http://$LB_IP/health"
echo "║    curl -X POST http://$LB_IP/observe -d '{\"url\":\"https://example.com\"}'"
echo "║"
echo "║  Update MCP server with:"
echo "║    BROWSER_POOL_URL=http://$LB_IP"
echo "╚══════════════════════════════════════════════════════════════╝"
