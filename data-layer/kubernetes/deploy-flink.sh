#!/bin/bash
# ==========================================================
# Flink Deployment Script for Argus
# ==========================================================
# Deploys Apache Flink to Vultr K8s with:
# - Cloudflare R2 for checkpoint storage
# - SASL authentication to Redpanda
# - Prometheus metrics
#
# Prerequisites:
#   - kubectl configured for your cluster
#   - Redpanda running in argus-data namespace
#
# Usage:
#   export CLOUDFLARE_ACCOUNT_ID=your-account-id
#   export CLOUDFLARE_R2_ACCESS_KEY_ID=your-access-key
#   export CLOUDFLARE_R2_SECRET_ACCESS_KEY=your-secret-key
#   ./deploy-flink.sh
# ==========================================================
set -e

NAMESPACE="argus-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2_BUCKET="argus-flink-checkpoints"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Argus Flink Deployment with R2                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ==========================================================
# Check Prerequisites
# ==========================================================
check_prerequisites() {
    echo "📋 Checking prerequisites..."

    # kubectl
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl not found"
        exit 1
    fi

    # helm
    if ! command -v helm &> /dev/null; then
        echo "❌ helm not found"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo "❌ Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check required environment variables
    if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
        echo "❌ CLOUDFLARE_ACCOUNT_ID not set"
        echo "   Export it: export CLOUDFLARE_ACCOUNT_ID=your-account-id"
        exit 1
    fi

    if [ -z "$CLOUDFLARE_R2_ACCESS_KEY_ID" ]; then
        echo "❌ CLOUDFLARE_R2_ACCESS_KEY_ID not set"
        echo "   Export it: export CLOUDFLARE_R2_ACCESS_KEY_ID=your-access-key"
        exit 1
    fi

    if [ -z "$CLOUDFLARE_R2_SECRET_ACCESS_KEY" ]; then
        echo "❌ CLOUDFLARE_R2_SECRET_ACCESS_KEY not set"
        echo "   Export it: export CLOUDFLARE_R2_SECRET_ACCESS_KEY=your-secret-key"
        exit 1
    fi

    # Check Redpanda
    if ! kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=redpanda --no-headers 2>/dev/null | grep -q Running; then
        echo "⚠️  Redpanda not running in $NAMESPACE (continuing anyway)"
    fi

    echo "✅ All prerequisites met"
    echo ""
}

# ==========================================================
# Create Namespace
# ==========================================================
create_namespace() {
    echo "📦 Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    echo ""
}

# ==========================================================
# Install cert-manager
# ==========================================================
install_cert_manager() {
    echo "🔐 Step 1/4: Installing cert-manager..."

    if kubectl get namespace cert-manager &> /dev/null; then
        echo "   cert-manager already installed"
    else
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml
        echo "   Waiting for cert-manager..."
        kubectl wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=180s
        kubectl wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=180s
    fi
    echo "✅ cert-manager ready"
    echo ""
}

# ==========================================================
# Install Flink Operator
# ==========================================================
install_flink_operator() {
    echo "⚙️  Step 2/4: Installing Flink Kubernetes Operator..."

    helm repo add flink-operator https://downloads.apache.org/flink/flink-kubernetes-operator-1.10.0/ 2>/dev/null || true
    helm repo update

    if helm status flink-kubernetes-operator -n $NAMESPACE &> /dev/null; then
        echo "   Flink operator already installed"
    else
        helm install flink-kubernetes-operator flink-operator/flink-kubernetes-operator \
            -n $NAMESPACE \
            --set webhook.create=true \
            --set metrics.port=9999 \
            --wait
    fi

    echo "   Waiting for operator..."
    kubectl wait --for=condition=Available deployment/flink-kubernetes-operator -n $NAMESPACE --timeout=180s

    echo "✅ Flink operator ready"
    echo ""
}

# ==========================================================
# Deploy R2 Credentials Secret
# ==========================================================
deploy_secrets() {
    echo "🔑 Step 3/4: Deploying R2 credentials secret..."

    kubectl create secret generic flink-r2-credentials \
        -n $NAMESPACE \
        --from-literal=AWS_ACCESS_KEY_ID="$CLOUDFLARE_R2_ACCESS_KEY_ID" \
        --from-literal=AWS_SECRET_ACCESS_KEY="$CLOUDFLARE_R2_SECRET_ACCESS_KEY" \
        --from-literal=CLOUDFLARE_ACCOUNT_ID="$CLOUDFLARE_ACCOUNT_ID" \
        --dry-run=client -o yaml | kubectl apply -f -

    echo "✅ Secrets deployed"
    echo ""
}

# ==========================================================
# Deploy Flink Cluster
# ==========================================================
deploy_flink() {
    echo "🚀 Step 4/4: Deploying Flink cluster..."

    # Create a temporary file with substituted values
    TEMP_FILE=$(mktemp)

    # Replace placeholders in the deployment file
    sed -e "s|REPLACE_WITH_R2_ACCESS_KEY_ID|$CLOUDFLARE_R2_ACCESS_KEY_ID|g" \
        -e "s|REPLACE_WITH_R2_SECRET_ACCESS_KEY|$CLOUDFLARE_R2_SECRET_ACCESS_KEY|g" \
        -e "s|REPLACE_WITH_ACCOUNT_ID|$CLOUDFLARE_ACCOUNT_ID|g" \
        -e "s|\${CLOUDFLARE_ACCOUNT_ID}|$CLOUDFLARE_ACCOUNT_ID|g" \
        "$SCRIPT_DIR/flink-deployment.yaml" > "$TEMP_FILE"

    # Apply the deployment
    kubectl apply -f "$TEMP_FILE"

    # Cleanup
    rm -f "$TEMP_FILE"

    echo "✅ Flink deployment created"
    echo ""
}

# ==========================================================
# Wait for Flink to be ready
# ==========================================================
wait_for_flink() {
    echo "⏳ Waiting for Flink cluster to be ready..."

    # Wait for JobManager
    echo "   Waiting for JobManager..."
    for i in {1..60}; do
        if kubectl get pods -n $NAMESPACE -l app=argus-flink,component=jobmanager --no-headers 2>/dev/null | grep -q Running; then
            echo "   ✅ JobManager is running"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "   ⚠️  Timeout waiting for JobManager - check logs"
            kubectl get pods -n $NAMESPACE -l app=argus-flink
            break
        fi
        echo "   Waiting... ($i/60)"
        sleep 5
    done

    # Wait for TaskManager
    echo "   Waiting for TaskManager..."
    for i in {1..60}; do
        if kubectl get pods -n $NAMESPACE -l app=argus-flink,component=taskmanager --no-headers 2>/dev/null | grep -q Running; then
            echo "   ✅ TaskManager is running"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "   ⚠️  Timeout waiting for TaskManager"
            kubectl get pods -n $NAMESPACE -l app=argus-flink
            break
        fi
        echo "   Waiting... ($i/60)"
        sleep 5
    done

    echo ""
}

# ==========================================================
# Verify Deployment
# ==========================================================
verify_deployment() {
    echo "🔍 Verifying deployment..."

    echo ""
    echo "FlinkDeployment Status:"
    kubectl get flinkdeployment -n $NAMESPACE

    echo ""
    echo "Flink Pods:"
    kubectl get pods -n $NAMESPACE -l app=argus-flink

    echo ""
}

# ==========================================================
# Print Summary
# ==========================================================
print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   DEPLOYMENT COMPLETE                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🔗 ACCESS FLINK WEB UI:"
    echo "   kubectl port-forward svc/flink-rest -n $NAMESPACE 8081:8081"
    echo "   Open: http://localhost:8081"
    echo ""
    echo "📊 VIEW LOGS:"
    echo "   JobManager: kubectl logs -n $NAMESPACE -l component=jobmanager -f"
    echo "   TaskManager: kubectl logs -n $NAMESPACE -l component=taskmanager -f"
    echo ""
    echo "💾 CHECKPOINT STORAGE (Cloudflare R2):"
    echo "   R2 Bucket: $R2_BUCKET"
    echo "   Endpoint: https://$CLOUDFLARE_ACCOUNT_ID.r2.cloudflarestorage.com"
    echo "   Checkpoints: s3://$R2_BUCKET/checkpoints"
    echo "   Savepoints: s3://$R2_BUCKET/savepoints"
    echo "   HA Data: s3://$R2_BUCKET/ha"
    echo ""
    echo "⚠️  IMPORTANT: Create the R2 bucket manually if not exists:"
    echo "   wrangler r2 bucket create $R2_BUCKET"
    echo "   Or via Cloudflare Dashboard > R2 > Create bucket"
    echo ""
    echo "🔧 USEFUL COMMANDS:"
    echo "   # Check Flink status"
    echo "   kubectl get flinkdeployment -n $NAMESPACE"
    echo ""
    echo "   # Trigger manual savepoint"
    echo "   kubectl annotate flinkdeployment/argus-flink -n $NAMESPACE \\"
    echo "     flink.apache.org/trigger-savepoint=\$(date +%s)"
    echo ""
    echo "   # Scale TaskManagers"
    echo "   kubectl patch flinkdeployment argus-flink -n $NAMESPACE \\"
    echo "     --type=merge -p '{\"spec\":{\"taskManager\":{\"replicas\":2}}}'"
    echo ""
}

# ==========================================================
# Main
# ==========================================================
main() {
    check_prerequisites
    create_namespace
    install_cert_manager
    install_flink_operator
    deploy_secrets
    deploy_flink
    wait_for_flink
    verify_deployment
    print_summary
}

main "$@"
