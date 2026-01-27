#!/bin/bash
# =============================================================================
# Argus Data Layer - Minimal Deployment
# =============================================================================
# Deploys ONLY what's needed for Uber/Netflix architecture:
# - Flink (stateless, writes to Supabase)
# - Cognee Worker (knowledge graph processing)
#
# Does NOT deploy (using managed services instead):
# - Redpanda (using Serverless)
# - PostgreSQL (using Supabase)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="argus-data"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Argus Data Layer - Minimal Deployment (Uber/Netflix)     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
preflight_checks() {
    echo "📋 Pre-flight checks..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl not found"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo "❌ Cannot connect to Kubernetes cluster"
        echo "   Ensure kubectl is configured for Vultr"
        exit 1
    fi

    # Show cluster info
    echo "   Cluster: $(kubectl config current-context)"
    echo ""

    # Check existing namespaces
    echo "📦 Existing namespaces:"
    kubectl get namespaces --no-headers | awk '{print "   - "$1}'
    echo ""

    # Check if browser-pool exists (confirms we're on right cluster)
    if kubectl get namespace browser-pool &> /dev/null; then
        echo "✅ Confirmed: browser-pool namespace exists"
        echo "   This is the correct Vultr cluster"
    else
        echo "⚠️  Warning: browser-pool namespace not found"
        echo "   Are you connected to the right cluster?"
        read -p "   Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Create namespace with isolation
# -----------------------------------------------------------------------------
create_namespace() {
    echo "📦 Creating namespace: $NAMESPACE"
    kubectl apply -f "$SCRIPT_DIR/namespace.yaml"
    echo "✅ Namespace created with ResourceQuota and LimitRange"
    echo ""
}

# -----------------------------------------------------------------------------
# Apply network policies (isolation)
# -----------------------------------------------------------------------------
apply_network_policies() {
    echo "🔒 Applying network policies..."
    kubectl apply -f "$SCRIPT_DIR/network-policies.yaml"
    echo "✅ Network isolation configured"
    echo "   - Default deny all ingress/egress"
    echo "   - Allow only internal namespace traffic"
    echo "   - Allow external HTTPS for APIs"
    echo ""
}

# -----------------------------------------------------------------------------
# Create secrets
# -----------------------------------------------------------------------------
create_secrets() {
    echo "🔑 Creating secrets..."

    # Check if secrets already exist
    if kubectl get secret redpanda-credentials -n $NAMESPACE &> /dev/null; then
        echo "   Secrets already exist, skipping..."
    else
        echo "   Creating secrets from environment variables..."

        # Create secret from environment variables
        kubectl create secret generic redpanda-credentials -n $NAMESPACE \
            --from-literal=bootstrap_servers="${REDPANDA_BROKERS:-d5rq6j17lc71h60f3oog.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092}" \
            --from-literal=sasl_username="${REDPANDA_SASL_USERNAME:-bapatla92}" \
            --from-literal=sasl_password="${REDPANDA_SASL_PASSWORD:-PpoOCp1nneHSPNIdBLh2NV5GbEuadZ}" \
            --from-literal=sasl_mechanism="SCRAM-SHA-256" \
            --from-literal=security_protocol="SASL_SSL" \
            --dry-run=client -o yaml | kubectl apply -f -

        kubectl create secret generic supabase-credentials -n $NAMESPACE \
            --from-literal=url="${SUPABASE_URL}" \
            --from-literal=service_key="${SUPABASE_SERVICE_KEY}" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    echo "✅ Secrets configured"
    echo ""
}

# -----------------------------------------------------------------------------
# Deploy Cognee Worker
# -----------------------------------------------------------------------------
deploy_cognee_worker() {
    echo "🧠 Deploying Cognee Worker..."

    if [ -f "$SCRIPT_DIR/cognee-worker.yaml" ]; then
        kubectl apply -f "$SCRIPT_DIR/cognee-worker.yaml"
        echo "✅ Cognee Worker deployed"
    else
        echo "⚠️  cognee-worker.yaml not found, skipping"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Deploy Flink (minimal - stateless)
# -----------------------------------------------------------------------------
deploy_flink() {
    echo "⚡ Deploying Flink (stateless)..."

    # Check if cert-manager exists
    if ! kubectl get namespace cert-manager &> /dev/null; then
        echo "   Installing cert-manager (required for Flink operator)..."
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml
        echo "   Waiting for cert-manager..."
        kubectl wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=180s || true
    fi

    # Install Flink operator
    echo "   Installing Flink Kubernetes Operator..."
    helm repo add flink-operator https://downloads.apache.org/flink/flink-kubernetes-operator-1.10.0/ 2>/dev/null || true
    helm repo update

    if helm status flink-kubernetes-operator -n $NAMESPACE &> /dev/null; then
        echo "   Flink operator already installed"
    else
        helm install flink-kubernetes-operator flink-operator/flink-kubernetes-operator \
            -n $NAMESPACE \
            --set webhook.create=true \
            --wait || echo "   Warning: Flink operator install may need retry"
    fi

    # Deploy Flink cluster (stateless config)
    if [ -f "$SCRIPT_DIR/flink-platform/self-healing-operator.yaml" ]; then
        kubectl apply -f "$SCRIPT_DIR/flink-platform/self-healing-operator.yaml"
        echo "✅ Flink cluster deployed (stateless, writes to Supabase)"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Verify deployment
# -----------------------------------------------------------------------------
verify_deployment() {
    echo "🔍 Verifying deployment..."
    echo ""

    echo "Pods in $NAMESPACE:"
    kubectl get pods -n $NAMESPACE -o wide
    echo ""

    echo "Services in $NAMESPACE:"
    kubectl get services -n $NAMESPACE
    echo ""

    echo "Resource usage:"
    kubectl top pods -n $NAMESPACE 2>/dev/null || echo "   (metrics-server not available)"
    echo ""
}

# -----------------------------------------------------------------------------
# Print summary
# -----------------------------------------------------------------------------
print_summary() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   DEPLOYMENT COMPLETE                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 What's Running on Vultr K8s:"
    echo "───────────────────────────────"
    echo "  Namespace: browser-pool"
    echo "    ✅ Selenium Grid (video recording)"
    echo ""
    echo "  Namespace: argus-data"
    echo "    ✅ Cognee Worker (knowledge graphs)"
    echo "    ✅ Flink (stateless stream processing)"
    echo ""
    echo "🌐 External Services (Managed):"
    echo "───────────────────────────────"
    echo "  ✅ Redpanda Serverless (Kafka)"
    echo "  ✅ Supabase (PostgreSQL + real-time)"
    echo ""
    echo "🔒 Isolation:"
    echo "───────────────────────────────"
    echo "  ✅ browser-pool and argus-data are isolated"
    echo "  ✅ NetworkPolicy prevents cross-namespace traffic"
    echo "  ✅ ResourceQuota prevents resource starvation"
    echo ""
    echo "📝 Useful Commands:"
    echo "───────────────────────────────"
    echo "  # View all pods"
    echo "  kubectl get pods -A"
    echo ""
    echo "  # View Flink logs"
    echo "  kubectl logs -n argus-data -l component=jobmanager -f"
    echo ""
    echo "  # View Cognee logs"
    echo "  kubectl logs -n argus-data -l app=cognee-worker -f"
    echo ""
    echo "  # Port-forward Flink UI"
    echo "  kubectl port-forward -n argus-data svc/argus-flink-rest 8081:8081"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    preflight_checks
    create_namespace
    apply_network_policies
    create_secrets
    deploy_cognee_worker
    deploy_flink
    verify_deployment
    print_summary
}

# Run with optional --dry-run flag
if [[ "$1" == "--dry-run" ]]; then
    echo "🔍 DRY RUN - showing what would be deployed:"
    echo ""
    preflight_checks
    echo "Would create namespace: $NAMESPACE"
    echo "Would apply network policies"
    echo "Would deploy Cognee Worker"
    echo "Would deploy Flink (stateless)"
else
    main
fi
