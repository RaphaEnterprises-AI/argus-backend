#!/bin/bash
# ==========================================================
# Argus Flink Platform - Complete Deployment Script
# ==========================================================
# Deploys a fully automated, self-healing Flink platform with:
# - Automatic scaling (KEDA)
# - Automatic checkpointing (S3/R2)
# - Automatic recovery
# - Full monitoring (Prometheus + Grafana)
# - Alerting (Slack/PagerDuty)
# ==========================================================
set -e

NAMESPACE="argus-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Argus Flink Platform - Automated Deployment            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."

    # kubectl
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl not found. Please install kubectl."
        exit 1
    fi

    # helm
    if ! command -v helm &> /dev/null; then
        echo "❌ helm not found. Please install helm."
        exit 1
    fi

    # Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        echo "❌ Cannot connect to Kubernetes cluster."
        echo "   Run: kubectl config current-context"
        exit 1
    fi

    echo "✅ All prerequisites met"
    echo ""
}

# Create namespace
create_namespace() {
    echo "📦 Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    echo ""
}

# Install cert-manager (required for Flink operator webhooks)
install_cert_manager() {
    echo "🔐 Step 1/6: Installing cert-manager..."

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

# Install KEDA (autoscaling)
install_keda() {
    echo "📈 Step 2/6: Installing KEDA (autoscaler)..."

    helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
    helm repo update

    if helm status keda -n keda &> /dev/null; then
        echo "   KEDA already installed, upgrading..."
        helm upgrade keda kedacore/keda -n keda
    else
        kubectl create namespace keda --dry-run=client -o yaml | kubectl apply -f -
        helm install keda kedacore/keda -n keda --wait
    fi

    echo "✅ KEDA ready"
    echo ""
}

# Install Prometheus + Grafana (monitoring)
install_monitoring() {
    echo "📊 Step 3/6: Installing monitoring stack..."

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo update

    if helm status prometheus -n monitoring &> /dev/null; then
        echo "   Prometheus stack already installed"
    else
        kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
        helm install prometheus prometheus-community/kube-prometheus-stack \
            -n monitoring \
            --set grafana.enabled=true \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --wait
    fi

    # Apply Flink-specific monitoring
    kubectl apply -f "$SCRIPT_DIR/monitoring.yaml"

    echo "✅ Monitoring stack ready"
    echo ""
}

# Install Flink Kubernetes Operator
install_flink_operator() {
    echo "⚙️  Step 4/6: Installing Flink Kubernetes Operator..."

    helm repo add flink-operator https://downloads.apache.org/flink/flink-kubernetes-operator-1.10.0/ 2>/dev/null || true
    helm repo update

    if helm status flink-kubernetes-operator -n $NAMESPACE &> /dev/null; then
        echo "   Flink operator already installed, upgrading..."
        helm upgrade flink-kubernetes-operator flink-operator/flink-kubernetes-operator \
            -n $NAMESPACE \
            --set webhook.create=true \
            --set metrics.port=9999
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

# Deploy secrets and config
deploy_config() {
    echo "🔑 Step 5/6: Deploying secrets and configuration..."

    # Apply checkpoint config
    kubectl apply -f "$SCRIPT_DIR/checkpoint-config.yaml"

    # Apply KEDA autoscaler
    kubectl apply -f "$SCRIPT_DIR/keda-autoscaler.yaml"

    echo "✅ Configuration deployed"
    echo ""
}

# Deploy Flink cluster
deploy_flink_cluster() {
    echo "🚀 Step 6/6: Deploying Flink cluster..."

    kubectl apply -f "$SCRIPT_DIR/self-healing-operator.yaml"

    echo "   Waiting for JobManager..."
    sleep 15

    # Wait for JobManager pod
    for i in {1..30}; do
        if kubectl get pods -n $NAMESPACE -l component=jobmanager --no-headers 2>/dev/null | grep -q Running; then
            echo "✅ Flink cluster deployed and running"
            break
        fi
        echo "   Waiting for JobManager... ($i/30)"
        sleep 10
    done

    echo ""
}

# Print summary
print_summary() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   DEPLOYMENT COMPLETE                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 WHAT'S RUNNING:"
    echo "─────────────────────────────────────────────────────────────────"
    kubectl get pods -n $NAMESPACE
    echo ""
    echo "🔗 ACCESS POINTS:"
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    echo "Flink Web UI:"
    echo "  kubectl port-forward svc/argus-flink-rest -n $NAMESPACE 8081:8081"
    echo "  Open: http://localhost:8081"
    echo ""
    echo "Grafana Dashboard:"
    echo "  kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80"
    echo "  Open: http://localhost:3000 (admin/prom-operator)"
    echo ""
    echo "Prometheus:"
    echo "  kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n monitoring 9090:9090"
    echo "  Open: http://localhost:9090"
    echo ""
    echo "🤖 AUTOMATION ENABLED:"
    echo "─────────────────────────────────────────────────────────────────"
    echo "✅ Auto-scaling: KEDA scales TaskManagers based on Kafka lag"
    echo "✅ Auto-recovery: Jobs restart from checkpoint on failure"
    echo "✅ Auto-checkpointing: Every 60 seconds to S3/R2"
    echo "✅ Monitoring: Prometheus scrapes metrics every 15 seconds"
    echo "✅ Alerting: PrometheusRules configured for backpressure, failures"
    echo "✅ HA: Kubernetes-native high availability"
    echo ""
    echo "📚 USEFUL COMMANDS:"
    echo "─────────────────────────────────────────────────────────────────"
    echo "# Check Flink status"
    echo "kubectl get flinkdeployment -n $NAMESPACE"
    echo ""
    echo "# View JobManager logs"
    echo "kubectl logs -n $NAMESPACE -l component=jobmanager -f"
    echo ""
    echo "# View TaskManager logs"
    echo "kubectl logs -n $NAMESPACE -l component=taskmanager -f"
    echo ""
    echo "# Trigger manual savepoint"
    echo "kubectl annotate flinkdeployment/argus-flink -n $NAMESPACE \\"
    echo "  flink.apache.org/trigger-savepoint=\$(date +%s)"
    echo ""
    echo "# Scale TaskManagers manually"
    echo "kubectl patch flinkdeployment argus-flink -n $NAMESPACE \\"
    echo "  --type=merge -p '{\"spec\":{\"taskManager\":{\"replicas\":4}}}'"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    create_namespace
    install_cert_manager
    install_keda
    install_monitoring
    install_flink_operator
    deploy_config
    deploy_flink_cluster
    print_summary
}

main "$@"
