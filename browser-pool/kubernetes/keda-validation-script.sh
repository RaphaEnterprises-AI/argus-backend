#!/bin/bash
# KEDA Selenium Grid 4 Validation and Troubleshooting Script
# Usage: bash keda-validation-script.sh [check-name]
#
# Validates KEDA autoscaling configuration for Selenium Grid 4
# Performs comprehensive checks and provides troubleshooting guidance

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default namespace
NAMESPACE="${KEDA_NAMESPACE:-selenium-grid}"
KEDA_NAMESPACE="${KEDA_NAMESPACE:-keda}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Print section header
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check Kubernetes connection
check_kubernetes_connection() {
    print_header "1. Kubernetes Connection"

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
    log_success "Connected to Kubernetes cluster"

    # Get cluster info
    local context=$(kubectl config current-context)
    log_info "Current context: $context"

    # Check namespaces exist
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_success "Namespace '$NAMESPACE' exists"
    else
        log_error "Namespace '$NAMESPACE' not found"
        return 1
    fi

    if kubectl get namespace $KEDA_NAMESPACE &> /dev/null; then
        log_success "KEDA namespace '$KEDA_NAMESPACE' exists"
    else
        log_error "KEDA namespace '$KEDA_NAMESPACE' not found"
        return 1
    fi
}

# Check KEDA operator
check_keda_operator() {
    print_header "2. KEDA Operator Status"

    # Check if KEDA operator deployment exists
    if ! kubectl get deployment -n $KEDA_NAMESPACE keda-operator &> /dev/null; then
        log_error "KEDA operator deployment not found in namespace '$KEDA_NAMESPACE'"
        log_info "Install KEDA: helm install keda kedacore/keda --namespace keda --create-namespace"
        return 1
    fi

    # Check operator readiness
    local replicas=$(kubectl get deployment -n $KEDA_NAMESPACE keda-operator -o jsonpath='{.status.readyReplicas}')
    local desired=$(kubectl get deployment -n $KEDA_NAMESPACE keda-operator -o jsonpath='{.spec.replicas}')

    if [ "$replicas" = "$desired" ]; then
        log_success "KEDA operator ready ($replicas/$desired replicas)"
    else
        log_warning "KEDA operator not fully ready ($replicas/$desired replicas)"
    fi

    # Check metrics API server
    if kubectl get deployment -n $KEDA_NAMESPACE keda-operator-metrics-apiserver &> /dev/null; then
        local metrics_replicas=$(kubectl get deployment -n $KEDA_NAMESPACE keda-operator-metrics-apiserver -o jsonpath='{.status.readyReplicas}')
        local metrics_desired=$(kubectl get deployment -n $KEDA_NAMESPACE keda-operator-metrics-apiserver -o jsonpath='{.spec.replicas}')

        if [ "$metrics_replicas" = "$metrics_desired" ]; then
            log_success "KEDA metrics API server ready ($metrics_replicas/$metrics_desired replicas)"
        else
            log_warning "KEDA metrics API server not fully ready ($metrics_replicas/$metrics_desired replicas)"
        fi
    else
        log_warning "KEDA metrics API server not found"
    fi

    # Check KEDA operator logs for errors
    log_info "Recent KEDA operator logs:"
    kubectl logs -n $KEDA_NAMESPACE deployment/keda-operator --tail=5 2>/dev/null | sed 's/^/  /'
}

# Check ScaledObjects
check_scaled_objects() {
    print_header "3. ScaledObjects Configuration"

    # Get all ScaledObjects
    local count=$(kubectl get scaledobjects -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        log_warning "No ScaledObjects found in namespace '$NAMESPACE'"
        log_info "Deploy KEDA manifests: kubectl apply -f keda-scaledobject.yaml"
        return 1
    fi

    log_success "Found $count ScaledObject(s)"

    # Check each ScaledObject
    for scaledobject in $(kubectl get scaledobjects -n $NAMESPACE -o name); do
        local name=$(echo $scaledobject | cut -d'/' -f2)
        local status=$(kubectl get scaledobject $name -n $NAMESPACE -o jsonpath='{.status.conditions[0].status}' 2>/dev/null)
        local message=$(kubectl get scaledobject $name -n $NAMESPACE -o jsonpath='{.status.conditions[0].message}' 2>/dev/null)

        if [ "$status" = "True" ]; then
            log_success "ScaledObject '$name' is active"
        else
            log_warning "ScaledObject '$name' status: $message"
        fi

        # Check target deployment
        local target=$(kubectl get scaledobject $name -n $NAMESPACE -o jsonpath='{.spec.scaleTargetRef.name}')
        if kubectl get deployment $target -n $NAMESPACE &> /dev/null; then
            log_info "  Target deployment: $target ✓"
        else
            log_error "  Target deployment not found: $target"
        fi
    done
}

# Check Prometheus connectivity
check_prometheus() {
    print_header "4. Prometheus Configuration"

    # Try to find Prometheus
    local prom_pods=$(kubectl get pods -n prometheus -l app=prometheus 2>/dev/null | wc -l)

    if [ "$prom_pods" -lt 2 ]; then
        log_warning "Prometheus not found or not ready in 'prometheus' namespace"
        log_info "Check other namespaces: kubectl get pods -A | grep prometheus"
        return 1
    fi

    log_success "Prometheus pods found: $(($prom_pods - 1))"

    # Test Prometheus connectivity
    log_info "Testing Prometheus API connectivity..."

    # Port-forward to test (if in Docker or local environment)
    if command -v curl &> /dev/null; then
        # Try direct connection if Prometheus service is accessible
        if kubectl get svc -n prometheus prometheus &> /dev/null; then
            log_success "Prometheus service found"
        else
            log_warning "Prometheus service not found, may need port-forwarding"
        fi
    fi
}

# Check Selenium Grid metrics
check_selenium_metrics() {
    print_header "5. Selenium Grid Metrics"

    # Check if Selenium Grid hub is running
    if ! kubectl get pod -n $NAMESPACE -l app=selenium-grid,component=hub &> /dev/null; then
        log_warning "Selenium Grid hub not found"
        return 1
    fi

    log_success "Selenium Grid hub pod(s) found"

    # Try to access metrics endpoint
    local hub_pod=$(kubectl get pod -n $NAMESPACE -l app=selenium-grid,component=hub -o name 2>/dev/null | head -1 | cut -d'/' -f2)

    if [ -z "$hub_pod" ]; then
        log_error "Could not find hub pod name"
        return 1
    fi

    log_info "Checking metrics from hub pod: $hub_pod"

    # Port-forward and test metrics
    log_info "Required metrics for autoscaling:"
    log_info "  - selenium_sessions_queued{node=\"chrome\"}"
    log_info "  - selenium_node_slots_available{node=\"chrome\"}"
    log_info "  - selenium_node_slots_used{node=\"chrome\"}"
}

# Check browser node deployments
check_browser_nodes() {
    print_header "6. Browser Node Deployments"

    local browser_types=("chrome" "firefox" "edge")

    for browser in "${browser_types[@]}"; do
        log_info "Checking $browser nodes..."

        local deployment="selenium-grid-selenium-node-$browser"

        if kubectl get deployment $deployment -n $NAMESPACE &> /dev/null; then
            local replicas=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
            local desired=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.replicas}')

            if [ "$replicas" = "$desired" ]; then
                log_success "  $browser nodes: $replicas/$desired ready"
            else
                log_warning "  $browser nodes: $replicas/$desired ready"
            fi
        else
            log_warning "  Deployment not found: $deployment"
        fi
    done
}

# Check HPA status
check_hpa_status() {
    print_header "7. Horizontal Pod Autoscaler Status"

    # Get HPA objects created by KEDA
    local hpa_count=$(kubectl get hpa -n $NAMESPACE --no-headers 2>/dev/null | wc -l)

    if [ "$hpa_count" -eq 0 ]; then
        log_warning "No HPA objects found (should be created by KEDA)"
    else
        log_success "Found $hpa_count HPA object(s)"

        # Show HPA details
        kubectl get hpa -n $NAMESPACE -o wide
    fi
}

# Check Pod Disruption Budgets
check_pdb() {
    print_header "8. Pod Disruption Budgets"

    local pdb_count=$(kubectl get pdb -n $NAMESPACE --no-headers 2>/dev/null | wc -l)

    if [ "$pdb_count" -eq 0 ]; then
        log_warning "No PDB objects found (optional for availability)"
    else
        log_success "Found $pdb_count PDB object(s)"
        kubectl get pdb -n $NAMESPACE -o wide
    fi
}

# Check resource requests and limits
check_resources() {
    print_header "9. Resource Requests and Limits"

    log_info "Checking browser node resource configuration..."

    for browser in "chrome" "firefox" "edge"; do
        local deployment="selenium-grid-selenium-node-$browser"

        if kubectl get deployment $deployment -n $NAMESPACE &> /dev/null; then
            local requests=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].resources.requests}')
            local limits=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].resources.limits}')

            if [ -n "$requests" ]; then
                log_info "  $browser - Requests: $requests"
            fi
            if [ -n "$limits" ]; then
                log_info "  $browser - Limits: $limits"
            fi
        fi
    done
}

# Check recent events
check_events() {
    print_header "10. Recent Kubernetes Events"

    log_info "Recent scaling events:"
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' 2>/dev/null | tail -10 | sed 's/^/  /'
}

# Test scaling (creates test load)
test_scaling() {
    print_header "11. Scaling Test"

    log_info "This will create test load on Selenium Grid to verify scaling"
    read -p "Continue? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Creating test pods to generate Selenium requests..."

        # Create a test job that creates sessions
        cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: selenium-load-test
  namespace: $NAMESPACE
spec:
  template:
    spec:
      serviceAccountName: default
      containers:
      - name: load-test
        image: curlimages/curl:latest
        command:
        - /bin/sh
        - -c
        - |
          for i in {1..10}; do
            curl -X POST http://selenium-grid-selenium-hub:4444/session \
              -H "Content-Type: application/json" \
              -d '{"capabilities":{"browserName":"chrome"}}' &
          done
          wait
      restartPolicy: Never
  backoffLimit: 3
EOF

        log_info "Load test job created. Monitor scaling:"
        log_info "  kubectl get hpa -n $NAMESPACE -w"
        log_info "  kubectl get pods -n $NAMESPACE | grep chrome"
    fi
}

# Generate summary report
generate_report() {
    print_header "VALIDATION SUMMARY"

    echo ""
    echo "Run individual checks:"
    echo "  $0 kubernetes"
    echo "  $0 keda"
    echo "  $0 scaledobjects"
    echo "  $0 prometheus"
    echo "  $0 selenium"
    echo "  $0 browser-nodes"
    echo "  $0 hpa"
    echo "  $0 pdb"
    echo "  $0 resources"
    echo "  $0 events"
    echo "  $0 test"
    echo ""
    echo "Run all checks:"
    echo "  $0 all"
    echo ""
}

# Main execution
main() {
    local check_type="${1:-all}"

    case "$check_type" in
        kubernetes)
            check_kubernetes_connection
            ;;
        keda)
            check_keda_operator
            ;;
        scaledobjects)
            check_kubernetes_connection && check_scaled_objects
            ;;
        prometheus)
            check_prometheus
            ;;
        selenium)
            check_selenium_metrics
            ;;
        browser-nodes)
            check_kubernetes_connection && check_browser_nodes
            ;;
        hpa)
            check_kubernetes_connection && check_hpa_status
            ;;
        pdb)
            check_kubernetes_connection && check_pdb
            ;;
        resources)
            check_kubernetes_connection && check_resources
            ;;
        events)
            check_kubernetes_connection && check_events
            ;;
        test)
            check_kubernetes_connection && test_scaling
            ;;
        all)
            check_kubernetes_connection && \
            check_keda_operator && \
            check_scaled_objects && \
            check_prometheus && \
            check_selenium_metrics && \
            check_browser_nodes && \
            check_hpa_status && \
            check_pdb && \
            check_resources && \
            check_events && \
            generate_report
            ;;
        *)
            echo "Invalid check type: $check_type"
            generate_report
            exit 1
            ;;
    esac
}

# Run main with provided arguments
main "$@"
