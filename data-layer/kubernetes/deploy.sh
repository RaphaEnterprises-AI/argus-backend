#!/bin/bash
# Argus Data Layer - Kubernetes Deployment Script
# Deploys Redpanda, FalkorDB, Valkey, and Cognee Worker to VKE
#
# Usage: ./deploy.sh [--generate-secrets] [--skip-helm-repo]
#
# Options:
#   --generate-secrets  Generate random passwords and update secrets.yaml
#   --skip-helm-repo    Skip adding Helm repository (use if already added)
#   --dry-run           Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="argus-data"
HELM_RELEASE="redpanda"
GENERATE_SECRETS=false
SKIP_HELM_REPO=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-secrets) GENERATE_SECRETS=true; shift ;;
        --skip-helm-repo) SKIP_HELM_REPO=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    log_error "helm is not installed"
    exit 1
fi

# Verify kubectl can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster. Check KUBECONFIG."
    exit 1
fi

log_success "Prerequisites check passed"

# Generate secrets if requested
if [ "$GENERATE_SECRETS" = true ]; then
    log_info "Generating random passwords..."

    FALKORDB_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
    VALKEY_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
    REDPANDA_ADMIN_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
    REDPANDA_SERVICE_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)

    log_info "Updating secrets.yaml with generated passwords..."

    # Create a copy with generated secrets
    SECRETS_FILE="${SCRIPT_DIR}/secrets.yaml"
    SECRETS_GENERATED="${SCRIPT_DIR}/secrets-generated.yaml"

    cp "$SECRETS_FILE" "$SECRETS_GENERATED"

    # Update FalkorDB password in multiple places
    sed -i.bak "s/falkordb-password: \"REPLACE_WITH_SECURE_PASSWORD\"/falkordb-password: \"${FALKORDB_PASS}\"/" "$SECRETS_GENERATED"

    # Update Valkey password
    sed -i.bak "s/valkey-password: \"REPLACE_WITH_SECURE_PASSWORD\"/valkey-password: \"${VALKEY_PASS}\"/" "$SECRETS_GENERATED"

    # Update Redpanda password
    sed -i.bak "s/redpanda-password: \"REPLACE_WITH_SECURE_PASSWORD\"/redpanda-password: \"${REDPANDA_SERVICE_PASS}\"/" "$SECRETS_GENERATED"

    # Update Redpanda superusers (need to handle multiline)
    sed -i.bak "s/admin:REPLACE_WITH_ADMIN_PASSWORD:SCRAM-SHA-512/admin:${REDPANDA_ADMIN_PASS}:SCRAM-SHA-512/" "$SECRETS_GENERATED"
    sed -i.bak "s/argus-service:REPLACE_WITH_SERVICE_PASSWORD:SCRAM-SHA-512/argus-service:${REDPANDA_SERVICE_PASS}:SCRAM-SHA-512/" "$SECRETS_GENERATED"

    # Update falkordb-auth secret
    # This one appears at the end of the file, need a different approach
    # Use awk to replace only the last occurrence
    awk -v pass="$FALKORDB_PASS" '
    /name: falkordb-auth/,/password:/ {
        if (/password:.*REPLACE/) {
            sub(/REPLACE_WITH_SECURE_PASSWORD/, pass)
        }
    }
    {print}
    ' "$SECRETS_GENERATED" > "${SECRETS_GENERATED}.tmp" && mv "${SECRETS_GENERATED}.tmp" "$SECRETS_GENERATED"

    # Update valkey-auth secret
    awk -v pass="$VALKEY_PASS" '
    /name: valkey-auth/,/password:/ {
        if (/password:.*REPLACE/) {
            sub(/REPLACE_WITH_SECURE_PASSWORD/, pass)
        }
    }
    {print}
    ' "$SECRETS_GENERATED" > "${SECRETS_GENERATED}.tmp" && mv "${SECRETS_GENERATED}.tmp" "$SECRETS_GENERATED"

    rm -f "${SECRETS_GENERATED}.bak"

    log_success "Generated passwords:"
    echo "  FalkorDB: ${FALKORDB_PASS}"
    echo "  Valkey: ${VALKEY_PASS}"
    echo "  Redpanda Admin: ${REDPANDA_ADMIN_PASS}"
    echo "  Redpanda Service: ${REDPANDA_SERVICE_PASS}"

    SECRETS_TO_APPLY="$SECRETS_GENERATED"
else
    SECRETS_TO_APPLY="${SCRIPT_DIR}/secrets.yaml"

    # Check if secrets have placeholder values
    if grep -q "REPLACE_WITH" "$SECRETS_TO_APPLY"; then
        log_warn "secrets.yaml contains placeholder values!"
        log_warn "Run with --generate-secrets to auto-generate passwords,"
        log_warn "or manually update secrets.yaml before deploying."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Step 1: Add Helm repository
if [ "$SKIP_HELM_REPO" = false ]; then
    log_info "Adding Redpanda Helm repository..."
    run_cmd helm repo add redpanda https://charts.redpanda.com/ || true
    run_cmd helm repo update
    log_success "Helm repository ready"
fi

# Step 2: Create namespace
log_info "Creating namespace and resource quotas..."
run_cmd kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"
log_success "Namespace created"

# Step 3: Apply secrets (MUST be before Helm install)
log_info "Applying secrets..."
run_cmd kubectl apply -f "$SECRETS_TO_APPLY"
log_success "Secrets applied"

# Step 4: Deploy FalkorDB
log_info "Deploying FalkorDB..."
run_cmd kubectl apply -f "${SCRIPT_DIR}/falkordb.yaml"
log_success "FalkorDB deployed"

# Step 5: Deploy Valkey
log_info "Deploying Valkey..."
run_cmd kubectl apply -f "${SCRIPT_DIR}/valkey.yaml"
log_success "Valkey deployed"

# Step 6: Wait for storage to be ready
log_info "Waiting for PVCs to be bound (this may take a few minutes)..."
run_cmd kubectl wait --for=condition=Bound pvc/data-falkordb-0 -n "$NAMESPACE" --timeout=300s || true
run_cmd kubectl wait --for=condition=Bound pvc/data-valkey-0 -n "$NAMESPACE" --timeout=300s || true
log_success "Storage ready"

# Step 7: Deploy Redpanda via Helm
log_info "Deploying Redpanda via Helm..."
log_info "This may take 10-15 minutes on a small cluster..."

# Check if release exists
if helm status "$HELM_RELEASE" -n "$NAMESPACE" &> /dev/null; then
    log_info "Redpanda release exists, upgrading..."
    run_cmd helm upgrade "$HELM_RELEASE" redpanda/redpanda \
        -n "$NAMESPACE" \
        -f "${SCRIPT_DIR}/redpanda-values.yaml" \
        --wait \
        --timeout 15m
else
    log_info "Installing Redpanda..."
    run_cmd helm install "$HELM_RELEASE" redpanda/redpanda \
        -n "$NAMESPACE" \
        -f "${SCRIPT_DIR}/redpanda-values.yaml" \
        --wait \
        --timeout 15m
fi
log_success "Redpanda deployed"

# Step 8: Apply network policies
log_info "Applying network policies..."
run_cmd kubectl apply -f "${SCRIPT_DIR}/network-policies.yaml"
log_success "Network policies applied"

# Step 9: Apply services
log_info "Applying services..."
run_cmd kubectl apply -f "${SCRIPT_DIR}/services.yaml"
log_success "Services applied"

# Step 10: Create Kafka topics
log_info "Creating Kafka topics..."
run_cmd kubectl exec -n "$NAMESPACE" "${HELM_RELEASE}-0" -- rpk topic create \
    argus.codebase.ingested \
    argus.codebase.analyzed \
    argus.test.created \
    argus.test.executed \
    argus.test.failed \
    argus.healing.requested \
    argus.healing.completed \
    argus.dlq \
    --partitions 3 \
    --replication-factor 1 \
    -X user=admin \
    -X pass="$(kubectl get secret redpanda-superusers -n $NAMESPACE -o jsonpath='{.data.users\.txt}' | base64 -d | grep admin | cut -d: -f2)" \
    -X sasl.mechanism=SCRAM-SHA-512 || log_warn "Some topics may already exist"
log_success "Kafka topics created"

# Step 11: Wait for storage pods to be ready
log_info "Waiting for FalkorDB and Valkey pods to be ready..."
run_cmd kubectl wait --for=condition=Ready pod/falkordb-0 -n "$NAMESPACE" --timeout=300s
run_cmd kubectl wait --for=condition=Ready pod/valkey-0 -n "$NAMESPACE" --timeout=300s
log_success "Storage pods ready"

# Step 12: Deploy Cognee worker
log_info "Deploying Cognee worker..."
run_cmd kubectl apply -f "${SCRIPT_DIR}/cognee-worker.yaml"
log_success "Cognee worker deployed"

# Final verification
log_info "Verifying deployment..."
echo ""
echo "=== Pod Status ==="
kubectl get pods -n "$NAMESPACE" -o wide
echo ""
echo "=== Service Status ==="
kubectl get svc -n "$NAMESPACE"
echo ""
echo "=== PVC Status ==="
kubectl get pvc -n "$NAMESPACE"
echo ""

log_success "Deployment complete!"
echo ""
echo "Next steps:"
echo "  1. Update database-url, supabase-url, supabase-service-key in secrets if using BYOK"
echo "  2. Monitor logs: kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=cognee-worker -f"
echo "  3. Check Redpanda health: kubectl exec -n $NAMESPACE ${HELM_RELEASE}-0 -- rpk cluster health"
echo ""
