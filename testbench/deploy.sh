#!/usr/bin/env bash
# Deploy Argus Testbench services to Railway.
# Usage: ./deploy.sh [OPTIONS]
#
# Prerequisites:
#   - Railway CLI installed and authenticated (`railway login`)
#   - Project "argus-testbench" exists on Railway

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RAILWAY_PROJECT="${RAILWAY_PROJECT:-argus-testbench}"

# Service definitions: name:directory:health_path
# Note: chaos-ctrl-testbench is last because it depends on other services.
SERVICES=(
    "jsonapi-testbench:json-api:/health"
    "chaos-testbench:chaos-app:/health"
    "conduit-testbench:conduit:/health"
    "plane-testbench:plane:/health"
    "chaos-ctrl-testbench:chaos-controller:/health"
)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy Argus Testbench services to Railway"
    echo ""
    echo "Options:"
    echo "  --only SERVICE     Deploy only a specific service"
    echo "  --skip SERVICE     Skip a specific service"
    echo "  --no-wait          Don't wait for health checks"
    echo "  --status           Just check health of all services"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Services: json-api, chaos-app, conduit, plane, chaos-controller"
}

# Parse args
ONLY=""
SKIP=""
NO_WAIT=false
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --only) ONLY="$2"; shift 2 ;;
        --skip) SKIP="$2"; shift 2 ;;
        --no-wait) NO_WAIT=true; shift ;;
        --status) STATUS_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

check_health() {
    local name=$1
    local health_url="https://${name}-production.up.railway.app/health"

    echo -n "  Checking ${name}... "
    local response
    response=$(curl -sf --max-time 10 "$health_url" 2>/dev/null) && {
        echo -e "${GREEN}healthy${NC}"
        return 0
    } || {
        echo -e "${RED}unhealthy${NC}"
        return 1
    }
}

wait_for_health() {
    local name=$1
    local max_attempts=30
    local attempt=0

    echo -n "  Waiting for ${name}..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf --max-time 5 "https://${name}-production.up.railway.app/health" >/dev/null 2>&1; then
            echo -e " ${GREEN}ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 10
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}timeout${NC}"
    return 1
}

deploy_service() {
    local name=$1
    local dir=$2
    local health_path=$3

    echo -e "${BLUE}Deploying ${name}...${NC}"

    # Check directory exists
    local service_dir="${SCRIPT_DIR}/${dir}"
    if [ ! -d "$service_dir" ]; then
        echo -e "  ${RED}Directory not found: ${service_dir}${NC}"
        return 1
    fi

    # Link to Railway project + service
    (cd "$service_dir" && railway link -p "$RAILWAY_PROJECT" -s "$name" 2>/dev/null || true)

    # Deploy
    (cd "$service_dir" && railway up --detach)

    echo -e "  ${GREEN}Deploy initiated${NC}"

    # Wait for health if not skipped
    if [ "$NO_WAIT" = false ]; then
        wait_for_health "$name"
    fi
}

# Status check mode
if [ "$STATUS_ONLY" = true ]; then
    echo -e "${BLUE}=== Testbench Service Status ===${NC}"
    echo ""
    healthy=0
    total=0
    for svc in "${SERVICES[@]}"; do
        IFS=':' read -r name dir health_path <<< "$svc"
        total=$((total + 1))
        if check_health "$name"; then
            healthy=$((healthy + 1))
        fi
    done
    echo ""
    echo -e "${healthy}/${total} services healthy"
    exit 0
fi

# Deploy mode
echo -e "${BLUE}=== Deploying Argus Testbench ===${NC}"
echo -e "Railway project: ${RAILWAY_PROJECT}"
echo ""

deployed=0
failed=0

for svc in "${SERVICES[@]}"; do
    IFS=':' read -r name dir health_path <<< "$svc"

    # Filter by --only / --skip
    if [ -n "$ONLY" ] && [ "$dir" != "$ONLY" ]; then
        continue
    fi
    if [ -n "$SKIP" ] && [ "$dir" = "$SKIP" ]; then
        echo -e "${YELLOW}Skipping ${name}${NC}"
        continue
    fi

    if deploy_service "$name" "$dir" "$health_path"; then
        deployed=$((deployed + 1))
    else
        failed=$((failed + 1))
    fi
    echo ""
done

# Summary
echo -e "${BLUE}=== Deployment Summary ===${NC}"
echo -e "  Deployed: ${GREEN}${deployed}${NC}"
if [ $failed -gt 0 ]; then
    echo -e "  Failed:   ${RED}${failed}${NC}"
fi
echo ""
echo "Service URLs:"
echo "  JSON API:          https://jsonapi-testbench-production.up.railway.app"
echo "  Chaos App:         https://chaos-testbench-production.up.railway.app"
echo "  Conduit:           https://conduit-testbench-production.up.railway.app"
echo "  Plane:             https://plane-testbench-production.up.railway.app"
echo "  Chaos Controller:  https://chaos-ctrl-testbench-production.up.railway.app"
