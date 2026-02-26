#!/usr/bin/env bash
# Deploy Argus data-layer services to Fly.io.
#
# Deploys: FalkorDB, Valkey, Cognee Worker, Stream Processor
#
# Prerequisites:
#   - fly CLI installed and authenticated (`fly auth login`)
#   - Org "skopaq" exists (`fly orgs list`)
#   - Secrets pre-set for each app (see comments below)
#
# Usage:
#   ./deploy.sh                    # Deploy all services
#   ./deploy.sh --only falkordb    # Deploy single service
#   ./deploy.sh --only valkey
#   ./deploy.sh --only cognee
#   ./deploy.sh --only stream
#   ./deploy.sh --skip-databases   # Deploy workers only (assumes DBs running)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLY="${FLYCTL:-fly}"

# App names
FALKORDB_APP="skopaq-falkordb"
VALKEY_APP="skopaq-valkey"
COGNEE_APP="skopaq-cognee-worker"
STREAM_APP="skopaq-stream-processor"

# Parse args
ONLY=""
SKIP_DBS=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip-databases) SKIP_DBS=true; shift ;;
    *) echo "Unknown arg: $1"; echo "Usage: $0 [--only falkordb|valkey|cognee|stream] [--skip-databases]"; exit 1 ;;
  esac
done

log()  { echo -e "\033[1;34m==>\033[0m $1"; }
ok()   { echo -e "\033[1;32m ✓\033[0m  $1"; }
err()  { echo -e "\033[1;31mERR\033[0m $1" >&2; }
warn() { echo -e "\033[1;33mWRN\033[0m $1"; }

should_deploy() {
  [[ -z "$ONLY" ]] || [[ "$ONLY" == "$1" ]]
}

ensure_app() {
  local app="$1"
  if ! $FLY apps list --org skopaq 2>/dev/null | grep -q "$app"; then
    log "Creating app $app..."
    $FLY apps create "$app" --org skopaq
  fi
}

ensure_volume() {
  local app="$1" vol_name="$2" size="$3"
  if ! $FLY volumes list --app "$app" 2>/dev/null | grep -q "$vol_name"; then
    log "Creating volume $vol_name (${size}GB) for $app..."
    $FLY volumes create "$vol_name" --size "$size" --region iad --app "$app" --yes
  else
    ok "Volume $vol_name already exists for $app"
  fi
}

wait_healthy_tcp() {
  local app="$1" port="$2" timeout="${3:-60}"
  log "Waiting for $app to become healthy (TCP :$port)..."
  local ip
  ip=$($FLY ips list --app "$app" --json 2>/dev/null | python3 -c "
import sys, json
ips = json.load(sys.stdin)
for ip in ips:
    if ip.get('type') == 'v4':
        print(ip['address'])
        break
" 2>/dev/null || echo "")

  if [[ -z "$ip" ]]; then
    warn "No IPv4 allocated for $app — skipping TCP health check"
    return 0
  fi

  for i in $(seq 1 $((timeout / 2))); do
    if redis-cli -h "$ip" -p "$port" PING 2>/dev/null | grep -q PONG; then
      ok "$app is healthy at $ip:$port"
      return 0
    fi
    sleep 2
  done
  warn "$app health check timed out (may still be starting)"
}

wait_healthy_http() {
  local url="$1" timeout="${2:-60}"
  log "Waiting for $url to become healthy..."
  for i in $(seq 1 $((timeout / 2))); do
    if curl -sf "$url" >/dev/null 2>&1; then
      ok "Healthy: $url"
      return 0
    fi
    sleep 2
  done
  warn "Health check timed out: $url (may still be starting)"
}

# ==========================================================================
# Phase 1a: FalkorDB
# ==========================================================================
if should_deploy "falkordb" && [[ "$SKIP_DBS" == "false" ]]; then
  log "Deploying FalkorDB ($FALKORDB_APP)..."
  ensure_app "$FALKORDB_APP"
  ensure_volume "$FALKORDB_APP" "falkordb_data" 10

  # Secrets reminder
  if ! $FLY secrets list --app "$FALKORDB_APP" 2>/dev/null | grep -q "FALKORDB_PASSWORD"; then
    warn "FALKORDB_PASSWORD secret not set! Run:"
    warn "  fly secrets set FALKORDB_PASSWORD=<64-char-password> --app $FALKORDB_APP"
  fi

  (cd "$SCRIPT_DIR/falkordb" && $FLY deploy --app "$FALKORDB_APP" --yes)

  # Ensure single instance (graph DB is stateful)
  $FLY scale count 1 --app "$FALKORDB_APP" --yes 2>/dev/null || true

  # Allocate dedicated IPv4 if not already
  if ! $FLY ips list --app "$FALKORDB_APP" --json 2>/dev/null | python3 -c "
import sys, json
ips = json.load(sys.stdin)
sys.exit(0 if any(ip.get('type') == 'v4' for ip in ips) else 1)
" 2>/dev/null; then
    log "Allocating dedicated IPv4 for $FALKORDB_APP..."
    $FLY ips allocate-v4 --app "$FALKORDB_APP" --yes
  fi

  ok "FalkorDB deployed"
fi

# ==========================================================================
# Phase 1b: Valkey
# ==========================================================================
if should_deploy "valkey" && [[ "$SKIP_DBS" == "false" ]]; then
  log "Deploying Valkey ($VALKEY_APP)..."
  ensure_app "$VALKEY_APP"
  ensure_volume "$VALKEY_APP" "valkey_data" 5

  if ! $FLY secrets list --app "$VALKEY_APP" 2>/dev/null | grep -q "VALKEY_PASSWORD"; then
    warn "VALKEY_PASSWORD secret not set! Run:"
    warn "  fly secrets set VALKEY_PASSWORD=<password> --app $VALKEY_APP"
  fi

  (cd "$SCRIPT_DIR/valkey" && $FLY deploy --app "$VALKEY_APP" --yes)
  $FLY scale count 1 --app "$VALKEY_APP" --yes 2>/dev/null || true

  if ! $FLY ips list --app "$VALKEY_APP" --json 2>/dev/null | python3 -c "
import sys, json
ips = json.load(sys.stdin)
sys.exit(0 if any(ip.get('type') == 'v4' for ip in ips) else 1)
" 2>/dev/null; then
    log "Allocating dedicated IPv4 for $VALKEY_APP..."
    $FLY ips allocate-v4 --app "$VALKEY_APP" --yes
  fi

  ok "Valkey deployed"
fi

# Wait for databases before deploying workers
if [[ "$SKIP_DBS" == "false" ]] && [[ -z "$ONLY" || "$ONLY" == "falkordb" || "$ONLY" == "valkey" ]]; then
  if should_deploy "falkordb"; then wait_healthy_tcp "$FALKORDB_APP" 6379 30; fi
  if should_deploy "valkey"; then wait_healthy_tcp "$VALKEY_APP" 6379 30; fi
fi

# ==========================================================================
# Phase 2: Cognee Worker
# ==========================================================================
if should_deploy "cognee"; then
  log "Deploying Cognee Worker ($COGNEE_APP)..."
  ensure_app "$COGNEE_APP"

  # Check required secrets
  MISSING_SECRETS=""
  for secret in KAFKA_BOOTSTRAP_SERVERS KAFKA_SASL_USERNAME KAFKA_SASL_PASSWORD \
                LLM_API_KEY EMBEDDING_API_KEY SUPABASE_URL SUPABASE_SERVICE_KEY \
                FALKORDB_PASSWORD GRAPH_DATABASE_PASSWORD; do
    if ! $FLY secrets list --app "$COGNEE_APP" 2>/dev/null | grep -q "$secret"; then
      MISSING_SECRETS="$MISSING_SECRETS $secret"
    fi
  done
  if [[ -n "$MISSING_SECRETS" ]]; then
    warn "Missing secrets for $COGNEE_APP:$MISSING_SECRETS"
    warn "Set them with: fly secrets set KEY=VALUE --app $COGNEE_APP"
  fi

  (cd "$SCRIPT_DIR/cognee-worker" && $FLY deploy --app "$COGNEE_APP" --yes)
  ok "Cognee Worker deployed"

  wait_healthy_http "https://${COGNEE_APP}.fly.dev/health" 90
fi

# ==========================================================================
# Phase 3: Stream Processor
# ==========================================================================
if should_deploy "stream"; then
  log "Deploying Stream Processor ($STREAM_APP)..."
  ensure_app "$STREAM_APP"

  MISSING_SECRETS=""
  for secret in KAFKA_BOOTSTRAP_SERVERS KAFKA_SASL_USERNAME KAFKA_SASL_PASSWORD \
                SUPABASE_URL SUPABASE_SERVICE_KEY; do
    if ! $FLY secrets list --app "$STREAM_APP" 2>/dev/null | grep -q "$secret"; then
      MISSING_SECRETS="$MISSING_SECRETS $secret"
    fi
  done
  if [[ -n "$MISSING_SECRETS" ]]; then
    warn "Missing secrets for $STREAM_APP:$MISSING_SECRETS"
    warn "Set them with: fly secrets set KEY=VALUE --app $STREAM_APP"
  fi

  (cd "$SCRIPT_DIR/stream-processor" && $FLY deploy --app "$STREAM_APP" --yes)
  ok "Stream Processor deployed"

  wait_healthy_http "https://${STREAM_APP}.fly.dev/health" 60
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
log "Data layer deployment complete"
echo ""

# Print IPs
for app in "$FALKORDB_APP" "$VALKEY_APP"; do
  IP=$($FLY ips list --app "$app" --json 2>/dev/null | python3 -c "
import sys, json
ips = json.load(sys.stdin)
for ip in ips:
    if ip.get('type') == 'v4':
        print(ip['address'])
        break
" 2>/dev/null || echo "N/A")
  echo "  $app IPv4: $IP"
done
echo ""
echo "  Cognee Worker:    https://${COGNEE_APP}.fly.dev/health"
echo "  Stream Processor: https://${STREAM_APP}.fly.dev/health"
echo ""
echo "  Railway env vars to update (after verifying health):"

FALKORDB_IP=$($FLY ips list --app "$FALKORDB_APP" --json 2>/dev/null | python3 -c "
import sys, json
ips = json.load(sys.stdin)
for ip in ips:
    if ip.get('type') == 'v4':
        print(ip['address'])
        break
" 2>/dev/null || echo "<falkordb-fly-ipv4>")

VALKEY_IP=$($FLY ips list --app "$VALKEY_APP" --json 2>/dev/null | python3 -c "
import sys, json
ips = json.load(sys.stdin)
for ip in ips:
    if ip.get('type') == 'v4':
        print(ip['address'])
        break
" 2>/dev/null || echo "<valkey-fly-ipv4>")

echo "    FALKORDB_HOST=$FALKORDB_IP"
echo "    GRAPH_DATABASE_URL=$FALKORDB_IP"
echo "    VALKEY_URL=redis://:PASSWORD@${VALKEY_IP}:6379/0"
echo ""
echo "  Manage:"
echo "    fly logs --app <app-name>"
echo "    fly status --app <app-name>"
echo "    fly scale count N --app <app-name>"
