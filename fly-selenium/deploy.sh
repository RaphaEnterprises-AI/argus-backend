#!/usr/bin/env bash
# Deploy Selenium Grid (hub + chrome nodes) to Fly.io.
# Usage: ./deploy.sh [--chrome-count N]
#
# Prerequisites:
#   - fly CLI installed and authenticated (`fly auth login`)
#   - Org "skopaq" exists (`fly orgs list`)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUB_APP="skopaq-selenium-hub"
CHROME_APP="skopaq-selenium-chrome"
HUB_URL="https://${HUB_APP}.fly.dev"
CHROME_COUNT=2

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chrome-count) CHROME_COUNT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

log() { echo -e "\033[1;34m==>\033[0m $1"; }
err() { echo -e "\033[1;31mERR\033[0m $1" >&2; }

# ---------- Hub ----------
log "Deploying hub ($HUB_APP)..."

# Create app if it doesn't exist yet
if ! fly apps list --org skopaq 2>/dev/null | grep -q "$HUB_APP"; then
  fly apps create "$HUB_APP" --org skopaq
fi

fly deploy --app "$HUB_APP" --config "$SCRIPT_DIR/hub/fly.toml" --yes

# Hub is a singleton — Selenium Grid is stateful.
# Fly auto-creates 2 machines; scale back to 1.
fly scale count 1 --app "$HUB_APP" --yes 2>/dev/null || true

log "Waiting for hub to become healthy..."
for i in $(seq 1 30); do
  if curl -sf "${HUB_URL}/status" >/dev/null 2>&1; then
    log "Hub is healthy."
    break
  fi
  if [[ $i -eq 30 ]]; then
    err "Hub did not become healthy within 60 seconds."
    fly logs --app "$HUB_APP" | tail -20
    exit 1
  fi
  sleep 2
done

# ---------- Chrome Nodes ----------
log "Deploying chrome nodes ($CHROME_APP, count=$CHROME_COUNT)..."

if ! fly apps list --org skopaq 2>/dev/null | grep -q "$CHROME_APP"; then
  fly apps create "$CHROME_APP" --org skopaq
fi

fly deploy --app "$CHROME_APP" --config "$SCRIPT_DIR/chrome/fly.toml" --yes
fly scale count "$CHROME_COUNT" --app "$CHROME_APP" --yes

log "Waiting for nodes to register with hub..."
for i in $(seq 1 30); do
  NODE_COUNT=$(curl -sf "${HUB_URL}/status" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('value',{}).get('nodes',[])))" 2>/dev/null || echo 0)
  if [[ "$NODE_COUNT" -ge "$CHROME_COUNT" ]]; then
    log "$NODE_COUNT node(s) registered."
    break
  fi
  if [[ $i -eq 30 ]]; then
    err "Only $NODE_COUNT/$CHROME_COUNT nodes registered after 60 seconds."
    err "Nodes may still be starting — check: fly logs --app $CHROME_APP"
    break
  fi
  sleep 2
done

# ---------- Summary ----------
echo ""
log "Selenium Grid deployed on Fly.io"
echo "  Hub URL:     $HUB_URL"
echo "  Hub status:  ${HUB_URL}/status"
echo "  Chrome nodes: $CHROME_COUNT"
echo ""
echo "  Set in Railway:"
echo "    SELENIUM_GRID_URL=${HUB_URL}"
echo ""
echo "  Scale nodes:  fly scale count N --app $CHROME_APP"
echo "  Hub logs:     fly logs --app $HUB_APP"
echo "  Node logs:    fly logs --app $CHROME_APP"
