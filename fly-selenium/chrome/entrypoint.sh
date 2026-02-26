#!/bin/bash
# Set the Selenium node host to the Fly.io WireGuard mesh address
# so the hub can reach back to this node for health checks.
if [ -n "$FLY_PRIVATE_IP" ]; then
  export SE_NODE_HOST="$FLY_PRIVATE_IP"
  echo "Set SE_NODE_HOST=$FLY_PRIVATE_IP (Fly WireGuard mesh)"
fi

# Delegate to the original Selenium entrypoint
exec /opt/bin/entry_point.sh
