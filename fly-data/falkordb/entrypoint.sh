#!/bin/sh
# Use the FalkorDB image's run.sh which handles module loading.
# Pass extra Redis args via REDIS_ARGS env var.
export REDIS_ARGS="--requirepass $FALKORDB_PASSWORD --maxmemory 1536mb --maxmemory-policy noeviction --appendonly yes --appendfsync everysec"
exec /FalkorDB/build/docker/run.sh
