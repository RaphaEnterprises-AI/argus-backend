#!/bin/sh
exec valkey-server \
  --requirepass "$VALKEY_PASSWORD" \
  --maxmemory 384mb \
  --maxmemory-policy allkeys-lru \
  --appendonly yes \
  --appendfsync everysec \
  --save "900 1" \
  --save "300 10"
