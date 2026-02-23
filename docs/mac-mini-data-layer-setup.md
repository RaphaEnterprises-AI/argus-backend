# Mac Mini Data Layer Setup — Complete Guide

> **Instructions for Claude Desktop**: Run these commands step-by-step on this Mac Mini.
> This sets up the Skopaq/Argus stateful data layer (Redpanda, FalkorDB, Valkey, Flink, Prometheus, Grafana) via Docker Compose with Cloudflare Tunnel for external access from Railway.

---

## Prerequisites Check

Run these first to verify the Mac Mini is ready:

```bash
# Check macOS version and architecture
uname -m  # Should show arm64 for M-series
sw_vers

# Check available disk space (need ~10GB free)
df -h /

# Check available RAM (need ~8GB free for all services)
sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'
```

---

## Step 1: Install Docker Desktop

```bash
# Check if Docker is already installed
docker --version 2>/dev/null && echo "Docker is installed" || echo "Need to install Docker"

# If not installed, download and install Docker Desktop for Apple Silicon:
# Option A: Via Homebrew (recommended)
brew install --cask docker

# Option B: Manual download
# Download from https://desktop.docker.com/mac/main/arm64/Docker.dmg
# Open the .dmg and drag Docker to Applications
# Launch Docker from Applications

# After installation, start Docker Desktop and wait for it to be ready
open -a Docker

# Verify Docker is running (wait ~30 seconds after launching)
docker info
```

**Docker Desktop Settings** (open Docker Desktop → Settings):
- Resources → Memory: Set to **8 GB** minimum (10 GB recommended)
- Resources → CPUs: Set to **4** minimum
- Resources → Disk: Set to **30 GB** minimum

---

## Step 2: Install Cloudflare Tunnel (cloudflared)

```bash
# Install cloudflared via Homebrew
brew install cloudflare/cloudflare/cloudflared

# Verify installation
cloudflared --version

# Login to Cloudflare (opens browser)
cloudflared login
# Select the domain you want to use (e.g., skopaq.ai)
# This saves credentials to ~/.cloudflared/cert.pem
```

---

## Step 3: Create Project Directory

```bash
# Create the project directory
mkdir -p ~/skopaq-data-layer
cd ~/skopaq-data-layer

# Create subdirectories for persistent data and configs
mkdir -p data/redpanda data/falkordb data/valkey data/flink/checkpoints data/prometheus data/grafana
mkdir -p config/prometheus config/grafana/provisioning/datasources config/grafana/provisioning/dashboards
```

---

## Step 4: Create Docker Compose Configuration

Create the file `~/skopaq-data-layer/docker-compose.yml`:

```yaml
version: "3.8"

services:
  # ============================================================
  # Redpanda (Kafka-compatible message broker)
  # ============================================================
  redpanda:
    image: docker.redpanda.com/redpandadata/redpanda:v24.3.1
    container_name: skopaq-redpanda
    command:
      - redpanda start
      - --smp 2
      - --memory 1G
      - --reserve-memory 0M
      - --overprovisioned
      - --node-id 0
      - --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      - --advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
      - --pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082
      - --advertise-pandaproxy-addr internal://redpanda:8082,external://localhost:18082
      - --schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081
      - --advertise-schema-registry-addr internal://redpanda:8081,external://localhost:18081
      - --set redpanda.enable_idempotence=true
      - --set redpanda.default_topic_replications=1
    ports:
      - "19092:19092"   # Kafka external
      - "18082:18082"   # Pandaproxy
      - "18081:18081"   # Schema Registry
      - "9644:9644"     # Admin API
    volumes:
      - ./data/redpanda:/var/lib/redpanda/data
    healthcheck:
      test: ["CMD", "rpk", "cluster", "health"]
      interval: 15s
      timeout: 10s
      retries: 5
    restart: unless-stopped

  # Create Kafka topics on startup
  redpanda-init:
    image: docker.redpanda.com/redpandadata/redpanda:v24.3.1
    container_name: skopaq-redpanda-init
    depends_on:
      redpanda:
        condition: service_healthy
    entrypoint: /bin/bash
    command: |
      -c "
      echo 'Creating Argus Kafka topics...'
      rpk topic create argus.codebase.ingested --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.codebase.analyzed --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.test.created --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.test.executed --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.test.failed --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.healing.requested --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.healing.completed --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.dlq --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.agent.request --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.agent.response --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.agent.broadcast --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      rpk topic create argus.agent.heartbeat --brokers redpanda:9092 --partitions 6 --config compression.type=zstd
      echo 'All topics created.'
      rpk topic list --brokers redpanda:9092
      "

  # ============================================================
  # Redpanda Console (Web UI for Kafka)
  # ============================================================
  redpanda-console:
    image: docker.redpanda.com/redpandadata/console:v2.8.0
    container_name: skopaq-redpanda-console
    depends_on:
      redpanda:
        condition: service_healthy
    environment:
      CONFIG_FILEPATH: ""
      KAFKA_BROKERS: redpanda:9092
      KAFKA_SCHEMAREGISTRY_ENABLED: "true"
      KAFKA_SCHEMAREGISTRY_URLS: http://redpanda:8081
    ports:
      - "8080:8080"
    restart: unless-stopped

  # ============================================================
  # FalkorDB (Graph database — Redis-compatible with graph queries)
  # ============================================================
  falkordb:
    image: falkordb/falkordb:latest
    container_name: skopaq-falkordb
    command: >
      --requirepass CHANGE_ME_FALKORDB_PASSWORD_64_CHARS_HERE____________
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save 60 1000
      --save 300 100
    ports:
      - "6379:6379"
    volumes:
      - ./data/falkordb:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "CHANGE_ME_FALKORDB_PASSWORD_64_CHARS_HERE____________", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # ============================================================
  # Valkey (Redis-compatible cache)
  # ============================================================
  valkey:
    image: valkey/valkey:8-alpine
    container_name: skopaq-valkey
    command: >
      valkey-server
      --requirepass CHANGE_ME_VALKEY_PASSWORD
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --save 60 1000
    ports:
      - "6380:6379"
    volumes:
      - ./data/valkey:/data
    healthcheck:
      test: ["CMD", "valkey-cli", "-a", "CHANGE_ME_VALKEY_PASSWORD", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # ============================================================
  # Apache Flink (Stream processing)
  # ============================================================
  flink-jobmanager:
    image: flink:1.20-java17
    container_name: skopaq-flink-jobmanager
    command: jobmanager
    ports:
      - "8081:8081"  # Flink Web UI
    environment:
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: flink-jobmanager
        jobmanager.memory.process.size: 1024m
        state.checkpoints.dir: file:///opt/flink/checkpoints
    volumes:
      - ./data/flink/checkpoints:/opt/flink/checkpoints
    restart: unless-stopped

  flink-taskmanager:
    image: flink:1.20-java17
    container_name: skopaq-flink-taskmanager
    command: taskmanager
    depends_on:
      - flink-jobmanager
    environment:
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: flink-jobmanager
        taskmanager.numberOfTaskSlots: 4
        taskmanager.memory.process.size: 1536m
        state.checkpoints.dir: file:///opt/flink/checkpoints
    volumes:
      - ./data/flink/checkpoints:/opt/flink/checkpoints
    restart: unless-stopped

  # ============================================================
  # Prometheus (Metrics collection)
  # ============================================================
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: skopaq-prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --storage.tsdb.retention.time=30d
      - --web.enable-lifecycle
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./data/prometheus:/prometheus
    restart: unless-stopped

  # ============================================================
  # Grafana (Dashboards)
  # ============================================================
  grafana:
    image: grafana/grafana:11.4.0
    container_name: skopaq-grafana
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: CHANGE_ME_GRAFANA_PASSWORD
      GF_USERS_ALLOW_SIGN_UP: "false"
    ports:
      - "3001:3000"
    volumes:
      - ./data/grafana:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
    restart: unless-stopped
```

---

## Step 5: Create Prometheus Configuration

Create `~/skopaq-data-layer/config/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "redpanda"
    static_configs:
      - targets: ["redpanda:9644"]
    metrics_path: /public_metrics

  - job_name: "flink"
    static_configs:
      - targets: ["flink-jobmanager:8081"]
    metrics_path: /metrics
```

---

## Step 6: Create Grafana Datasource Config

Create `~/skopaq-data-layer/config/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

---

## Step 7: Set Passwords

**IMPORTANT: Replace all placeholder passwords before starting.**

```bash
cd ~/skopaq-data-layer

# Generate a 64-character FalkorDB password
FALKORDB_PASS=$(openssl rand -hex 32)
echo "FalkorDB password: $FALKORDB_PASS"

# Generate Valkey password
VALKEY_PASS=$(openssl rand -hex 16)
echo "Valkey password: $VALKEY_PASS"

# Generate Grafana password
GRAFANA_PASS=$(openssl rand -base64 16)
echo "Grafana password: $GRAFANA_PASS"

# Replace passwords in docker-compose.yml
sed -i '' "s/CHANGE_ME_FALKORDB_PASSWORD_64_CHARS_HERE____________/$FALKORDB_PASS/g" docker-compose.yml
sed -i '' "s/CHANGE_ME_VALKEY_PASSWORD/$VALKEY_PASS/g" docker-compose.yml
sed -i '' "s/CHANGE_ME_GRAFANA_PASSWORD/$GRAFANA_PASS/g" docker-compose.yml

# Save passwords to a local env file (DO NOT commit this)
cat > .env.passwords <<EOF
FALKORDB_PASSWORD=$FALKORDB_PASS
VALKEY_PASSWORD=$VALKEY_PASS
GRAFANA_PASSWORD=$GRAFANA_PASS
EOF

echo ""
echo "=== SAVE THESE PASSWORDS ==="
cat .env.passwords
echo "==========================="
```

---

## Step 8: Start All Services

```bash
cd ~/skopaq-data-layer

# Pull all images first
docker compose pull

# Start everything
docker compose up -d

# Watch logs during startup (Ctrl+C to stop watching)
docker compose logs -f --tail=20
```

---

## Step 9: Verify All Services Are Running

```bash
cd ~/skopaq-data-layer

# Check all containers are healthy
docker compose ps

# Test Redpanda
docker compose exec redpanda rpk cluster health
docker compose exec redpanda rpk topic list

# Test FalkorDB
source .env.passwords
docker compose exec falkordb redis-cli -a "$FALKORDB_PASSWORD" PING
docker compose exec falkordb redis-cli -a "$FALKORDB_PASSWORD" GRAPH.LIST

# Test Valkey
docker compose exec valkey valkey-cli -a "$VALKEY_PASSWORD" PING

# Test Flink
curl -s http://localhost:8081/overview | python3 -m json.tool

# Test Prometheus
curl -s http://localhost:9090/-/healthy

# Test Grafana
curl -s http://localhost:3001/api/health

# Test Redpanda Console
curl -s http://localhost:8080/api/health
```

Expected output: All containers UP, all health checks passing.

---

## Step 10: Set Up Cloudflare Tunnel

This creates a secure tunnel so Railway can reach the Mac Mini services.

```bash
# Create a tunnel (pick a meaningful name)
cloudflared tunnel create skopaq-data-layer

# Note the Tunnel ID from the output — you'll need it
# It looks like: a1b2c3d4-e5f6-7890-abcd-ef1234567890

# List tunnels to confirm
cloudflared tunnel list
```

Create the tunnel config at `~/skopaq-data-layer/cloudflare-tunnel.yml`:

```yaml
# Replace TUNNEL_ID with your actual tunnel ID from the step above
tunnel: TUNNEL_ID
credentials-file: /Users/YOUR_USERNAME/.cloudflared/TUNNEL_ID.json

ingress:
  # Redpanda Kafka (TCP)
  - hostname: kafka.skopaq.ai
    service: tcp://localhost:19092

  # FalkorDB (TCP)
  - hostname: falkordb.skopaq.ai
    service: tcp://localhost:6379

  # Valkey (TCP)
  - hostname: valkey.skopaq.ai
    service: tcp://localhost:6380

  # Redpanda Console (HTTP)
  - hostname: redpanda-console.skopaq.ai
    service: http://localhost:8080

  # Flink UI (HTTP)
  - hostname: flink.skopaq.ai
    service: http://localhost:8081

  # Prometheus (HTTP)
  - hostname: prometheus.skopaq.ai
    service: http://localhost:9090

  # Grafana (HTTP)
  - hostname: grafana.skopaq.ai
    service: http://localhost:3001

  # Catch-all (required)
  - service: http_status:404
```

**IMPORTANT**: Replace `TUNNEL_ID` and `YOUR_USERNAME` in the file above.

```bash
# Create DNS records for each service
# Replace TUNNEL_ID with your actual tunnel ID

cloudflared tunnel route dns TUNNEL_ID kafka.skopaq.ai
cloudflared tunnel route dns TUNNEL_ID falkordb.skopaq.ai
cloudflared tunnel route dns TUNNEL_ID valkey.skopaq.ai
cloudflared tunnel route dns TUNNEL_ID redpanda-console.skopaq.ai
cloudflared tunnel route dns TUNNEL_ID flink.skopaq.ai
cloudflared tunnel route dns TUNNEL_ID prometheus.skopaq.ai
cloudflared tunnel route dns TUNNEL_ID grafana.skopaq.ai
```

### Start the tunnel

```bash
# Test the tunnel first (foreground, Ctrl+C to stop)
cloudflared tunnel --config ~/skopaq-data-layer/cloudflare-tunnel.yml run

# If it works, install as a macOS service (runs on boot)
sudo cloudflared service install --config ~/skopaq-data-layer/cloudflare-tunnel.yml
sudo launchctl start com.cloudflare.cloudflared
```

### Important: TCP services need cloudflared on the client side too

For TCP services (Kafka, FalkorDB, Valkey), Railway can't connect directly via hostname. Instead, use **Cloudflare Access + `cloudflared access`** or use a different approach:

**Recommended alternative for TCP: Use Tailscale instead**

```bash
# Install Tailscale on Mac Mini
brew install tailscale

# Start Tailscale
sudo tailscaled &
tailscale up

# Note the Tailscale IP (e.g., 100.x.y.z)
tailscale ip -4
```

Then on the Railway side, install Tailscale in the Docker container and connect to the same Tailnet. The Mac Mini services will be reachable at `100.x.y.z:PORT`.

**Simpler alternative: Use Cloudflare Tunnel for HTTP services only, and use direct IP + port forwarding for TCP services.**

---

## Step 11: Configure Automatic Startup on Boot

```bash
# Docker Desktop auto-starts on login (enable in Docker Desktop Settings → General → "Start Docker Desktop when you sign in")

# Create a launchd plist for docker compose
cat > ~/Library/LaunchAgents/com.skopaq.data-layer.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.skopaq.data-layer</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/docker</string>
        <string>compose</string>
        <string>-f</string>
        <string>/Users/YOUR_USERNAME/skopaq-data-layer/docker-compose.yml</string>
        <string>up</string>
        <string>-d</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StartInterval</key>
    <integer>60</integer>
    <key>StandardOutPath</key>
    <string>/tmp/skopaq-data-layer.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/skopaq-data-layer.err</string>
</dict>
</plist>
EOF

# Replace YOUR_USERNAME
sed -i '' "s/YOUR_USERNAME/$(whoami)/g" ~/Library/LaunchAgents/com.skopaq.data-layer.plist

# Load the service
launchctl load ~/Library/LaunchAgents/com.skopaq.data-layer.plist
```

---

## Step 12: Update Railway Environment Variables

Once everything is running and the tunnel is active, update these env vars in Railway for the `argus-brain` service:

**If using Tailscale** (recommended for TCP):
```
REDPANDA_BROKERS=100.x.y.z:19092          # Tailscale IP
FALKORDB_HOST=100.x.y.z                    # Tailscale IP
FALKORDB_PORT=6379
FALKORDB_PASSWORD=<from .env.passwords>
GRAPH_DATABASE_PROVIDER=falkor
GRAPH_DATABASE_URL=100.x.y.z               # Tailscale IP
GRAPH_DATABASE_PORT=6379
GRAPH_DATABASE_PASSWORD=<from .env.passwords>
VALKEY_URL=redis://:PASSWORD@100.x.y.z:6380
```

**If using port forwarding** (simpler but less secure):
```
REDPANDA_BROKERS=YOUR_PUBLIC_IP:19092
FALKORDB_HOST=YOUR_PUBLIC_IP
FALKORDB_PORT=6379
# ... same pattern
```

---

## Useful Management Commands

```bash
cd ~/skopaq-data-layer

# Check status of all services
docker compose ps

# View logs for a specific service
docker compose logs -f redpanda
docker compose logs -f falkordb
docker compose logs -f valkey

# Restart a specific service
docker compose restart falkordb

# Stop everything
docker compose down

# Stop and remove all data (DESTRUCTIVE)
docker compose down -v

# Update images
docker compose pull && docker compose up -d

# Check resource usage
docker stats --no-stream

# Redpanda topic management
docker compose exec redpanda rpk topic list
docker compose exec redpanda rpk topic describe argus.healing.requested
docker compose exec redpanda rpk group list
docker compose exec redpanda rpk group describe argus-cognee-workers

# FalkorDB graph queries
source .env.passwords
docker compose exec falkordb redis-cli -a "$FALKORDB_PASSWORD" GRAPH.LIST
docker compose exec falkordb redis-cli -a "$FALKORDB_PASSWORD" GRAPH.QUERY knowledge "MATCH (n) RETURN count(n)"

# Valkey cache stats
docker compose exec valkey valkey-cli -a "$VALKEY_PASSWORD" INFO memory
docker compose exec valkey valkey-cli -a "$VALKEY_PASSWORD" DBSIZE
```

---

## Service URLs (Local)

| Service | URL | Purpose |
|---------|-----|---------|
| Redpanda Console | http://localhost:8080 | Kafka topic browser |
| Flink UI | http://localhost:8081 | Stream processing jobs |
| Prometheus | http://localhost:9090 | Metrics queries |
| Grafana | http://localhost:3001 | Dashboards (admin/PASSWORD) |
| Redpanda Kafka | localhost:19092 | Kafka broker |
| FalkorDB | localhost:6379 | Graph database |
| Valkey | localhost:6380 | Cache |

---

## Troubleshooting

### Docker won't start
```bash
# Check if Docker daemon is running
docker info
# If not, open Docker Desktop app manually
open -a Docker
```

### Container keeps restarting
```bash
# Check logs for the failing container
docker compose logs --tail=50 SERVICE_NAME
```

### Port conflict
```bash
# Check what's using a port
lsof -i :6379  # or whatever port
```

### Out of disk space
```bash
# Clean up Docker
docker system prune -a --volumes
```

### Redpanda topics not created
```bash
# Run the init container manually
docker compose run --rm redpanda-init
```
