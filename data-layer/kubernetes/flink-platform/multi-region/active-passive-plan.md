# Active-Passive Multi-Region Plan for Argus

## Overview
Deploy a standby Flink cluster in a second region that can take over within 5-15 minutes if the primary fails.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Cloudflare DNS                               │
│                    (Health checks + Failover)                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                                               ▼
┌───────────────────────┐               ┌───────────────────────┐
│  ap-south-1 (Primary) │               │  ap-south-2 (Standby) │
│  Vultr Mumbai         │               │  Vultr Singapore      │
├───────────────────────┤               ├───────────────────────┤
│                       │               │                       │
│  ┌─────────────────┐  │               │  ┌─────────────────┐  │
│  │  Flink Cluster  │  │               │  │  Flink Cluster  │  │
│  │  (RUNNING)      │  │               │  │  (STANDBY)      │  │
│  └────────┬────────┘  │               │  └─────────────────┘  │
│           │           │               │                       │
│  ┌────────▼────────┐  │   Mirror      │  ┌─────────────────┐  │
│  │ Redpanda        │──┼──────────────▶│  │ Redpanda        │  │
│  │ Serverless      │  │   (Async)     │  │ (Self-hosted)   │  │
│  └─────────────────┘  │               │  └─────────────────┘  │
│                       │               │                       │
│  ┌─────────────────┐  │   S3 CRR      │  ┌─────────────────┐  │
│  │ R2 Checkpoints  │──┼──────────────▶│  │ R2 Checkpoints  │  │
│  │ (Primary)       │  │               │  │ (Replica)       │  │
│  └─────────────────┘  │               │  └─────────────────┘  │
│                       │               │                       │
└───────────────────────┘               └───────────────────────┘
```

## Implementation Steps

### Phase 1: Checkpoint Replication (Week 1)
- [ ] Enable Cloudflare R2 bucket replication to second region
- [ ] Test checkpoint restore in standby region
- [ ] Document restore procedure

### Phase 2: Kafka Mirroring (Week 2-3)
- [ ] Deploy self-hosted Redpanda in standby region
- [ ] Configure MirrorMaker 2 or Redpanda topic mirroring
- [ ] Test consumer offset synchronization
- [ ] Validate message ordering and exactly-once

### Phase 3: Standby Cluster (Week 3-4)
- [ ] Deploy Flink cluster in standby region (stopped)
- [ ] Create failover scripts
- [ ] Test manual failover procedure
- [ ] Measure RTO (Recovery Time Objective)

### Phase 4: Automated Failover (Week 4-5)
- [ ] Configure Cloudflare health checks
- [ ] Set up DNS failover rules
- [ ] Create automated startup scripts for standby
- [ ] Test automated failover

### Phase 5: DR Testing (Week 5-6)
- [ ] Schedule monthly DR drills
- [ ] Create runbook documentation
- [ ] Train team on failover procedures
- [ ] Set up alerting for replication lag

## Failover Procedure

### Automatic (via Cloudflare)
1. Health check detects primary region down
2. DNS automatically routes to standby
3. Alert sent to on-call engineer
4. Engineer manually starts Flink jobs in standby

### Manual Failover Script
```bash
#!/bin/bash
# failover-to-standby.sh

# 1. Stop primary (if reachable)
kubectl --context=primary-cluster delete flinkdeployment argus-flink -n argus-data || true

# 2. Get latest checkpoint from R2
LATEST_CHECKPOINT=$(aws s3 ls s3://argus-flink-checkpoints-replica/checkpoints/ \
  | sort | tail -1 | awk '{print $4}')

# 3. Update standby Flink config with checkpoint path
kubectl --context=standby-cluster patch configmap flink-config -n argus-data \
  --patch "{\"data\":{\"restore-path\":\"s3://argus-flink-checkpoints-replica/checkpoints/$LATEST_CHECKPOINT\"}}"

# 4. Start Flink in standby
kubectl --context=standby-cluster apply -f flink-cluster-standby.yaml

# 5. Update DNS (if not automatic)
# cloudflare api call to switch traffic

echo "Failover complete. Verify at https://flink.standby.argus.ai"
```

## Cost Estimate

| Component | Primary | Standby | Total |
|-----------|---------|---------|-------|
| Vultr K8s | $100/mo | $50/mo (smaller) | $150/mo |
| Redpanda | $50/mo | $30/mo (self-hosted) | $80/mo |
| R2 Storage | $10/mo | $5/mo (replica) | $15/mo |
| Cloudflare | $0 | $0 | $0 |
| **Total** | **$160/mo** | **$85/mo** | **$245/mo** |

## Recovery Objectives

| Metric | Target | Notes |
|--------|--------|-------|
| RTO (Recovery Time) | 5-15 min | Time to restore service |
| RPO (Data Loss) | < 1 min | Async replication lag |
| Failover Frequency | Monthly drill | Test regularly |

## Monitoring

### Replication Lag Alert
```yaml
- alert: KafkaMirrorLagHigh
  expr: kafka_mirror_records_lag > 10000
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Kafka mirror replication lag high"
```

### Checkpoint Sync Alert
```yaml
- alert: CheckpointSyncDelayed
  expr: time() - s3_last_sync_timestamp > 300
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Checkpoint replication delayed >5 min"
```
