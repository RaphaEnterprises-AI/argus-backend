# Uber/Netflix Style Architecture for Argus

## Core Principle: Stateless Flink + Global State Store

Instead of storing state inside Flink (which is hard to replicate), we store state in a **globally-replicated database** that handles multi-region automatically.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GLOBAL LAYER                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Supabase (PostgreSQL + pgvector)                     │ │
│  │                    OR CockroachDB Serverless                            │ │
│  │                                                                          │ │
│  │    • Auto-replicates across regions                                     │ │
│  │    • Handles consistency automatically                                   │ │
│  │    • You already have Supabase!                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Redpanda Serverless                                  │ │
│  │                    (Kafka - already multi-AZ)                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                       │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              │                              ▼
┌───────────────────┐                  │                  ┌───────────────────┐
│   Region A        │                  │                  │   Region B        │
│   (ap-south-1)    │                  │                  │   (ap-southeast-1)│
│                   │                  │                  │                   │
│  ┌─────────────┐  │                  │                  │  ┌─────────────┐  │
│  │ Flink Jobs  │  │                  │                  │  │ Flink Jobs  │  │
│  │ (STATELESS) │  │    Both run      │                  │  │ (STATELESS) │  │
│  │             │  │    simultaneously│                  │  │             │  │
│  │ • Read from │  │◀─────────────────┼─────────────────▶│  │ • Read from │  │
│  │   Kafka     │  │                  │                  │  │   Kafka     │  │
│  │ • Write to  │  │                  │                  │  │ • Write to  │  │
│  │   Supabase  │  │                  │                  │  │   Supabase  │  │
│  └─────────────┘  │                  │                  │  └─────────────┘  │
│                   │                  │                  │                   │
└───────────────────┘                  │                  └───────────────────┘
                                       │
                            Global State in Supabase
                            handles deduplication
                            via idempotency keys
```

## Why This Is Simpler

| Challenge | Traditional | Uber/Netflix Style |
|-----------|-------------|-------------------|
| State replication | You build it | Supabase handles it |
| Failover | Complex scripts | Automatic (both regions active) |
| Consistency | Two-phase commit | Idempotency keys |
| Recovery | Restore from checkpoint | Nothing to restore (stateless) |
| Scaling | Careful checkpoint sizing | Just add pods |

## Components We Use

1. **Supabase** (you already have this!)
   - PostgreSQL with built-in replication
   - Real-time subscriptions for dashboard
   - Row-level security for multi-tenancy

2. **Redpanda Serverless** (you already have this!)
   - Multi-AZ by default
   - No replication to manage

3. **Stateless Flink Jobs** (new design)
   - Read from Kafka
   - Compute aggregations in memory (short windows only)
   - Write results to Supabase with idempotency keys

## The Magic: Idempotency Keys

```sql
-- Every write includes an idempotency key
-- If both regions process the same event, only one write succeeds

CREATE TABLE test_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key TEXT UNIQUE,  -- event_id + window_start
    org_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    total_tests BIGINT,
    passed_tests BIGINT,
    failed_tests BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Upsert with idempotency (both regions can write, duplicates ignored)
INSERT INTO test_metrics (idempotency_key, org_id, project_id, ...)
VALUES ('event123_2024-01-01T00:00:00Z', 'org1', 'proj1', ...)
ON CONFLICT (idempotency_key) DO NOTHING;
```

## Effort Comparison

| Approach | Design | Implement | Test | Total |
|----------|--------|-----------|------|-------|
| Active-Passive (then Active-Active) | 1 week | 4 weeks | 2 weeks | 7 weeks + rebuild later |
| Uber/Netflix from start | 2 weeks | 3 weeks | 2 weeks | 7 weeks (no rebuild) |

## Files in This Directory

- `stateless-flink-jobs.py` - Flink jobs that use external state
- `supabase-schema.sql` - Database schema with idempotency
- `multi-region-deployment.yaml` - K8s manifests for both regions
- `traffic-routing.yaml` - Cloudflare/Route53 configuration
