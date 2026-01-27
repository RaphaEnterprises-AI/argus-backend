# Argus Flink Platform - Self-Healing Stream Processing

A lightweight "Flink-as-a-Service" platform that automates all operational challenges.

## What This Solves

| Challenge | Solution | Automation Level |
|-----------|----------|------------------|
| Checkpointing | Auto-configured with S3 backend | Fully automated |
| Backpressure | KEDA scales based on Kafka lag | Fully automated |
| Exactly-Once | Pre-configured transactional sinks | Built-in |
| State Recovery | Automatic restart from checkpoint | Fully automated |
| Monitoring | Prometheus + Grafana dashboards | Pre-configured |
| Alerting | PagerDuty/Slack integration | Pre-configured |
| Upgrades | Blue-green with savepoint | Semi-automated |
| Multi-region | Active-passive failover | Configurable |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Argus Flink Platform                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   KEDA       │  │  Flink K8s   │  │  Prometheus + Grafana    │  │
│  │  Autoscaler  │  │  Operator    │  │  (Monitoring)            │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                   │
│         ▼                 ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Flink Jobs                                │   │
│  │  • Auto-restart on failure                                   │   │
│  │  • Auto-scale on Kafka lag                                   │   │
│  │  • Auto-checkpoint to S3                                     │   │
│  │  • Metrics exported to Prometheus                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
└──────────────────────────────┼───────────────────────────────────────┘
                               ▼
                    ┌───────────────────┐
                    │ Redpanda Serverless│
                    └───────────────────┘
```

## Quick Start

```bash
./deploy-platform.sh
```

## Components

1. `flink-operator.yaml` - Flink Kubernetes Operator
2. `keda-scaler.yaml` - Auto-scaling based on Kafka consumer lag
3. `checkpoint-config.yaml` - S3 checkpoint backend
4. `monitoring/` - Prometheus rules + Grafana dashboards
5. `alerting/` - Slack/PagerDuty alert rules
