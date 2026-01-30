# Business Continuity Plan

**Argus E2E Testing Platform**
**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Classification:** Internal

---

## 1. Purpose

This Business Continuity Plan (BCP) ensures the Argus platform can maintain critical operations during disruptions, recover from disasters, and resume normal operations with minimal impact to customers.

---

## 2. Scope

This plan covers:
- Infrastructure failures (cloud provider outages)
- Data center disasters
- Cyber attacks
- Personnel unavailability
- Third-party vendor failures
- Natural disasters affecting operations

---

## 3. Critical Business Functions

### 3.1 Priority Classification

| Priority | Function | RTO | RPO | Description |
|----------|----------|-----|-----|-------------|
| P1 | API Services | 15 min | 0 | Core API for test execution |
| P1 | Authentication | 15 min | 0 | User login and API key validation |
| P1 | Database | 30 min | 5 min | Primary data store |
| P2 | Dashboard | 1 hour | 15 min | Web UI for users |
| P2 | Test Execution | 1 hour | N/A | Active test runs |
| P3 | Reporting | 4 hours | 1 hour | Report generation |
| P3 | Integrations | 4 hours | 1 hour | Third-party connections |
| P4 | Analytics | 24 hours | 24 hours | Usage analytics |

**RTO** = Recovery Time Objective (max acceptable downtime)
**RPO** = Recovery Point Objective (max acceptable data loss)

### 3.2 System Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                     EXTERNAL DEPENDENCIES                    │
├─────────────────────────────────────────────────────────────┤
│  Anthropic API    │  Google Cloud   │  OpenAI API          │
│  (LLM Provider)   │  (Vision/LLM)   │  (Fallback LLM)      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  Railway (API)    │  Supabase (DB)  │  Cloudflare (CDN)    │
│  99.99% SLA       │  99.95% SLA     │  99.99% SLA          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER (VKE)                        │
├─────────────────────────────────────────────────────────────┤
│  Redpanda         │  FalkorDB       │  Valkey              │
│  (Event Stream)   │  (Graph DB)     │  (Cache)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Risk Assessment

### 4.1 Threat Matrix

| Threat | Likelihood | Impact | Risk Score | Mitigation |
|--------|------------|--------|------------|------------|
| Cloud provider outage | Medium | High | High | Multi-region, failover |
| Database corruption | Low | Critical | High | Backups, replication |
| DDoS attack | Medium | Medium | Medium | Cloudflare protection |
| Key personnel unavailable | Medium | Medium | Medium | Documentation, cross-training |
| Third-party API failure | Medium | Medium | Medium | Fallback providers |
| Data breach | Low | Critical | High | Encryption, access controls |
| Natural disaster | Low | High | Medium | Distributed infrastructure |

### 4.2 Single Points of Failure

| Component | Risk | Mitigation |
|-----------|------|------------|
| Supabase Database | High | Point-in-time recovery, daily backups |
| Railway API | Medium | Auto-scaling, health checks |
| Anthropic API | Medium | OpenRouter fallback (300+ models) |
| Kubernetes Cluster | Medium | Node redundancy, pod replicas |

---

## 5. Recovery Strategies

### 5.1 API Services (Railway)

**Primary Strategy: Auto-Scaling with Health Checks**
- Railway automatically restarts unhealthy containers
- Multiple replicas for high availability
- Automatic rollback on failed deployments

**Failover Procedure:**
1. Railway auto-detects failure via health checks
2. Unhealthy instances replaced automatically
3. If platform-wide outage: Deploy to backup region

**Backup Region:** Render.com or Fly.io (pre-configured)

### 5.2 Database (Supabase)

**Primary Strategy: Point-in-Time Recovery**
- Continuous WAL archiving
- Daily full backups
- 30-day backup retention

**Recovery Procedure:**
```bash
# 1. Access Supabase Dashboard
# 2. Navigate to Database → Backups
# 3. Select restoration point
# 4. Initiate point-in-time recovery

# Post-recovery verification:
curl -s https://argus-brain-production.up.railway.app/api/v1/health/database
```

**RTO:** 30 minutes
**RPO:** 5 minutes (WAL-based recovery)

### 5.3 Data Layer (Kubernetes)

**Primary Strategy: Pod Redundancy**
- Redpanda: 3-node cluster with replication factor 3
- FalkorDB: Replicated deployment
- Valkey: Sentinel for automatic failover

**Recovery Procedure:**
```bash
# Check cluster health
kubectl get pods -n argus-data

# Restart failed pods
kubectl rollout restart deployment/<name> -n argus-data

# Scale up if needed
kubectl scale deployment/<name> --replicas=3 -n argus-data
```

### 5.4 LLM Provider Failover

**Primary:** Anthropic Claude
**Fallback Chain:** OpenRouter → Google → OpenAI

```python
# src/core/model_registry.py - Automatic failover
FALLBACK_PROVIDERS = [
    Provider.ANTHROPIC,    # Primary
    Provider.OPENROUTER,   # 300+ models
    Provider.GOOGLE,       # Gemini
    Provider.OPENAI,       # GPT-4
]
```

---

## 6. Communication Plan

### 6.1 Internal Communication

| Scenario | Channel | Audience |
|----------|---------|----------|
| Initial detection | Slack #incidents | On-call team |
| Escalation | Phone call | Management |
| Status updates | Slack #incidents | All engineering |
| Resolution | Email + Slack | All company |

### 6.2 External Communication

| Scenario | Channel | Timeline |
|----------|---------|----------|
| Service degradation | Status page | Within 15 minutes |
| Major outage | Status page + Email | Within 30 minutes |
| Data incident | Direct customer contact | Per IRP |
| Resolution | All channels | When confirmed |

**Status Page:** https://status.heyargus.ai

### 6.3 Communication Templates

**Outage Announcement:**
```
Title: Service Disruption - [Component]

We are currently experiencing issues with [component].
Our team is actively investigating.

Impact: [Description of affected functionality]
Started: [Timestamp]
Status: Investigating

We will provide updates every [30 minutes / hour].
```

**Resolution Announcement:**
```
Title: Resolved - [Component]

The issues with [component] have been resolved.

Impact duration: [X hours/minutes]
Root cause: [Brief description]
Resolution: [What was done]

We apologize for any inconvenience.
```

---

## 7. Recovery Procedures

### 7.1 Complete Service Recovery

**Phase 1: Assessment (0-15 min)**
1. Identify scope of outage
2. Determine root cause category
3. Activate appropriate recovery procedure
4. Notify stakeholders

**Phase 2: Infrastructure Recovery (15-60 min)**
1. Verify cloud provider status
2. Check database connectivity
3. Restart failed services
4. Verify inter-service communication

**Phase 3: Application Recovery (60-120 min)**
1. Deploy last known-good version
2. Run health checks on all endpoints
3. Verify authentication working
4. Test critical user flows

**Phase 4: Data Verification (2-4 hours)**
1. Verify database integrity
2. Check for data loss (compare RPO)
3. Run reconciliation if needed
4. Verify audit logs intact

**Phase 5: Return to Normal (4-8 hours)**
1. Monitor for recurring issues
2. Gradual traffic restoration
3. Customer communication (all clear)
4. Schedule post-mortem

### 7.2 Database Recovery

```bash
# Step 1: Verify backup availability
# Access Supabase Dashboard → Backups

# Step 2: Create new project from backup (if needed)
# Supabase Dashboard → Restore to new project

# Step 3: Update connection strings
# Railway Dashboard → Environment Variables
# Update DATABASE_URL, SUPABASE_URL

# Step 4: Verify data integrity
psql $DATABASE_URL -c "SELECT COUNT(*) FROM tests;"
psql $DATABASE_URL -c "SELECT COUNT(*) FROM organizations;"

# Step 5: Restart API services
railway up --detach
```

### 7.3 Kubernetes Cluster Recovery

```bash
# Check cluster status
kubectl cluster-info
kubectl get nodes

# If nodes unhealthy, recreate
# Vultr Dashboard → Kubernetes → Node Pools

# Restore from manifests
kubectl apply -f kubernetes/argus-data/

# Verify pods running
kubectl get pods -n argus-data -w

# Check Redpanda health
kubectl exec -n argus-data redpanda-0 -- rpk cluster health
```

---

## 8. Testing and Maintenance

### 8.1 Testing Schedule

| Test Type | Frequency | Participants | Duration |
|-----------|-----------|--------------|----------|
| Tabletop exercise | Quarterly | All engineering | 2 hours |
| Database failover | Semi-annually | Database team | 4 hours |
| Full DR test | Annually | All teams | 1 day |
| Communication test | Quarterly | All company | 1 hour |

### 8.2 Test Scenarios

**Quarterly Tabletop:**
1. Railway complete outage
2. Supabase database corruption
3. API key compromise
4. DDoS attack

**Semi-Annual Practical:**
1. Database restore from backup
2. Failover to backup region
3. LLM provider failover

### 8.3 Maintenance Requirements

| Task | Frequency | Responsible |
|------|-----------|-------------|
| Backup verification | Weekly | DevOps |
| BCP document review | Quarterly | Security |
| Contact list update | Monthly | Operations |
| Runbook validation | Quarterly | Engineering |

---

## 9. Roles and Responsibilities

### 9.1 BCP Team

| Role | Primary Responsibility |
|------|------------------------|
| BCP Coordinator | Plan maintenance, testing coordination |
| Technical Lead | Infrastructure recovery |
| Security Lead | Security incident coordination |
| Communications Lead | Customer/public communication |
| Executive Sponsor | Resource allocation, escalation |

### 9.2 Emergency Contact List

| Role | Contact Method | Backup |
|------|----------------|--------|
| On-call Engineer | PagerDuty | Slack #engineering |
| Technical Lead | Phone + Slack | Email |
| Security Lead | Phone + Slack | Email |
| Executive | Phone | Email |

---

## 10. Resource Requirements

### 10.1 Technology Resources

| Resource | Purpose | Location |
|----------|---------|----------|
| Backup credentials | Service access | Secure vault |
| Recovery scripts | Automated recovery | Git repository |
| Configuration backups | Service configs | Encrypted storage |
| Documentation | Procedures | This document + wiki |

### 10.2 Vendor Contacts

| Vendor | Service | Support Contact |
|--------|---------|-----------------|
| Railway | API hosting | support@railway.app |
| Supabase | Database | support@supabase.io |
| Cloudflare | CDN/Security | Enterprise support |
| Vultr | Kubernetes | support@vultr.com |
| Anthropic | LLM | Enterprise support |

---

## 11. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-30 | Engineering | Initial release |

**Review Schedule:** Quarterly
**Next Review:** 2026-04-30
**Document Owner:** Engineering Leadership

---

## Appendix A: Quick Reference Card

### Critical Contacts
- On-call: PagerDuty
- Escalation: Slack #incidents
- Executive: cto@heyargus.ai

### Critical URLs
- API Health: https://argus-brain-production.up.railway.app/health
- Status Page: https://status.heyargus.ai
- Railway Dashboard: https://railway.app
- Supabase Dashboard: https://supabase.com/dashboard

### Quick Commands
```bash
# Check API status
curl -s https://argus-brain-production.up.railway.app/health

# View Railway logs
railway logs --tail 100

# Check Kubernetes
kubectl get pods -n argus-data

# Redpanda health
kubectl exec -n argus-data redpanda-0 -- rpk cluster health
```

## Appendix B: Related Documents

- Incident Response Plan (`docs/INCIDENT_RESPONSE_PLAN.md`)
- SOC 2 Evidence (`docs/SOC2_EVIDENCE.md`)
- Architecture Documentation (`docs/architecture/`)
