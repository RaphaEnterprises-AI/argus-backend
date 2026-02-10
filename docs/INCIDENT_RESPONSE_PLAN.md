# Incident Response Plan

**Skopaq E2E Testing Platform**
**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Classification:** Internal

---

## 1. Purpose

This Incident Response Plan (IRP) establishes procedures for detecting, responding to, and recovering from security incidents affecting the Skopaq platform. It ensures consistent, effective responses that minimize impact and protect customer data.

---

## 2. Scope

This plan covers:
- Security breaches and unauthorized access
- Data leaks or exposure
- Service disruptions and outages
- Malware or ransomware incidents
- Denial of service attacks
- Insider threats
- Third-party vendor incidents

---

## 3. Incident Severity Levels

| Level | Name | Description | Response Time | Examples |
|-------|------|-------------|---------------|----------|
| P1 | Critical | Active data breach, complete service outage | < 15 minutes | Unauthorized data access, production down |
| P2 | High | Partial service degradation, potential breach | < 1 hour | API errors affecting users, suspicious activity |
| P3 | Medium | Limited impact, contained issue | < 4 hours | Single component failure, minor vulnerability |
| P4 | Low | Minimal impact, informational | < 24 hours | Failed login attempts, minor policy violation |

---

## 4. Incident Response Team

### 4.1 Core Team Roles

| Role | Responsibilities | Contact |
|------|------------------|---------|
| **Incident Commander** | Overall coordination, stakeholder communication | On-call rotation |
| **Technical Lead** | Technical investigation and remediation | Engineering team |
| **Security Analyst** | Forensics, threat analysis | Security team |
| **Communications Lead** | Customer/public communication | Product team |
| **Legal Liaison** | Regulatory compliance, legal guidance | Legal counsel |

### 4.2 Escalation Path

```
L1: On-call Engineer
    ↓ (15 min no resolution)
L2: Technical Lead + Security Analyst
    ↓ (P1/P2 or 1 hour no resolution)
L3: Incident Commander + Executive Team
    ↓ (Data breach confirmed)
L4: Legal + External Response Team
```

---

## 5. Incident Response Phases

### Phase 1: Detection and Identification

#### Detection Sources
- Automated monitoring alerts (Prometheus/Alertmanager)
- Sentry error tracking
- Customer reports via support channels
- Security scanning tools
- Employee observations
- Third-party notifications

#### Initial Assessment Checklist
- [ ] What systems are affected?
- [ ] What data may be compromised?
- [ ] How many users/customers are impacted?
- [ ] Is the incident ongoing or contained?
- [ ] What is the severity level (P1-P4)?
- [ ] Who discovered the incident and when?

### Phase 2: Containment

#### Immediate Actions (First 15 minutes)

**For Active Breaches:**
```bash
# Revoke compromised credentials
# Location: Railway dashboard or Supabase admin

# Block suspicious IPs
# Add to CORS blocklist in src/main.py

# Disable affected API endpoints (if necessary)
# Set MAINTENANCE_MODE=true in environment
```

**For Service Disruptions:**
```bash
# Check service status
curl https://skopaq-brain-production.up.railway.app/health

# View recent deployments
railway logs --filter error

# Rollback if deployment-related
railway rollback
```

#### Short-Term Containment
1. Isolate affected systems
2. Preserve evidence (logs, snapshots)
3. Block attack vectors
4. Enable enhanced monitoring

### Phase 3: Eradication

#### Root Cause Analysis
1. Review audit logs (`src/services/audit_logger.py`)
2. Analyze Sentry error traces
3. Check Langfuse for unusual AI usage patterns
4. Review database access logs in Supabase

#### Remediation Steps
- [ ] Patch identified vulnerabilities
- [ ] Update affected credentials/keys
- [ ] Remove malicious code/access
- [ ] Verify remediation effectiveness

### Phase 4: Recovery

#### System Restoration
1. Verify systems are clean
2. Restore from known-good backups if needed
3. Gradually restore services
4. Monitor for recurrence

#### Validation Checklist
- [ ] All security patches applied
- [ ] Credentials rotated
- [ ] Services functioning normally
- [ ] No signs of persistent threat
- [ ] Monitoring enhanced for affected areas

### Phase 5: Post-Incident Review

#### Timeline Documentation
| Time | Event | Action Taken | By Whom |
|------|-------|--------------|---------|
| T+0 | Incident detected | Initial assessment | On-call |
| T+X | Containment started | Isolated systems | Tech Lead |
| ... | ... | ... | ... |

#### Post-Mortem Template
1. **Incident Summary**: What happened?
2. **Timeline**: Detailed chronology
3. **Root Cause**: Why did it happen?
4. **Impact**: What was affected?
5. **Response Evaluation**: What went well/poorly?
6. **Lessons Learned**: What can we improve?
7. **Action Items**: Preventive measures

---

## 6. Communication Procedures

### 6.1 Internal Communication

**Slack Channels:**
- `#incidents` - Active incident coordination
- `#security` - Security team discussions
- `#engineering` - Technical updates

**Communication Templates:**

**Initial Alert:**
```
🚨 INCIDENT ALERT - [SEVERITY]
Issue: [Brief description]
Affected: [Systems/users impacted]
Status: [Investigating/Contained/Resolved]
IC: [Incident Commander name]
Updates: Every [15/30/60] minutes
```

**Status Update:**
```
📊 INCIDENT UPDATE - [SEVERITY]
Current Status: [Status]
Actions Taken: [List]
Next Steps: [List]
ETA to Resolution: [Time estimate]
```

### 6.2 Customer Communication

**For P1/P2 Incidents:**
1. Status page update within 30 minutes
2. Email notification to affected customers
3. Regular updates every hour until resolved
4. Post-incident summary within 24 hours

**Status Page Updates:**
- URL: https://status.skopaq.ai
- Update frequency based on severity

### 6.3 Regulatory Notification

**Data Breach Requirements:**
| Regulation | Notification Window | Authority |
|------------|---------------------|-----------|
| GDPR | 72 hours | Supervisory authority |
| CCPA | "Without unreasonable delay" | CA Attorney General |
| HIPAA | 60 days | HHS OCR |

---

## 7. Evidence Preservation

### 7.1 What to Preserve
- System logs (last 30 days minimum)
- Network traffic captures
- Database query logs
- User access logs
- Audit trail entries
- Screenshots of anomalous behavior
- Communication records

### 7.2 Chain of Custody
1. Document who collected evidence
2. Timestamp all evidence collection
3. Store in secure, access-controlled location
4. Maintain integrity hashes

### 7.3 Log Locations

| Log Type | Location | Retention |
|----------|----------|-----------|
| Application Logs | Railway logs | 30 days |
| Audit Logs | Supabase `audit_logs` table | 1 year |
| Security Events | Sentry | 90 days |
| AI Usage | Langfuse | 30 days |
| Infrastructure | Prometheus/Grafana | 15 days |

---

## 8. Specific Incident Playbooks

### 8.1 Unauthorized Data Access

1. **Identify**: Which data was accessed? By whom?
2. **Contain**: Revoke access, change credentials
3. **Assess**: Determine data sensitivity, scope
4. **Notify**: Legal, affected customers, regulators (if required)
5. **Remediate**: Patch access control gaps
6. **Review**: Enhance monitoring, update policies

### 8.2 API Key Compromise

1. **Immediate**: Revoke compromised key
   ```sql
   UPDATE api_keys SET revoked = true WHERE key_hash = 'HASH';
   ```
2. **Audit**: Check key usage in audit logs
3. **Assess**: Determine actions taken with key
4. **Notify**: Key owner, security team
5. **Issue**: New key to legitimate user
6. **Review**: Key rotation policy

### 8.3 DDoS Attack

1. **Detect**: Monitor traffic patterns, error rates
2. **Activate**: Cloudflare DDoS protection
3. **Scale**: Increase backend capacity
4. **Communicate**: Status page update
5. **Analyze**: Attack patterns, origin IPs
6. **Harden**: Update rate limits, WAF rules

### 8.4 Service Outage

1. **Assess**: Which components are affected?
2. **Diagnose**: Check Railway, Supabase, Kubernetes status
3. **Restore**: Restart services, rollback if needed
4. **Verify**: Test all endpoints
5. **Communicate**: Customer notification
6. **Review**: Add monitoring for failure mode

---

## 9. Tools and Resources

### 9.1 Incident Response Tools

| Tool | Purpose | Access |
|------|---------|--------|
| Railway Dashboard | Deployment, logs | team@skopaq.ai |
| Supabase Dashboard | Database, auth | team@skopaq.ai |
| Sentry | Error tracking | team@skopaq.ai |
| Grafana | Metrics dashboards | monitoring.skopaq.ai |
| Cloudflare | WAF, DDoS protection | team@skopaq.ai |

### 9.2 Quick Commands

```bash
# Check API health
curl -s https://skopaq-brain-production.up.railway.app/health | jq .

# View recent logs
railway logs --tail 100

# Check Kubernetes pods
kubectl get pods -n argus-data

# View Redpanda health
kubectl exec -n argus-data redpanda-0 -- rpk cluster health
```

---

## 10. Training and Testing

### 10.1 Training Requirements
- All engineers: Annual security awareness training
- On-call staff: Incident response procedures quarterly
- Security team: Advanced incident handling

### 10.2 Testing Schedule
- Tabletop exercises: Quarterly
- Simulated incidents: Semi-annually
- Full disaster recovery test: Annually

---

## 11. Document Maintenance

| Review Frequency | Responsible Party |
|------------------|-------------------|
| Quarterly review | Security Team |
| Post-incident updates | Incident Commander |
| Annual full revision | Engineering Leadership |

---

## Appendix A: Contact Information

| Role | Primary | Backup |
|------|---------|--------|
| On-call Engineer | PagerDuty rotation | Slack #engineering |
| Security Lead | security@skopaq.ai | Slack #security |
| Executive Contact | cto@skopaq.ai | ceo@skopaq.ai |
| Legal | legal@skopaq.ai | External counsel |

## Appendix B: Related Documents

- Business Continuity Plan (`docs/BUSINESS_CONTINUITY_PLAN.md`)
- SOC 2 Evidence (`docs/SOC2_EVIDENCE.md`)
- Data Flow Diagrams (`docs/DATA_FLOW_DIAGRAMS.md`)
