# Incident Response Plan

**Document ID**: IRP-001
**Version**: 1.0
**Effective Date**: [DATE]
**Last Review**: [DATE]
**Next Review**: [DATE + 1 year]
**Owner**: [Security Lead]
**Classification**: INTERNAL

---

## 1. Purpose

This Incident Response Plan establishes procedures for detecting, responding to, and recovering from security incidents. It ensures consistent, effective handling of incidents to minimize impact and prevent recurrence.

## 2. Scope

This plan covers:
- Security incidents affecting Argus systems, data, or customers
- Suspected or confirmed unauthorized access
- Data breaches or data loss
- Service disruptions caused by security events
- Malware or ransomware attacks
- Social engineering attacks

## 3. Incident Classification

### 3.1 Severity Levels

| Level | Name | Description | Response Time | Example |
|-------|------|-------------|---------------|---------|
| P1 | Critical | Active breach, data exfiltration, complete service outage | Immediate (15 min) | Customer data breach, ransomware |
| P2 | High | Confirmed security compromise, significant service impact | 1 hour | Unauthorized access to production, critical vulnerability exploited |
| P3 | Medium | Potential security issue, limited impact | 4 hours | Failed intrusion attempt, suspicious activity |
| P4 | Low | Minor security issue, no immediate impact | 24 hours | Policy violation, minor misconfiguration |

### 3.2 Incident Categories

- **Unauthorized Access**: Access to systems or data without authorization
- **Data Breach**: Unauthorized disclosure of sensitive data
- **Malware**: Malicious software detected on systems
- **Denial of Service**: Attack affecting system availability
- **Social Engineering**: Phishing or other manipulation attempts
- **Insider Threat**: Malicious activity by authorized users
- **System Compromise**: Complete or partial control by attacker
- **Policy Violation**: Violation of security policies

## 4. Incident Response Team

### 4.1 Core Team

| Role | Responsibilities | Primary | Backup |
|------|-----------------|---------|--------|
| **Incident Commander** | Overall incident coordination, decision authority | [Security Lead] | [CTO] |
| **Technical Lead** | Technical investigation and remediation | [Senior Engineer] | [Backend Lead] |
| **Communications Lead** | Internal/external communications | [CEO] | [Head of Product] |
| **Documentation Lead** | Incident logging and evidence preservation | [Security Lead] | [Engineering Manager] |

### 4.2 Extended Team (as needed)

- Legal counsel
- External security consultants
- Law enforcement liaison
- Customer support lead
- HR representative

## 5. Response Phases

### 5.1 Detection

**Objective**: Identify potential security incidents promptly.

**Detection Sources**:
- Sentry alerts (production errors, anomalies)
- Audit log analysis (unusual access patterns)
- Customer reports
- Security scanning tools
- Third-party notifications
- Employee observations

**Actions**:
1. Monitor Sentry dashboards for anomalies
2. Review audit logs for suspicious activity
3. Respond to security alerts within 15 minutes
4. Document initial observations

### 5.2 Analysis

**Objective**: Confirm incident, determine scope and severity.

**Actions**:
1. Assign incident number: `INC-YYYYMMDD-XXX`
2. Determine severity level (P1-P4)
3. Identify affected systems and data
4. Estimate number of affected users/customers
5. Identify attack vector if known
6. Preserve evidence (logs, screenshots, artifacts)

**Evidence Preservation Checklist**:
- [ ] Capture relevant audit logs
- [ ] Screenshot affected systems
- [ ] Export Sentry error details
- [ ] Preserve network traffic logs (if available)
- [ ] Document timeline of events
- [ ] Secure any malware samples (sandboxed)

### 5.3 Containment

**Objective**: Limit incident impact and prevent spread.

**Short-term Containment**:
1. Isolate affected systems (if applicable)
2. Revoke compromised credentials immediately
3. Block malicious IP addresses
4. Disable compromised accounts
5. Implement emergency firewall rules

**Long-term Containment**:
1. Patch vulnerable systems
2. Reset all potentially compromised credentials
3. Implement additional monitoring
4. Apply network segmentation if needed

**API Key Compromise Response**:
```bash
# 1. Revoke compromised key immediately
# (via admin dashboard or API)

# 2. Notify affected customer

# 3. Generate new key for customer

# 4. Update audit logs
```

### 5.4 Eradication

**Objective**: Remove threat and eliminate vulnerability.

**Actions**:
1. Identify root cause
2. Remove malware or malicious code
3. Patch vulnerable software
4. Close unauthorized access paths
5. Verify complete removal of threat
6. Update security controls

### 5.5 Recovery

**Objective**: Restore normal operations safely.

**Actions**:
1. Restore from clean backups if needed
2. Verify system integrity before restoration
3. Implement additional monitoring
4. Gradually restore services
5. Confirm normal operation
6. Maintain heightened monitoring for 7 days

**Recovery Verification Checklist**:
- [ ] All malicious artifacts removed
- [ ] Affected credentials rotated
- [ ] Systems patched and updated
- [ ] Backup integrity verified
- [ ] Monitoring enhanced
- [ ] Service functionality confirmed

### 5.6 Lessons Learned

**Objective**: Improve security posture and prevent recurrence.

**Actions** (within 7 days of resolution):
1. Conduct post-incident review meeting
2. Document incident timeline
3. Identify what worked well
4. Identify areas for improvement
5. Define remediation actions
6. Update policies/procedures as needed
7. Share lessons with team (anonymized if needed)

**Post-Incident Report Template**:
```
Incident ID: INC-YYYYMMDD-XXX
Severity: P1/P2/P3/P4
Duration: [start] to [end]

Summary: [Brief description]

Timeline:
- [timestamp]: [event]
- [timestamp]: [event]

Root Cause: [description]

Impact:
- Systems affected: [list]
- Data affected: [description]
- Customers affected: [count/scope]

Response Actions:
1. [action taken]
2. [action taken]

Remediation:
- [ ] [action item] - Owner: [name] - Due: [date]
- [ ] [action item] - Owner: [name] - Due: [date]

Lessons Learned:
- [insight 1]
- [insight 2]
```

## 6. Communication Plan

### 6.1 Internal Escalation

| Severity | Notify Within | Notify |
|----------|---------------|--------|
| P1 | Immediately | CEO, CTO, Security Lead, all engineers |
| P2 | 1 hour | CTO, Security Lead, affected team leads |
| P3 | 4 hours | Security Lead, affected team |
| P4 | 24 hours | Security Lead |

**Communication Channels**:
- Primary: Slack #security-incidents
- Secondary: Phone/SMS for P1/P2
- Email for formal notifications

### 6.2 External Communication

**Customer Notification** (for confirmed data breaches):
- Notify within 72 hours of confirmation
- Include: what happened, data affected, actions taken, customer actions, contact info
- Template available in Appendix A

**Regulatory Notification**:
- GDPR: 72 hours to supervisory authority
- CCPA: As required
- Other: Per contractual requirements

**Law Enforcement**:
- Contact for significant criminal activity
- Coordinate with legal counsel first
- Preserve evidence chain of custody

### 6.3 Status Updates

| Phase | Update Frequency |
|-------|-----------------|
| Active P1 | Every 30 minutes |
| Active P2 | Every 2 hours |
| Active P3/P4 | Every 8 hours |
| Resolution | Final update within 24 hours |

## 7. Tools and Resources

### 7.1 Incident Response Tools

| Tool | Purpose | Access |
|------|---------|--------|
| Sentry | Error monitoring, alerting | All engineers |
| Supabase Dashboard | Audit logs, database access | Security Lead, DBAs |
| Cloudflare Dashboard | Traffic analysis, blocking | Security Lead, Platform team |
| GitHub | Code review, PR history | All engineers |
| Linear | Incident tracking | All team members |

### 7.2 Contact List

| Role | Name | Phone | Email | Slack |
|------|------|-------|-------|-------|
| Security Lead | [Name] | [Phone] | [Email] | @[handle] |
| CTO | [Name] | [Phone] | [Email] | @[handle] |
| CEO | [Name] | [Phone] | [Email] | @[handle] |
| Legal Counsel | [Name] | [Phone] | [Email] | N/A |

### 7.3 External Contacts

| Organization | Purpose | Contact |
|--------------|---------|---------|
| Supabase Support | Database incidents | support@supabase.com |
| Cloudflare Support | CDN/WAF incidents | enterprise-support@cloudflare.com |
| Clerk Support | Auth incidents | support@clerk.com |
| [Law Firm] | Legal guidance | [contact] |
| [Incident Response Firm] | External forensics | [contact] |

## 8. Testing and Training

### 8.1 Plan Testing

- **Tabletop Exercises**: Quarterly
- **Technical Drills**: Bi-annually
- **Full Simulation**: Annually

### 8.2 Training

All team members must complete:
- Incident response overview (onboarding)
- Role-specific training (as assigned)
- Annual refresher training

## 9. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | [DATE] | [Author] | Initial release |

---

## Appendix A: Customer Notification Template

```
Subject: Security Notification from Argus

Dear [Customer Name],

We are writing to inform you of a security incident that may have
affected your data.

WHAT HAPPENED
[Brief description of the incident]

WHAT INFORMATION WAS INVOLVED
[Description of data types affected]

WHAT WE ARE DOING
[Actions taken to address the incident]

WHAT YOU CAN DO
[Recommended customer actions]

FOR MORE INFORMATION
If you have questions, please contact us at security@heyargus.ai.

We sincerely apologize for any inconvenience this may cause.

[Signature]
```

## Appendix B: Incident Log Template

| Field | Value |
|-------|-------|
| Incident ID | INC-YYYYMMDD-XXX |
| Reported By | |
| Reported At | |
| Severity | P1 / P2 / P3 / P4 |
| Category | |
| Status | Open / In Progress / Resolved / Closed |
| Incident Commander | |
| Summary | |
| Systems Affected | |
| Data Affected | |
| Timeline | |
| Actions Taken | |
| Resolution | |
| Lessons Learned | |
