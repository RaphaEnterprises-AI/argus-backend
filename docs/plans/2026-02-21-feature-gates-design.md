# Feature Gates Design — Tier-Based Soft Gates

**Date**: 2026-02-21
**Status**: Approved

## Decision

Tier-based feature gating with soft enforcement (visual indicators only, no API blocking). Single shared TypeScript config as source of truth.

## Tiers

| | Free | Pro ($99/mo) | Enterprise |
|---|------|-------------|------------|
| Projects | 1 | 10 | Unlimited |
| Test runs/mo | 200 | 10,000 | Unlimited |
| Managed tests | 25 | 500 | Unlimited |
| Team members | 2 | 25 | Unlimited |
| Agents | Core 5 | All 30+ | All + custom |
| Self-healing | View only | Auto-apply | Auto + rules |
| Integrations | GitHub + 2 | All 60+ | All + custom |
| Data retention | 7 days | 90 days | Custom |

Core 5 agents (Free): CodeAnalyzer, TestPlanner, UITester, APITester, Reporter.

## Architecture

```
lib/feature-gates.ts (source of truth)
  ├── PLAN_LIMITS: { free, pro, enterprise }
  ├── canAccess(plan, feature) → boolean
  ├── getLimit(plan, key) → number | Infinity
  ├── isAgentAvailable(plan, agentId) → boolean
  └── AGENT_TIERS: { agentId → 'free' | 'pro' | 'enterprise' }

components/ui/plan-gate.tsx
  ├── PlanProvider (React context, provides current plan)
  └── PlanGate (wrapper: shows children or upgrade prompt)

components/ui/pro-badge.tsx
  └── Small "PRO" badge for nav items and cards
```

## Enforcement Level

**Soft gates only** (no API blocking):
- PRO badges on locked sidebar items
- Lock overlays on pro-only agent/integration cards
- Usage bars showing limits (e.g., "180/200 test runs")
- Upgrade modals when clicking locked features
- Everything still works — no 403s

## Backend Fix

Tenant middleware hardcodes `plan = "free"` (line 155 of `src/api/middleware/tenant.py`). Fix: read actual plan from organizations table.

## Files

| File | Action |
|------|--------|
| `dashboard/lib/feature-gates.ts` | Create |
| `dashboard/components/ui/plan-gate.tsx` | Create |
| `dashboard/components/ui/pro-badge.tsx` | Create |
| `dashboard/components/layout/sidebar.tsx` | Modify |
| `src/api/middleware/tenant.py` | Modify |

## Not in scope

- Stripe integration
- Hard API enforcement (403s)
- Usage metering/counting
- Per-org feature overrides
