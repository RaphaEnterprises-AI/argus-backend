# Argus Test Bench

Sample apps deployed on Railway to dogfood Argus end-to-end.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Conduit | conduit-testbench-production.up.railway.app | Simple blog (RealWorld) |
| Plane | plane-testbench-production.up.railway.app | Project management SaaS |
| JSON API | jsonapi-testbench-production.up.railway.app | Pure REST with OpenAPI spec |
| Chaos App | chaos-testbench-production.up.railway.app | E-commerce with 30 toggleable bugs |
| Chaos Controller | chaos-ctrl-testbench-production.up.railway.app | Scenario orchestrator |

## Quick Start

```bash
# Deploy everything to Railway
./testbench/deploy.sh

# Run a scenario
curl -X POST https://chaos-ctrl-testbench-production.up.railway.app/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{"scenario": "golden-path"}'

# Toggle a bug
curl -X POST https://chaos-testbench-production.up.railway.app/chaos/bugs/selector-login-id/enable

# Reset all bugs
curl -X POST https://chaos-testbench-production.up.railway.app/chaos/bugs/reset
```

## Architecture

See `docs/plans/2026-02-27-testbench-design.md` for full design document.
