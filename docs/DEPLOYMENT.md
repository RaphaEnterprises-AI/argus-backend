# Argus Deployment Guide

This guide covers deploying the full Argus E2E Testing Agent stack.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DASHBOARD (Vercel)                             │
│                      Next.js + Tailwind + Clerk Auth                     │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────┐
│    ARGUS BRAIN        │  │   BROWSER WORKER      │  │    MCP SERVER     │
│  (Railway/Render)     │  │   (Cloudflare)        │  │   (Cloudflare)    │
├───────────────────────┤  ├───────────────────────┤  ├───────────────────┤
│ • LangGraph Orchestr. │  │ • Playwright Browser  │  │ • AI IDE Integr.  │
│ • Code Analysis       │  │ • Self-Healing        │  │ • Claude Code     │
│ • Test Planning       │  │ • Screenshot Capture  │  │ • Cursor/Windsurf │
│ • Visual AI           │  │ • Pattern Learning    │  │                   │
│ • Report Generation   │  │ • Smart Waits         │  │                   │
└───────────────────────┘  └───────────────────────┘  └───────────────────┘
        Python                    TypeScript                TypeScript
```

## Components

| Component | Location | Platform | Status |
|-----------|----------|----------|--------|
| Dashboard | `dashboard/` | Vercel | Deployed |
| Browser Worker | `cloudflare-worker/` | Cloudflare Workers | Deployed |
| MCP Server | `argus-mcp-server/` | Cloudflare Workers | Deployed |
| **Argus Brain** | `src/` | Railway/Render/Fly.io | **NEEDS DEPLOYMENT** |

## 1. Deploy Argus Brain (Python Backend)

The Brain handles orchestration, code analysis, and advanced AI features.

### Option A: Railway (Recommended)

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Create a new project
railway init

# 4. Deploy
railway up

# 5. Set environment variables
railway variables set ANTHROPIC_API_KEY=sk-ant-...
railway variables set BROWSER_WORKER_URL=https://argus-api.samuelvinay-kumar.workers.dev

# 6. Get your deployment URL
railway status
# Example: https://argus-brain-production.up.railway.app
```

### Option B: Render

```bash
# 1. Connect your GitHub repo to Render
# 2. Create a new Web Service
# 3. Select Docker deployment
# 4. Set environment variables in Render dashboard
# 5. Deploy
```

Or use the Blueprint:
```bash
# render.yaml is already configured
# Just connect repo and deploy
```

### Option C: Fly.io

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Login
fly auth login

# 3. Launch (uses fly.toml config)
fly launch

# 4. Set secrets
fly secrets set ANTHROPIC_API_KEY=sk-ant-...

# 5. Deploy
fly deploy
```

### Option D: Local Development

```bash
# 1. Install Python dependencies
pip install -e .

# 2. Set environment variables
export ANTHROPIC_API_KEY=sk-ant-...
export BROWSER_WORKER_URL=https://argus-api.samuelvinay-kumar.workers.dev

# 3. Run the server
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

## 2. Connect Dashboard to Brain

Update your dashboard environment:

```bash
# dashboard/.env.local

# Point to your deployed Brain
ARGUS_BACKEND_URL=https://your-argus-brain.railway.app

# Browser Worker (already deployed)
E2E_WORKER_URL=https://argus-api.samuelvinay-kumar.workers.dev
```

Then redeploy the dashboard:

```bash
cd dashboard
vercel --prod
```

## 3. Verify Deployment

### Test Brain Health

```bash
curl https://your-argus-brain.railway.app/health
# Should return: {"status": "healthy", "version": "0.1.0", ...}
```

### Test Browser Worker Health

```bash
curl https://argus-api.samuelvinay-kumar.workers.dev/health
# Should return: {"status": "ok", ...}
```

### Test Full Integration

In the dashboard chat, type:
```
Check system status
```

You should see both components connected:
- Argus Brain (Python/LangGraph): connected
- Argus Worker (Cloudflare/Browser): connected

## Environment Variables Reference

### Argus Brain (Python)

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `BROWSER_WORKER_URL` | Yes | URL of Browser Worker |
| `GITHUB_TOKEN` | No | GitHub token for PR integration |
| `DATABASE_URL` | No | PostgreSQL connection string |
| `SLACK_WEBHOOK_URL` | No | Slack notifications |
| `COST_LIMIT_PER_RUN` | No | Max cost per run (default: $10) |

### Dashboard (Next.js)

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `ARGUS_BACKEND_URL` | Yes | URL of Argus Brain |
| `E2E_WORKER_URL` | Yes | URL of Browser Worker |
| `NEXT_PUBLIC_CLERK_*` | Yes | Clerk auth config |
| `NEXT_PUBLIC_SUPABASE_*` | No | Supabase config |

### Browser Worker (Cloudflare)

Set via `wrangler secret put`:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `OPENAI_API_KEY` | No | OpenAI API key |
| `TESTINGBOT_KEY` | No | TestingBot credentials |
| `TESTINGBOT_SECRET` | No | TestingBot credentials |

## Deployment Checklist

- [ ] Deploy Argus Brain to Railway/Render/Fly.io
- [ ] Set all required environment variables
- [ ] Test Brain health endpoint
- [ ] Update dashboard with Brain URL
- [ ] Redeploy dashboard
- [ ] Test full integration via chat
- [ ] Verify MCP Server works with Claude Code

## Troubleshooting

### Brain Not Connecting

1. Check if Brain is deployed: `curl $ARGUS_BACKEND_URL/health`
2. Verify CORS is enabled (should be by default)
3. Check environment variables are set correctly

### Browser Actions Failing

1. Check Worker health: `curl $E2E_WORKER_URL/health`
2. Verify Anthropic API key is set in Worker
3. Check Cloudflare dashboard for errors

### MCP Server Not Working

1. Verify MCP config in Claude Code/Cursor
2. Test SSE endpoint: `curl https://argus-mcp.samuelvinay-kumar.workers.dev/sse`
3. Check MCP server logs in Cloudflare dashboard

## Cost Considerations

| Component | Estimated Cost |
|-----------|----------------|
| Railway (Brain) | $5-20/month |
| Cloudflare Workers | Free tier (100k req/day) |
| Vercel (Dashboard) | Free tier |
| Anthropic API | Pay-per-use |

Total infrastructure cost: **~$5-20/month** (excluding API usage)

## Scaling

### Horizontal Scaling

- **Brain**: Railway/Render auto-scale based on traffic
- **Workers**: Cloudflare edge locations globally
- **Dashboard**: Vercel edge functions

### Performance Tips

1. Enable self-healing caching in Browser Worker
2. Use Vectorize for pattern learning (already configured)
3. Set appropriate timeouts for long-running tests
4. Use background jobs for large test suites
