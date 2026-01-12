# Railway Deployment Guide

This guide explains how to deploy Argus on Railway with a microservice architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAILWAY PROJECT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────┐     ┌─────────────────────────┐       │
│  │   argus-backend         │     │   argus-crawlee         │       │
│  │   (Python/FastAPI)      │────▶│   (Node.js/Crawlee)     │       │
│  │   Port: 8000            │     │   Port: 3000            │       │
│  │   ~200MB image          │     │   ~1.5GB image          │       │
│  └─────────────────────────┘     └─────────────────────────┘       │
│             │                              │                        │
│             │                              │                        │
│             ▼                              ▼                        │
│  ┌─────────────────────────┐     ┌─────────────────────────┐       │
│  │   PostgreSQL            │     │   (Browser in Container)│       │
│  │   (Supabase/Railway)    │     │   Chromium for crawling │       │
│  └─────────────────────────┘     └─────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. Railway CLI installed: `npm install -g @railway/cli`
2. Railway account and project created
3. Environment variables configured

## Deployment Steps

### 1. Create Railway Project

```bash
# Login to Railway
railway login

# Create new project (if not exists)
railway init

# Or link to existing project
railway link
```

### 2. Create Services

Railway supports multiple services in one project. Create two services:

#### Backend Service (argus-backend)

```bash
# From project root
railway service create argus-backend

# Set service to use root Dockerfile
railway service set --name argus-backend

# Set environment variables
railway variables set ENVIRONMENT=production
railway variables set ANTHROPIC_API_KEY=your-key
railway variables set SUPABASE_URL=your-supabase-url
railway variables set SUPABASE_SERVICE_KEY=your-key
railway variables set CRAWLEE_SERVICE_URL=http://argus-crawlee.railway.internal:3000
```

#### Crawlee Service (argus-crawlee)

```bash
# Create crawler service
railway service create argus-crawlee

# Configure to use crawlee-service directory
railway service set --name argus-crawlee --source crawlee-service

# Set environment variables
railway variables set NODE_ENV=production
railway variables set PORT=3000
railway variables set LOG_LEVEL=info
```

### 3. Configure Internal Networking

Railway provides internal networking between services. The backend connects to the Crawlee service via:

```
http://argus-crawlee.railway.internal:3000
```

### 4. Deploy

```bash
# Deploy all services
railway up

# Or deploy specific service
railway up --service argus-backend
railway up --service argus-crawlee
```

## Environment Variables

### Backend (argus-backend)

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service role key |
| `CRAWLEE_SERVICE_URL` | Yes | Internal URL to Crawlee service |
| `ENVIRONMENT` | No | `production` or `development` |
| `LOG_LEVEL` | No | Logging level (default: `info`) |

### Crawlee Service (argus-crawlee)

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | No | Server port (default: 3000) |
| `NODE_ENV` | No | `production` or `development` |
| `LOG_LEVEL` | No | Logging level (default: `info`) |

## API Endpoints

### Crawlee Service

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/crawl/discovery` | POST | Run discovery crawl |
| `/api/v1/capture/screenshot` | POST | Capture screenshot |
| `/api/v1/capture/responsive` | POST | Responsive capture |
| `/api/v1/execute/test` | POST | Execute test |
| `/api/v1/extract/elements` | POST | Extract page elements |

## Scaling

Railway allows horizontal scaling:

```bash
# Scale backend
railway service scale argus-backend --replicas 2

# Scale crawler (more memory-intensive)
railway service scale argus-crawlee --replicas 3
```

## Resource Recommendations

| Service | CPU | Memory | Notes |
|---------|-----|--------|-------|
| argus-backend | 0.5 vCPU | 512MB | Lightweight API |
| argus-crawlee | 1 vCPU | 2GB | Browser requires more RAM |

## Monitoring

Railway provides built-in logging and metrics:

```bash
# View logs
railway logs --service argus-backend
railway logs --service argus-crawlee

# Follow logs
railway logs -f
```

## Troubleshooting

### Crawlee Service Out of Memory

Increase memory allocation in Railway dashboard or use:
```bash
railway service set --memory 4096 --service argus-crawlee
```

### Connection Timeout to Crawlee Service

1. Check service is running: `railway status`
2. Verify internal URL is correct
3. Check Crawlee service logs for errors

### Browser Crashes

The Crawlee container includes Chromium. If browser crashes:
1. Check memory allocation
2. Review screenshot/page limits
3. Consider reducing concurrency in crawler config

## Local Development

For local testing of the microservice architecture:

```bash
# Terminal 1: Start Crawlee service
cd crawlee-service
npm install
npm run dev

# Terminal 2: Start backend
export CRAWLEE_SERVICE_URL=http://localhost:3000
cd ..
python -m uvicorn src.main:app --reload
```

## Docker Compose Alternative

For local development, you can use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CRAWLEE_SERVICE_URL=http://crawlee:3000
    depends_on:
      - crawlee

  crawlee:
    build: ./crawlee-service
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
```
