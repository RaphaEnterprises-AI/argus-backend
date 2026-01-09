# Argus Deployment Checklist

## Prerequisites

- [ ] PostgreSQL database (Supabase recommended)
- [ ] Anthropic API key
- [ ] OpenAI API key (for embeddings)
- [ ] Cloudflare Workers account (for browser automation)

## Database Setup

1. Run migrations:
   ```bash
   # Using Supabase CLI
   supabase db push

   # Or manually
   psql $DATABASE_URL -f supabase/migrations/20260109000000_langgraph_checkpoints.sql
   psql $DATABASE_URL -f supabase/migrations/20260109000001_langgraph_memory_store.sql
   ```

2. Enable pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. Verify tables created:
   - langgraph_checkpoints
   - langgraph_checkpoint_writes
   - langgraph_memory_store
   - test_failure_patterns

## Environment Variables

Required:
- [ ] ANTHROPIC_API_KEY
- [ ] DATABASE_URL
- [ ] SUPABASE_URL
- [ ] SUPABASE_SERVICE_KEY

Optional:
- [ ] OPENAI_API_KEY (for embeddings)
- [ ] ENABLE_HUMAN_APPROVAL
- [ ] E2E_WORKER_URL

## Backend Deployment (Railway)

1. Connect repository
2. Set environment variables
3. Deploy with:
   ```bash
   python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   ```

## Dashboard Deployment (Vercel)

1. Connect dashboard directory
2. Set environment variables:
   - NEXT_PUBLIC_SUPABASE_URL
   - NEXT_PUBLIC_SUPABASE_ANON_KEY
   - ARGUS_BACKEND_URL
3. Deploy

## Browser Worker Deployment (Cloudflare)

1. Navigate to argus-mcp-server
2. Deploy:
   ```bash
   wrangler deploy
   ```

## Verification

- [ ] Health check: GET /health
- [ ] Run test: POST /api/v1/test/run
- [ ] Stream test: POST /api/v1/stream/test
- [ ] Time travel: GET /api/v1/time-travel/history/{thread_id}

## Monitoring

- [ ] Set up Supabase monitoring
- [ ] Configure Railway logs
- [ ] Set up Cloudflare analytics
