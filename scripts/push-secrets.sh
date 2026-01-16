#!/bin/bash
# Secure Secret Migration Script
# This script reads your local .env files and pushes secrets to Vercel and Cloudflare
#
# IMPORTANT: Run this BEFORE revoking your API keys!
#
# Usage: ./scripts/push-secrets.sh

set -e

echo "🔐 Argus Secret Migration Script"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_DIR="$ROOT_DIR/dashboard"
WORKER_DIR="$ROOT_DIR/cloudflare-worker"

# Load environment variables from .env files
load_env() {
    local env_file="$1"
    if [[ -f "$env_file" ]]; then
        echo -e "${GREEN}Loading $env_file${NC}"
        set -a
        source "$env_file"
        set +a
        return 0
    else
        echo -e "${YELLOW}Warning: $env_file not found${NC}"
        return 1
    fi
}

# Push secret to Cloudflare Workers
push_to_cloudflare() {
    local name="$1"
    local value="$2"

    if [[ -z "$value" || "$value" == "sk-..."* || "$value" == "AIza..."* || "$value" == "your-"* ]]; then
        echo -e "${YELLOW}  Skipping $name (not set or placeholder)${NC}"
        return
    fi

    echo -e "  Pushing $name to Cloudflare Workers..."
    echo "$value" | npx wrangler secret put "$name" --cwd "$WORKER_DIR" 2>/dev/null && \
        echo -e "${GREEN}  ✓ $name pushed to Cloudflare${NC}" || \
        echo -e "${RED}  ✗ Failed to push $name to Cloudflare${NC}"
}

# Push secret to Vercel
push_to_vercel() {
    local name="$1"
    local value="$2"
    local env="${3:-production}"  # production, preview, development

    if [[ -z "$value" || "$value" == "sk-..."* || "$value" == "AIza..."* || "$value" == "your-"* ]]; then
        echo -e "${YELLOW}  Skipping $name (not set or placeholder)${NC}"
        return
    fi

    echo -e "  Pushing $name to Vercel ($env)..."
    cd "$DASHBOARD_DIR"
    echo "$value" | npx vercel env add "$name" "$env" 2>/dev/null && \
        echo -e "${GREEN}  ✓ $name pushed to Vercel ($env)${NC}" || \
        echo -e "${YELLOW}  ⚠ $name may already exist in Vercel${NC}"
    cd "$ROOT_DIR"
}

echo "Step 1: Loading environment files..."
echo "-------------------------------------"
load_env "$ROOT_DIR/.env" || true
load_env "$DASHBOARD_DIR/.env.local" || true

echo ""
echo "Step 2: Pushing secrets to Cloudflare Workers..."
echo "-------------------------------------------------"

# Cloudflare Worker Secrets
push_to_cloudflare "ANTHROPIC_API_KEY" "$ANTHROPIC_API_KEY"
push_to_cloudflare "OPENAI_API_KEY" "$OPENAI_API_KEY"
push_to_cloudflare "GOOGLE_API_KEY" "$GOOGLE_API_KEY"
push_to_cloudflare "SUPABASE_URL" "$SUPABASE_URL"
push_to_cloudflare "SUPABASE_SERVICE_KEY" "$SUPABASE_SERVICE_KEY"
push_to_cloudflare "GITHUB_TOKEN" "$GITHUB_TOKEN"
push_to_cloudflare "TESTINGBOT_KEY" "$TESTINGBOT_KEY"
push_to_cloudflare "TESTINGBOT_SECRET" "$TESTINGBOT_SECRET"

echo ""
echo "Step 3: Pushing secrets to Vercel (Dashboard)..."
echo "-------------------------------------------------"

# Vercel Dashboard Secrets (push to all environments)
for env in production preview development; do
    echo ""
    echo "Environment: $env"
    push_to_vercel "NEXT_PUBLIC_SUPABASE_URL" "$SUPABASE_URL" "$env"
    push_to_vercel "NEXT_PUBLIC_SUPABASE_ANON_KEY" "$SUPABASE_ANON_KEY" "$env"
    push_to_vercel "SUPABASE_SERVICE_ROLE_KEY" "$SUPABASE_SERVICE_KEY" "$env"
    push_to_vercel "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY" "$NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY" "$env"
    push_to_vercel "CLERK_SECRET_KEY" "$CLERK_SECRET_KEY" "$env"
    push_to_vercel "ANTHROPIC_API_KEY" "$ANTHROPIC_API_KEY" "$env"
    push_to_vercel "OPENAI_API_KEY" "$OPENAI_API_KEY" "$env"
    push_to_vercel "GOOGLE_API_KEY" "$GOOGLE_API_KEY" "$env"
done

echo ""
echo "================================================"
echo -e "${GREEN}✓ Secret migration complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Verify secrets in Vercel dashboard: https://vercel.com/dashboard"
echo "2. Verify secrets in Cloudflare: npx wrangler secret list --cwd $WORKER_DIR"
echo "3. Test your deployments"
echo "4. THEN rotate your API keys in each provider's console"
echo ""
echo -e "${RED}⚠️  IMPORTANT: Do NOT commit this script output to git!${NC}"
