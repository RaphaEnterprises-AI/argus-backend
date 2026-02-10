#!/bin/bash
# ARGUS Documentation Deployment Script
# Deploys to Cloudflare Pages with Wrangler

set -e

echo "🚀 ARGUS Documentation Deployment"
echo "=================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if logged in to Cloudflare
echo -e "${BLUE}Checking Cloudflare authentication...${NC}"
if ! npx wrangler whoami &>/dev/null; then
    echo -e "${YELLOW}Not logged in. Starting login...${NC}"
    npx wrangler login
fi

echo -e "${GREEN}✓ Authenticated with Cloudflare${NC}"

# Build MkDocs
echo -e "${BLUE}Building documentation...${NC}"
pip install mkdocs-material mkdocs-minify-plugin -q
mkdocs build

echo -e "${GREEN}✓ Documentation built successfully${NC}"

# Deploy to Cloudflare Pages
echo -e "${BLUE}Deploying to Cloudflare Pages...${NC}"
npx wrangler pages deploy site --project-name=argus-docs --branch=main

echo ""
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Go to Cloudflare Dashboard → Workers & Pages → argus-docs"
echo "2. Add custom domain: docs.skopaq.ai"
echo "3. Set up Cloudflare Access for private access"
echo ""
echo "Run: ./scripts/setup-access.sh to configure private access"
