#!/bin/bash
# ARGUS Documentation - Cloudflare Access Setup
# Sets up private access control for docs.heyargus.ai

set -e

echo "🔒 ARGUS Documentation Access Control Setup"
echo "============================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
DOMAIN="heyargus.ai"
SUBDOMAIN="docs"
APP_NAME="ARGUS Documentation"

echo ""
echo -e "${YELLOW}This script will help you set up Cloudflare Access.${NC}"
echo -e "${YELLOW}You'll need to complete some steps in the Cloudflare Dashboard.${NC}"
echo ""

# Check authentication
echo -e "${BLUE}Step 1: Checking Cloudflare authentication...${NC}"
if ! npx wrangler whoami &>/dev/null; then
    echo -e "${YELLOW}Not logged in. Starting login...${NC}"
    npx wrangler login
fi
echo -e "${GREEN}✓ Authenticated${NC}"

# Get account info
echo ""
echo -e "${BLUE}Step 2: Getting account information...${NC}"
ACCOUNT_INFO=$(npx wrangler whoami 2>/dev/null)
echo "$ACCOUNT_INFO"

echo ""
echo -e "${BLUE}Step 3: Manual Cloudflare Access Setup${NC}"
echo "========================================"
echo ""
echo "Cloudflare Access (Zero Trust) must be configured via the dashboard."
echo "Follow these steps:"
echo ""
echo -e "${YELLOW}1. Go to Cloudflare Zero Trust Dashboard:${NC}"
echo "   https://one.dash.cloudflare.com"
echo ""
echo -e "${YELLOW}2. Navigate to Access → Applications${NC}"
echo ""
echo -e "${YELLOW}3. Click 'Add an application' → 'Self-hosted'${NC}"
echo ""
echo -e "${YELLOW}4. Configure the application:${NC}"
echo "   • Application name: $APP_NAME"
echo "   • Session duration: 24 hours"
echo "   • Subdomain: $SUBDOMAIN"
echo "   • Domain: $DOMAIN"
echo ""
echo -e "${YELLOW}5. Create an access policy:${NC}"
echo "   • Policy name: Only Me"
echo "   • Action: Allow"
echo "   • Include → Emails → [YOUR EMAIL]"
echo ""
echo -e "${YELLOW}6. Click 'Add application'${NC}"
echo ""
echo "========================================"
echo ""
read -p "Press Enter when you've completed the setup..."

echo ""
echo -e "${GREEN}✓ Access control setup complete!${NC}"
echo ""
echo "Your documentation is now private at:"
echo -e "${BLUE}https://docs.heyargus.ai${NC}"
echo ""
echo "Only authenticated users in your Access policy can view it."
