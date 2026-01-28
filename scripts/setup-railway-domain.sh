#!/bin/bash
# Setup custom domain api.heyargus.ai for Railway backend
# This configures both Railway and Cloudflare DNS

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Setup api.heyargus.ai → Railway Backend                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for Railway CLI
check_railway() {
    echo -e "${YELLOW}Checking Railway CLI...${NC}"
    if ! command -v railway &> /dev/null; then
        echo -e "${RED}Railway CLI not found. Installing...${NC}"
        npm install -g @railway/cli
    fi
    echo -e "${GREEN}✓ Railway CLI installed${NC}"
    railway --version
    echo ""
}

# Login and link project
railway_login() {
    echo -e "${YELLOW}Step 1: Railway Authentication${NC}"

    if ! railway whoami &> /dev/null 2>&1; then
        echo "Opening browser for Railway login..."
        railway login
    else
        echo -e "${GREEN}✓ Already logged in:${NC}"
        railway whoami
    fi
    echo ""

    echo -e "${YELLOW}Linking to project...${NC}"
    railway link || {
        echo -e "${RED}Please select your Railway project when prompted${NC}"
        railway link
    }
    echo ""
}

# Add custom domain to Railway
add_railway_domain() {
    echo -e "${YELLOW}Step 2: Add Custom Domain to Railway${NC}"

    DOMAIN="api.heyargus.ai"

    echo "Adding domain: $DOMAIN to argus-backend service..."

    # Try to add domain
    railway domain add $DOMAIN --service argus-backend 2>/dev/null || {
        echo -e "${YELLOW}Domain might already exist or needs manual setup.${NC}"
    }

    # Get the CNAME target
    echo ""
    echo -e "${GREEN}Railway domain configuration:${NC}"
    railway domain list --service argus-backend 2>/dev/null || echo "Run 'railway domain list' to see domains"
    echo ""
}

# Cloudflare DNS setup instructions
cloudflare_dns() {
    echo -e "${YELLOW}Step 3: Configure Cloudflare DNS${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Add a CNAME record in Cloudflare DNS:"
    echo ""
    echo "  Type:    CNAME"
    echo "  Name:    api"
    echo "  Target:  argus-brain-production.up.railway.app"
    echo "  Proxy:   OFF (DNS only - grey cloud)"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Proxy must be OFF for Railway SSL to work!${NC}"
    echo ""
    echo "Cloudflare Dashboard URL:"
    echo "  https://dash.cloudflare.com → heyargus.ai → DNS → Add record"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Automated Cloudflare DNS setup (if token available)
cloudflare_auto_dns() {
    if [ -n "$CLOUDFLARE_API_TOKEN" ]; then
        echo -e "${YELLOW}Attempting automatic Cloudflare DNS setup...${NC}"

        # Get zone ID
        ZONE_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=heyargus.ai" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
            -H "Content-Type: application/json" | jq -r '.result[0].id')

        if [ "$ZONE_ID" != "null" ] && [ -n "$ZONE_ID" ]; then
            echo "Found zone ID: $ZONE_ID"

            # Check if record exists
            EXISTING=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=api.heyargus.ai" \
                -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
                -H "Content-Type: application/json" | jq -r '.result[0].id')

            if [ "$EXISTING" != "null" ] && [ -n "$EXISTING" ]; then
                echo -e "${YELLOW}DNS record already exists. Updating...${NC}"
                curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$EXISTING" \
                    -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
                    -H "Content-Type: application/json" \
                    --data '{
                        "type": "CNAME",
                        "name": "api",
                        "content": "argus-brain-production.up.railway.app",
                        "proxied": false,
                        "ttl": 1
                    }' | jq -r 'if .success then "✓ DNS record updated" else "✗ Error: " + (.errors[0].message // "Unknown") end'
            else
                echo "Creating new DNS record..."
                curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
                    -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
                    -H "Content-Type: application/json" \
                    --data '{
                        "type": "CNAME",
                        "name": "api",
                        "content": "argus-brain-production.up.railway.app",
                        "proxied": false,
                        "ttl": 1
                    }' | jq -r 'if .success then "✓ DNS record created" else "✗ Error: " + (.errors[0].message // "Unknown") end'
            fi
            echo ""
        else
            echo -e "${RED}Could not find zone for heyargus.ai${NC}"
        fi
    else
        echo -e "${YELLOW}CLOUDFLARE_API_TOKEN not set. Manual DNS setup required.${NC}"
        cloudflare_dns
    fi
}

# Update codebase references
update_codebase() {
    echo -e "${YELLOW}Step 4: Update Codebase References${NC}"
    echo ""
    echo "The following files reference the Railway URL:"
    echo ""
    grep -r "argus-brain-production.up.railway.app" --include="*.py" --include="*.ts" --include="*.json" --include="*.md" --include="*.yaml" . 2>/dev/null | head -20 || true
    echo ""
    echo -e "${YELLOW}Consider updating these to use api.heyargus.ai${NC}"
    echo ""
}

# Verify setup
verify_setup() {
    echo -e "${YELLOW}Step 5: Verify Setup${NC}"
    echo ""
    echo "Waiting for DNS propagation (this may take a few minutes)..."
    echo ""

    # Check DNS
    echo "Checking DNS resolution..."
    DNS_RESULT=$(dig api.heyargus.ai +short 2>/dev/null || echo "")
    if [ -n "$DNS_RESULT" ]; then
        echo -e "${GREEN}✓ DNS resolves: api.heyargus.ai → $DNS_RESULT${NC}"
    else
        echo -e "${YELLOW}⏳ DNS not propagated yet. Check again in a few minutes.${NC}"
    fi
    echo ""

    # Test endpoint
    echo "Testing API endpoint..."
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://api.heyargus.ai/health" 2>/dev/null || echo "000")
    if [ "$HTTP_STATUS" = "200" ]; then
        echo -e "${GREEN}✓ API responding: https://api.heyargus.ai/health (HTTP $HTTP_STATUS)${NC}"
    else
        echo -e "${YELLOW}⏳ API not ready yet (HTTP $HTTP_STATUS). SSL certificate may be provisioning.${NC}"
    fi
    echo ""
}

# Summary
summary() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Setup Complete!                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "URLs:"
    echo "  • Old: https://argus-brain-production.up.railway.app"
    echo "  • New: https://api.heyargus.ai"
    echo ""
    echo "API Endpoints:"
    echo "  • Health:  https://api.heyargus.ai/health"
    echo "  • API v1:  https://api.heyargus.ai/api/v1/"
    echo "  • Docs:    https://api.heyargus.ai/docs"
    echo ""
    echo "SSL Certificate:"
    echo "  Railway will automatically provision a Let's Encrypt certificate."
    echo "  This may take 2-5 minutes after DNS is configured."
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Update environment variables to use api.heyargus.ai"
    echo "  2. Update frontend/dashboard to point to new URL"
    echo "  3. Update MCP server configuration"
    echo "  4. (Optional) Set up docs.heyargus.ai for API documentation"
}

# Main
main() {
    check_railway
    railway_login
    add_railway_domain
    cloudflare_auto_dns
    update_codebase
    verify_setup
    summary
}

main "$@"
