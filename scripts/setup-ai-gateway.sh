#!/bin/bash
# ARGUS - Cloudflare AI Gateway Setup
# Creates and configures AI Gateway for LLM response caching
#
# Prerequisites:
#   - CLOUDFLARE_API_TOKEN environment variable (with AI Gateway permissions)
#   - CLOUDFLARE_ACCOUNT_ID environment variable
#
# Usage:
#   ./scripts/setup-ai-gateway.sh
#   ./scripts/setup-ai-gateway.sh --gateway-id custom-gateway-name
#   ./scripts/setup-ai-gateway.sh --cache-ttl 7200

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default configuration
GATEWAY_ID="${CLOUDFLARE_AI_GATEWAY_ID:-argus-gateway}"
CACHE_TTL="${AI_GATEWAY_CACHE_TTL:-3600}"  # 1 hour default
COLLECT_LOGS="true"
CACHE_INVALIDATE_ON_UPDATE="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gateway-id)
            GATEWAY_ID="$2"
            shift 2
            ;;
        --cache-ttl)
            CACHE_TTL="$2"
            shift 2
            ;;
        --no-logs)
            COLLECT_LOGS="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gateway-id NAME   Gateway identifier (default: argus-gateway)"
            echo "  --cache-ttl SECS    Cache TTL in seconds (default: 3600)"
            echo "  --no-logs           Disable request logging"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BLUE}🚀 ARGUS AI Gateway Setup${NC}"
echo "=================================="
echo ""

# Check for required environment variables
if [[ -z "$CLOUDFLARE_API_TOKEN" ]]; then
    echo -e "${RED}Error: CLOUDFLARE_API_TOKEN environment variable is required${NC}"
    echo ""
    echo "Create an API token at: https://dash.cloudflare.com/profile/api-tokens"
    echo "Required permissions: AI Gateway:Edit"
    echo ""
    echo "Then set it:"
    echo "  export CLOUDFLARE_API_TOKEN=your_token_here"
    exit 1
fi

if [[ -z "$CLOUDFLARE_ACCOUNT_ID" ]]; then
    echo -e "${YELLOW}CLOUDFLARE_ACCOUNT_ID not set, attempting to fetch from wrangler...${NC}"

    # Try to get account ID from wrangler
    if command -v wrangler &> /dev/null; then
        ACCOUNT_INFO=$(wrangler whoami 2>/dev/null || true)
        CLOUDFLARE_ACCOUNT_ID=$(echo "$ACCOUNT_INFO" | grep -oP '(?<=Account ID: )[a-f0-9]+' | head -1 || true)

        if [[ -z "$CLOUDFLARE_ACCOUNT_ID" ]]; then
            # Alternative: try to extract from wrangler config
            CLOUDFLARE_ACCOUNT_ID=$(wrangler whoami --json 2>/dev/null | jq -r '.accounts[0].id' 2>/dev/null || true)
        fi
    fi

    if [[ -z "$CLOUDFLARE_ACCOUNT_ID" ]]; then
        echo -e "${RED}Error: Could not determine CLOUDFLARE_ACCOUNT_ID${NC}"
        echo ""
        echo "Set it manually:"
        echo "  export CLOUDFLARE_ACCOUNT_ID=your_account_id"
        echo ""
        echo "Find your Account ID at: https://dash.cloudflare.com → Overview → Account ID"
        exit 1
    fi

    echo -e "${GREEN}✓ Found Account ID: ${CLOUDFLARE_ACCOUNT_ID:0:8}...${NC}"
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  Gateway ID: $GATEWAY_ID"
echo "  Cache TTL: ${CACHE_TTL}s"
echo "  Collect Logs: $COLLECT_LOGS"
echo "  Account ID: ${CLOUDFLARE_ACCOUNT_ID:0:8}..."
echo ""

# Check if gateway already exists
echo -e "${BLUE}Step 1: Checking if gateway exists...${NC}"
EXISTING_GATEWAY=$(curl -s -X GET \
    "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai-gateway/gateways/${GATEWAY_ID}" \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    -H "Content-Type: application/json")

GATEWAY_EXISTS=$(echo "$EXISTING_GATEWAY" | jq -r '.success' 2>/dev/null || echo "false")

if [[ "$GATEWAY_EXISTS" == "true" ]]; then
    echo -e "${YELLOW}Gateway '${GATEWAY_ID}' already exists. Updating configuration...${NC}"

    # Update existing gateway
    RESPONSE=$(curl -s -X PUT \
        "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai-gateway/gateways/${GATEWAY_ID}" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"cache_invalidate_on_update\": ${CACHE_INVALIDATE_ON_UPDATE},
            \"cache_ttl\": ${CACHE_TTL},
            \"collect_logs\": ${COLLECT_LOGS},
            \"rate_limiting_interval\": 60,
            \"rate_limiting_limit\": 1000,
            \"rate_limiting_technique\": \"sliding\"
        }")

    ACTION="updated"
else
    echo -e "${BLUE}Step 2: Creating new AI Gateway...${NC}"

    # Create new gateway
    RESPONSE=$(curl -s -X POST \
        "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai-gateway/gateways" \
        -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"id\": \"${GATEWAY_ID}\",
            \"cache_invalidate_on_update\": ${CACHE_INVALIDATE_ON_UPDATE},
            \"cache_ttl\": ${CACHE_TTL},
            \"collect_logs\": ${COLLECT_LOGS},
            \"rate_limiting_interval\": 60,
            \"rate_limiting_limit\": 1000,
            \"rate_limiting_technique\": \"sliding\"
        }")

    ACTION="created"
fi

# Check response
SUCCESS=$(echo "$RESPONSE" | jq -r '.success' 2>/dev/null || echo "false")

if [[ "$SUCCESS" == "true" ]]; then
    echo -e "${GREEN}✓ Gateway ${ACTION} successfully!${NC}"
    echo ""

    # Extract gateway details
    GATEWAY_URL="https://gateway.ai.cloudflare.com/v1/${CLOUDFLARE_ACCOUNT_ID}/${GATEWAY_ID}"

    echo -e "${BLUE}Gateway Details:${NC}"
    echo "=================================="
    echo ""
    echo "Gateway ID: $GATEWAY_ID"
    echo "Base URL:   $GATEWAY_URL"
    echo ""
    echo -e "${BLUE}Provider-specific URLs:${NC}"
    echo "  Anthropic: ${GATEWAY_URL}/anthropic"
    echo "  OpenAI:    ${GATEWAY_URL}/openai"
    echo "  Workers AI: ${GATEWAY_URL}/workers-ai"
    echo ""
    echo -e "${BLUE}Universal Endpoint:${NC}"
    echo "  ${GATEWAY_URL}/universal"
    echo ""

    # Show environment variable to set
    echo -e "${YELLOW}Add these to your .env file:${NC}"
    echo ""
    echo "CLOUDFLARE_AI_GATEWAY_ID=${GATEWAY_ID}"
    echo "CLOUDFLARE_AI_GATEWAY_URL=${GATEWAY_URL}"
    echo ""

    # Show usage example
    echo -e "${BLUE}Usage Example (Python):${NC}"
    echo ""
    cat << 'EOF'
import anthropic

# Route through AI Gateway for caching
client = anthropic.Anthropic(
    base_url=f"https://gateway.ai.cloudflare.com/v1/{ACCOUNT_ID}/argus-gateway/anthropic"
)

# Requests are now cached at the edge!
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
EOF
    echo ""

    # Show dashboard link
    echo -e "${BLUE}View in Dashboard:${NC}"
    echo "  https://dash.cloudflare.com/${CLOUDFLARE_ACCOUNT_ID}/ai-gateway/${GATEWAY_ID}"
    echo ""

    echo -e "${GREEN}✓ AI Gateway setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Update src/services/cloudflare_storage.py with gateway config"
    echo "  2. Set CLOUDFLARE_AI_GATEWAY_ID in your environment"
    echo "  3. Monitor cache hit rates in the Cloudflare dashboard"

else
    echo -e "${RED}✗ Failed to ${ACTION/d/} gateway${NC}"
    echo ""
    echo "Error response:"
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    echo ""

    # Common error handling
    ERROR_CODE=$(echo "$RESPONSE" | jq -r '.errors[0].code' 2>/dev/null || echo "")
    ERROR_MSG=$(echo "$RESPONSE" | jq -r '.errors[0].message' 2>/dev/null || echo "")

    if [[ "$ERROR_CODE" == "10000" ]]; then
        echo -e "${YELLOW}Hint: Check your API token permissions.${NC}"
        echo "Required: AI Gateway:Edit"
    elif [[ "$ERROR_MSG" == *"already exists"* ]]; then
        echo -e "${YELLOW}Hint: Gateway already exists. Try updating instead.${NC}"
    fi

    exit 1
fi
