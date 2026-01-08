#!/bin/bash
# Cloudflare Domain Configuration Script for Argus
# This script configures DNS, redirects, and email routing for heyargus.ai and heyargus.com

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Argus Cloudflare Domain Setup ===${NC}\n"

# Check for required environment variables
if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo -e "${RED}Error: CLOUDFLARE_API_TOKEN is not set${NC}"
    echo "Create a token at: https://dash.cloudflare.com/profile/api-tokens"
    echo "Required permissions: Zone:DNS:Edit, Zone:Zone Settings:Edit, Email Routing:Edit"
    echo ""
    echo "Export it with: export CLOUDFLARE_API_TOKEN=your_token_here"
    exit 1
fi

# Configuration
PRIMARY_DOMAIN="heyargus.ai"
SECONDARY_DOMAIN="heyargus.com"
FORWARD_EMAIL="team@youremail.com"  # Change this to your actual email

# Get Zone IDs
echo -e "${YELLOW}Fetching Zone IDs...${NC}"

get_zone_id() {
    local domain=$1
    curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=$domain" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json" | jq -r '.result[0].id'
}

ZONE_ID_AI=$(get_zone_id $PRIMARY_DOMAIN)
ZONE_ID_COM=$(get_zone_id $SECONDARY_DOMAIN)

if [ "$ZONE_ID_AI" == "null" ] || [ -z "$ZONE_ID_AI" ]; then
    echo -e "${RED}Error: Could not find zone for $PRIMARY_DOMAIN${NC}"
    echo "Make sure the domain is added to your Cloudflare account"
    exit 1
fi

if [ "$ZONE_ID_COM" == "null" ] || [ -z "$ZONE_ID_COM" ]; then
    echo -e "${RED}Error: Could not find zone for $SECONDARY_DOMAIN${NC}"
    echo "Make sure the domain is added to your Cloudflare account"
    exit 1
fi

echo -e "${GREEN}Found Zone IDs:${NC}"
echo "  $PRIMARY_DOMAIN: $ZONE_ID_AI"
echo "  $SECONDARY_DOMAIN: $ZONE_ID_COM"
echo ""

# Function to add DNS record
add_dns_record() {
    local zone_id=$1
    local type=$2
    local name=$3
    local content=$4
    local proxied=${5:-true}

    echo -e "Adding $type record: $name -> $content"

    curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$zone_id/dns_records" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json" \
        --data "{
            \"type\": \"$type\",
            \"name\": \"$name\",
            \"content\": \"$content\",
            \"proxied\": $proxied,
            \"ttl\": 1
        }" | jq -r 'if .success then "  ✓ Success" else "  ✗ Error: " + (.errors[0].message // "Unknown error") end'
}

# Function to setup email routing
setup_email_routing() {
    local zone_id=$1
    local domain=$2
    local forward_to=$3

    echo -e "\n${YELLOW}Setting up email routing for $domain...${NC}"

    # Enable email routing
    curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$zone_id/email/routing/enable" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json" | jq -r 'if .success then "  ✓ Email routing enabled" else "  ✗ Error enabling email routing" end'

    # Add email forwarding rules
    local emails=("hello" "support" "sales" "team" "security" "billing" "privacy" "legal")

    for prefix in "${emails[@]}"; do
        echo -e "  Adding $prefix@$domain -> $forward_to"
        curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$zone_id/email/routing/rules" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
            -H "Content-Type: application/json" \
            --data "{
                \"actions\": [{\"type\": \"forward\", \"value\": [\"$forward_to\"]}],
                \"matchers\": [{\"type\": \"literal\", \"field\": \"to\", \"value\": \"$prefix@$domain\"}],
                \"enabled\": true,
                \"name\": \"Forward $prefix to team\"
            }" | jq -r 'if .success then "    ✓" else "    ✗" end'
    done
}

# Function to create redirect rule (heyargus.com -> heyargus.ai)
create_redirect_rule() {
    local zone_id=$1
    local from_domain=$2
    local to_domain=$3

    echo -e "\n${YELLOW}Creating redirect rule: $from_domain -> $to_domain${NC}"

    # Create a redirect rule using Cloudflare Rules
    curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$zone_id/rulesets" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json" \
        --data "{
            \"name\": \"Redirect to primary domain\",
            \"kind\": \"zone\",
            \"phase\": \"http_request_dynamic_redirect\",
            \"rules\": [{
                \"action\": \"redirect\",
                \"action_parameters\": {
                    \"from_value\": {
                        \"status_code\": 301,
                        \"target_url\": {
                            \"expression\": \"concat(\\\"https://$to_domain\\\", http.request.uri.path)\"
                        },
                        \"preserve_query_string\": true
                    }
                },
                \"expression\": \"true\",
                \"description\": \"Redirect all traffic to $to_domain\"
            }]
        }" | jq -r 'if .success then "  ✓ Redirect rule created" else "  ✗ Error: " + (.errors[0].message // "Unknown error") end'
}

# ============================================
# MAIN SETUP
# ============================================

echo -e "\n${GREEN}=== Step 1: DNS Records for $PRIMARY_DOMAIN ===${NC}"
# Add your server IP or Cloudflare Pages here
# For Cloudflare Pages, you typically use CNAME records
# add_dns_record $ZONE_ID_AI "A" "@" "YOUR_SERVER_IP"
# add_dns_record $ZONE_ID_AI "CNAME" "www" "$PRIMARY_DOMAIN"

echo -e "${YELLOW}Skipping DNS records - configure manually or uncomment above${NC}"

echo -e "\n${GREEN}=== Step 2: DNS Records for $SECONDARY_DOMAIN ===${NC}"
# These point to Cloudflare for the redirect to work
add_dns_record $ZONE_ID_COM "A" "@" "192.0.2.1" true
add_dns_record $ZONE_ID_COM "CNAME" "www" "$SECONDARY_DOMAIN" true

echo -e "\n${GREEN}=== Step 3: Redirect Rule ($SECONDARY_DOMAIN -> $PRIMARY_DOMAIN) ===${NC}"
create_redirect_rule $ZONE_ID_COM $SECONDARY_DOMAIN $PRIMARY_DOMAIN

echo -e "\n${GREEN}=== Step 4: Email Routing ===${NC}"
echo -e "${YELLOW}Note: Change FORWARD_EMAIL in the script to your actual email${NC}"
echo ""

read -p "Enter the email to forward all addresses to: " FORWARD_EMAIL

if [ -n "$FORWARD_EMAIL" ]; then
    setup_email_routing $ZONE_ID_COM $SECONDARY_DOMAIN $FORWARD_EMAIL
    setup_email_routing $ZONE_ID_AI $PRIMARY_DOMAIN $FORWARD_EMAIL
else
    echo -e "${YELLOW}Skipping email setup${NC}"
fi

echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Summary:"
echo "  - $PRIMARY_DOMAIN: Primary domain (configure DNS manually for your hosting)"
echo "  - $SECONDARY_DOMAIN: Redirects to $PRIMARY_DOMAIN"
echo "  - Email addresses forward to: $FORWARD_EMAIL"
echo ""
echo "Email addresses created:"
echo "  - hello@heyargus.com    -> $FORWARD_EMAIL"
echo "  - support@heyargus.com  -> $FORWARD_EMAIL"
echo "  - sales@heyargus.com    -> $FORWARD_EMAIL"
echo "  - team@heyargus.com     -> $FORWARD_EMAIL"
echo "  - security@heyargus.com -> $FORWARD_EMAIL"
echo "  - billing@heyargus.com  -> $FORWARD_EMAIL"
echo "  - privacy@heyargus.com  -> $FORWARD_EMAIL"
echo "  - legal@heyargus.com    -> $FORWARD_EMAIL"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Verify email routing in Cloudflare dashboard"
echo "2. Add the destination email to Cloudflare Email Routing (verification required)"
echo "3. Test by sending an email to hello@heyargus.com"
