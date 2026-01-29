#!/bin/bash
# Setup script for api.heyargus.ai (API Documentation)
# This creates a Cloudflare Pages project and configures the custom domain

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Argus API Documentation Setup (api.heyargus.ai)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
check_prereqs() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    if ! command -v npx &> /dev/null; then
        echo -e "${RED}Error: npx not found. Install Node.js first.${NC}"
        exit 1
    fi

    if ! npx wrangler --version &> /dev/null; then
        echo -e "${YELLOW}Installing wrangler...${NC}"
        npm install -g wrangler
    fi

    echo -e "${GREEN}✓ Prerequisites OK${NC}\n"
}

# Login to Cloudflare
login_cloudflare() {
    echo -e "${YELLOW}Step 1: Cloudflare Authentication${NC}"
    echo "If not logged in, a browser will open for authentication."
    echo ""

    if ! npx wrangler whoami &> /dev/null; then
        npx wrangler login
    else
        echo -e "${GREEN}✓ Already logged in to Cloudflare${NC}"
        npx wrangler whoami
    fi
    echo ""
}

# Create Pages project
create_pages_project() {
    echo -e "${YELLOW}Step 2: Create Cloudflare Pages Project${NC}"

    PROJECT_NAME="argus-api-docs"

    # Check if project exists
    if npx wrangler pages project list 2>/dev/null | grep -q "$PROJECT_NAME"; then
        echo -e "${GREEN}✓ Project '$PROJECT_NAME' already exists${NC}"
    else
        echo "Creating Pages project: $PROJECT_NAME"
        npx wrangler pages project create "$PROJECT_NAME" --production-branch main 2>/dev/null || true
        echo -e "${GREEN}✓ Project created${NC}"
    fi
    echo ""
}

# Generate sample docs for initial deployment
generate_sample_docs() {
    echo -e "${YELLOW}Step 3: Generate Initial Documentation${NC}"

    # Create api-docs directory
    mkdir -p api-docs/v1

    # Check if we have openapi.json
    if [ -f "openapi.json" ]; then
        echo "Found openapi.json, generating docs..."

        # Install redocly if not present
        if ! command -v redocly &> /dev/null; then
            npm install -g @redocly/cli
        fi

        # Generate docs
        redocly build-docs openapi.json -o api-docs/v1/index.html
        cp openapi.json api-docs/v1/openapi.json
        echo -e "${GREEN}✓ Generated ReDoc documentation${NC}"
    else
        echo -e "${YELLOW}openapi.json not found, creating placeholder...${NC}"
        cat > api-docs/v1/index.html << 'PLACEHOLDER'
<!DOCTYPE html>
<html>
<head>
    <title>Argus API v1 - Coming Soon</title>
    <style>
        body { font-family: system-ui; background: #1a1a2e; color: white;
               display: flex; justify-content: center; align-items: center;
               height: 100vh; margin: 0; }
        .container { text-align: center; }
        h1 { background: linear-gradient(90deg, #00d4ff, #7b2ff7);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Argus API v1</h1>
        <p>Documentation coming soon...</p>
        <p style="color: #666;">Run the GitHub Actions workflow to deploy full docs.</p>
    </div>
</body>
</html>
PLACEHOLDER
        echo -e "${GREEN}✓ Created placeholder page${NC}"
    fi

    # Create version index
    cat > api-docs/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Argus API Documentation</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
               min-height: 100vh; color: #fff; padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { font-size: 2.5rem; margin-bottom: 0.5rem;
             background: linear-gradient(90deg, #00d4ff, #7b2ff7);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #8892b0; margin-bottom: 3rem; }
        .version-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
                        border-left: 4px solid #00d4ff; transition: all 0.3s ease; }
        .version-card:hover { background: rgba(255,255,255,0.08); transform: translateY(-2px); }
        .version-name { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; }
        .badge { font-size: 0.75rem; padding: 0.25rem 0.75rem; border-radius: 20px;
                 background: #00d4ff; color: #000; margin-left: 1rem; }
        .version-desc { color: #8892b0; margin-bottom: 1rem; }
        .links a { color: #00d4ff; text-decoration: none; margin-right: 1.5rem; }
        .links a:hover { text-decoration: underline; }
        .footer { margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1);
                  color: #8892b0; font-size: 0.875rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Argus API</h1>
        <p class="subtitle">AI-Powered E2E Testing Platform</p>

        <div class="version-card">
            <div class="version-name">v1 <span class="badge">Current</span></div>
            <p class="version-desc">Latest stable API version with full feature support.</p>
            <div class="links">
                <a href="/v1/">Documentation</a>
                <a href="/v1/openapi.json">OpenAPI Spec</a>
            </div>
        </div>

        <div class="footer">
            <p>Base URL: <code>https://app.heyargus.ai/api/v1/</code></p>
            <p style="margin-top: 0.5rem;">
                <a href="https://heyargus.ai" style="color: #00d4ff;">heyargus.ai</a>
            </p>
        </div>
    </div>
</body>
</html>
EOF

    echo -e "${GREEN}✓ Created version index page${NC}"
    echo ""
}

# Deploy to Cloudflare Pages
deploy_pages() {
    echo -e "${YELLOW}Step 4: Deploy to Cloudflare Pages${NC}"

    npx wrangler pages deploy api-docs --project-name=argus-api-docs

    echo -e "${GREEN}✓ Deployed successfully!${NC}"
    echo ""
}

# Configure custom domain
configure_domain() {
    echo -e "${YELLOW}Step 5: Configure Custom Domain${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}MANUAL STEP REQUIRED:${NC}"
    echo ""
    echo "1. Go to Cloudflare Dashboard:"
    echo "   https://dash.cloudflare.com → Pages → argus-api-docs"
    echo ""
    echo "2. Click 'Custom domains' tab"
    echo ""
    echo "3. Click 'Set up a custom domain'"
    echo ""
    echo "4. Enter: api.heyargus.ai"
    echo ""
    echo "5. Cloudflare will auto-configure DNS (since you own heyargus.ai on CF)"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Add GitHub secrets instructions
github_secrets() {
    echo -e "${YELLOW}Step 6: GitHub Actions Setup${NC}"
    echo ""
    echo "Add these secrets to GitHub (Settings → Secrets → Actions):"
    echo ""
    echo "  CLOUDFLARE_API_TOKEN"
    echo "  └── Create at: https://dash.cloudflare.com/profile/api-tokens"
    echo "  └── Template: 'Edit Cloudflare Workers' (includes Pages)"
    echo ""
    echo "  CLOUDFLARE_ACCOUNT_ID"
    echo "  └── Find at: https://dash.cloudflare.com → Right sidebar"
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
    echo "  • Cloudflare Pages: https://argus-api-docs.pages.dev"
    echo "  • Custom Domain:    https://api.heyargus.ai (after DNS setup)"
    echo ""
    echo "Auto-deployment:"
    echo "  • Push to main branch → Triggers GitHub Actions → Updates docs"
    echo ""
    echo "Next time you push API changes, docs will auto-update!"
}

# Main execution
main() {
    check_prereqs
    login_cloudflare
    create_pages_project
    generate_sample_docs
    deploy_pages
    configure_domain
    github_secrets
    summary
}

main "$@"
