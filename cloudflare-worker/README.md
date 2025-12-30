# E2E Testing Agent - Cloudflare Worker

Deploy Stagehand-powered browser automation to Cloudflare's global edge network.

## Quick Deploy (One-Click)

[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/anthropics/e2e-testing-agent/tree/main/cloudflare-worker)

## Manual Deployment

### Prerequisites

1. [Cloudflare Account](https://dash.cloudflare.com/sign-up) (free tier works!)
2. [Node.js 18+](https://nodejs.org/)
3. [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/install-and-update/)

### Step 1: Install Dependencies

```bash
cd cloudflare-worker
npm install
```

### Step 2: Login to Cloudflare

```bash
npx wrangler login
```

### Step 3: (Optional) Configure API Keys

If you want to use OpenAI or Anthropic instead of free Workers AI:

```bash
# For OpenAI
npx wrangler secret put OPENAI_API_KEY
# Enter your OpenAI API key when prompted

# For Anthropic
npx wrangler secret put ANTHROPIC_API_KEY
# Enter your Anthropic API key when prompted

# For API authentication (recommended for production)
npx wrangler secret put API_TOKEN
# Enter a secure token for authenticating requests
```

### Step 4: Deploy

```bash
npm run deploy
```

Your Worker will be deployed to: `https://e2e-testing-agent.<your-subdomain>.workers.dev`

## API Usage

### Run a Test

```bash
curl -X POST https://e2e-testing-agent.your-subdomain.workers.dev/test \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "url": "https://example.com/login",
    "steps": [
      "Click the Sign In button",
      "Type '\''test@example.com'\'' in the email field",
      "Type '\''password123'\'' in the password field",
      "Click Submit"
    ],
    "extract": {
      "welcome_message": "string",
      "username": "string"
    },
    "screenshot": true
  }'
```

Response:
```json
{
  "success": true,
  "steps": [
    { "instruction": "Click the Sign In button", "success": true },
    { "instruction": "Type 'test@example.com' in the email field", "success": true },
    { "instruction": "Type 'password123' in the password field", "success": true },
    { "instruction": "Click Submit", "success": true }
  ],
  "extracted": {
    "welcome_message": "Welcome back, Test User!",
    "username": "test@example.com"
  },
  "screenshot": "base64-encoded-png...",
  "stats": {
    "totalDuration": 4523,
    "cachedActions": 2,
    "healedActions": 0
  }
}
```

### Extract Data from a Page

```bash
curl -X POST https://e2e-testing-agent.your-subdomain.workers.dev/extract \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/product/123",
    "schema": {
      "title": "string",
      "price": "number",
      "in_stock": "boolean",
      "features": "array"
    },
    "instruction": "Extract the main product information"
  }'
```

### AI Page Observation

```bash
curl -X POST https://e2e-testing-agent.your-subdomain.workers.dev/observe \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/checkout",
    "instruction": "What forms and buttons are visible? Are there any error messages?"
  }'
```

### Health Check

```bash
curl https://e2e-testing-agent.your-subdomain.workers.dev/health
```

## Model Providers

### Workers AI (Default - FREE!)

Uses Cloudflare's built-in Workers AI with Llama models. No API key required!

```toml
# wrangler.toml
[vars]
DEFAULT_MODEL_PROVIDER = "workers-ai"
```

### OpenAI

```toml
# wrangler.toml
[vars]
DEFAULT_MODEL_PROVIDER = "openai"
```

```bash
npx wrangler secret put OPENAI_API_KEY
```

### Anthropic (Claude)

```toml
# wrangler.toml
[vars]
DEFAULT_MODEL_PROVIDER = "anthropic"
```

```bash
npx wrangler secret put ANTHROPIC_API_KEY
```

## Observability with AI Gateway

Enable AI Gateway for logging, caching, and cost tracking:

1. Go to [Cloudflare Dashboard → AI → Gateway](https://dash.cloudflare.com/?to=/:account/ai/ai-gateway)
2. Create a new gateway (e.g., `e2e-testing-gateway`)
3. Update `src/index.ts` to use gateway:

```typescript
const stagehand = new Stagehand({
  // ...
  llmClient: new WorkersAIClient(env.AI, {
    gateway: {
      id: "e2e-testing-gateway"
    }
  }),
});
```

## Local Development

```bash
npm run dev
```

This starts a local development server at `http://localhost:8787`

## Limits

### Free Tier
- Browser Rendering: 2,000 requests/month
- Workers AI: 10,000 neurons/day (free)
- Worker Invocations: 100,000/day

### Paid Plan ($5/month Workers Paid)
- Browser Rendering: 20,000 requests/month (included)
- Workers AI: Pay-per-use after free tier
- Worker Invocations: 10 million/month

## Troubleshooting

### "Browser binding not found"

Make sure your `wrangler.toml` includes:

```toml
[browser]
binding = "BROWSER"
```

### "AI binding not found"

Make sure your `wrangler.toml` includes:

```toml
[ai]
binding = "AI"
```

### Timeout errors

Increase the timeout in your request:

```json
{
  "url": "https://slow-site.com",
  "steps": ["..."],
  "timeout": 60000
}
```

## Integration with Main Agent

The Python agent can call this Worker API:

```python
from src.browser import StagehandClient

# Configure to use Cloudflare Worker
client = StagehandClient(
    endpoint="https://e2e-testing-agent.your-subdomain.workers.dev",
    api_token="your-api-token"
)
```
