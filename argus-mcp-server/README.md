# Argus MCP Server

Model Context Protocol (MCP) server for Argus E2E Testing Agent. This allows AI coding assistants to interact with Argus testing capabilities directly.

## Supported AI IDEs

- **Claude Code** - Anthropic's CLI for Claude
- **Cursor** - AI-first code editor
- **Windsurf** - Codeium's AI IDE
- **VS Code** - With MCP extension
- **Any MCP-compatible client**

## Available Tools

| Tool | Description |
|------|-------------|
| `argus_health` | Check Argus API status |
| `argus_discover` | Discover interactive elements on a page |
| `argus_act` | Execute browser actions (click, type, navigate) |
| `argus_test` | Run multi-step E2E tests with screenshots |
| `argus_extract` | Extract structured data from pages |
| `argus_agent` | Autonomous task completion |
| `argus_generate_test` | Generate test steps from natural language |

## Installation

### For Claude Code / Claude Desktop

Add to your MCP settings (`~/.claude/claude_desktop_config.json` or Claude Code settings):

```json
{
  "mcpServers": {
    "argus": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://argus-mcp.samuelvinay-kumar.workers.dev/sse"]
    }
  }
}
```

### For Cursor

Add to Cursor's MCP settings:

```json
{
  "mcpServers": {
    "argus": {
      "url": "https://argus-mcp.samuelvinay-kumar.workers.dev/sse"
    }
  }
}
```

## Usage Examples

### Discover Page Elements
```
Use argus_discover to analyze https://example.com and find all interactive elements
```

### Run E2E Test
```
Use argus_test to test login on https://uitestingplayground.com/sampleapp with these steps:
1. Type "TestUser" in the username field
2. Type "pwd" in the password field
3. Click the Log In button
4. Verify the welcome message appears
```

### Execute Single Action
```
Use argus_act to click the "Sign Up" button on https://example.com/signup
```

### Extract Data
```
Use argus_extract to get all product names and prices from https://example.com/products
```

### Autonomous Agent
```
Use argus_agent to complete the checkout flow on https://example.com/shop starting from the homepage
```

## Development

### Prerequisites

- Node.js 18+
- Wrangler CLI

### Setup

```bash
cd argus-mcp-server
npm install
```

### Local Development

```bash
npm run dev
```

### Deploy

```bash
npm run deploy
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI IDE (Claude/Cursor)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol (SSE)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Argus MCP Server                           │
│              (Cloudflare Workers + agents/mcp)               │
│                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │argus_discover│ │ argus_test  │ │ argus_agent │           │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
└─────────┼───────────────┼───────────────┼───────────────────┘
          │               │               │
          └───────────────┼───────────────┘
                          │ REST API
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Argus API Worker                          │
│            (cloudflare-worker/src/index.ts)                  │
│                                                              │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐                 │
│  │/observe │  │  /test   │  │  /agent    │                 │
│  └────┬────┘  └────┬─────┘  └─────┬──────┘                 │
└───────┼────────────┼──────────────┼─────────────────────────┘
        │            │              │
        └────────────┼──────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Browser (CF/TestingBot)  │
         └───────────────────────┘
```

## Pass Rate Optimization

Argus achieves 95%+ pass rate through:

1. **Self-Healing Selectors** - AI-powered selector recovery
2. **Smart Waits** - Intelligent element detection
3. **Retry Logic** - Automatic retry on transient failures
4. **Screenshot Capture** - Visual verification at each step
5. **Multi-Browser Support** - Cross-browser compatibility

## License

MIT
