# E2E Testing Agent - User Workflows

## What is this?

The E2E Testing Agent is an **autonomous AI system** that:
1. Analyzes your codebase to understand what needs testing
2. Generates test plans automatically
3. Executes UI, API, and database tests
4. Self-heals broken tests when selectors change
5. Reports results

**You don't write tests** - the AI does everything.

## Quick Start (30 seconds)

```bash
# 1. Install
pip install -e .

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run (your app must be running)
e2e-agent --codebase ./my-app --app-url http://localhost:3000
```

That's it. The agent will:
- Scan your code
- Generate tests
- Run them
- Give you a report

---

## User Workflows

### Workflow 1: Full Automated Testing (Most Common)

**Use case**: "Test my entire application"

```bash
# Your app must be running first!
npm start  # or docker-compose up, etc.

# Run the agent
e2e-agent --codebase /path/to/your/app --app-url http://localhost:3000
```

**What happens**:
```
┌─────────────────────────────────────────────────────────────┐
│ 1. ANALYZE CODE                                              │
│    Agent reads your codebase, finds:                        │
│    - UI pages and components                                │
│    - API endpoints                                          │
│    - Database models                                        │
│    - Existing tests (to avoid duplication)                  │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PLAN TESTS                                               │
│    Agent creates prioritized test plan:                     │
│    - Critical paths (login, checkout, etc.)                 │
│    - API endpoint validation                                │
│    - Database integrity checks                              │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EXECUTE TESTS                                            │
│    Agent runs each test:                                    │
│    - Opens browser (Playwright)                             │
│    - Performs actions (click, type, etc.)                   │
│    - Takes screenshots                                      │
│    - Validates assertions                                   │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. SELF-HEAL (if tests fail)                                │
│    If a test fails:                                         │
│    - Agent analyzes why                                     │
│    - Fixes selector if it changed                           │
│    - Retries the test                                       │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. REPORT                                                   │
│    Agent outputs:                                           │
│    - Summary (passed/failed/skipped)                        │
│    - Detailed results JSON                                  │
│    - Screenshots of failures                                │
│    - Cost breakdown                                         │
└─────────────────────────────────────────────────────────────┘
```

**Output**:
```
./test-results/
├── results.json        # Full test results
├── report.html         # Human-readable report
└── screenshots/        # Failure screenshots
    ├── login-failed.png
    └── checkout-error.png
```

---

### Workflow 2: PR/CI Integration

**Use case**: "Test only what changed in this PR"

```bash
# In GitHub Actions or CI
e2e-agent \
  --codebase . \
  --app-url $PREVIEW_URL \
  --pr 123 \
  --changed-files src/auth/login.tsx src/api/users.ts
```

**What happens**:
- Agent focuses only on files that changed
- Generates targeted tests for affected functionality
- Faster and cheaper than full run

**GitHub Actions Example**:
```yaml
name: AI E2E Tests
on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start preview
        run: docker-compose up -d

      - name: Get changed files
        id: changes
        run: |
          echo "files=$(git diff --name-only origin/main)" >> $GITHUB_OUTPUT

      - name: Run AI Tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          pip install e2e-testing-agent
          e2e-agent \
            --codebase . \
            --app-url http://localhost:3000 \
            --pr ${{ github.event.number }} \
            --changed-files ${{ steps.changes.outputs.files }}
```

---

### Workflow 3: Python API (Programmatic)

**Use case**: "Integrate into my existing test framework"

```python
import asyncio
from e2e_testing_agent import TestingOrchestrator

async def main():
    # Initialize
    orchestrator = TestingOrchestrator(
        codebase_path="/path/to/my/app",
        app_url="http://localhost:3000",
    )

    # Run all tests
    results = await orchestrator.run()

    # Check results
    print(f"Passed: {results['passed_count']}")
    print(f"Failed: {results['failed_count']}")

    # Access detailed results
    for test_result in results['test_results']:
        if test_result['status'] == 'failed':
            print(f"FAILED: {test_result['name']}")
            print(f"  Error: {test_result['error_message']}")

asyncio.run(main())
```

---

### Workflow 4: Single Test Execution

**Use case**: "Just run this one specific test"

```python
orchestrator = TestingOrchestrator(
    codebase_path="/path/to/app",
    app_url="http://localhost:3000",
)

# Define a specific test
test_spec = {
    "id": "login-test",
    "name": "User Login Flow",
    "type": "ui",
    "steps": [
        {"action": "goto", "target": "/login"},
        {"action": "fill", "selector": "#email", "value": "test@example.com"},
        {"action": "fill", "selector": "#password", "value": "password123"},
        {"action": "click", "selector": "#submit"},
    ],
    "assertions": [
        {"type": "url_contains", "value": "/dashboard"},
        {"type": "element_visible", "selector": "#welcome-message"},
    ]
}

result = await orchestrator.run_single_test(test_spec)
```

---

## Browser Automation Options

The agent supports multiple ways to control the browser:

### Option 1: Playwright (Default) - RECOMMENDED

```python
# This is automatic - no configuration needed
e2e-agent --codebase ./app --app-url http://localhost:3000
```

**Pros**: Fastest, most reliable, works in CI
**Cons**: Can be detected as bot by some sites

### Option 2: Chrome Extension (Real Browser)

Use when you need:
- Your existing browser session (cookies, auth)
- To avoid bot detection
- To capture console logs

```python
from src.tools import create_browser

# Start with extension (requires Chrome + extension installed)
browser = await create_browser("extension")
await browser.goto("https://example.com")

# Unique feature: capture console logs
logs = await browser.get_console_logs()
```

**Setup**:
1. Load `extension/` folder in Chrome as unpacked extension
2. Keep Chrome open
3. Agent connects via WebSocket

### Option 3: Computer Use (Visual AI)

Use when:
- Selectors are unreliable
- UI changes frequently
- You want "human-like" testing

```python
browser = await create_browser("computer_use")

# No selectors! Natural language descriptions
await browser.click("the blue Login button")
await browser.fill("the email input field", "test@example.com")
```

**Pros**: Works with any UI, no selectors needed
**Cons**: Slower (2-5s per action), higher API cost

### Option 4: Hybrid (Best of Both)

```python
browser = await create_browser("hybrid")

# Tries Playwright first (fast)
# Falls back to Computer Use if selector fails
await browser.click("#login-btn")  # Playwright tries first
# If #login-btn doesn't exist, Computer Use finds it visually
```

---

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Models (optional)
DEFAULT_MODEL=claude-sonnet-4-5         # Main testing model
VERIFICATION_MODEL=claude-haiku-4-5     # Quick checks
DEBUGGING_MODEL=claude-opus-4-5         # Complex analysis

# Cost controls (optional)
COST_LIMIT_PER_RUN=10.00     # Max $ per full run
COST_LIMIT_PER_TEST=1.00     # Max $ per test
MAX_ITERATIONS=50            # Safety limit

# Self-healing (optional)
SELF_HEAL_ENABLED=true
SELF_HEAL_MAX_ATTEMPTS=3
```

### Config File (.env)

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-xxx
DEFAULT_MODEL=claude-sonnet-4-5
COST_LIMIT_PER_RUN=15.00
SELF_HEAL_ENABLED=true
```

---

## Cost Estimates

| Action | Approx Cost |
|--------|-------------|
| Analyze small codebase (< 50 files) | $0.10-0.30 |
| Analyze large codebase (500+ files) | $0.50-2.00 |
| Generate test plan | $0.05-0.20 |
| Run 1 UI test (10 steps) | $0.10-0.20 |
| Run 1 API test | $0.02-0.05 |
| Self-heal 1 failure | $0.05-0.15 |
| Generate report | $0.02-0.05 |

**Typical full run**: $1-5 depending on app size

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                            USER                                      │
│                  e2e-agent --codebase ./app                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TESTING ORCHESTRATOR                            │
│                        (LangGraph FSM)                              │
│                                                                      │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│   │ Analyze  │ → │  Plan    │ → │ Execute  │ → │  Report  │        │
│   │   Code   │   │  Tests   │   │  Tests   │   │          │        │
│   └──────────┘   └──────────┘   └────┬─────┘   └──────────┘        │
│                                      │                              │
│                                      ▼                              │
│                               ┌──────────┐                          │
│                               │Self-Heal │ (on failure)             │
│                               └──────────┘                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │UI Tester │ │API Tester│ │DB Tester │
             └────┬─────┘ └────┬─────┘ └────┬─────┘
                  │            │            │
                  ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │Browser   │ │  httpx   │ │SQLAlchemy│
             │Automation│ │          │ │          │
             └──────────┘ └──────────┘ └──────────┘
```

---

## Troubleshooting

### "No testable surfaces found"
- Make sure your app has recognizable patterns (routes, API endpoints, etc.)
- Check that codebase path is correct

### Tests timing out
- Increase timeout: `TIMEOUT_MS=60000`
- Make sure app is fully loaded before testing starts

### High costs
- Use `--changed-files` for targeted testing
- Lower `COST_LIMIT_PER_RUN`
- Use Haiku for verification: `VERIFICATION_MODEL=claude-haiku-4-5`

### Bot detection issues
- Use Chrome Extension instead of Playwright
- Or use Computer Use mode

---

## Summary: When to Use What

| Scenario | Recommendation |
|----------|----------------|
| CI/CD pipeline | Default (Playwright) |
| Testing authenticated app | Chrome Extension |
| Flaky selectors | Hybrid mode |
| Any website/no selectors | Computer Use |
| Quick PR checks | `--changed-files` flag |
| Full regression | Full run, maybe overnight |
