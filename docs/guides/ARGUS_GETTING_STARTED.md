# Getting Started with ARGUS

**Your First Test in Under 5 Minutes**

---

## Welcome to Argus

Argus is an AI-powered autonomous testing platform that lets you create and run E2E tests using plain English. No coding required.

This guide will help you:
1. Create your first project
2. Run your first test
3. Explore key features
4. Choose your deployment mode

---

## Table of Contents

1. [Quick Start (5 Minutes)](#quick-start-5-minutes)
2. [Account Setup](#account-setup)
3. [Creating Your First Project](#creating-your-first-project)
4. [Running Your First Test](#running-your-first-test)
5. [Understanding Test Results](#understanding-test-results)
6. [Auto-Discovery: Generate Tests Automatically](#auto-discovery-generate-tests-automatically)
7. [Choosing Your Deployment Mode](#choosing-your-deployment-mode)
8. [Setting Up Integrations](#setting-up-integrations)
9. [Next Steps](#next-steps)

---

## Quick Start (5 Minutes)

### The Fastest Way to Your First Test

```bash
# 1. Sign up at argus.dev (free tier available)

# 2. Create a new project with your app URL

# 3. Write a test in plain English:
"Go to the homepage and verify the login button is visible"

# 4. Click Run - That's it!
```

**Argus handles everything else:** browser automation, assertions, screenshots, and reporting.

---

## Account Setup

### Step 1: Sign Up

1. Go to [argus.dev](https://argus.dev)
2. Click **"Get Started Free"**
3. Sign up with:
   - Google
   - GitHub
   - Email/Password

### Step 2: Verify Your Email

If you signed up with email, check your inbox for a verification link.

### Step 3: Complete Your Profile

```
┌─────────────────────────────────────────────────────────────┐
│  Welcome to Argus!                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Your Name:     [________________________]                  │
│                                                             │
│  Company:       [________________________] (optional)       │
│                                                             │
│  Role:          [ Developer ▼ ]                             │
│                                                             │
│  How did you    [ Search Engine ▼ ]                         │
│  hear about us?                                             │
│                                                             │
│                              [ Continue → ]                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Creating Your First Project

### Step 1: Click "New Project"

From your dashboard, click the **"+ New Project"** button.

### Step 2: Enter Project Details

```
┌─────────────────────────────────────────────────────────────┐
│  Create New Project                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Project Name:  [My First Project___________]               │
│                                                             │
│  App URL:       [https://myapp.com__________]               │
│                                                             │
│  Description:   [Testing my web application_] (optional)    │
│                                                             │
│  Environment:   ○ Production                                │
│                 ● Staging                                   │
│                 ○ Development                               │
│                                                             │
│                              [ Create Project → ]           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Tips:**
- Use your **staging/dev environment** for testing (not production)
- The URL should be accessible from the internet
- For localhost testing, see [Local Development Testing](#testing-localhost)

### Step 3: Project Created!

You'll see your new project dashboard:

```
┌─────────────────────────────────────────────────────────────┐
│  My First Project                        [ Settings ⚙️ ]    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  App URL: https://myapp.com                                 │
│  Tests: 0  |  Last Run: Never  |  Status: Ready             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │   🎯 Ready to create your first test!              │   │
│  │                                                     │   │
│  │   [ + Create Test ]    [ 🔍 Auto-Discover ]        │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Running Your First Test

### Option A: Write a Test in Plain English (Recommended)

Click **"+ Create Test"** and describe what you want to test:

```
┌─────────────────────────────────────────────────────────────┐
│  Create New Test                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Name:     [Homepage Login Button Test____]            │
│                                                             │
│  Describe your test in plain English:                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Go to the homepage and verify that:                 │   │
│  │ - The page title contains "Welcome"                 │   │
│  │ - The login button is visible                       │   │
│  │ - The signup link exists                            │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│                    [ Create & Run → ]                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Example test descriptions:**

| What You Type | What Argus Does |
|---------------|-----------------|
| "Login with email test@example.com and password demo123" | Fills login form, submits, verifies success |
| "Add a product to the shopping cart" | Navigates to product, clicks add to cart, verifies |
| "Search for 'laptop' and verify results appear" | Uses search box, waits for results, validates |
| "Fill out the contact form and submit" | Fills all fields, submits, checks confirmation |

### Option B: Use the AI Chat Interface

Click the **Chat** icon to describe tests conversationally:

```
┌─────────────────────────────────────────────────────────────┐
│  Argus AI Assistant                              [ × ]      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🤖 Hi! I can help you create tests. What would you        │
│     like to test today?                                     │
│                                                             │
│  👤 I want to test the checkout flow                        │
│                                                             │
│  🤖 I'll create a checkout flow test for you. Let me       │
│     ask a few questions:                                    │
│                                                             │
│     1. Do users need to be logged in?                       │
│     2. What payment methods should I test?                  │
│     3. Should I verify the confirmation email?              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Type your message...                         [Send] │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Option C: Record a Test (Coming Soon)

Use our browser extension to record your actions and convert them to tests.

---

## Understanding Test Results

### Test Execution View

When your test runs, you'll see real-time progress:

```
┌─────────────────────────────────────────────────────────────┐
│  Test: Homepage Login Button Test           ▶️ Running...   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Steps:                                                     │
│  ✅ 1. Navigate to https://myapp.com                       │
│  ✅ 2. Wait for page to load                               │
│  ✅ 3. Verify page title contains "Welcome"                │
│  🔄 4. Check login button is visible                       │
│  ⏳ 5. Verify signup link exists                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │              [ Live Screenshot ]                    │   │
│  │                                                     │   │
│  │         Shows current browser state                 │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Duration: 3.2s  |  AI Cost: $0.02                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Test Results Summary

After completion:

```
┌─────────────────────────────────────────────────────────────┐
│  ✅ Test Passed                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Homepage Login Button Test                                 │
│                                                             │
│  Duration:     4.8 seconds                                  │
│  Steps:        5/5 passed                                   │
│  Screenshots:  3 captured                                   │
│  AI Cost:      $0.03                                        │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Screenshot 1│ │ Screenshot 2│ │ Screenshot 3│           │
│  │   Start     │ │   Middle    │ │    End      │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                             │
│  [ View Details ]  [ Re-run ]  [ Edit Test ]               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### When Tests Fail

If a test fails, Argus shows you exactly what went wrong:

```
┌─────────────────────────────────────────────────────────────┐
│  ❌ Test Failed                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Failed Step: Check login button is visible                 │
│                                                             │
│  Error: Element not found: button#login-btn                 │
│                                                             │
│  🔧 Self-Healing Suggestion:                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Found similar element: button[data-testid="login"]  │   │
│  │ Confidence: 94%                                     │   │
│  │                                                     │   │
│  │ [ Apply Fix ]  [ Ignore ]  [ Report Bug ]          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Screenshot at failure:                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │    [ Screenshot showing current page state ]        │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Auto-Discovery: Generate Tests Automatically

Let Argus crawl your app and generate tests for you.

### Step 1: Click "Auto-Discover"

From your project dashboard, click **"🔍 Auto-Discover"**

### Step 2: Configure Discovery

```
┌─────────────────────────────────────────────────────────────┐
│  Auto-Discovery Settings                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Starting URL:    [https://myapp.com____________]           │
│                                                             │
│  Discovery Depth: [ Medium (20 pages) ▼ ]                   │
│                                                             │
│  Options:                                                   │
│  ☑️ Include login flows                                     │
│  ☑️ Test form submissions                                   │
│  ☑️ Check broken links                                      │
│  ☐ Test file uploads                                        │
│                                                             │
│  Exclude URLs:    [/admin/*, /api/*___________]             │
│                                                             │
│                       [ Start Discovery → ]                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 3: Review Generated Tests

Argus will crawl your app and suggest tests:

```
┌─────────────────────────────────────────────────────────────┐
│  Discovery Complete! Found 12 test suggestions              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Pages Crawled: 18                                          │
│  Forms Found: 4                                             │
│  User Flows Detected: 6                                     │
│                                                             │
│  Suggested Tests:                                           │
│                                                             │
│  ☑️ Homepage loads correctly                                │
│  ☑️ Login form submission                                   │
│  ☑️ Registration flow                                       │
│  ☑️ Search functionality                                    │
│  ☑️ Add to cart flow                                        │
│  ☑️ Checkout process                                        │
│  ☐ Newsletter signup (optional)                             │
│  ☐ Contact form submission (optional)                       │
│                                                             │
│  [ Select All ]  [ Create Selected Tests → ]                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Choosing Your Deployment Mode

Argus offers three deployment modes. **You can start with DOM-Only (no setup required) and upgrade later.**

### Quick Comparison

| Mode | Code Access | Healing Accuracy | Setup Time |
|------|-------------|------------------|------------|
| **DOM-Only** | None | 90-95% | Instant |
| **Git-Aware** | Read-only | 99.9% | 5 minutes |
| **Self-Hosted** | On-premise | 99.9% | Days/Weeks |

### DOM-Only Mode (Default)

**No setup required.** This is the default mode when you create a project.

✅ Best for:
- Getting started quickly
- Testing third-party apps
- Privacy-conscious teams
- Agencies testing client apps

### Git-Aware Mode (Recommended for Teams)

Connect your repository for maximum healing accuracy.

**Setup:**

1. Go to **Project Settings > Integrations**
2. Click **"Connect Repository"**
3. Select your Git provider (GitHub, GitLab, Bitbucket)
4. Authorize read-only access
5. Select your repository

```
┌─────────────────────────────────────────────────────────────┐
│  Connect Repository                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Select Provider:                                           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  GitHub  │  │  GitLab  │  │Bitbucket │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│                                                             │
│  ⓘ We only request READ-ONLY access.                       │
│    Your code is never stored permanently.                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

✅ Best for:
- Development teams
- Frequent refactoring
- Maximum test reliability

### Self-Hosted Mode (Enterprise)

Contact sales@argus.dev for on-premise deployment.

✅ Best for:
- Regulated industries
- Strict data residency requirements
- Air-gapped environments

---

## Setting Up Integrations

### CI/CD Integration

#### GitHub Actions

```yaml
# .github/workflows/argus-tests.yml
name: Argus E2E Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run Argus Tests
        uses: argus-testing/github-action@v1
        with:
          api-key: ${{ secrets.ARGUS_API_KEY }}
          project-id: ${{ secrets.ARGUS_PROJECT_ID }}
```

#### GitLab CI

```yaml
# .gitlab-ci.yml
argus-tests:
  stage: test
  image: argus/cli:latest
  script:
    - argus run --project $ARGUS_PROJECT_ID
  variables:
    ARGUS_API_KEY: $ARGUS_API_KEY
```

### Slack Notifications

1. Go to **Project Settings > Notifications**
2. Click **"Connect Slack"**
3. Select your channel
4. Choose notification triggers:
   - ☑️ Test failures
   - ☑️ Daily summary
   - ☐ All test runs

### GitHub PR Comments

1. Go to **Project Settings > Integrations**
2. Enable **"GitHub PR Comments"**
3. Argus will automatically comment on PRs with test results

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 Argus Test Results                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ 12 tests passed                                         │
│  ⚠️ 1 test healed automatically                             │
│  ❌ 0 tests failed                                          │
│                                                             │
│  Duration: 45s | Cost: $0.28                                │
│                                                             │
│  [View Full Report →]                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Localhost

To test applications running on localhost:

### Option 1: Use Argus CLI (Recommended)

```bash
# Install CLI
npm install -g @argus/cli

# Start tunnel
argus tunnel --port 3000

# Your app is now accessible at:
# https://your-project.argus.dev
```

### Option 2: Use ngrok

```bash
# Start ngrok tunnel
ngrok http 3000

# Copy the https URL and use it as your App URL in Argus
```

### Option 3: Deploy to Staging

Use a staging environment URL instead of localhost.

---

## IDE Integration (MCP)

Use Argus directly in your AI-powered IDE.

### VS Code / Cursor / Windsurf Setup

1. Install the Argus MCP extension
2. Add to your MCP config:

```json
{
  "mcpServers": {
    "argus": {
      "command": "npx",
      "args": ["@argus/mcp-server"],
      "env": {
        "ARGUS_API_KEY": "your-api-key"
      }
    }
  }
}
```

3. Use Argus commands in your IDE:

```
> argus_discover - Find interactive elements on a page
> argus_test - Run a test from description
> argus_generate - Generate test from user story
> argus_health - Check API status
```

---

## Common First-Time Issues

### "My test can't find the login button"

**Solution:** Be more specific in your description:
```
❌ "Click login"
✅ "Click the blue Login button in the top navigation bar"
```

### "The test is too slow"

**Solution:** Tests include AI processing time. Typical tests take 5-15 seconds. For faster execution, use the Starter plan or above.

### "I can't test my localhost app"

**Solution:** Use `argus tunnel` or deploy to a staging environment. See [Testing Localhost](#testing-localhost).

### "Self-healing suggested wrong element"

**Solution:**
1. Click "Report Issue" on the healing suggestion
2. Manually specify the correct selector
3. Argus learns from your feedback

### "My test passes locally but fails in Argus"

**Common causes:**
- Different screen resolution (Argus uses 1920x1080)
- Missing test data in staging environment
- Authentication tokens not set up

---

## Next Steps

### 1. Create More Tests

Start building your test suite:
- **Critical paths** - Login, checkout, core features
- **User journeys** - Complete workflows
- **Edge cases** - Error handling, validation

### 2. Set Up CI/CD

Integrate Argus into your deployment pipeline:
- Run tests on every PR
- Block deploys on test failures
- Get Slack notifications

### 3. Enable Git-Aware Mode

If you haven't already, connect your repository for 99.9% healing accuracy.

### 4. Explore Advanced Features

| Feature | Where to Find |
|---------|---------------|
| Visual Regression Testing | Project > Visual |
| API Testing | Project > API Tests |
| Database Validation | Project > DB Tests |
| Performance Metrics | Project > Quality |
| Accessibility Testing | Project > Quality |

### 5. Invite Your Team

Go to **Settings > Team** to invite colleagues.

---

## Getting Help

### Documentation
- Full docs: [docs.argus.dev](https://docs.argus.dev)
- API reference: [docs.argus.dev/api](https://docs.argus.dev/api)
- FAQ: [docs.argus.dev/faq](https://docs.argus.dev/faq)

### Support
- **Community Discord:** [discord.gg/argus](https://discord.gg/argus)
- **Email:** support@argus.dev
- **In-app chat:** Click the chat icon in the dashboard

### Video Tutorials
- [Your First Test in 5 Minutes](https://youtube.com/argus)
- [Setting Up CI/CD Integration](https://youtube.com/argus)
- [Advanced Test Patterns](https://youtube.com/argus)

---

## Quick Reference Card

### Test Description Cheat Sheet

| Action | Example Description |
|--------|---------------------|
| Navigate | "Go to the products page" |
| Click | "Click the Submit button" |
| Type | "Type 'hello' in the search box" |
| Select | "Select 'Large' from the size dropdown" |
| Verify text | "Verify the page shows 'Success'" |
| Verify element | "Verify the cart icon is visible" |
| Wait | "Wait for the loading spinner to disappear" |
| Scroll | "Scroll to the footer" |
| Screenshot | "Take a screenshot of the current page" |

### Keyboard Shortcuts (Dashboard)

| Shortcut | Action |
|----------|--------|
| `N` | New test |
| `R` | Run selected test |
| `D` | Open discovery |
| `S` | Open settings |
| `?` | Show help |

---

**Welcome to Argus!** 🎯

You're ready to start testing. If you have questions, our team is here to help.

---

*Last Updated: January 2026*
