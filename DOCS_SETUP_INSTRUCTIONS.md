# ARGUS Documentation Setup Instructions

## Private Documentation Site on Cloudflare Pages

This guide will help you deploy your private documentation to `docs.heyargus.ai`

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   GitHub Repo ──► GitHub Actions ──► Cloudflare Pages            │
│        │                                      │                  │
│        │                                      ▼                  │
│   docs-site/           builds MkDocs     docs.heyargus.ai        │
│   ARGUS_*.md                                  │                  │
│                                               ▼                  │
│                                    Cloudflare Access             │
│                                    (Email Authentication)        │
│                                               │                  │
│                                               ▼                  │
│                                         Only YOU can access      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Cost: $0 (completely free)**

---

## Step 1: Create Cloudflare Pages Project

### 1.1 Go to Cloudflare Dashboard

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Select your account
3. Go to **Workers & Pages** → **Pages**
4. Click **Create a project** → **Connect to Git**

### 1.2 Connect GitHub

1. Select **GitHub**
2. Authorize Cloudflare to access your repos
3. Select your `e2e-testing-agent` repository

### 1.3 Configure Build Settings

```
Project name:       argus-docs
Production branch:  main
Build command:      pip install mkdocs-material mkdocs-minify-plugin && mkdocs build
Build output:       site
Root directory:     /
```

### 1.4 Environment Variables (Optional)

No environment variables needed for basic setup.

### 1.5 Deploy

Click **Save and Deploy**. Cloudflare will build your docs.

Your site will be live at: `argus-docs.pages.dev`

---

## Step 2: Add Custom Domain (docs.heyargus.ai)

### 2.1 In Cloudflare Pages

1. Go to your `argus-docs` project
2. Click **Custom domains**
3. Click **Set up a custom domain**
4. Enter: `docs.heyargus.ai`
5. Click **Continue**

### 2.2 DNS Configuration

Since your DNS is already on Cloudflare, it will auto-configure!

If prompted, add this DNS record:
```
Type:    CNAME
Name:    docs
Content: argus-docs.pages.dev
Proxy:   Enabled (orange cloud)
```

### 2.3 Wait for SSL

SSL certificate is automatically provisioned. Takes 1-5 minutes.

---

## Step 3: Protect with Cloudflare Access (Zero Trust)

**This is what makes it private - only you can access!**

### 3.1 Go to Zero Trust

1. In Cloudflare Dashboard, click **Zero Trust** (left sidebar)
2. Or go to: `https://one.dash.cloudflare.com`

### 3.2 Create Access Application

1. Go to **Access** → **Applications**
2. Click **Add an application**
3. Select **Self-hosted**

### 3.3 Configure Application

```
Application name:   ARGUS Docs
Session duration:   24 hours (or your preference)

Application domain:
  Subdomain: docs
  Domain:    heyargus.ai
```

### 3.4 Create Access Policy

Click **Next** and create a policy:

```
Policy name:        Only Me
Action:            Allow
Session duration:  24 hours

Include:
  - Selector: Emails
  - Value:    your-email@example.com
```

**Replace `your-email@example.com` with your actual email.**

### 3.5 (Optional) Add More Allowed Users

To allow team members, add more emails:

```
Include:
  - Emails: your-email@example.com
  - Emails: teammate@example.com
```

Or use email domain:
```
Include:
  - Emails ending in: @yourcompany.com
```

### 3.6 Save and Deploy

Click **Add application**

---

## Step 4: GitHub Actions Setup (Auto-Deploy)

### 4.1 Get Cloudflare API Token

1. Go to Cloudflare Dashboard → **My Profile** → **API Tokens**
2. Click **Create Token**
3. Use template: **Edit Cloudflare Workers**
4. Or create custom with:
   - **Cloudflare Pages: Edit**
   - **Account: Read**
5. Copy the token

### 4.2 Get Cloudflare Account ID

1. Go to Cloudflare Dashboard
2. Select any domain
3. On the right sidebar, find **Account ID**
4. Copy it

### 4.3 Add GitHub Secrets

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Add these secrets:

```
CLOUDFLARE_API_TOKEN:   (paste your API token)
CLOUDFLARE_ACCOUNT_ID:  (paste your account ID)
```

### 4.4 Trigger Deployment

Push any change to `docs-site/` or `ARGUS_*.md` files, and it will auto-deploy!

Or manually trigger:
1. Go to **Actions** tab
2. Select **Deploy Documentation**
3. Click **Run workflow**

---

## Step 5: Test Access

### 5.1 Visit Your Docs

Go to: `https://docs.heyargus.ai`

### 5.2 You Should See

1. Cloudflare Access login page
2. Enter your email
3. Check email for code
4. Enter code
5. Access granted!

### 5.3 Verify Privacy

Try from incognito or different browser - it should require authentication.

---

## Local Development

### Preview Docs Locally

```bash
# Install MkDocs
pip install mkdocs-material mkdocs-minify-plugin

# Serve locally
mkdocs serve

# Open: http://localhost:8000
```

### Build Static Site

```bash
mkdocs build
# Output in: site/
```

---

## File Structure

```
e2e-testing-agent/
├── mkdocs.yml                    # MkDocs configuration
├── docs-site/                    # Documentation source
│   ├── index.md                  # Homepage
│   ├── getting-started/
│   │   └── index.md              # Getting Started Guide
│   ├── product/
│   │   ├── executive-summary.md
│   │   ├── pitch-deck.md
│   │   └── knowledge-base.md
│   ├── deployment/
│   │   ├── index.md              # Deployment Options
│   │   └── faq.md                # Deployment FAQ
│   ├── sales/
│   │   ├── battlecard.md
│   │   └── competitive-analysis.md
│   ├── technical/
│   │   ├── architecture.md
│   │   ├── implementation.md
│   │   └── security.md
│   ├── stylesheets/
│   │   └── extra.css
│   └── javascripts/
│       └── extra.js
├── ARGUS_*.md                    # Source docs (auto-synced)
└── .github/workflows/
    └── deploy-docs.yml           # Deployment workflow
```

---

## Updating Documentation

### Option 1: Edit Source Files

Edit the `ARGUS_*.md` files in the root. They auto-sync to `docs-site/` during deployment.

### Option 2: Edit docs-site Directly

Edit files in `docs-site/` folder directly for faster iteration.

### Option 3: Use the Docs Site

After setup, you can edit directly in Cloudflare Pages or use the MkDocs admin (if enabled).

---

## Troubleshooting

### Build Fails

Check that all files referenced in `mkdocs.yml` nav exist in `docs-site/`.

### Access Denied

1. Check Cloudflare Access policy includes your email
2. Clear browser cookies and try again
3. Check email for authentication link

### DNS Not Working

1. Verify CNAME record exists: `docs` → `argus-docs.pages.dev`
2. Ensure proxy is enabled (orange cloud)
3. Wait 5 minutes for propagation

### SSL Certificate Error

1. Wait up to 15 minutes for certificate provisioning
2. Check domain is correctly configured in Cloudflare Pages

---

## Security Notes

1. **Cloudflare Access** provides enterprise-grade authentication
2. **All traffic is encrypted** via Cloudflare SSL
3. **No passwords stored** - email-based one-time codes
4. **Audit logs available** in Zero Trust dashboard
5. **Revoke access anytime** by removing email from policy

---

## Cost Summary

| Service | Cost |
|---------|------|
| Cloudflare Pages | Free |
| Cloudflare Access (up to 50 users) | Free |
| Custom Domain SSL | Free |
| **Total** | **$0/month** |

---

## Quick Reference

| What | Where |
|------|-------|
| Docs URL | `https://docs.heyargus.ai` |
| Cloudflare Pages | `dash.cloudflare.com` → Workers & Pages |
| Cloudflare Access | `one.dash.cloudflare.com` → Access |
| GitHub Workflow | `.github/workflows/deploy-docs.yml` |
| MkDocs Config | `mkdocs.yml` |
| Docs Source | `docs-site/` |

---

**Setup Complete!**

Your private documentation is now live at `docs.heyargus.ai` - accessible only to you.

---

*Last Updated: January 2026*
