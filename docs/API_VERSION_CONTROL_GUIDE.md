# API Version Control Guide

**Document Version:** 1.0
**Date:** 2026-01-28

---

## Overview

This guide explains how to set up and use API version control for the Argus backend. The system automatically:

1. **Detects breaking changes** in pull requests
2. **Generates changelogs** for API modifications
3. **Updates documentation** on merge to main
4. **Blocks merges** for undocumented breaking changes

---

## Quick Start

### 1. Install Optic Locally (for development)

```bash
npm install -g @useoptic/optic

# Verify installation
optic --version
```

### 2. Check for Breaking Changes Before Pushing

```bash
# Generate current spec
python scripts/generate_openapi.py > openapi.new.json 2>/dev/null

# Compare with committed spec
optic diff openapi.json openapi.new.json

# If changes look good, update the committed spec
mv openapi.new.json openapi.json
git add openapi.json
git commit -m "chore: update OpenAPI spec"
```

---

## Tool Setup (100% FREE Options)

### Option A: Cloudflare Pages + ReDoc (Recommended - FREE)

**ReDoc** provides beautiful, responsive API documentation hosted for free on Cloudflare Pages.

**Live URL:** `https://api.heyargus.ai`

#### Setup Steps

1. **Create Cloudflare Pages Project**
   ```bash
   # One-time setup
   npx wrangler pages project create argus-api-docs
   ```

2. **Add GitHub Secrets**
   - Go to repo Settings → Secrets → Actions
   - Add `CLOUDFLARE_API_TOKEN` (create at dash.cloudflare.com → API Tokens)
   - Add `CLOUDFLARE_ACCOUNT_ID` (find in Cloudflare dashboard URL)

3. **Configure Custom Domain**
   - Cloudflare Dashboard → Pages → argus-api-docs → Custom domains
   - Add `api.heyargus.ai`
   - DNS is auto-configured (you're already on Cloudflare)

4. **Local Preview**
   ```bash
   # Install redocly CLI
   npm install -g @redocly/cli

   # Preview docs locally
   redocly preview-docs openapi.json

   # Generate standalone HTML
   redocly build-docs openapi.json -o api-docs/v1/index.html
   ```

5. **Features (FREE)**
   | Feature | Description |
   |---------|-------------|
   | Auto-generated Docs | Beautiful, responsive API documentation |
   | Search | Built-in search functionality |
   | Custom Domain | api.heyargus.ai (free on Cloudflare) |
   | Global CDN | Cloudflare's edge network |
   | Zero Cost | Unlimited bandwidth, free forever |

### Option B: oasdiff (FREE - Changelog Generation)

**oasdiff** is an open-source tool for OpenAPI diff and changelog generation.

```bash
# Install (Go required)
go install github.com/tufin/oasdiff@latest

# Or via Docker (no Go needed)
docker pull tufin/oasdiff

# Generate changelog
oasdiff changelog openapi.old.json openapi.new.json --format markdown

# Check for breaking changes
oasdiff breaking openapi.old.json openapi.new.json

# Get summary diff
oasdiff diff openapi.old.json openapi.new.json --format text
```

#### oasdiff Features (FREE)

| Feature | Description |
|---------|-------------|
| Breaking Detection | Identifies breaking API changes |
| Changelog | Generates markdown/json/yaml changelogs |
| CI Integration | Exit codes for CI/CD pipelines |
| Docker Support | No local installation needed |
| 100% Open Source | MIT licensed, free forever |

### Option C: Optic CLI (FREE for Local Use)

**Optic CLI** is free for local development and CI/CD pipelines.

```bash
# Install
npm install -g @useoptic/optic

# Compare specs
optic diff openapi.old.json openapi.new.json

# Check governance rules
optic lint openapi.json

# Run with our config
optic diff openapi.json openapi.new.json --check
```

### Option D: Swagger UI (FREE - Self-Hosted)

For teams wanting the classic Swagger interface:

```bash
# Docker - instant API docs
docker run -p 8080:8080 -e SWAGGER_JSON=/app/openapi.json \
  -v $(pwd)/openapi.json:/app/openapi.json swaggerapi/swagger-ui

# Or static files
npm install swagger-ui-dist
```

### Option E: Redocly CLI (FREE Tier)

For enterprise-grade linting and docs:

```bash
npm install -g @redocly/cli

# Lint API spec
redocly lint openapi.json

# Preview docs locally
redocly preview-docs openapi.json

# Build static HTML
redocly build-docs openapi.json -o docs/
```

---

## CI/CD Integration

### GitHub Actions Workflow

The workflow (`.github/workflows/api-versioning.yml`) runs on:
- **Pull Requests**: Detects breaking changes, comments on PR
- **Push to Main**: Updates docs, commits spec changes

### Workflow Jobs

```
┌─────────────────────┐
│  generate-openapi   │ → Regenerates spec from code
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌─────────┐
│ breaking│  │ update  │ (main only)
│ change  │  │ docs    │
│detection│  └─────────┘
└────┬────┘
     │
     ▼
┌─────────┐
│ comment │ → PR gets API change summary
│ on PR   │
└─────────┘
```

### Breaking Change Detection

The CI will **fail** if these changes are detected:

| Change Type | Example | Severity |
|-------------|---------|----------|
| Remove endpoint | DELETE `/api/v1/tests/{id}` removed | 🔴 Error |
| Add required parameter | New required `project_id` query param | 🔴 Error |
| Remove required parameter | Remove `name` from request body | 🔴 Error |
| Change parameter type | `id: string` → `id: integer` | 🔴 Error |
| Remove response property | Remove `created_at` from response | 🔴 Error |
| Change response type | `count: integer` → `count: string` | 🔴 Error |

**Non-breaking changes** (warnings only):

| Change Type | Example | Severity |
|-------------|---------|----------|
| Add endpoint | New `/api/v1/analytics` endpoint | ℹ️ Info |
| Add optional parameter | New optional `limit` query param | ℹ️ Info |
| Add response property | New `updated_at` in response | ℹ️ Info |
| Deprecate endpoint | Mark `/api/v1/old-endpoint` deprecated | ⚠️ Warn |

---

## Local Development Workflow

### Before Making API Changes

```bash
# 1. Create a backup of current spec
cp openapi.json openapi.backup.json

# 2. Make your code changes to src/api/**

# 3. Regenerate the spec
python scripts/generate_openapi.py > openapi.json 2>/dev/null

# 4. Check what changed
optic diff openapi.backup.json openapi.json

# 5. If breaking changes, update API version
#    Edit src/api/server.py: API_VERSION = "2.11.0" → "3.0.0"
```

### Semantic Versioning for APIs

Follow [SemVer](https://semver.org/) for API versions:

| Version Bump | When to Use | Example |
|--------------|-------------|---------|
| **MAJOR** (3.0.0) | Breaking changes | Remove endpoint, change types |
| **MINOR** (2.11.0) | New features, backwards compatible | Add endpoint, add optional param |
| **PATCH** (2.10.1) | Bug fixes, docs | Fix response schema, update descriptions |

---

## API Governance Rules

The `optic.yml` configuration enforces these rules:

### Required (Errors)

- ✅ All operations must have `operationId`
- ✅ No breaking changes without version bump
- ✅ Response schemas must be defined

### Recommended (Warnings)

- ⚠️ All operations should have descriptions
- ⚠️ All parameters should have descriptions
- ⚠️ Operations should have tags

### Check Governance Locally

```bash
# Run governance checks
optic lint openapi.json

# Auto-fix some issues
optic lint openapi.json --fix
```

---

## Generating Changelogs

### Automatic (CI/CD)

The GitHub Actions workflow automatically generates changelogs on each push to main.
View at: `https://your-org.github.io/your-repo/api-docs/CHANGELOG.md`

### Manual (Local)

```bash
# Using oasdiff (recommended - FREE)
go install github.com/tufin/oasdiff@latest
oasdiff changelog openapi.v2.10.json openapi.v2.11.json --format markdown > CHANGELOG_API.md

# Using Docker (no Go needed)
docker run --rm -v $(pwd):/specs tufin/oasdiff changelog /specs/openapi.old.json /specs/openapi.new.json

# Using Optic CLI
optic diff openapi.v2.10.json openapi.v2.11.json
```

### Example Changelog Output

```markdown
## API Changelog: v2.10.0 → v2.11.0

### Added
- `POST /api/v1/analytics/custom` - Create custom analytics query
- `GET /api/v1/projects/{id}/metrics` - Get project metrics
- Added optional `include_archived` parameter to `GET /api/v1/tests`

### Changed
- `GET /api/v1/reports` now returns `total_count` in response

### Deprecated
- `GET /api/v1/legacy/tests` - Use `GET /api/v1/tests` instead
```

---

## SDK Generation

### TypeScript SDK

```bash
# Install generator
npm install -g @openapitools/openapi-generator-cli

# Generate SDK
openapi-generator-cli generate \
  -i openapi.json \
  -g typescript-fetch \
  -o sdk/typescript \
  --additional-properties=npmName=@argus/api-client

# Use in your project
npm install ./sdk/typescript
```

### Python SDK

```bash
openapi-generator-cli generate \
  -i openapi.json \
  -g python \
  -o sdk/python \
  --additional-properties=packageName=argus_client
```

### Using Speakeasy (Type-Safe SDKs)

[Speakeasy](https://speakeasy.com) generates production-quality SDKs:

```bash
# Install
brew install speakeasy-api/homebrew-tap/speakeasy

# Generate TypeScript SDK
speakeasy generate sdk \
  --schema openapi.json \
  --lang typescript \
  --out sdk/typescript-speakeasy
```

---

## Troubleshooting

### CI Failing: "Breaking changes detected"

1. **Check the PR comment** for specific changes
2. **If intentional**: Bump the major version in `src/api/server.py`
3. **If accidental**: Revert your changes and fix

### OpenAPI Spec Not Generating

```bash
# Check for Python errors
python scripts/generate_openapi.py 2>&1 | head -50

# Common issues:
# - Missing mock for new dependency → Add to MOCK_MODULES in generate_openapi.py
# - Import error → Check circular imports
```

### Optic Not Finding Changes

```bash
# Ensure both files are valid JSON
python -c "import json; json.load(open('openapi.json'))"
python -c "import json; json.load(open('openapi.new.json'))"

# Run with verbose output
optic diff openapi.json openapi.new.json --verbose
```

---

## Best Practices

### DO ✅

- **Version your API** using semantic versioning
- **Document all changes** in PR descriptions
- **Add deprecation notices** before removing endpoints
- **Use feature flags** for gradual rollouts
- **Keep specs in version control**

### DON'T ❌

- **Don't make breaking changes** without major version bump
- **Don't remove properties** from responses without deprecation
- **Don't change types** without migration path
- **Don't skip CI checks** for "small" changes

---

## Resources

| Resource | URL |
|----------|-----|
| Optic Documentation | https://www.useoptic.com/docs |
| oasdiff (FREE changelog) | https://github.com/tufin/oasdiff |
| ReDoc | https://github.com/Redocly/redoc |
| Redocly CLI | https://redocly.com/docs/cli |
| OpenAPI Specification | https://spec.openapis.org/oas/latest.html |
| Semantic Versioning | https://semver.org |
| OpenAPI Generator | https://openapi-generator.tech |
| Swagger UI | https://swagger.io/tools/swagger-ui |

---

## Quick Reference

```bash
# Generate OpenAPI spec
python scripts/generate_openapi.py > openapi.json 2>/dev/null

# Check for breaking changes (Optic)
optic diff openapi.old.json openapi.json

# Check for breaking changes (oasdiff - FREE)
oasdiff breaking openapi.old.json openapi.json

# Generate changelog (FREE)
oasdiff changelog openapi.old.json openapi.json --format markdown

# Lint the spec
optic lint openapi.json
redocly lint openapi.json

# Preview docs locally (FREE)
redocly preview-docs openapi.json

# Build static docs (FREE)
redocly build-docs openapi.json -o api-docs/

# Generate TypeScript SDK
openapi-generator-cli generate -i openapi.json -g typescript-fetch -o sdk/ts
```

---

*End of API Version Control Guide*
