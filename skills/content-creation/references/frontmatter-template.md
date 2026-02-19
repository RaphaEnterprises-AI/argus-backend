# Frontmatter Templates

## Blog Post Frontmatter

```yaml
---
title: "Post Title — Clear, Specific, Under 70 Characters"
slug: "post-slug-matching-filename"
date: "YYYY-MM-DD"
author:
  name: "Skopaq Engineering"
  avatar: "/icons/skopaq-logo.svg"
category: "engineering"           # engineering | benchmarks | changelog | case-study
tags: ["tag1", "tag2"]            # lowercase, hyphenated, 2-5 tags
excerpt: "One-sentence summary for cards and SEO. Under 160 characters."
readingTime: 8                    # word count / 250, rounded up
featured: false                   # true = appears in hero section on /blog
coverImage: "/blog/slug-cover.png" # optional, must exist in dashboard/public/blog/
---
```

### Field Details

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| `title` | Yes | string | Under 70 chars for SEO. No period at end. |
| `slug` | Yes | string | Must match filename (minus date and `.mdx`). Lowercase, hyphenated. |
| `date` | Yes | string | ISO 8601 (`YYYY-MM-DD`). Publication date. |
| `author.name` | Yes | string | Default: `"Skopaq Engineering"`. Use individual names for guest posts. |
| `author.avatar` | Yes | string | Default: `"/icons/skopaq-logo.svg"`. Path relative to `public/`. |
| `category` | Yes | string | One of: `engineering`, `benchmarks`, `changelog`, `case-study`. |
| `tags` | Yes | string[] | 2-5 tags. Lowercase, hyphenated. Used for filtering and SEO. |
| `excerpt` | Yes | string | Under 160 chars. Shown on post cards and in `<meta name="description">`. |
| `readingTime` | Yes | number | Minutes. Calculate as `ceil(word_count / 250)`. |
| `featured` | Yes | boolean | Only 1-2 posts should be `true` at any time. |
| `coverImage` | No | string | Path under `public/`. Omit if no cover image. Recommended: 1200x630px. |

### Category Definitions

| Category | Slug | Purpose | Example Topics |
|----------|------|---------|----------------|
| Engineering | `engineering` | Technical deep dives, architecture decisions | "How We Built the Self-Healing Pipeline", "Event-Driven Architecture with Kafka" |
| Benchmarks | `benchmarks` | Agent reliability reports, CLEAR scorecards | "Agent Reliability: February 2026", "Self-Healing Pass@8 Analysis" |
| Changelog | `changelog` | Product updates grouped by Added/Changed/Fixed | "v2.1: GitHub Integration & Visual AI", "January 2026 Updates" |
| Case Studies | `case-study` | Customer outcomes with real metrics | "How Acme Reduced Flaky Tests by 80%", "Enterprise CI/CD Integration" |

### Tag Guidelines

Common tags (reuse these before inventing new ones):

```
# Agents
self-healing, code-analysis, sre, hallucination-detection, visual-ai, security

# Frameworks
clear-framework, langgraph, cognee, playwright

# Topics
reliability, benchmarks, transparency, performance, cost-efficiency

# Product
changelog, new-feature, improvement, bugfix
```

## Changelog Entry Frontmatter

Changelog entries use the same frontmatter but with `category: "changelog"`:

```yaml
---
title: "v2.1 — GitHub Integration & Visual AI"
slug: "v2-1"
date: "2026-02-19"
author:
  name: "Skopaq Engineering"
  avatar: "/icons/skopaq-logo.svg"
category: "changelog"
tags: ["changelog", "new-feature"]
excerpt: "GitHub-based code-aware healing, visual AI comparison, and 12 bug fixes."
readingTime: 3
featured: false
---
```

### Changelog Body Structure

```mdx
## Added

- **GitHub Code-Aware Healing** — Self-healing agent now analyzes git history via GitHub API for production environments where local repo access isn't available.
- **Visual AI Comparison** — Side-by-side screenshot diffing with perceptual hashing.

## Changed

- **Kafka Consumer** — Now checks `REDPANDA_BROKERS` first, falls back to `KAFKA_BOOTSTRAP_SERVERS`.

## Fixed

- **pgvector Type Mismatch** — Search functions now return `DOUBLE PRECISION` instead of `FLOAT`.
- **isoformat URL Encoding** — DateTime values in PostgREST queries use `Z` suffix instead of `+00:00`.

## Removed

- **Neo4j Legacy Config** — Removed stale Neo4j connection strings from Cognee configuration.
```

## Example: Complete Blog Post

```yaml
---
title: "Agent Reliability Benchmarks: February 2026"
slug: "agent-reliability-benchmarks-feb-2026"
date: "2026-02-19"
author:
  name: "Skopaq Engineering"
  avatar: "/icons/skopaq-logo.svg"
category: "benchmarks"
tags: ["benchmarks", "clear-framework", "reliability", "transparency"]
excerpt: "We ran 26 scenarios across 4 agent suites. Here are the results."
readingTime: 8
featured: true
coverImage: "/blog/benchmarks-feb-2026-cover.png"
---
```
