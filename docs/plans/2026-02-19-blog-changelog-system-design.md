# Blog & Changelog System Design

**Date**: 2026-02-19
**Status**: Approved
**Author**: Skopaq Engineering

## Problem

Skopaq has 28+ AI agents, benchmark data proving they work, and zero public presence showing it. There's a dead `/blog` link in the dashboard footer pointing to nothing. Competitors like Anthropic, Vercel, and Linear publish engineering content, changelogs, and transparency reports that build customer trust. We need the same.

## Decision

Build a blog and changelog system in the Next.js dashboard using MDX files stored in the repo. No CMS — content is version-controlled, committed, and deployed via Vercel.

## Architecture

### Content Storage

```
dashboard/
  content/
    blog/
      2026-02-19-agent-benchmarks.mdx
    changelog/
      2026-02-19-v1.mdx
```

### Routes

```
/blog              → Blog index (grid of post cards, category filters)
/blog/[slug]       → Individual post (prose layout, author, reading time)
/changelog         → Changelog timeline (Linear-style, grouped by month)
```

### MDX Frontmatter Standard

```yaml
---
title: "Post Title"
slug: "post-slug"
date: "2026-02-19"
author:
  name: "Skopaq Engineering"
  avatar: "/icons/skopaq-logo.svg"
category: "benchmarks"       # engineering | benchmarks | changelog | case-study
tags: ["tag1", "tag2"]
excerpt: "Short description for cards and SEO."
readingTime: 8
featured: true
coverImage: "/blog/cover.png"
---
```

### Categories

| Category | Purpose |
|----------|---------|
| Engineering | Technical deep dives, architecture decisions, methodology |
| Benchmarks | Agent reliability reports, CLEAR scorecards, comparisons |
| Changelog | Product updates, new features, improvements, fixes |
| Case Studies | Customer success stories, real-world outcomes |

### New Files

| File | Purpose |
|------|---------|
| `dashboard/lib/content.ts` | MDX loading, frontmatter parsing, sorting |
| `dashboard/app/blog/page.tsx` | Blog index page |
| `dashboard/app/blog/[slug]/page.tsx` | Individual post page |
| `dashboard/app/changelog/page.tsx` | Changelog timeline |
| `dashboard/components/blog/post-card.tsx` | Post preview card |
| `dashboard/components/blog/post-layout.tsx` | Full post layout |
| `dashboard/components/blog/category-filter.tsx` | Category filter |
| `dashboard/components/blog/changelog-entry.tsx` | Changelog entry |
| `dashboard/components/blog/blog-header.tsx` | Shared public nav |
| `dashboard/components/blog/mdx-components.tsx` | Custom MDX components |
| `dashboard/content/blog/*.mdx` | Blog posts |
| `dashboard/content/changelog/*.mdx` | Changelog entries |

### Modified Files

| File | Change |
|------|--------|
| `dashboard/middleware.ts` | Add `/blog(.*)` and `/changelog(.*)` to public routes |
| `dashboard/package.json` | Add `next-mdx-remote`, `gray-matter`, `@tailwindcss/typography`, `reading-time`, `feed` |
| `dashboard/tailwind.config.ts` | Add typography plugin |
| `dashboard/components/landing/landing-page.tsx` | Add "Latest from the Blog" section |

### Custom MDX Components

| Component | Use Case |
|-----------|----------|
| `<BenchmarkChart />` | Render benchmark data visualization |
| `<CLEARScorecard />` | CLEAR framework 5-axis display |
| `<Callout />` | Info/warning/tip callout boxes |
| `<ComparisonTable />` | Product or before/after comparisons |

### Design

Follows the `/trust` page pattern — standalone public pages:
- No sidebar, own header/footer
- Dark-first, electric teal accent
- Glass cards for post previews
- `max-w-3xl` content width for readability
- Inter body, Plus Jakarta Sans for titles
- `prose prose-neutral dark:prose-invert` for post content

### Dependencies

```
next-mdx-remote          # Server-side MDX rendering
gray-matter               # Frontmatter parsing
@tailwindcss/typography   # Prose styling
reading-time              # Auto-calculate reading time
feed                      # RSS feed generation
```

### Public Route Config

Add to `middleware.ts` `isPublicRoute` matcher:
```ts
"/blog(.*)",
"/changelog(.*)",
```

### Launch Content

First post: "Agent Reliability Benchmarks: February 2026" — the benchmark data we just generated, turned into a transparency report with CLEAR scorecard, per-agent breakdowns, and methodology.
