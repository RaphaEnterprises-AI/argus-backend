---
name: content-creation
description: This skill should be used when the user asks to "write a blog post", "create a changelog entry", "add content to the blog", "publish an update", "write about benchmarks", "create a case study", or mentions creating MDX content for the Skopaq blog or changelog. Provides the full workflow, frontmatter template, quality checklist, and writing standards.
version: 1.0.0
author: Skopaq Engineering
---

# Skopaq Content Creation

Create blog posts and changelog entries for the Skopaq public website. All content lives as MDX files in the Next.js dashboard, version-controlled in git, deployed via Vercel.

## Content Locations

```
dashboard/content/blog/{YYYY-MM-DD}-{slug}.mdx       # Blog posts
dashboard/content/changelog/{YYYY-MM-DD}-{slug}.mdx   # Changelog entries
```

## Workflow

### 1. Determine Content Type

| Type | Category | When |
|------|----------|------|
| Technical deep dive | `engineering` | Architecture decisions, methodology, how-we-built-it |
| Reliability report | `benchmarks` | Agent performance data, CLEAR scorecards, comparisons |
| Product update | `changelog` | New features, improvements, fixes |
| Customer story | `case-study` | Real-world outcomes, success metrics |

### 2. Create MDX File

Every file requires complete frontmatter. Refer to `references/frontmatter-template.md` for the full template and field descriptions.

**Critical rules:**
- `slug` must match the filename (minus date prefix and `.mdx`)
- `date` must be ISO 8601 format (`YYYY-MM-DD`)
- `category` must be one of: `engineering`, `benchmarks`, `changelog`, `case-study`
- `readingTime` is estimated at ~250 words per minute
- `coverImage` path starts with `/blog/` and the image must exist in `dashboard/public/blog/`
- `author.avatar` defaults to `/icons/skopaq-logo.svg`

### 3. Write Content Body

**Voice and tone:**
- Write in first-person plural ("we") for Skopaq Engineering posts
- Technical but accessible — assume the reader is a software engineer, not an AI specialist
- Lead with the outcome/result, then explain how
- Include real data — never use placeholder metrics

**Structure for blog posts:**
1. Opening hook (1-2 sentences with the key finding or announcement)
2. Context (why this matters)
3. Technical content (methodology, implementation, results)
4. Conclusion with forward-looking statement

**Structure for changelog entries:**
1. Version/date heading
2. Grouped by: Added, Changed, Fixed, Removed
3. Each item: one-line summary + optional detail paragraph

### 4. Use Custom MDX Components

Available components (defined in `dashboard/components/blog/mdx-components.tsx`):

| Component | Usage |
|-----------|-------|
| `<Callout type="info\|warning\|tip">` | Highlighted callout boxes |
| `<BenchmarkTable data={[...]} />` | Agent performance data tables |

Standard markdown elements (h1-h3, links, code blocks, tables, images) are styled automatically via the prose typography plugin.

### 5. Quality Checklist

Before committing any content:

- [ ] Frontmatter is complete (all required fields present)
- [ ] `slug` matches filename
- [ ] `category` is one of the 4 valid values
- [ ] `excerpt` is under 160 characters (SEO meta description length)
- [ ] `readingTime` is calculated (word count / 250, rounded up)
- [ ] No placeholder data — all metrics, dates, and claims are real
- [ ] Cover image exists at the specified path (or `coverImage` is omitted)
- [ ] Code blocks have language annotations (```python, ```bash, etc.)
- [ ] Links use relative paths for internal pages (`/blog/other-post`)
- [ ] No "Argus" references — use "Skopaq" everywhere (per branding rules)
- [ ] Content renders correctly (`cd dashboard && npm run build`)

### 6. Verify Build

After creating content, always run:

```bash
cd dashboard && npm run build
```

This catches MDX parsing errors, missing imports, and broken frontmatter.

## Additional Resources

### Reference Files

- **`references/frontmatter-template.md`** — Complete frontmatter template with field descriptions and examples for both blog posts and changelog entries

## Naming Conventions

| Item | Pattern | Example |
|------|---------|---------|
| Blog filename | `{YYYY-MM-DD}-{slug}.mdx` | `2026-02-19-agent-reliability-benchmarks.mdx` |
| Changelog filename | `{YYYY-MM-DD}-{slug}.mdx` | `2026-02-19-v1.mdx` |
| Cover image | `/blog/{slug}-cover.png` | `/blog/agent-reliability-benchmarks-cover.png` |
| Tags | lowercase, hyphenated | `self-healing`, `clear-framework` |
