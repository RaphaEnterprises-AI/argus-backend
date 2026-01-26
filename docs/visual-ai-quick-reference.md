# Visual AI Feature - Quick Reference Guide

## Overview

The Visual AI feature provides Applitools-like visual regression testing using Claude Vision AI. It's fully integrated between the frontend dashboard and backend API.

## Quick Start

### Running a Visual Test

```typescript
// Frontend hook usage
const runVisualTest = useRunVisualTest();

await runVisualTest.mutateAsync({
  projectId: 'project-123',
  url: 'https://example.com',
  name: 'homepage-test', // optional
  viewport: '1920x1080', // optional, default: 1440x900
  threshold: 0.1, // optional, 0-1 scale
});
```

### Running a Responsive Test

```typescript
const runResponsiveTest = useRunResponsiveTest();

await runResponsiveTest.mutateAsync({
  projectId: 'project-123',
  url: 'https://example.com',
  viewports: [
    { name: 'Mobile', width: 375, height: 667 },
    { name: 'Tablet', width: 768, height: 1024 },
    { name: 'Desktop', width: 1920, height: 1080 },
  ],
  threshold: 0.1,
});
```

### Running Cross-Browser Tests

```typescript
const crossBrowserTest = useCrossBrowserTest();

await crossBrowserTest.mutateAsync({
  url: 'https://example.com',
  browsers: ['chromium', 'firefox', 'webkit'],
  projectId: 'project-123',
  viewport: { width: 1920, height: 1080 },
  compareAfterCapture: true, // Compare browsers against each other
});
```

### Running Accessibility Analysis

```typescript
const accessibilityAnalysis = useAccessibilityAnalysis();

const result = await accessibilityAnalysis.mutateAsync({
  url: 'https://example.com',
  projectId: 'project-123',
  wcagLevel: 'AA', // 'A', 'AA', or 'AAA'
});

// Result includes:
// - overall_score: 0-100
// - level_compliance: 'A' | 'AA' | 'AAA' | 'None'
// - issues: Array of accessibility issues
// - passed_criteria: Array of passed WCAG criteria
```

### Getting AI Explanation

```typescript
const { data: aiExplanation } = useAIExplain(comparisonId, comparison);

// Returns:
// - summary: Overall summary
// - changes_explained: Array of change details with risk levels
// - recommendations: Array of actionable recommendations
// - overall_assessment: Whether to approve or investigate
```

## Backend API Endpoints

### Screenshot Capture

```bash
POST /api/v1/visual/capture
{
  "url": "https://example.com",
  "viewport": { "width": 1920, "height": 1080 },
  "browser": "chromium",
  "project_id": "project-123",
  "name": "test-name",
  "wait_for": "#element-id", // optional
  "full_page": false
}
```

### Responsive Testing

```bash
# Step 1: Capture
POST /api/v1/visual/responsive/capture
{
  "url": "https://example.com",
  "viewports": [
    { "name": "mobile", "width": 375, "height": 667 },
    { "name": "desktop", "width": 1920, "height": 1080 }
  ],
  "project_id": "project-123"
}

# Step 2: Compare
POST /api/v1/visual/responsive/compare
{
  "url": "https://example.com",
  "viewports": [...], // Same as capture
  "project_id": "project-123",
  "threshold": 0.1
}
```

### Cross-Browser Testing

```bash
# Step 1: Capture
POST /api/v1/visual/browsers/capture
{
  "url": "https://example.com",
  "browsers": ["chromium", "firefox", "webkit"],
  "viewport": { "width": 1920, "height": 1080 },
  "project_id": "project-123"
}

# Step 2: Compare
POST /api/v1/visual/browsers/compare
{
  "url": "https://example.com",
  "browsers": ["chromium", "firefox", "webkit"],
  "project_id": "project-123"
}
```

### Accessibility Analysis

```bash
# Step 1: Capture screenshot (if not already done)
POST /api/v1/visual/capture
{
  "url": "https://example.com",
  "viewport": { "width": 1920, "height": 1080 },
  "browser": "chromium"
}

# Step 2: Analyze
POST /api/v1/visual/accessibility/analyze?snapshot_id={id}&wcag_level=AA
```

### AI Explanation

```bash
POST /api/v1/visual/ai/explain?comparison_id={id}
```

### Baseline Management

```bash
# Create/Update baseline
POST /api/v1/visual/baselines
{
  "url": "https://example.com",
  "name": "homepage-baseline",
  "project_id": "project-123",
  "viewport": { "width": 1920, "height": 1080 },
  "browser": "chromium"
}

# Get baseline
GET /api/v1/visual/baselines/{baseline_id}

# Get baseline history
GET /api/v1/visual/baselines/{baseline_id}/history?limit=20

# List baselines
GET /api/v1/visual/baselines?project_id={id}&limit=50
```

### Approval Workflow

```bash
# Approve changes
POST /api/v1/visual/comparisons/{comparison_id}/approve
{
  "change_ids": ["change-1", "change-2"], // optional, null = approve all
  "notes": "Looks good",
  "update_baseline": true
}

# Reject changes
POST /api/v1/visual/comparisons/{comparison_id}/reject
{
  "notes": "Regression detected",
  "create_issue": true
}
```

## Browser Mapping

Frontend browser names map to backend browser names:

```typescript
const browserMapping = {
  'chrome': 'chromium',
  'firefox': 'firefox',
  'safari': 'webkit',
  'edge': 'chromium',
};
```

## Viewport Presets

Default viewport configurations:

```typescript
const VIEWPORTS = [
  { name: 'Mobile S', width: 320, height: 568 },
  { name: 'Mobile M', width: 375, height: 667 },
  { name: 'Mobile L', width: 425, height: 812 },
  { name: 'Tablet', width: 768, height: 1024 },
  { name: 'Laptop', width: 1024, height: 768 },
  { name: 'Desktop', width: 1440, height: 900 },
  { name: 'Desktop L', width: 1920, height: 1080 },
];
```

## Status Types

Visual comparison statuses:

```typescript
type ComparisonStatus = 'match' | 'mismatch' | 'new' | 'pending' | 'error';
```

- **match**: Visual appearance matches baseline
- **mismatch**: Visual differences detected
- **new**: First capture (no baseline exists)
- **pending**: Awaiting review
- **error**: Capture or comparison failed

## Error Handling

All hooks return React Query mutation objects with error states:

```typescript
const { mutateAsync, isPending, isError, error } = useRunVisualTest();

try {
  await mutateAsync({ projectId, url });
} catch (error) {
  console.error('Test failed:', error);
  // Handle error (show toast, etc.)
}
```

## Cost Tracking

All API calls are tracked for cost:

```typescript
// Each comparison response includes:
{
  "cost_usd": 0.15, // API cost in USD
  "analysis_cost_usd": 0.10 // AI analysis cost
}
```

## Best Practices

### 1. Use Appropriate Thresholds

```typescript
// Strict comparison (0.1% tolerance)
threshold: 0.001

// Standard comparison (1% tolerance)
threshold: 0.01

// Lenient comparison (5% tolerance)
threshold: 0.05
```

### 2. Name Your Tests Clearly

```typescript
await runVisualTest.mutateAsync({
  projectId,
  url: 'https://example.com/pricing',
  name: 'pricing-page-logged-out-desktop',
  viewport: '1920x1080',
});
```

### 3. Use Context for Better AI Analysis

```typescript
await compareSnapshots({
  baseline_id,
  current_url,
  context: 'Updated pricing to reflect new tier structure',
  pr_description: 'Adds enterprise tier pricing',
  git_diff: '...', // Recent code changes
});
```

### 4. Batch Responsive Tests

```typescript
// Good: Single call for multiple viewports
await runResponsiveTest.mutateAsync({
  url,
  viewports: [mobile, tablet, desktop],
});

// Avoid: Multiple individual calls
for (const vp of viewports) {
  await runVisualTest.mutateAsync({ url, viewport: vp });
}
```

## TypeScript Types

Key types available:

```typescript
import type {
  VisualComparison,
  VisualBaseline,
  ViewportConfig,
  ResponsiveCompareResult,
  CrossBrowserTestResult,
  AccessibilityAnalysisResult,
  AIExplainResponse,
} from '@/lib/hooks/use-visual';
```

## Component Usage

```tsx
import { useRunVisualTest } from '@/lib/hooks/use-visual';

function MyComponent() {
  const runTest = useRunVisualTest();

  const handleTest = async () => {
    try {
      const result = await runTest.mutateAsync({
        projectId: 'my-project',
        url: 'https://example.com',
      });

      console.log('Test complete:', result);
    } catch (error) {
      console.error('Test failed:', error);
    }
  };

  return (
    <button
      onClick={handleTest}
      disabled={runTest.isPending}
    >
      {runTest.isPending ? 'Running...' : 'Run Test'}
    </button>
  );
}
```

## Common Patterns

### Loading State

```tsx
{runTest.isPending && (
  <Loader2 className="animate-spin" />
)}
```

### Error State

```tsx
{runTest.isError && (
  <Alert variant="error">
    {runTest.error.message}
  </Alert>
)}
```

### Success State

```tsx
{runTest.isSuccess && runTest.data && (
  <p>Test completed: {runTest.data.status}</p>
)}
```

## Debugging

Enable debug logging:

```typescript
// In browser console
localStorage.setItem('debug', 'visual-ai:*');

// View logs
// visual-ai:capture Screenshot captured
// visual-ai:compare Comparison complete
// visual-ai:ai AI explanation generated
```

## Performance Tips

1. **Cache screenshots** - Baselines are automatically versioned
2. **Use appropriate viewports** - Don't test every possible size
3. **Batch operations** - Use responsive/cross-browser endpoints
4. **Set timeouts** - Long captures may need longer timeouts
5. **Monitor costs** - Track API usage in dashboard

## Support

For issues or questions:
- Check backend logs: `src/api/visual_ai.py`
- Review hook implementation: `dashboard/lib/hooks/use-visual.ts`
- See test examples: `dashboard/__tests__/visual-ai-integration.test.tsx`
- Read full docs: `VISUAL_AI_FIXES.md`
