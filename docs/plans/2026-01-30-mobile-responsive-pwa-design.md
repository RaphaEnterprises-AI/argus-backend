# Mobile Responsive & PWA Design for Skopaq Dashboard

**Date**: 2026-01-30
**Status**: Approved
**Goal**: Transform Skopaq dashboard into a mobile-first, progressive web app that excels at "monitor and respond" on mobile while preserving full desktop power.

## Overview

This design addresses comprehensive mobile responsiveness issues and adds full PWA capabilities to the Skopaq dashboard. The approach is systematic - building a design system foundation, then migrating pages, then layering PWA features.

### Mobile Design Philosophy

> **"Monitor and respond, don't create or debug"**

| Activity | Mobile Fit | Rationale |
|----------|------------|-----------|
| "Did the build pass?" | Perfect | 5-second glance |
| "What failed overnight?" | Perfect | Scan alerts, triage |
| "Trigger a re-run" | Good | One-tap action |
| "Approve a healed test" | Good | Review + confirm |
| "Write a new test" | Poor | Needs keyboard, precision |
| "Debug a failure" | Poor | Needs screen real estate |
| "Analyze correlations" | Poor | Complex data exploration |

## Architecture

### Three-Phase Approach

```
Phase 1: Foundation (Design System)
├── Responsive tokens (spacing, breakpoints)
├── Layout primitives (ResponsiveGrid, MobileStack)
├── Mobile-optimized data display components
└── ~1 week

Phase 2: Page Migration
├── Dashboard (priority)
├── Tests, Test Results
├── Notifications, Activity
├── Settings, Profile
├── Complex pages (simplified mobile views)
└── ~2 weeks

Phase 3: PWA Layer
├── Web App Manifest
├── Service Worker (offline + caching)
├── Push Notifications
├── Install prompt UX
├── Background sync for actions
└── ~1 week
```

### Mobile Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Touch-first** | 44px minimum tap targets, swipe gestures |
| **Thumb-zone aware** | Critical actions in bottom 40% of screen |
| **Progressive disclosure** | Summary → tap for details |
| **Offline-resilient** | Cached data + queued actions |
| **Notification-driven** | Push for failures, healings, approvals |

---

## Phase 1: Design System Foundation

### Breakpoint Strategy (Mobile-First)

```typescript
// tailwind.config.ts - Enhanced breakpoints
screens: {
  'xs': '375px',   // Small phones (iPhone SE)
  'sm': '640px',   // Large phones / small tablets
  'md': '768px',   // Tablets portrait
  'lg': '1024px',  // Tablets landscape / small laptops
  'xl': '1280px',  // Desktops
  '2xl': '1536px', // Large desktops
}
```

**Key change**: Add `xs` for small phones, design base styles for 320px minimum.

### Spacing Scale (8px Grid)

```typescript
// Consistent spacing tokens
spacing: {
  'page': {
    'x': 'px-4 sm:px-6 lg:px-8',      // Page horizontal padding
    'y': 'py-4 sm:py-6 lg:py-8',      // Page vertical padding
  },
  'section': 'space-y-4 sm:space-y-6', // Between sections
  'card': 'p-4 sm:p-6',                // Card internal padding
  'stack': 'space-y-3 sm:space-y-4',   // Stacked elements
  'inline': 'gap-2 sm:gap-3',          // Inline elements
}
```

### Layout Tokens

```typescript
// New CSS custom properties in globals.css
:root {
  --sidebar-width: 256px;
  --sidebar-collapsed: 64px;
  --header-height: 56px;
  --mobile-header-height: 48px;
  --bottom-nav-height: 64px;
  --safe-area-bottom: env(safe-area-inset-bottom, 0px);
}
```

### Grid System

| Viewport | Columns | Use Case |
|----------|---------|----------|
| Base (< sm) | 1 | Single column stack |
| sm (640px+) | 2 | Side-by-side cards |
| md (768px+) | 2-3 | Tablet layouts |
| lg (1024px+) | 3-4 | Desktop grids |
| xl+ | 4-6 | Data-dense views |

### Typography Scale (Mobile-Adjusted)

```css
/* Mobile-first type scale */
.text-page-title {
  @apply text-xl sm:text-2xl lg:text-3xl font-bold tracking-tight;
}
.text-section-title {
  @apply text-lg sm:text-xl font-semibold;
}
.text-card-title {
  @apply text-base sm:text-lg font-medium;
}
.text-body {
  @apply text-sm sm:text-base;
}
.text-caption {
  @apply text-xs sm:text-sm text-muted-foreground;
}
```

---

## Phase 1: Responsive Component Primitives

### Core Layout Components

#### 1. PageContainer - Consistent page wrapper

```typescript
// components/layout/page-container.tsx
interface PageContainerProps {
  children: React.ReactNode;
  header?: React.ReactNode;      // Sticky header content
  bottomNav?: boolean;           // Show mobile bottom nav
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
}

// Handles: sidebar offset, safe areas, scroll behavior
```

#### 2. ResponsiveGrid - Adaptive grid layout

```typescript
// components/layout/responsive-grid.tsx
interface ResponsiveGridProps {
  children: React.ReactNode;
  cols?: {
    base?: 1 | 2;      // Mobile: usually 1
    sm?: 1 | 2;        // Large phone: 1-2
    md?: 2 | 3;        // Tablet: 2-3
    lg?: 3 | 4;        // Desktop: 3-4
  };
  gap?: 'sm' | 'md' | 'lg';
}

// Example usage:
<ResponsiveGrid cols={{ base: 1, sm: 2, lg: 4 }}>
  <MetricCard />
  <MetricCard />
  <MetricCard />
  <MetricCard />
</ResponsiveGrid>
```

#### 3. MobileStack - Vertical stacking with smart spacing

```typescript
// components/layout/mobile-stack.tsx
interface MobileStackProps {
  children: React.ReactNode;
  spacing?: 'tight' | 'normal' | 'loose';
  dividers?: boolean;            // Add dividers between items
  collapsible?: boolean;         // Allow collapse on mobile
}
```

### Data Display Components

#### 4. ResponsiveTable - Cards on mobile, table on desktop

```typescript
// components/ui/responsive-table.tsx
interface ResponsiveTableProps<T> {
  data: T[];
  columns: ColumnDef<T>[];
  mobileCard: (item: T) => React.ReactNode;  // Card renderer
  breakpoint?: 'sm' | 'md';                   // Switch point
  onRowClick?: (item: T) => void;
}

// Mobile: renders mobileCard for each row
// Desktop: renders full DataTable
```

#### 5. StatBar - Horizontal scrolling stats for mobile

```typescript
// components/ui/stat-bar.tsx
interface StatBarProps {
  stats: Array<{
    label: string;
    value: string | number;
    icon?: React.ReactNode;
    trend?: 'up' | 'down' | 'neutral';
  }>;
  variant?: 'pills' | 'compact' | 'cards';
}

// Horizontal scroll on mobile, flex-wrap on desktop
```

#### 6. CollapsibleSection - Progressive disclosure

```typescript
// components/ui/collapsible-section.tsx
interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  mobileCollapsed?: boolean;  // Auto-collapse on mobile only
  count?: number;             // Badge showing item count
}
```

### Navigation Components

#### 7. MobileBottomNav - Thumb-zone navigation

```typescript
// components/layout/mobile-bottom-nav.tsx
// Fixed bottom bar with 4-5 primary actions
// Only visible on mobile (< lg breakpoint)

items: [
  { icon: Home, label: 'Home', href: '/dashboard' },
  { icon: TestTube, label: 'Tests', href: '/tests' },
  { icon: Bell, label: 'Alerts', href: '/notifications', badge: unreadCount },
  { icon: MessageSquare, label: 'Chat', href: '/chat' },
  { icon: Menu, label: 'More', onClick: openDrawer },
]
```

#### 8. SwipeableDrawer - Mobile-native feel

```typescript
// components/ui/swipeable-drawer.tsx
// Replaces Sheet for mobile - adds swipe-to-dismiss
// Used for: filters, details panels, quick actions
```

---

## Phase 2: Page Migration Strategy

### Priority Order

| Priority | Page | Reason |
|----------|------|--------|
| P0 | Dashboard | First thing users see, most visited |
| P0 | Notifications | Critical for mobile "respond" use case |
| P1 | Tests | Core functionality |
| P1 | Test Results | View outcomes on the go |
| P2 | Projects | Management basics |
| P2 | Settings/Profile | Account access |
| P3 | Complex pages | Simplified or "Continue on desktop" |

### Dashboard Migration (Before → After)

**Current Issues:**
- Grid jumps 1 → 3 columns
- Project selector bar wraps awkwardly
- No mobile quick actions

**New Structure:**

```
Mobile (< sm)                    Desktop (lg+)
┌─────────────────────┐         ┌─────────────────────────────────┐
│ ▼ Project Selector  │         │ Project ▼ │ Stats │ Refresh    │
├─────────────────────┤         ├───────────┴───────┴─────────────┤
│ Quality Score Card  │         │ Hero Card (full width)          │
│ (hero, full width)  │         ├──────────┬──────────┬───────────┤
├─────────────────────┤         │ Metric   │ Metric   │ Metric    │
│ ← Stat Pills Scroll→│         ├──────────┴──────────┼───────────┤
├─────────────────────┤         │ Chart (2 cols)      │ Activity  │
│ Chart (collapsible) │         │                     │ Feed      │
├─────────────────────┤         ├─────────────────────┤           │
│ Recent Runs         │         │ Recent Runs Table   │           │
│ (card list)         │         ├─────────────────────┴───────────┤
├─────────────────────┤         │ Quick Actions                   │
│ [Bottom Nav]        │         └─────────────────────────────────┘
└─────────────────────┘
```

### Tests Page Migration

**Current Issues:**
- Header has two rows that wrap unpredictably
- Stats hidden entirely on mobile
- DataTable not mobile-friendly
- Activity sidebar fixed width

**New Structure:**

```
Mobile                           Desktop
┌─────────────────────┐         ┌──────────────────────┬──────────┐
│ ▼ Project │ + New   │         │ Project▼ │Stats│Live│ + New    │
├─────────────────────┤         ├──────────────────────┼──────────┤
│ [Live] 2 running    │         │                      │ Activity │
├─────────────────────┤         │  Tests DataTable     │ Feed     │
│ ← Recent Runs Scroll│         │  (full columns)      │ (toggle) │
├─────────────────────┤         │                      │          │
│ ┌─────────────────┐ │         │                      │          │
│ │ Test Card       │ │         │                      │          │
│ │ Name    [Run ▶] │ │         │                      │          │
│ │ 5 steps • high  │ │         │                      │          │
│ └─────────────────┘ │         │                      │          │
│ ┌─────────────────┐ │         │                      │          │
│ │ Test Card       │ │         │                      │          │
│ └─────────────────┘ │         │                      │          │
├─────────────────────┤         └──────────────────────┴──────────┘
│ [Bottom Nav]        │
└─────────────────────┘
```

### Complex Pages Strategy

For pages like **Orchestrator**, **Correlations**, **Visual Diff**:

```typescript
// Pattern: SimplifiedMobileView wrapper
<SimplifiedMobileView
  title="Orchestrator"
  summary={<OrchestratorSummaryCard session={session} />}
  desktopFeature="interactive graph visualization"
>
  {/* Full desktop component */}
  <OrchestratorVisualizer />
</SimplifiedMobileView>

// Mobile shows: summary card + "Open on desktop for full view" link
// Desktop shows: full interactive component
```

---

## Phase 3: PWA Implementation

### Web App Manifest

```json
// dashboard/public/manifest.json
{
  "name": "Skopaq - E2E Testing Agent",
  "short_name": "Skopaq",
  "description": "Autonomous end-to-end testing with AI",
  "start_url": "/dashboard",
  "display": "standalone",
  "background_color": "#0a0a0b",
  "theme_color": "#14b8a6",
  "orientation": "portrait-primary",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/icons/icon-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ],
  "screenshots": [
    { "src": "/screenshots/dashboard-mobile.png", "sizes": "390x844", "type": "image/png", "form_factor": "narrow" },
    { "src": "/screenshots/dashboard-desktop.png", "sizes": "1920x1080", "type": "image/png", "form_factor": "wide" }
  ],
  "shortcuts": [
    { "name": "Run Tests", "url": "/tests?action=run", "icon": "/icons/play.png" },
    { "name": "View Alerts", "url": "/notifications", "icon": "/icons/bell.png" }
  ]
}
```

### Service Worker Strategy

```typescript
// dashboard/public/sw.js (using Workbox)

// 1. CACHING STRATEGY
const strategies = {
  // Static assets: Cache-first (fonts, icons, images)
  static: new CacheFirst({
    cacheName: 'argus-static-v1',
    maxEntries: 100,
    maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
  }),

  // API data: Network-first with fallback
  api: new NetworkFirst({
    cacheName: 'argus-api-v1',
    maxEntries: 50,
    maxAgeSeconds: 5 * 60, // 5 min stale data acceptable
  }),

  // Pages: Stale-while-revalidate
  pages: new StaleWhileRevalidate({
    cacheName: 'argus-pages-v1',
  }),
};

// 2. OFFLINE FALLBACK
// Show cached dashboard with "You're offline" banner
// Queue actions for sync when back online
```

### Offline Data Model

| Data Type | Offline Behavior |
|-----------|------------------|
| Dashboard stats | Show cached, mark "Last updated: X" |
| Test list | Full offline access (cached) |
| Test results | Cached recent results |
| Notifications | Cached, queue mark-as-read |
| Run test action | Queue for background sync |
| Create/edit | Queue with optimistic UI |

### Background Sync

```typescript
// Queue actions when offline, sync when online
// dashboard/lib/pwa/background-sync.ts

interface QueuedAction {
  id: string;
  type: 'run_test' | 'approve_healing' | 'dismiss_alert';
  payload: Record<string, unknown>;
  timestamp: number;
}

// Register sync event
navigator.serviceWorker.ready.then((reg) => {
  return reg.sync.register('argus-action-sync');
});

// In service worker: process queue on sync event
self.addEventListener('sync', (event) => {
  if (event.tag === 'argus-action-sync') {
    event.waitUntil(processActionQueue());
  }
});
```

### Push Notifications

```typescript
// dashboard/lib/pwa/push-notifications.ts

// 1. NOTIFICATION TYPES
type NotificationType =
  | 'test_failed'      // Critical - immediate
  | 'test_passed'      // Optional - batched
  | 'healing_complete' // Important - immediate
  | 'approval_needed'  // Action required - immediate
  | 'schedule_started' // Informational - batched

// 2. USER PREFERENCES (stored in Supabase)
interface NotificationPrefs {
  enabled: boolean;
  test_failed: boolean;      // Default: true
  test_passed: boolean;      // Default: false
  healing_complete: boolean; // Default: true
  approval_needed: boolean;  // Default: true
  quiet_hours?: { start: string; end: string };
}

// 3. BACKEND INTEGRATION
// Push sent from: src/services/push_notification_service.py
// Uses: web-push library with VAPID keys
```

### Install Prompt UX

```typescript
// components/pwa/install-prompt.tsx

// Smart prompt timing:
// - Not on first visit
// - After 2+ sessions OR 3+ test runs
// - Dismissable, remembers "not now" for 7 days
// - Shows benefits: "Get notified of failures instantly"

<InstallPrompt
  trigger="engagement"  // or "manual" for settings page
  benefits={[
    "Instant push notifications for test failures",
    "Quick access from home screen",
    "Works offline - view results anywhere",
  ]}
/>
```

### PWA Status Component

```typescript
// components/pwa/pwa-status.tsx
// Shows in settings page

<PWAStatus />
// Displays:
// - Installation status (installed / not installed)
// - Notification permission (granted / denied / prompt)
// - Offline data size (e.g., "12.4 MB cached")
// - Last sync time
// - "Clear cache" action
```

---

## Testing & Error Handling

### Responsive Testing Strategy

#### Viewport Testing Matrix

| Device | Width | Test Focus |
|--------|-------|------------|
| iPhone SE | 375px | Minimum viable, no overflow |
| iPhone 14 | 390px | Standard mobile |
| iPhone 14 Pro Max | 430px | Large phone |
| iPad Mini | 768px | Tablet portrait |
| iPad Pro | 1024px | Tablet landscape / sidebar toggle |
| Desktop | 1280px+ | Full experience |

#### Playwright Visual Tests

```typescript
// tests/e2e/responsive.spec.ts

const viewports = [
  { name: 'mobile', width: 375, height: 667 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1280, height: 800 },
];

for (const vp of viewports) {
  test(`dashboard renders correctly on ${vp.name}`, async ({ page }) => {
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.goto('/dashboard');

    // No horizontal overflow
    const body = await page.locator('body');
    const scrollWidth = await body.evaluate(el => el.scrollWidth);
    const clientWidth = await body.evaluate(el => el.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth);

    // Visual snapshot
    await expect(page).toHaveScreenshot(`dashboard-${vp.name}.png`);
  });
}
```

### PWA Testing

#### Lighthouse CI Thresholds

```yaml
# lighthouserc.js
ci:
  assert:
    assertions:
      categories:pwa: ['error', { minScore: 0.9 }]
      installable-manifest: 'error'
      service-worker: 'error'
      offline-start-url: 'error'
      maskable-icon: 'warn'
```

#### Offline Testing

```typescript
// tests/e2e/pwa-offline.spec.ts

test('shows cached dashboard when offline', async ({ page, context }) => {
  // Visit while online to populate cache
  await page.goto('/dashboard');
  await page.waitForSelector('[data-testid="dashboard-loaded"]');

  // Go offline
  await context.setOffline(true);

  // Reload - should show cached content
  await page.reload();
  await expect(page.locator('[data-testid="offline-banner"]')).toBeVisible();
  await expect(page.locator('[data-testid="metric-card"]')).toHaveCount(4);
});
```

### Error Handling

#### Offline Error States

```typescript
// components/ui/offline-aware.tsx

interface OfflineAwareProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  showBanner?: boolean;
}

// Wraps components that need network
// Shows graceful fallback when offline
```

#### Network Error Handling

| Scenario | User Feedback | Recovery |
|----------|---------------|----------|
| Offline | Banner: "You're offline" | Auto-dismiss on reconnect |
| API timeout | Toast: "Request timed out" | Retry button |
| 401/403 | Redirect to login | Preserve intended destination |
| 500 | Error card with retry | Exponential backoff |
| Queued action failed | Toast: "Action failed to sync" | Manual retry in queue UI |

---

## File Structure

```
dashboard/
├── public/
│   ├── manifest.json            # NEW: PWA manifest
│   ├── sw.js                    # NEW: Service worker
│   ├── icons/
│   │   ├── icon-192.png         # NEW: PWA icons
│   │   ├── icon-512.png
│   │   └── icon-maskable.png
│   └── screenshots/             # NEW: Install prompt screenshots
│
├── app/
│   ├── layout.tsx               # MODIFY: Add PWA meta tags, bottom nav
│   ├── globals.css              # MODIFY: Add responsive tokens
│   └── (pages...)               # MODIFY: Migrate to new components
│
├── components/
│   ├── layout/
│   │   ├── page-container.tsx   # NEW: Consistent page wrapper
│   │   ├── responsive-grid.tsx  # NEW: Adaptive grid
│   │   ├── mobile-stack.tsx     # NEW: Vertical stacking
│   │   ├── mobile-bottom-nav.tsx# NEW: Bottom navigation
│   │   └── sidebar.tsx          # MODIFY: Better mobile behavior
│   │
│   ├── ui/
│   │   ├── responsive-table.tsx # NEW: Cards on mobile
│   │   ├── stat-bar.tsx         # NEW: Scrolling stats
│   │   ├── collapsible-section.tsx # NEW: Progressive disclosure
│   │   ├── swipeable-drawer.tsx # NEW: Mobile drawers
│   │   ├── offline-aware.tsx    # NEW: Offline wrapper
│   │   └── simplified-mobile-view.tsx # NEW: Complex page wrapper
│   │
│   ├── pwa/
│   │   ├── install-prompt.tsx   # NEW: Install UX
│   │   ├── pwa-status.tsx       # NEW: Settings status
│   │   ├── offline-banner.tsx   # NEW: Offline indicator
│   │   ├── update-prompt.tsx    # NEW: SW update UX
│   │   └── notification-settings.tsx # NEW: Push preferences
│   │
│   └── dashboard/
│       └── (existing...)        # MODIFY: Use new primitives
│
├── lib/
│   ├── pwa/
│   │   ├── service-worker.ts    # NEW: SW registration
│   │   ├── push-notifications.ts# NEW: Push subscription
│   │   ├── background-sync.ts   # NEW: Action queue
│   │   └── cache-manager.ts     # NEW: Cache utilities
│   │
│   └── hooks/
│       ├── use-media-query.ts   # NEW: Responsive hooks
│       ├── use-online-status.ts # NEW: Network detection
│       └── use-install-prompt.ts# NEW: PWA install
│
├── tailwind.config.ts           # MODIFY: Add tokens
│
└── tests/
    ├── e2e/
    │   ├── responsive.spec.ts   # NEW: Viewport tests
    │   └── pwa-offline.spec.ts  # NEW: PWA tests
    └── components/
        └── responsive-*.test.tsx # NEW: Component tests
```

---

## Implementation Timeline

```
Week 1: Foundation
├── Day 1-2: Tailwind tokens + globals.css updates
├── Day 3-4: Layout primitives (PageContainer, ResponsiveGrid, MobileStack)
├── Day 5: Data components (ResponsiveTable, StatBar)
└── Day 6-7: Navigation (MobileBottomNav, SwipeableDrawer)

Week 2: Page Migration
├── Day 1-2: Dashboard page
├── Day 3: Notifications page
├── Day 4-5: Tests + Test Results pages
├── Day 6: Projects, Settings, Profile
└── Day 7: Complex pages (simplified views)

Week 3: PWA Layer
├── Day 1-2: Manifest + Service Worker + Caching
├── Day 3-4: Offline support + Background sync
├── Day 5-6: Push notifications (frontend + backend)
└── Day 7: Install prompt + Testing + Polish
```

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Lighthouse Mobile Performance | ~60 | 90+ |
| Lighthouse PWA Score | 0 | 90+ |
| Mobile usability errors | Multiple | 0 |
| First Contentful Paint (mobile) | ~2.5s | <1.5s |
| Time to Interactive (mobile) | ~4s | <3s |
| Offline functionality | None | Full dashboard |

---

## Backend Changes Required

```python
# src/services/push_notification_service.py (NEW)
# - VAPID key management
# - Push subscription storage (Supabase)
# - Notification dispatch for: test_failed, healing_complete, approval_needed

# src/api/notifications.py (MODIFY)
# - Add push subscription endpoints
# - Add notification preferences endpoints

# supabase/migrations/XXXXXX_push_subscriptions.sql (NEW)
# - push_subscriptions table
# - notification_preferences table
```
