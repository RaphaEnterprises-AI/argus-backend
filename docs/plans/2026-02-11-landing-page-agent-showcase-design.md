# Landing Page: Agent Network + Mission Control Design

**Date**: 2026-02-11
**Status**: Draft — awaiting approval
**Scope**: Two new sections for the Skopaq landing page

---

## Problem

The current landing page mentions agents but doesn't *show* the system. Visitors read about 6 agents in static cards, but the actual platform has 28 agents with A2A communication, multi-agent consensus, and autonomous workflows. The page undersells the technical depth and provides no interactive proof of the system working.

## Solution: Two New Sections

### Section 1: The Agent Network (Interactive Constellation)

A living, interactive visualization of all 28 agents as a network constellation.

### Section 2: Mission Control (Live Agent Demo)

An interactive replay of real agent sessions — visitors watch agents solve problems in real-time.

---

## Section 1: Agent Network

### Visual Design

- **Canvas**: Full-width section, 80vh min-height, dark background with radial gradient from center
- **Technology**: SVG + Framer Motion (DOM-based for accessibility, no WebGL)
- **Layout**: 28 agent nodes in a loose constellation, spatially grouped by category
  - **Supervisor** at center (larger, pulsing)
  - **Testing** cluster (left): CodeAnalyzer, TestPlanner, UITester, APITester, DBTester, NLPCreator, SessionAnalyzer, VisualAI
  - **Intelligence** cluster (top): SelfHealer (near center), RootCause, CorrectiveRAG, AdaptiveRAG, TestImpact, FlakyDetector, HallucinationDetector, QualityAuditor
  - **DevOps** cluster (right): SREAgent, MRAnalyzer, Reporter, PerfAnalyzer, ToolDiscovery, ToolCreator
  - **Security** cluster (bottom): SecurityScanner, A11yChecker, AgentJudge, MetaJudge

### Node Design

- 56x56px circles with radial gradient (category color)
- Lucide icon centered (16px)
- Name label below (12px, visible on hover)
- Idle animation: slow vertical float (2-4px, randomized per node)

### Connection Lines

- SVG quadratic bezier curves (not straight lines)
- Resting state: 1px stroke, white at 8% opacity
- Pulse animation: 6px bright dot traveling along path (3-6s cycle, staggered)
- ~40 total connections reflecting real A2A communication paths

### Category Colors

| Category | Gradient |
|----------|----------|
| Testing | `violet-500` → `purple-500` |
| Intelligence | `emerald-500` → `teal-500` |
| DevOps | `amber-500` → `orange-500` |
| Security | `cyan-500` → `blue-500` |

### Interaction Model

**Layer 1 — Ambient (passive)**: Constellation animates on scroll-in. Pulses flow between agents continuously. No interaction needed to communicate "living system."

**Layer 2 — Hover exploration**:
- Node scales to 1.3x (spring animation)
- All connections FROM hovered agent brighten to 30% opacity
- Connected agents get highlight ring
- Non-connected agents dim to 40%
- Tooltip appears: name, one-line description, connected agents, "Click to explore"

**Layer 3 — Click deep-dive**:
- 480px slide-in panel from right
- Constellation shifts left, dims slightly
- Panel contents: full description, capabilities list, communication partners (clickable), model/cost info

### Scenario Simulator

Horizontal bar above constellation with 4 preset scenarios:

| Scenario | Description |
|----------|-------------|
| PR Submitted | Agents analyze, test, heal, and report on a code change |
| Test Failed | Root cause analysis → self-healing → verification |
| Production Error | SRE correlation → fix generation → incident resolution |
| New App Setup | Zero-config discovery → test generation → initial score |

Clicking a scenario triggers an animated sequence:
- Agents light up in order (active agent pulses, completed shows checkmark, pending dimmed)
- Pulses travel the actual path between agents
- Narration strip below shows plain-English description of current step
- Sequence runs 5-8 seconds at 1x, loops or pauses at end

### Mobile (< 768px)

Constellation collapses into a vertical scrollable list grouped by category with inline expand/collapse. Scenario simulator becomes a vertical timeline animation.

---

## Section 2: Mission Control

### Layout

Full-width, 80vh min-height. Split-screen:

| Panel | Width | Content |
|-------|-------|---------|
| Left | 30% | Mini constellation (only active agents) |
| Right | 70% | Agent Log Feed (terminal-style) |
| Bottom | 100% | Playback controls |

### Mini Constellation (Left)

Simplified network showing only agents involved in the current scenario (~6-8 nodes):
- Currently active agent: pulsing, enlarged
- Completed agents: green checkmark overlay
- Pending agents: dimmed
- Connection lines show the flow direction

### Agent Log Feed (Right)

Terminal-style scrolling log with entries that type in progressively:

Each entry includes:
- Timestamp (monospace, muted)
- Agent name badge (category-colored pill)
- Action description (plain English)
- Expandable "Details" block (collapsed by default)
- Result indicator: ✓ green (success), ✗ red (failure), ↻ amber→green (healed)

**Typing effect**: Text appears at ~30 chars/second at 1x speed. Instant at 4x.

### Playback Controls

- Play/Pause toggle
- Speed: 1x, 2x, 4x
- Progress bar with step indicator ("Step 8 of 14")
- Skip forward/back by step

### Demo Scenarios (3)

**Scenario A: PR Auth Refactor** (14 steps, ~45s at 1x)
1. Supervisor: New PR detected
2. CodeAnalyzer: Parse 12 changed files
3. CodeAnalyzer: Found 4 route changes, 2 middleware updates
4. TestPlanner: Risk analysis (auth.py: 0.94, middleware.py: 0.87)
5. TestPlanner: Generated 8 UI + 5 API + 2 DB tests
6. UITester: Login flow... passed
7. UITester: Signup flow... passed
8. UITester: Password reset... FAILED (#reset-btn not found)
9. SelfHealer: Analyzing failure, reading git diff
10. SelfHealer: Found rename in commit a3f42, fix applied
11. UITester: Password reset (retry)... passed (healed)
12. APITester: All 5 endpoint tests passed
13. Reporter: 15/15 passed (1 self-healed)
14. Reporter: PR comment posted with full results

**Scenario B: Production Error** (11 steps, ~35s at 1x)
1. Supervisor: Sentry alert — TypeError in checkout (42 users)
2. SREAgent: Correlating with deploy #312 (2h ago)
3. CodeAnalyzer: Analyzing deploy diff — payment.js modified
4. RootCauseAnalyzer: Null check missing on payment.provider
5. TestPlanner: Generating regression tests
6. APITester: /api/checkout with null provider → 500 confirmed
7. SelfHealer: Generating fix — null guard at payment.js:42
8. AgentJudge: Evaluating fix quality — 0.96 (approved)
9. Reporter: Creating Linear issue with fix + test results
10. Reporter: Slack notification to #engineering
11. Supervisor: Resolved. MTTR: 4m 12s

**Scenario C: New App Setup** (10 steps, ~30s at 1x)
1. Supervisor: New project connected (GitHub repo)
2. CodeAnalyzer: Cloning and analyzing
3. CodeAnalyzer: Found Next.js, 47 routes, 12 APIs, Clerk auth
4. AutoDiscovery: Crawling app URL
5. AutoDiscovery: 23 pages, 8 forms, 3 payment flows
6. TestPlanner: Generated 156 test cases
7. UITester: Running top 20 by risk
8. UITester: 18/20 passed, 2 flaky
9. FlakyDetector: Flagged 2 tests for monitoring
10. Reporter: Quality score B+ (87/100), dashboard ready

### CTA After Completion

When demo finishes:
```
That took [X] minutes [Y] seconds.
Without agents, this takes your team 2-3 days.

[Deploy Your Agents →]  [Replay Demo]
```

### Mobile (< 768px)

- Mini constellation hidden
- Log feed becomes full-width vertical timeline
- Playback controls pinned to bottom
- Each agent step is a card in a vertical scroll

### Technical Notes

- Pure client-side — all scenario data is static JSON
- No backend calls, works offline, never breaks
- Framer Motion for all animations
- IntersectionObserver: animate only when visible
- Estimated bundle: ~15KB scenario data + ~8KB components (after tree-shaking)

---

## Component Structure

```
dashboard/components/landing/
├── agent-network/
│   ├── AgentNetwork.tsx           # Main section wrapper
│   ├── ConstellationCanvas.tsx    # SVG + positioned nodes
│   ├── AgentNode.tsx              # Individual agent circle
│   ├── ConnectionLines.tsx        # SVG bezier paths + pulses
│   ├── ScenarioBar.tsx            # Scenario selector
│   ├── NarrationStrip.tsx         # Step description
│   ├── AgentDetailPanel.tsx       # Slide-in detail view
│   └── agent-network-data.ts      # Agent definitions + positions
├── mission-control/
│   ├── MissionControl.tsx         # Main section wrapper
│   ├── MiniConstellation.tsx      # Simplified left panel
│   ├── AgentLogFeed.tsx           # Terminal-style log
│   ├── LogEntry.tsx               # Individual log row
│   ├── PlaybackControls.tsx       # Play/pause/speed/progress
│   ├── MobileTimeline.tsx         # Mobile vertical timeline
│   └── mission-control-data.ts    # Scenario scripts
└── landing-page.tsx               # Updated to include new sections
```

## Data Types

```typescript
// agent-network-data.ts
type AgentCategory = 'testing' | 'intelligence' | 'devops' | 'security';

type AgentDef = {
  id: string;
  name: string;
  icon: string;  // lucide icon name
  category: AgentCategory;
  position: { x: number; y: number };  // percentage 0-100
  description: string;
  tagline: string;
  capabilities: string[];
  connections: string[];  // agent IDs this agent communicates with
  model: string;
  taskType: string;
  avgCost: string;
};

type ScenarioStep = {
  agentId: string;
  action: 'activate' | 'send' | 'complete';
  targetAgentId?: string;
  narration: string;
  durationMs: number;
};

type NetworkScenario = {
  id: string;
  name: string;
  subtitle: string;
  steps: ScenarioStep[];
};

// mission-control-data.ts
type LogEntry = {
  timestamp: string;
  agentId: string;
  agentName: string;
  category: AgentCategory;
  message: string;
  details?: { label: string; content: string }[];
  result?: 'success' | 'failure' | 'healed' | 'info';
  durationMs: number;
};

type DemoScenario = {
  id: string;
  title: string;
  subtitle: string;
  entries: LogEntry[];
};
```

## Page Placement

Replace the current "Meet Your Testing Agents" section (6 static cards) with:

1. **Agent Network** (interactive constellation)
2. **Mission Control** (live demo replay)

These two sections go between the "How It Works" section and the "Features" section.

## Replaces

- The existing 6-agent card grid (lines 708-793 in landing-page.tsx)

## Keeps

- All other sections unchanged (hero, stats, pricing, FAQ, etc.)
- The 6 featured agents become the "highlighted" nodes in the constellation (slightly larger, always showing name labels)
