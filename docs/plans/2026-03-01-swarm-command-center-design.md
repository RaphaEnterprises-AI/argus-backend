# Swarm Command Center — Chat Page Overhaul

**Date:** 2026-03-01
**Status:** Design approved
**Route:** Replaces `/chat/[id]`

## Summary

Replace the traditional chat interface at `/chat/[id]` with a Railway-style agent grid that serves as the primary swarm command center. Chat input remains at the bottom as a command bar, but the main viewport shows agents as interactive cards in a responsive grid. Clicking any card opens a slide-over detail panel with live browser views, streaming execution logs, and findings.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Layout model | Grid replaces chat | Agents ARE the interface; chat thread moves to history drawer |
| Detail panel | Slide-over Sheet (both: live view + findings) | Layered: browser/screenshot at top, logs in middle, findings at bottom |
| Idle state | Launch prompt + recent runs history | Users see both the action prompt and past run context |
| Route | Replace `/chat/[id]` | Single command center rather than fragmented pages |
| Visual style | Railway-style rectangular grid | Clean, familiar, proven; uses existing GlassCard + GlowEffect system |

## Layout (3 zones)

```
┌────────────────────────────────────────────────────────────┐
│ HEADER BAR                                                  │
│ [◀ History] [Swarm: full_crawl] [4/7 agents] [$0.04] [⚙]  │
├────────────────────────────────────────────────────────────┤
│                                                             │
│              AGENT GRID (main area, scrollable)             │
│                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│  │ CodeAna │ │UITester │ │APITest  │ │SecScan  │         │
│  │ ● run   │ │ ◐ 60%   │ │ ○ wait  │ │ ✓ done  │         │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                      │
│  │FlakyDet │ │VisualAI │ │MRAnalyz │                      │
│  └─────────┘ └─────────┘ └─────────┘                      │
│                                                             │
├────────────────────────────────────────────────────────────┤
│ COMMAND BAR  [/full_crawl https://example.com]  [▶ Launch] │
└────────────────────────────────────────────────────────────┘

Click any card → slide-over detail panel from right
```

## Agent Cards

Each card uses `GlassCard` + `GlowEffect`:

| State | Glass | Glow | Icon | Border |
|-------|-------|------|------|--------|
| Pending | subtle | none | Clock (muted) | default, opacity-60 |
| Running | medium | pulsing, agent color | Spinner (animated) | agent color/50 |
| Complete | subtle | brief green fade | CheckCircle (green) | emerald/30 |
| Error | subtle | none | XCircle (red) | red/30 |

Card contents:
- Agent icon (from `getAgentConfig`) + name
- Status line (phase + message)
- Progress bar (running only)
- Mini stats row: findings count, duration, cost (complete only)

Responsive grid: 1 col mobile, 2 cols sm, 3 cols lg, 4 cols xl.

## Detail Panel (Sheet, right side, ~500px)

Opens when clicking an agent card. Scrollable sections:

1. **Header**: Agent name + icon + status badge + close button
2. **Live View** (conditional): If agent has browser session (`ui_tester`, `visual_ai`), show latest screenshot. Auto-refreshes during execution.
3. **Execution Log**: Real-time streaming tool calls and decisions. Uses `tool_call_start`/`tool_call_end` SSE events filtered by `agent_id`.
4. **Findings**: Cards per finding (severity, description, confidence). Visible when agent completes.
5. **Metrics Bar**: Duration | Cost | Confidence gauge | Findings count

## Idle State (No Active Swarm)

### A. Launch Prompt (centered, prominent)

- Heading: "What should we test?"
- Mode selector pills: `Full Crawl` | `PR Analysis` | `Discovery` | `Targeted Blitz`
- Contextual input: URL field for crawl/blitz, PR number for PR analysis
- Launch button

### B. Recent Runs (below prompt)

- Grid of past swarm run cards (from Supabase)
- Each shows: mode badge, date, agent count, consensus score, cost
- Click → re-view completed swarm results in the grid

## Data Flow

```
User types in Command Bar
    ↓
POST /api/v1/swarms/launch (existing)
    ↓
Returns stream_url → useSwarmStream hook (existing, no changes)
    ↓
SSE events update SwarmState → grid re-renders
    ↓
Click card → Sheet opens, filters events by agent_id
```

No new backend endpoints required.

## Component Architecture

```
app/chat/[id]/page.tsx  (rewritten)
  └── SwarmCommandCenter
       ├── CommandCenterHeader
       │    ├── History toggle (Sheet)
       │    ├── Swarm status (mode, agent count, cost)
       │    └── Settings button
       ├── SwarmAgentGrid
       │    └── AgentGridCard (GlassCard + GlowEffect)
       ├── AgentDetailSheet
       │    ├── AgentDetailHeader
       │    ├── LiveBrowserView (screenshot/iframe)
       │    ├── ExecutionLog (streaming tool calls)
       │    ├── FindingsList
       │    └── AgentMetricsBar
       ├── IdleState
       │    ├── LaunchPrompt (mode pills + input)
       │    └── RecentRunsGrid
       ├── CommandBar (input + mode + launch button)
       └── ConversationDrawer (history access)
```

## Existing Code Reuse

| Existing | Reused As |
|----------|-----------|
| `useSwarmStream` hook | Core data source (no changes) |
| `GlassCard` + `GlowEffect` | Agent card styling |
| `getAgentConfig` (icons/colors) | Card icons and colors |
| `SwarmLauncher` mode logic | CommandBar mode selector |
| `SwarmEventLog` event rendering | Detail panel ExecutionLog |
| `AgentBadge` status icons | Card status indicators |
| `Sheet` (Radix) | Detail panel + history drawer |
| `ConversationList` | History drawer content |

## Files to Create/Modify

### New files
- `dashboard/components/command-center/SwarmCommandCenter.tsx` — top-level orchestrator
- `dashboard/components/command-center/CommandCenterHeader.tsx` — header bar
- `dashboard/components/command-center/SwarmAgentGrid.tsx` — responsive grid
- `dashboard/components/command-center/AgentGridCard.tsx` — individual card
- `dashboard/components/command-center/AgentDetailSheet.tsx` — slide-over panel
- `dashboard/components/command-center/LiveBrowserView.tsx` — screenshot/iframe
- `dashboard/components/command-center/ExecutionLog.tsx` — streaming log
- `dashboard/components/command-center/FindingsList.tsx` — findings cards
- `dashboard/components/command-center/AgentMetricsBar.tsx` — stats bar
- `dashboard/components/command-center/IdleState.tsx` — launch prompt + history
- `dashboard/components/command-center/CommandBar.tsx` — bottom input
- `dashboard/components/command-center/RecentRunsGrid.tsx` — past runs
- `dashboard/components/command-center/index.ts` — barrel export

### Modified files
- `dashboard/app/chat/[id]/page.tsx` — rewrite to use SwarmCommandCenter
- `dashboard/app/chat/page.tsx` — redirect to new experience or show idle state

### Preserved (not deleted)
- All existing `chat-workspace/` and `chat/` components — they still work for non-swarm conversations and can be accessed via the history drawer
- `useSwarmStream` hook — used directly, no changes
- `/swarms` page — can eventually be removed, but no urgency
