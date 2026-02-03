# Chat Workspace Redesign - AI-Native Glassmorphic Workspace

**Date:** 2026-02-03
**Status:** Design Complete
**Linear Issue:** [RAP-345](https://linear.app/raphatech/issue/RAP-345/chat-page-complete-overhaul-ai-native-glassmorphic-workspace)
**Project:** Argus Dashboard

---

## Executive Summary

Complete redesign of the Argus chat page from a basic chatbot interface to an **AI-Native Glassmorphic Workspace** - a next-gen command center for the Argus Quality Intelligence Platform.

### The Problem

The current chat page is:
- **Cluttered** - Two sidebars (nav + conversation history) steal focus
- **Disconnected** - Chat doesn't integrate with platform capabilities
- **Dated** - 2021 ChatGPT-era patterns, basic message bubbles
- **Not representative** - Doesn't reflect Argus as a full SDLC/STLC intelligence platform

### The Solution

An adaptive workspace where:
- Chat is the **command surface** for intent
- Contextual panels are the **execution layer** for results
- Layout responds intelligently to conversation context
- Glassmorphic design creates depth and premium feel

---

## Design Principles

| Principle | Meaning |
|-----------|---------|
| **Contextual, not cluttered** | Show only what's relevant to the current conversation |
| **Progressive disclosure** | Start minimal, expand as complexity grows |
| **Glassmorphic depth** | Layered surfaces with blur, glow, and spatial hierarchy |
| **AI-driven layout** | The AI decides what panels to spawn based on user intent |
| **Keyboard-first, mouse-friendly** | Power users get shortcuts, everyone gets intuitive clicks |

---

## Layout Architecture

### The Adaptive Split System

The workspace has three states that transition fluidly based on conversation context:

#### State 1: Focused Chat (Default)

```
┌─────────────────────────────────────────────────────────────────┐
│ Nav │                      Chat                                 │
│     │                   (Full width)                            │
│     │                                                           │
│     │              Maximum focus on conversation                │
└─────────────────────────────────────────────────────────────────┘
```

- Clean, distraction-free
- Full width for chat content
- Input strip at bottom with contextual chips

#### State 2: Split View (Contextual)

```
┌─────────────────────────────────────────────────────────────────┐
│ Nav │      Chat (50-60%)      │    Context Panel (40-50%)      │
│     │                         │  ┌─────────────────────────┐   │
│     │  Conversation flows     │  │ ░░ Glassmorphic ░░░░░░ │   │
│     │  naturally here         │  │ Test Results / Report  │   │
│     │                         │  │ Visual Diff / Code     │   │
│     │                         │  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

- Triggered when AI spawns content (test results, reports, diffs)
- Smooth animation: chat slides left, panel slides in from right
- Resizable divider between chat and panel

#### State 3: Multi-Panel (Power User)

```
┌─────────────────────────────────────────────────────────────────┐
│ Nav │      Chat        │ [Tab1] [Tab2] [Tab3]  ← stacked tabs  │
│     │                  │  ┌─────────────────────────┐          │
│     │                  │  │ Active tab content     │          │
│     │                  │  └─────────────────────────┘          │
│     │                  │          ┌──────────────┐             │
│     │                  │          │ Floating     │ ← pop-out   │
│     │                  │          │ comparison   │             │
│     │                  │          └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

- Multiple panels stack as tabs
- Any panel can pop-out to float for comparison
- Pinned panels persist across messages

### Panel Triggers

Panels spawn automatically when the AI:

| AI Action | Panel Spawned |
|-----------|---------------|
| Runs tests | Test Results Panel |
| Generates quality report | Quality Score Panel |
| Shows visual diff | Visual Comparison Panel |
| Analyzes code | Code Viewer Panel |
| Fetches CI/CD status | Pipeline Panel |
| Finds correlations | Correlation Graph Panel |

---

## Chat Experience

### Message Design

Messages evolve from basic bubbles to **rich interactive cards**:

#### User Messages
Simple, right-aligned bubbles with subtle glassmorphic background.

#### AI Messages (Rich Cards)

```
┌─────────────────────────────────────────────────────────────────┐
│ [AI]  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│       │ I've started the test suite on staging.                │ │
│       │                                                        │ │
│       │ ┌──────────────────────────────────────────────────┐  │ │
│       │ │ 🧪 Test Run #1847                    [View →]    │  │ │
│       │ │ ████████████░░░░░░░░  58% • 47/81 passed        │  │ │
│       │ │ ⏱ 2m 34s elapsed                                 │  │ │
│       │ └──────────────────────────────────────────────────┘  │ │
│       │                                                        │ │
│       │ 3 failures detected. Want me to analyze them?         │ │
│       │                                                        │ │
│       │ [Analyze Failures]  [View Full Report]  [Stop Run]    │ │
│       └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Inline Rich Content

| Content Type | Rendering |
|--------------|-----------|
| Test results | Live progress card with pass/fail counts |
| Quality scores | Circular gauge with trend indicator |
| Code snippets | Syntax-highlighted with copy button |
| Visual diffs | Thumbnail that expands to panel on click |
| CI/CD status | Pipeline diagram with stage indicators |
| Errors | Collapsible stack trace with "Fix" action |

### Thinking States

```
┌─────────────────────────────────────────────────────────────────┐
│ [AI]  ┌────────────────────────────────────────────────────┐   │
│       │ ⚡ Analyzing 47 test failures...                    │   │
│       │                                                     │   │
│       │ ░░░░░ Checking error patterns                      │   │
│       │ ░░░░░ Correlating with recent commits              │   │
│       │ ░░░░░ Identifying root causes                      │   │
│       └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input Experience

### Contextual Command Strip

#### Idle State

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                    [ What would you like to do? ]                │
│                  ░░░░░ glassmorphic, soft glow ░░░░░             │
│                                                                  │
│       ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│       │ 🧪 Test  │  │ 📊 Quality│  │ 🔍 Analyze│  │ 🚀 Deploy │   │
│       └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Focused State

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  │
│  │                                                           │  │
│  │  Run all critical tests on staging and show quality_      │  │
│  │                                                           │  │
│  │  ┌────┐ ┌────┐ ┌────┐                    ┌─────┐ ┌─────┐ │  │
│  │  │ 📎 │ │ 🎤 │ │ /  │                    │ ⚡️  │ │ ➤  │ │  │
│  │  └────┘ └────┘ └────┘                    └─────┘ └─────┘ │  │
│  │  attach  voice  commands                  model   send   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Contextual Chips

Chips adapt to conversation state:

| Context | Chips Shown |
|---------|-------------|
| Empty chat | "Test", "Quality", "Analyze", "Deploy" |
| After test failure | "Retry", "Heal", "Explain", "Skip" |
| After report | "Export", "Share", "Drill down", "Compare" |
| After error | "Fix", "Retry", "Explain", "Ignore" |

---

## History Management

### No Sidebar - Contextual Drawer

Remove the persistent conversation sidebar. History is accessed via:

1. **History icon** in header → opens glassmorphic drawer
2. **⌘K command palette** → search history
3. **Recent pills** → appear when starting new chat

### History Drawer

```
┌──────────────────────────────────────┐
│ ░░░ glassmorphic drawer ░░░░░░░░░░░░ │
│                                      │
│  🔍 Search conversations...          │
│                                      │
│  TODAY                               │
│  ├─ Auth flow testing on staging     │
│  └─ Visual regression for homepage   │
│                                      │
│  YESTERDAY                           │
│  ├─ Quality score for v2.3 release   │
│  ├─ API test failures investigation  │
│  └─ Security scan results review     │
│                                      │
│  LAST WEEK                           │
│  └─ ... more ...                     │
└──────────────────────────────────────┘
```

### Recent Pills (New Chat)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                      Recent conversations                        │
│          ┌─────────────────────────────────────────────┐        │
│          │  💬 "Auth flow testing on staging..."   2h  │        │
│          │  💬 "Quality score for v2.3 release"    1d  │        │
│          │  💬 "CI pipeline failure analysis"      2d  │        │
│          └─────────────────────────────────────────────┘        │
│                                                                  │
│                  [ What would you like to do? ]                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Contextual Panels

### Panel Types

| Panel | Trigger | Content |
|-------|---------|---------|
| **Test Results** | AI runs tests | Live progress, pass/fail breakdown, failure details |
| **Quality Report** | AI analyzes quality | Score gauge, trends, coverage, recommendations |
| **Visual Diff** | AI compares screenshots | Side-by-side, overlay, slider comparison modes |
| **Code Viewer** | AI shows code/tests | Syntax-highlighted editor with line numbers |
| **Pipeline Status** | AI checks CI/CD | Stage diagram, build logs, deployment status |
| **Correlation Graph** | AI finds connections | Interactive graph of related events |
| **Browser Preview** | AI executes UI test | Live browser view with element highlights |

### Panel Anatomy

```
┌─────────────────────────────────────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░ Glassmorphic Panel ░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │  📊 Quality Report              [📌 Pin] [↗ Pop] [✕ Close] │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │                                                             │ │
│ │     ┌───────────┐                                          │ │
│ │     │    87     │   Quality Score                          │ │
│ │     │   /100    │   ▲ +3 from last run                     │ │
│ │     └───────────┘                                          │ │
│ │                                                             │ │
│ │  Coverage    ████████████░░░░░░  78%                       │ │
│ │  Flaky Rate  ██░░░░░░░░░░░░░░░░   4%                       │ │
│ │  Pass Rate   █████████████████░  94%                       │ │
│ │                                                             │ │
│ │  ┌─────────────────────────────────────────────────────┐   │ │
│ │  │ 💡 Recommendation: Add tests for auth edge cases    │   │ │
│ │  └─────────────────────────────────────────────────────┘   │ │
│ │                                                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  [Export PDF]  [Share]  [View History]         Updated 2s ago  │
└─────────────────────────────────────────────────────────────────┘
```

### Panel Behaviors

| Action | Behavior |
|--------|----------|
| **Spawn** | Slides in from right with spring animation |
| **Resize** | Drag divider between chat and panel |
| **Pin** | Panel persists across new messages |
| **Pop-out** | Becomes floating window for comparison |
| **Stack** | Multiple panels become tabs |
| **Close** | Fades out, chat expands back |

---

## Visual Design System

### Color Palette (Dark Mode Primary)

```
Background Layers:
██████  #09090b  Base (near black)
██████  #18181b  Surface Level 1
██████  #27272a  Surface Level 2

Glass Surfaces:
░░░░░░  rgba(255,255,255,0.03)  Subtle glass
░░░░░░  rgba(255,255,255,0.06)  Medium glass
░░░░░░  rgba(255,255,255,0.10)  Prominent glass

Accents:
██████  #6366f1  Primary (Indigo)
██████  #8b5cf6  Secondary (Violet)
██████  #22c55e  Success
██████  #ef4444  Error
██████  #f59e0b  Warning
```

### Glassmorphic CSS

```css
/* Glassmorphic Panel */
.glass-panel {
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  box-shadow:
    0 4px 24px rgba(0, 0, 0, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

/* Floating Panel (more prominent) */
.glass-floating {
  background: rgba(24, 24, 27, 0.85);
  backdrop-filter: blur(32px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow:
    0 8px 40px rgba(0, 0, 0, 0.5),
    0 0 80px rgba(99, 102, 241, 0.1);
}

/* Input Strip Glow */
.input-glow {
  box-shadow:
    0 0 0 1px rgba(99, 102, 241, 0.3),
    0 0 20px rgba(99, 102, 241, 0.15);
}
```

### Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Chat message | Inter | 15px | 400 |
| Panel title | Inter | 14px | 600 |
| Code blocks | JetBrains Mono | 13px | 400 |
| Buttons | Inter | 13px | 500 |
| Labels/chips | Inter | 12px | 500 |

### Micro-interactions

| Element | Animation |
|---------|-----------|
| Panel spawn | `spring(1, 80, 10)` slide + fade |
| Button hover | Scale 1.02, glow intensify |
| Send button | Pulse glow on active |
| Chip select | Scale bounce + color shift |
| Panel close | Fade out + slide right |
| Tab switch | Cross-fade content |

---

## Component Architecture

### AI-Native Libraries (2026)

| Library | Purpose | Why |
|---------|---------|-----|
| **[Vercel AI Elements](https://github.com/vercel/ai-elements)** | Core chat components | 20+ production-ready components for streaming, tool calls, reasoning panels |
| **[assistant-ui](https://www.assistant-ui.com/)** | Artifacts & advanced features | Y Combinator backed. Has Claude Artifacts clone with live preview panels |
| **[AI SDK 6](https://vercel.com/blog/ai-sdk-6)** | Chat hooks & streaming | Upgrade for better streaming & generative UI |
| **[shadcn/ui AI](https://www.shadcn.io/ai)** | Additional AI components | 25+ components: AI Artifact, Chain of Thought, Reasoning, Web Preview |

### Component Structure

```
dashboard/components/chat-workspace/
├── ChatWorkspace.tsx              # Main orchestrator
├── layout/
│   ├── WorkspaceLayout.tsx        # Adaptive split
│   ├── ArtifactPanel.tsx          # Uses assistant-ui Artifact
│   └── FloatingPanel.tsx          # Custom glassmorphic float
├── chat/
│   ├── ChatThread.tsx             # Uses AI Elements Conversation
│   ├── StreamingMessage.tsx       # Uses AI Elements Response
│   ├── ReasoningDisplay.tsx       # Uses AI Elements Reasoning
│   └── ThinkingIndicator.tsx      # Uses AI Elements Shimmer
├── input/
│   ├── CommandStrip.tsx           # Extends AI Elements PromptInput
│   ├── ContextualChips.tsx        # Custom smart suggestions
│   └── ModelBadge.tsx             # Uses AI Elements ModelSelector
├── panels/
│   ├── TestResultsPanel.tsx       # Custom Argus panel
│   ├── QualityReportPanel.tsx     # Custom Argus panel
│   ├── VisualDiffPanel.tsx        # Custom with assistant-ui Artifact
│   ├── CodeViewerPanel.tsx        # Uses shadcn/ui AI Code Block
│   └── BrowserPreviewPanel.tsx    # Uses assistant-ui sandboxed iframe
├── glass/
│   ├── GlassCard.tsx              # Glassmorphic wrapper
│   ├── GlassOverlay.tsx           # Blur backdrop
│   └── GlowEffect.tsx             # Animated glow
└── hooks/
    ├── useAdaptiveLayout.ts       # Layout state machine
    ├── usePanelOrchestrator.ts    # AI-driven panel spawning
    └── useContextualSuggestions.ts
```

### Package Dependencies

```json
{
  "dependencies": {
    "ai": "^6.0.0",
    "@assistant-ui/react": "^0.7.x",
    "@ai-elements/react": "^1.x",
    "framer-motion": "^11.x",
    "@floating-ui/react": "^0.26.x"
  }
}
```

---

## Implementation Plan

### Phase Breakdown

| Phase | Focus | Duration |
|-------|-------|----------|
| **Phase 1** | Foundation - Layout system, glassmorphic primitives | Week 1 |
| **Phase 2** | Chat Core - Message stream, input strip, streaming | Week 2 |
| **Phase 3** | Panel System - Contextual panels, tabs, floating | Week 3 |
| **Phase 4** | Integration - Wire to Argus backend, AI-driven spawning | Week 4 |
| **Phase 5** | Polish - Animations, transitions, edge cases | Week 5 |

### Phase 1: Foundation

- Install AI Elements, assistant-ui, upgrade AI SDK to v6
- Create GlassCard, GlassOverlay, GlowEffect primitives
- Build WorkspaceLayout with adaptive split logic
- Implement ResizeDivider for panel resizing
- Set up color tokens and CSS variables for glassmorphism
- Create useAdaptiveLayout hook (state machine)

### Phase 2: Chat Core

- Build ChatThread using AI Elements Conversation
- Create StreamingMessage with AI Elements Response
- Build CommandStrip (glassmorphic input)
- Implement ContextualChips with smart suggestions
- Add ReasoningDisplay for thinking states
- Wire up useChat from AI SDK 6
- Implement history drawer (HistoryDrawer, RecentPills)

### Phase 3: Panel System

- Create PanelContainer with tab management
- Build panel registry (type → component mapping)
- Implement TestResultsPanel, QualityReportPanel
- Add VisualDiffPanel with comparison modes
- Build CodeViewerPanel with syntax highlighting
- Create FloatingPanel with drag support
- Implement panel spawn animations
- Add pin/pop-out/close behaviors

### Phase 4: Integration

- Create usePanelOrchestrator hook
- Map AI tool calls → panel spawns
- Wire test execution → TestResultsPanel
- Wire quality reports → QualityReportPanel
- Wire visual diffs → VisualDiffPanel
- Connect to existing Argus API endpoints
- Implement panel data refresh/streaming

### Phase 5: Polish

- Add spring animations for panel transitions
- Implement keyboard shortcuts (⌘K, Esc, etc.)
- Add responsive breakpoints for mobile
- Optimize virtualization for long chats
- Add loading skeletons (Shimmer)
- Accessibility audit (focus management, ARIA)
- Performance profiling and optimization

### Migration Strategy

**Parallel Development** - Build new workspace at `/workspace/[id]` without touching current `/chat/[id]`:

1. Feature flag to toggle between old/new
2. Beta test with subset of users
3. Gradual rollout
4. Deprecate old chat page

### Success Metrics

| Metric | Target |
|--------|--------|
| Time to first message | < 500ms |
| Panel spawn animation | 60fps, < 300ms |
| Streaming latency (perceived) | No flicker |
| Bundle size increase | < 50KB gzipped |
| Lighthouse performance | > 90 |

---

## References

- [Vercel AI Elements](https://github.com/vercel/ai-elements)
- [AI Elements Changelog](https://vercel.com/changelog/introducing-ai-elements)
- [assistant-ui](https://www.assistant-ui.com/)
- [assistant-ui Artifacts](https://www.assistant-ui.com/examples/artifacts)
- [AI SDK 6](https://vercel.com/blog/ai-sdk-6)
- [shadcn/ui AI Components](https://www.shadcn.io/ai)
- [React + AI Stack 2026](https://www.builder.io/blog/react-ai-stack-2026)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-03 | Initial design document created |
