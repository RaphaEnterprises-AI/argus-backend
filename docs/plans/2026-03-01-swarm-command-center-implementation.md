# Swarm Command Center Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `/chat/[id]` page with a Railway-style agent grid that serves as the primary swarm command center, with clickable agent cards opening detail panels.

**Architecture:** New `command-center/` component directory with 13 components. The page renders `SwarmCommandCenter` which shows `IdleState` when no swarm is running, and `SwarmAgentGrid` + `CommandBar` when a swarm is active. Clicking a card opens `AgentDetailSheet`. All state comes from the existing `useSwarmStream` SSE hook — no new backend endpoints.

**Tech Stack:** Next.js 15, React 19, Tailwind CSS, Radix UI (Sheet), Framer Motion, existing `GlassCard`/`GlowEffect`, existing `useSwarmStream` hook, existing `getAgentConfig` for agent icons/colors.

---

### Task 1: Barrel export + AgentGridCard component

**Files:**
- Create: `dashboard/components/command-center/index.ts`
- Create: `dashboard/components/command-center/AgentGridCard.tsx`

**Step 1: Create the barrel export**

```ts
// dashboard/components/command-center/index.ts
export { AgentGridCard } from './AgentGridCard';
```

**Step 2: Create AgentGridCard**

This is the core visual unit — a `GlassCard` + `GlowEffect` that renders one swarm worker.

```tsx
// dashboard/components/command-center/AgentGridCard.tsx
'use client';

import { memo } from 'react';
import { motion } from 'framer-motion';
import { Loader2, CheckCircle, XCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { GlassCard, GlowEffect, glowColors } from '@/components/chat-workspace';
import { getAgentConfig } from '@/lib/chat/agent-config';
import type { SwarmWorker } from '@/lib/hooks/use-swarm-stream';

const STATUS_GLOW: Record<string, { color: string; animated: boolean; intensity: 'subtle' | 'medium' | 'strong' }> = {
  running: { color: glowColors.info, animated: true, intensity: 'medium' },
  complete: { color: glowColors.success, animated: false, intensity: 'subtle' },
  error: { color: glowColors.error, animated: false, intensity: 'subtle' },
  pending: { color: 'transparent', animated: false, intensity: 'subtle' },
};

export const AgentGridCard = memo(function AgentGridCard({
  worker,
  onClick,
}: {
  worker: SwarmWorker;
  onClick: () => void;
}) {
  const config = getAgentConfig(worker.agentType);
  const Icon = config.icon;
  const glow = STATUS_GLOW[worker.status] ?? STATUS_GLOW.pending;

  return (
    <GlowEffect color={glow.color} animated={glow.animated} intensity={glow.intensity} active={worker.status === 'running'}>
      <GlassCard
        hoverable
        variant={worker.status === 'running' ? 'medium' : 'subtle'}
        padding="md"
        onClick={onClick}
        className={cn(
          'min-h-[120px] transition-all duration-300',
          worker.status === 'running' && 'border-blue-500/50',
          worker.status === 'complete' && 'border-emerald-500/30',
          worker.status === 'error' && 'border-red-500/30',
          worker.status === 'pending' && 'opacity-60',
        )}
      >
        {/* Header row: icon + name + status */}
        <div className="flex items-center gap-3 mb-3">
          <div className={cn('p-2 rounded-lg', config.bgColor)}>
            <Icon className={cn('w-4 h-4', config.color)} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm truncate">{config.name}</div>
            <div className="text-xs text-muted-foreground truncate">{worker.message || worker.phase}</div>
          </div>
          {/* Status icon */}
          {worker.status === 'pending' && <Clock className="w-4 h-4 text-muted-foreground" />}
          {worker.status === 'running' && <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />}
          {worker.status === 'complete' && <CheckCircle className="w-4 h-4 text-emerald-500" />}
          {worker.status === 'error' && <XCircle className="w-4 h-4 text-red-500" />}
        </div>

        {/* Progress bar (running only) */}
        {worker.status === 'running' && (
          <div className="h-1.5 bg-muted rounded-full overflow-hidden mb-2">
            <motion.div
              className="h-full bg-blue-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${worker.progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        )}

        {/* Stats row (complete only) */}
        {worker.status === 'complete' && (
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span>{worker.findingsCount} findings</span>
            <span>{(worker.durationMs / 1000).toFixed(1)}s</span>
            {worker.costUsd > 0 && <span>${worker.costUsd.toFixed(3)}</span>}
          </div>
        )}

        {/* Error message */}
        {worker.status === 'error' && (
          <p className="text-xs text-red-400 truncate">{worker.resultSummary}</p>
        )}
      </GlassCard>
    </GlowEffect>
  );
});
```

**Step 3: Verify it compiles**

Run: `cd dashboard && npx tsc --noEmit --pretty 2>&1 | head -20`

**Step 4: Commit**

```bash
git add dashboard/components/command-center/
git commit -m "feat(dashboard): add AgentGridCard with GlassCard + GlowEffect"
```

---

### Task 2: SwarmAgentGrid — responsive grid container

**Files:**
- Create: `dashboard/components/command-center/SwarmAgentGrid.tsx`
- Modify: `dashboard/components/command-center/index.ts`

**Step 1: Create SwarmAgentGrid**

```tsx
// dashboard/components/command-center/SwarmAgentGrid.tsx
'use client';

import { AnimatePresence, motion } from 'framer-motion';
import type { SwarmWorker } from '@/lib/hooks/use-swarm-stream';
import { AgentGridCard } from './AgentGridCard';

export function SwarmAgentGrid({
  workers,
  onSelectAgent,
}: {
  workers: SwarmWorker[];
  onSelectAgent: (agentId: string) => void;
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 p-6">
      <AnimatePresence mode="popLayout">
        {workers.map((worker, i) => (
          <motion.div
            key={worker.agentId || worker.agentType}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3, delay: i * 0.05 }}
          >
            <AgentGridCard
              worker={worker}
              onClick={() => onSelectAgent(worker.agentId || worker.agentType)}
            />
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
```

**Step 2: Add to barrel export**

Add `export { SwarmAgentGrid } from './SwarmAgentGrid';` to `index.ts`.

**Step 3: Commit**

```bash
git add dashboard/components/command-center/
git commit -m "feat(dashboard): add SwarmAgentGrid responsive grid"
```

---

### Task 3: AgentDetailSheet — slide-over panel

**Files:**
- Create: `dashboard/components/command-center/AgentDetailSheet.tsx`
- Create: `dashboard/components/command-center/ExecutionLog.tsx`
- Create: `dashboard/components/command-center/FindingsList.tsx`
- Create: `dashboard/components/command-center/AgentMetricsBar.tsx`
- Modify: `dashboard/components/command-center/index.ts`

**Step 1: Create ExecutionLog**

Renders filtered SSE events for a single agent.

```tsx
// dashboard/components/command-center/ExecutionLog.tsx
'use client';

import { useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import type { RawSwarmEvent } from '@/lib/hooks/use-swarm-stream';

export function ExecutionLog({
  events,
  agentId,
}: {
  events: RawSwarmEvent[];
  agentId: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Filter to this agent's events
  const agentEvents = events.filter(
    (e) => (e.data.agent_id as string) === agentId || e.type === 'run_started'
  );

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [agentEvents.length]);

  if (agentEvents.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        Waiting for events...
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="max-h-[300px] overflow-y-auto space-y-1 font-mono text-xs">
      {agentEvents.map((event) => (
        <div
          key={event.id}
          className={cn(
            'px-3 py-1.5 rounded border-l-2',
            event.type === 'step_started' && 'border-l-blue-500 bg-blue-500/5',
            event.type === 'step_finished' && 'border-l-emerald-500 bg-emerald-500/5',
            event.type === 'state_delta' && 'border-l-muted-foreground/30',
            event.type === 'tool_call_start' && 'border-l-amber-500 bg-amber-500/5',
            event.type === 'tool_call_end' && 'border-l-amber-500/50',
            event.type === 'run_error' && 'border-l-red-500 bg-red-500/5',
          )}
        >
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground shrink-0">
              {new Date(event.timestamp).toLocaleTimeString()}
            </span>
            <span className="text-muted-foreground">{event.type.replace(/_/g, ' ')}</span>
          </div>
          {(event.data.message || event.data.phase) && (
            <div className="text-foreground/80 mt-0.5 truncate">
              {(event.data.message as string) || (event.data.phase as string)}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
```

**Step 2: Create FindingsList**

```tsx
// dashboard/components/command-center/FindingsList.tsx
'use client';

import { Shield, AlertTriangle, Info } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Finding {
  type?: string;
  severity?: string;
  description?: string;
  [key: string]: unknown;
}

const SEVERITY_CONFIG = {
  high: { icon: AlertTriangle, color: 'text-red-500', bg: 'bg-red-500/10' },
  medium: { icon: Shield, color: 'text-amber-500', bg: 'bg-amber-500/10' },
  low: { icon: Info, color: 'text-blue-500', bg: 'bg-blue-500/10' },
} as const;

export function FindingsList({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No findings
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {findings.map((finding, i) => {
        const severity = (finding.severity || 'low') as keyof typeof SEVERITY_CONFIG;
        const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.low;
        const Icon = config.icon;

        return (
          <div key={i} className="flex items-start gap-3 p-3 rounded-lg border bg-card/50">
            <div className={cn('p-1.5 rounded-md', config.bg)}>
              <Icon className={cn('w-3.5 h-3.5', config.color)} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium">{finding.type || 'Finding'}</div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {finding.description || JSON.stringify(finding)}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

**Step 3: Create AgentMetricsBar**

```tsx
// dashboard/components/command-center/AgentMetricsBar.tsx
'use client';

import { Clock, DollarSign, Target, FileSearch } from 'lucide-react';
import type { SwarmWorker } from '@/lib/hooks/use-swarm-stream';

export function AgentMetricsBar({ worker }: { worker: SwarmWorker }) {
  return (
    <div className="flex items-center gap-4 p-3 rounded-lg bg-muted/50 text-sm">
      <div className="flex items-center gap-1.5">
        <Clock className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="font-mono">{(worker.durationMs / 1000).toFixed(1)}s</span>
      </div>
      {worker.costUsd > 0 && (
        <div className="flex items-center gap-1.5">
          <DollarSign className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="font-mono">${worker.costUsd.toFixed(3)}</span>
        </div>
      )}
      <div className="flex items-center gap-1.5">
        <FileSearch className="w-3.5 h-3.5 text-muted-foreground" />
        <span>{worker.findingsCount} findings</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Target className="w-3.5 h-3.5 text-muted-foreground" />
        <span>{worker.progress}%</span>
      </div>
    </div>
  );
}
```

**Step 4: Create AgentDetailSheet**

```tsx
// dashboard/components/command-center/AgentDetailSheet.tsx
'use client';

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { getAgentConfig } from '@/lib/chat/agent-config';
import type { SwarmWorker, RawSwarmEvent } from '@/lib/hooks/use-swarm-stream';
import { ExecutionLog } from './ExecutionLog';
import { FindingsList } from './FindingsList';
import { AgentMetricsBar } from './AgentMetricsBar';

const STATUS_BADGE = {
  pending: { label: 'Pending', variant: 'outline' as const },
  running: { label: 'Running', variant: 'default' as const },
  complete: { label: 'Complete', variant: 'default' as const },
  error: { label: 'Error', variant: 'destructive' as const },
};

export function AgentDetailSheet({
  worker,
  events,
  open,
  onOpenChange,
}: {
  worker: SwarmWorker | null;
  events: RawSwarmEvent[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  if (!worker) return null;

  const config = getAgentConfig(worker.agentType);
  const Icon = config.icon;
  const badge = STATUS_BADGE[worker.status];

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[500px] sm:max-w-[500px] flex flex-col p-0">
        {/* Header */}
        <SheetHeader className="p-6 border-b shrink-0">
          <div className="flex items-center gap-3">
            <div className={cn('p-2.5 rounded-xl', config.bgColor)}>
              <Icon className={cn('w-5 h-5', config.color)} />
            </div>
            <div className="flex-1">
              <SheetTitle className="flex items-center gap-2">
                {config.name}
                <Badge variant={badge.variant} className="text-xs">
                  {badge.label}
                </Badge>
              </SheetTitle>
              <SheetDescription>{worker.message || worker.phase || 'Waiting...'}</SheetDescription>
            </div>
          </div>
        </SheetHeader>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Metrics (visible when not pending) */}
          {worker.status !== 'pending' && <AgentMetricsBar worker={worker} />}

          {/* Execution Log */}
          <div>
            <h3 className="text-sm font-medium mb-3">Execution Log</h3>
            <ExecutionLog events={events} agentId={worker.agentId} />
          </div>

          {/* Findings (visible when complete) */}
          {worker.status === 'complete' && worker.findingsCount > 0 && (
            <div>
              <h3 className="text-sm font-medium mb-3">Findings</h3>
              <FindingsList findings={[]} />
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
```

**Step 5: Update barrel export**

```ts
export { AgentGridCard } from './AgentGridCard';
export { SwarmAgentGrid } from './SwarmAgentGrid';
export { AgentDetailSheet } from './AgentDetailSheet';
export { ExecutionLog } from './ExecutionLog';
export { FindingsList } from './FindingsList';
export { AgentMetricsBar } from './AgentMetricsBar';
```

**Step 6: Commit**

```bash
git add dashboard/components/command-center/
git commit -m "feat(dashboard): add AgentDetailSheet with ExecutionLog, FindingsList, MetricsBar"
```

---

### Task 4: CommandBar — bottom input with mode selector

**Files:**
- Create: `dashboard/components/command-center/CommandBar.tsx`
- Modify: `dashboard/components/command-center/index.ts`

**Step 1: Create CommandBar**

The command bar sits at the bottom. It has mode pills, a URL/PR input, and a launch button.

```tsx
// dashboard/components/command-center/CommandBar.tsx
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Zap, GitPullRequest, Compass, Search, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { GlassCard } from '@/components/chat-workspace';

const MODES = [
  { id: 'full_crawl', name: 'Full Crawl', icon: Compass, input: 'url' as const },
  { id: 'targeted_blitz', name: 'Blitz', icon: Zap, input: 'url' as const },
  { id: 'pr_analysis', name: 'PR Analysis', icon: GitPullRequest, input: 'pr' as const },
  { id: 'discovery_swarm', name: 'Discovery', icon: Search, input: 'url' as const },
] as const;

export function CommandBar({
  onLaunch,
  isLaunching,
  disabled,
}: {
  onLaunch: (mode: string, config: { targetUrl?: string; prNumber?: number }) => void;
  isLaunching: boolean;
  disabled?: boolean;
}) {
  const [mode, setMode] = useState('full_crawl');
  const [inputValue, setInputValue] = useState('');

  const selectedMode = MODES.find((m) => m.id === mode)!;

  const handleLaunch = () => {
    onLaunch(mode, {
      targetUrl: selectedMode.input === 'url' ? inputValue || undefined : undefined,
      prNumber: selectedMode.input === 'pr' ? parseInt(inputValue, 10) || undefined : undefined,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLaunching) {
      e.preventDefault();
      handleLaunch();
    }
  };

  return (
    <GlassCard variant="prominent" padding="sm" className="mx-4 mb-4 sm:mx-6">
      <div className="flex items-center gap-2">
        {/* Mode pills */}
        <div className="hidden sm:flex items-center gap-1 shrink-0">
          {MODES.map((m) => {
            const ModeIcon = m.icon;
            return (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                className={cn(
                  'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors',
                  mode === m.id
                    ? 'bg-primary/20 text-primary'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted',
                )}
              >
                <ModeIcon className="w-3.5 h-3.5" />
                {m.name}
              </button>
            );
          })}
        </div>

        {/* Input */}
        <Input
          placeholder={selectedMode.input === 'pr' ? 'PR number...' : 'https://your-app.com'}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || isLaunching}
          className="flex-1 bg-transparent border-0 focus-visible:ring-0 focus-visible:ring-offset-0"
        />

        {/* Launch button */}
        <Button
          onClick={handleLaunch}
          disabled={disabled || isLaunching}
          size="sm"
          className="shrink-0 gap-1.5"
        >
          {isLaunching ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Zap className="w-3.5 h-3.5" />
          )}
          Launch
        </Button>
      </div>
    </GlassCard>
  );
}
```

**Step 2: Update barrel, commit**

```bash
git add dashboard/components/command-center/
git commit -m "feat(dashboard): add CommandBar with mode pills and launch input"
```

---

### Task 5: CommandCenterHeader + IdleState

**Files:**
- Create: `dashboard/components/command-center/CommandCenterHeader.tsx`
- Create: `dashboard/components/command-center/IdleState.tsx`
- Modify: `dashboard/components/command-center/index.ts`

**Step 1: Create CommandCenterHeader**

Shows swarm status when active, or just branding when idle.

```tsx
// dashboard/components/command-center/CommandCenterHeader.tsx
'use client';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger, SheetDescription } from '@/components/ui/sheet';
import { Tooltip } from '@/components/ui/tooltip';
import { PanelLeft, History, Loader2, CheckCircle, XCircle, Zap } from 'lucide-react';
import type { SwarmState } from '@/lib/hooks/use-swarm-stream';

export function CommandCenterHeader({
  state,
  historyContent,
}: {
  state: SwarmState;
  historyContent: React.ReactNode;
}) {
  const isActive = state.status !== 'idle';
  const completedCount = state.workers.filter((w) => w.status === 'complete').length;

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-14 px-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex items-center gap-2">
        {/* History drawer */}
        <Sheet>
          <Tooltip content="History" side="bottom">
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="h-9 w-9">
                <PanelLeft className="h-5 w-5" />
              </Button>
            </SheetTrigger>
          </Tooltip>
          <SheetContent side="left" className="w-[320px] p-0">
            <SheetHeader className="p-4 border-b">
              <SheetTitle className="flex items-center gap-2">
                <History className="h-4 w-4" />
                History
              </SheetTitle>
              <SheetDescription className="sr-only">View past conversations and swarm runs</SheetDescription>
            </SheetHeader>
            {historyContent}
          </SheetContent>
        </Sheet>

        {/* Title / Swarm status */}
        <div className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-primary" />
          <span className="font-semibold">Skopaq</span>
          {isActive && (
            <>
              <span className="text-muted-foreground">/</span>
              <Badge variant="outline" className="gap-1 text-xs font-normal">
                {state.mode.replace(/_/g, ' ')}
              </Badge>
            </>
          )}
        </div>
      </div>

      {/* Right: status indicators */}
      {isActive && (
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="gap-1.5 text-xs font-normal">
            {state.status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
            {state.status === 'complete' && <CheckCircle className="w-3 h-3 text-emerald-500" />}
            {state.status === 'error' && <XCircle className="w-3 h-3 text-red-500" />}
            {completedCount}/{state.workers.length} agents
          </Badge>
          {state.totalCostUsd > 0 && (
            <Badge variant="outline" className="text-xs font-mono">
              ${state.totalCostUsd.toFixed(3)}
            </Badge>
          )}
        </div>
      )}
    </header>
  );
}
```

**Step 2: Create IdleState**

Shown when no swarm is running. Launch prompt centered, recent runs below.

```tsx
// dashboard/components/command-center/IdleState.tsx
'use client';

import { Zap, Compass, GitPullRequest, Search, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { GlassCard, GlowEffect, glowColors } from '@/components/chat-workspace';

const QUICK_ACTIONS = [
  { mode: 'full_crawl', name: 'Full Crawl', desc: 'Analyze entire app', icon: Compass, color: glowColors.violet },
  { mode: 'targeted_blitz', name: 'Targeted Blitz', desc: 'Test a specific flow', icon: Zap, color: glowColors.warning },
  { mode: 'pr_analysis', name: 'PR Analysis', desc: 'Analyze a pull request', icon: GitPullRequest, color: glowColors.success },
  { mode: 'discovery_swarm', name: 'Discovery', desc: 'Crawl and discover', icon: Search, color: glowColors.cyan },
];

export function IdleState({
  onQuickLaunch,
}: {
  onQuickLaunch: (mode: string) => void;
}) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-4">
          <Zap className="w-8 h-8 text-primary" />
        </div>
        <h1 className="text-2xl font-bold mb-2">What should we test?</h1>
        <p className="text-muted-foreground max-w-md">
          Deploy a swarm of AI agents to analyze, test, and secure your application.
        </p>
      </motion.div>

      {/* Quick action cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-4xl w-full">
        {QUICK_ACTIONS.map((action, i) => {
          const Icon = action.icon;
          return (
            <motion.div
              key={action.mode}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.05 }}
            >
              <GlowEffect color={action.color} intensity="subtle" active={false}>
                <GlassCard
                  hoverable
                  variant="subtle"
                  padding="lg"
                  onClick={() => onQuickLaunch(action.mode)}
                  className="text-center group"
                >
                  <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10 mb-3">
                    <Icon className="w-5 h-5 text-primary" />
                  </div>
                  <h3 className="font-medium text-sm mb-1">{action.name}</h3>
                  <p className="text-xs text-muted-foreground mb-3">{action.desc}</p>
                  <ArrowRight className="w-4 h-4 mx-auto text-muted-foreground group-hover:text-primary transition-colors" />
                </GlassCard>
              </GlowEffect>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
```

**Step 3: Update barrel, commit**

```bash
git add dashboard/components/command-center/
git commit -m "feat(dashboard): add CommandCenterHeader and IdleState"
```

---

### Task 6: SwarmCommandCenter — top-level orchestrator

**Files:**
- Create: `dashboard/components/command-center/SwarmCommandCenter.tsx`
- Modify: `dashboard/components/command-center/index.ts`

**Step 1: Create SwarmCommandCenter**

This is the main component that wires everything together.

```tsx
// dashboard/components/command-center/SwarmCommandCenter.tsx
'use client';

import { useState, useCallback } from 'react';
import { useSwarmStream } from '@/lib/hooks/use-swarm-stream';
import { useAuthApi } from '@/lib/hooks/use-auth-api';
import { CommandCenterHeader } from './CommandCenterHeader';
import { SwarmAgentGrid } from './SwarmAgentGrid';
import { AgentDetailSheet } from './AgentDetailSheet';
import { CommandBar } from './CommandBar';
import { IdleState } from './IdleState';

export function SwarmCommandCenter({
  projectId,
  historyContent,
}: {
  projectId: string;
  historyContent?: React.ReactNode;
}) {
  const { fetchJson } = useAuthApi();
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const { state, events, reset } = useSwarmStream(streamUrl);

  const isActive = state.status !== 'idle';

  const handleLaunch = useCallback(
    async (mode: string, config: { targetUrl?: string; prNumber?: number }) => {
      setIsLaunching(true);
      reset();
      setSelectedAgentId(null);

      try {
        const response = await fetchJson<{
          swarm_id: string;
          stream_url: string;
        }>('/api/v1/swarms/launch', {
          method: 'POST',
          body: JSON.stringify({
            mode,
            project_id: projectId,
            target_url: config.targetUrl,
            pr_number: config.prNumber,
          }),
        });

        if (response.data?.stream_url) {
          setStreamUrl(response.data.stream_url);
        }
      } catch (err) {
        console.error('Failed to launch swarm:', err);
      } finally {
        setIsLaunching(false);
      }
    },
    [fetchJson, projectId, reset],
  );

  const handleQuickLaunch = useCallback(
    (mode: string) => {
      // For quick launch, just set the mode — user still needs to fill URL in CommandBar
      // We could auto-focus the command bar input here
    },
    [],
  );

  const selectedWorker = state.workers.find(
    (w) => w.agentId === selectedAgentId || w.agentType === selectedAgentId,
  ) ?? null;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <CommandCenterHeader state={state} historyContent={historyContent} />

      {/* Main content */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {isActive ? (
          <>
            {/* Agent grid */}
            <SwarmAgentGrid
              workers={state.workers}
              onSelectAgent={setSelectedAgentId}
            />

            {/* Summary */}
            {state.summary && (
              <div className="mx-6 mb-4 p-4 rounded-xl bg-muted/50 border text-sm">
                {state.summary}
              </div>
            )}

            {/* Error */}
            {state.error && (
              <div className="mx-6 mb-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-sm text-red-400">
                {state.error}
              </div>
            )}
          </>
        ) : (
          <IdleState onQuickLaunch={handleQuickLaunch} />
        )}
      </div>

      {/* Command bar (always visible at bottom) */}
      <CommandBar
        onLaunch={handleLaunch}
        isLaunching={isLaunching}
        disabled={state.status === 'running'}
      />

      {/* Detail sheet */}
      <AgentDetailSheet
        worker={selectedWorker}
        events={events}
        open={selectedAgentId !== null}
        onOpenChange={(open) => {
          if (!open) setSelectedAgentId(null);
        }}
      />
    </div>
  );
}
```

**Step 2: Update barrel export to include SwarmCommandCenter**

**Step 3: Commit**

```bash
git add dashboard/components/command-center/
git commit -m "feat(dashboard): add SwarmCommandCenter orchestrator component"
```

---

### Task 7: Rewrite `/chat/[id]` page to use SwarmCommandCenter

**Files:**
- Modify: `dashboard/app/chat/[id]/page.tsx`

**Step 1: Rewrite the page**

Replace the current `ChatPageContent` with `SwarmCommandCenter`. Preserve the Clerk auth wrapper and conversation history in the drawer.

The key change: instead of rendering `ChatWorkspace`, render `SwarmCommandCenter`. The conversation list moves into the `historyContent` prop.

```tsx
// dashboard/app/chat/[id]/page.tsx
'use client';

import { useParams, useRouter } from 'next/navigation';
import { SignedIn, SignedOut, RedirectToSignIn } from '@clerk/nextjs';
import { Sidebar } from '@/components/layout/sidebar';
import { SwarmCommandCenter } from '@/components/command-center';
import { useConversations, useCreateConversation, useDeleteConversation } from '@/lib/hooks/use-chat';
import { Button } from '@/components/ui/button';
import { Loader2, MessageSquarePlus, MessageSquare, Clock, Trash2 } from 'lucide-react';
import { safeFormatDistanceToNow, cn } from '@/lib/utils';

function HistoryList({
  onNewChat,
}: {
  onNewChat: () => void;
}) {
  const { data: conversations = [], isLoading } = useConversations();
  const deleteConversation = useDeleteConversation();
  const router = useRouter();

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b shrink-0">
        <Button onClick={onNewChat} className="w-full" size="sm">
          <MessageSquarePlus className="h-4 w-4 mr-2" />
          New Session
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {isLoading ? (
          <div className="flex items-center justify-center p-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : conversations.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center p-4">No history yet</p>
        ) : (
          conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => router.push(`/chat/${conv.id}`)}
              className={cn(
                'w-full text-left p-3 rounded-lg transition-colors group hover:bg-muted',
              )}
            >
              <div className="flex items-start gap-2">
                <MessageSquare className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate">{conv.title || 'Untitled'}</div>
                  <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {safeFormatDistanceToNow(conv.updated_at, { addSuffix: true })}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation.mutateAsync(conv.id);
                  }}
                >
                  <Trash2 className="h-3 w-3 text-muted-foreground" />
                </Button>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}

function ChatPageContent() {
  const params = useParams();
  const router = useRouter();
  const conversationId = params.id as string;
  const createConversation = useCreateConversation();

  const handleNewChat = async () => {
    const conversation = await createConversation.mutateAsync({});
    router.push(`/chat/${conversation.id}`);
  };

  return (
    <div className="flex min-h-screen overflow-x-hidden">
      <Sidebar />
      <main className="flex-1 lg:ml-64 min-w-0 flex flex-col h-screen">
        <SwarmCommandCenter
          projectId={conversationId}
          historyContent={<HistoryList onNewChat={handleNewChat} />}
        />
      </main>
    </div>
  );
}

export default function ChatPage() {
  return (
    <>
      <SignedOut>
        <RedirectToSignIn />
      </SignedOut>
      <SignedIn>
        <ChatPageContent />
      </SignedIn>
    </>
  );
}
```

**Step 2: Verify it compiles**

Run: `cd dashboard && npx tsc --noEmit --pretty 2>&1 | head -30`

**Step 3: Run dev server and test manually**

Run: `cd dashboard && npm run dev`
Navigate to `/chat/some-uuid` and verify:
- Idle state shows with launch prompt + quick action cards
- CommandBar at bottom with mode pills
- History drawer opens from left
- Launching a swarm transitions to the agent grid

**Step 4: Commit**

```bash
git add dashboard/app/chat/ dashboard/components/command-center/
git commit -m "feat(dashboard): replace chat page with SwarmCommandCenter"
```

---

### Task 8: Update `/chat` index page

**Files:**
- Modify: `dashboard/app/chat/page.tsx`

**Step 1: Simplify the index to redirect or show idle state**

The `/chat` page (without an ID) should create a new session and redirect to `/chat/[id]`.

```tsx
// Simplify to auto-redirect on mount
'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { SignedIn, SignedOut, RedirectToSignIn } from '@clerk/nextjs';
import { useCreateConversation } from '@/lib/hooks/use-chat';
import { Loader2 } from 'lucide-react';

function ChatRedirect() {
  const router = useRouter();
  const createConversation = useCreateConversation();

  useEffect(() => {
    createConversation.mutateAsync({}).then((conv) => {
      router.replace(`/chat/${conv.id}`);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex min-h-screen items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );
}

export default function ChatPage() {
  return (
    <>
      <SignedOut><RedirectToSignIn /></SignedOut>
      <SignedIn><ChatRedirect /></SignedIn>
    </>
  );
}
```

**Step 2: Commit**

```bash
git add dashboard/app/chat/page.tsx
git commit -m "feat(dashboard): simplify /chat to auto-create session and redirect"
```

---

### Task 9: Visual polish pass

**Files:**
- Modify: Various command-center components

**Step 1: Test on different screen sizes**

Open Chrome DevTools, test at:
- Mobile (375px): Cards stack 1-col, mode pills hidden, CommandBar simplified
- Tablet (768px): 2-col grid
- Desktop (1280px): 3-4 col grid, full CommandBar

**Step 2: Fix any layout/overflow issues found**

**Step 3: Test the detail sheet**

- Click a card → sheet opens from right
- Verify execution log shows filtered events
- Verify metrics bar shows correct data
- Close sheet → selected state clears

**Step 4: Final commit**

```bash
git add dashboard/components/command-center/
git commit -m "fix(dashboard): polish SwarmCommandCenter responsive layout"
```

---

## Execution Summary

| Task | Component | Est. LOC |
|------|-----------|----------|
| 1 | AgentGridCard | ~100 |
| 2 | SwarmAgentGrid | ~35 |
| 3 | AgentDetailSheet + ExecutionLog + FindingsList + MetricsBar | ~200 |
| 4 | CommandBar | ~90 |
| 5 | CommandCenterHeader + IdleState | ~150 |
| 6 | SwarmCommandCenter (orchestrator) | ~120 |
| 7 | Rewrite `/chat/[id]` page | ~100 |
| 8 | Update `/chat` index | ~30 |
| 9 | Visual polish | ~varies |

**Total: ~825 new LOC across 13 new files + 2 modified pages**
