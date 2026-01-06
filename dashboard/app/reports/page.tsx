'use client';

import { useState, useMemo } from 'react';
import { formatDistanceToNow } from 'date-fns';
import {
  BarChart3,
  TrendingUp,
  Clock,
  Download,
  Calendar,
  ArrowUpRight,
  CheckCircle2,
  XCircle,
  Activity,
  Loader2,
} from 'lucide-react';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { useProjects } from '@/lib/hooks/use-projects';
import { useReportsStats, useRecentRuns } from '@/lib/hooks/use-reports';
import { cn } from '@/lib/utils';
import { StatusDot, Badge } from '@/components/ui/data-table';

function formatDuration(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${secs}s`;
}

export default function ReportsPage() {
  const [selectedPeriod, setSelectedPeriod] = useState<7 | 30 | 90>(7);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const currentProject = selectedProjectId || projects[0]?.id;

  const { data: stats, isLoading: statsLoading } = useReportsStats(currentProject || null, selectedPeriod);
  const { data: recentRuns = [], isLoading: runsLoading } = useRecentRuns(currentProject || null);

  const maxBar = useMemo(() => {
    if (!stats?.dailyStats) return 1;
    return Math.max(...stats.dailyStats.map((s) => s.passed + s.failed), 1);
  }, [stats?.dailyStats]);

  const isLoading = projectsLoading || statsLoading || runsLoading;

  // No project state
  if (!projectsLoading && projects.length === 0) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <BarChart3 className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">No Projects Yet</h2>
            <p className="text-muted-foreground">
              Create a project first to start seeing reports and analytics.
            </p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-64">
        {/* Header */}
        <header className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex h-14 items-center gap-4 px-6">
            {/* Project Selector */}
            <select
              value={currentProject || ''}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="h-9 rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>

            <div className="flex-1" />

            {/* Period Selector */}
            <div className="flex rounded-lg border p-1">
              {([7, 30, 90] as const).map((period) => (
                <button
                  key={period}
                  onClick={() => setSelectedPeriod(period)}
                  className={cn(
                    'px-3 py-1.5 text-sm rounded-md transition-colors',
                    selectedPeriod === period
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  {period} Days
                </button>
              ))}
            </div>

            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </header>

        <div className="p-6 space-y-6">
          {/* Overview Stats */}
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">
                    {isLoading ? (
                      <Loader2 className="h-6 w-6 animate-spin" />
                    ) : (
                      stats?.totalRuns || 0
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">Total Runs</div>
                </div>
                <BarChart3 className="h-8 w-8 text-muted-foreground/50" />
              </div>
            </div>

            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl font-bold text-success">
                      {isLoading ? (
                        <Loader2 className="h-6 w-6 animate-spin" />
                      ) : (
                        `${stats?.avgPassRate?.toFixed(1) || 0}%`
                      )}
                    </span>
                    <span className="flex items-center text-xs text-success">
                      <ArrowUpRight className="h-3 w-3" />
                    </span>
                  </div>
                  <div className="text-sm text-muted-foreground">Pass Rate</div>
                </div>
                <TrendingUp className="h-8 w-8 text-success/50" />
              </div>
            </div>

            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">
                    {isLoading ? (
                      <Loader2 className="h-6 w-6 animate-spin" />
                    ) : (
                      formatDuration(stats?.avgDuration || 0)
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">Avg Duration</div>
                </div>
                <Clock className="h-8 w-8 text-muted-foreground/50" />
              </div>
            </div>

            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold text-primary">
                    {isLoading ? (
                      <Loader2 className="h-6 w-6 animate-spin" />
                    ) : (
                      stats?.failedRuns?.length || 0
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">Failed Runs</div>
                </div>
                <Activity className="h-8 w-8 text-primary/50" />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Weekly Chart */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="mb-4">
                <h3 className="font-medium">Test Results by Day</h3>
                <p className="text-sm text-muted-foreground">Pass/fail distribution over the selected period</p>
              </div>
              <div className="flex items-end gap-2 h-48">
                {isLoading ? (
                  <div className="flex-1 flex items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : stats?.dailyStats?.length ? (
                  stats.dailyStats.map((stat, index) => (
                    <div key={stat.date} className="flex-1 flex flex-col items-center gap-2">
                      <div className="w-full flex flex-col-reverse gap-0.5" style={{ height: '160px' }}>
                        <div
                          className="w-full bg-success rounded-b transition-all duration-300"
                          style={{ height: `${(stat.passed / maxBar) * 100}%` }}
                        />
                        <div
                          className="w-full bg-error rounded-t transition-all duration-300"
                          style={{ height: `${(stat.failed / maxBar) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground">{stat.day}</span>
                    </div>
                  ))
                ) : (
                  <div className="flex-1 flex items-center justify-center text-muted-foreground">
                    No data available
                  </div>
                )}
              </div>
              <div className="flex items-center justify-center gap-6 mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-success" />
                  <span className="text-muted-foreground">Passed</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-error" />
                  <span className="text-muted-foreground">Failed</span>
                </div>
              </div>
            </div>

            {/* Recent Failures */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="mb-4">
                <h3 className="font-medium">Recent Failures</h3>
                <p className="text-sm text-muted-foreground">Test runs that need attention</p>
              </div>
              <div className="space-y-3">
                {isLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : stats?.failedRuns?.length ? (
                  stats.failedRuns.map((run) => (
                    <div
                      key={run.id}
                      className="flex items-center gap-3 p-3 rounded-lg border border-error/20 bg-error/5"
                    >
                      <XCircle className="h-5 w-5 text-error flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate">{run.name || 'Test Run'}</div>
                        <div className="text-sm text-muted-foreground">
                          {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                        </div>
                      </div>
                      <Badge variant="error">{run.failed_tests} failed</Badge>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No failed runs - great job!
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Recent Runs */}
          <div className="p-4 rounded-lg border bg-card">
            <div className="mb-4">
              <h3 className="font-medium">Recent Test Runs</h3>
              <p className="text-sm text-muted-foreground">History of all test executions</p>
            </div>
            <div className="space-y-2">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : recentRuns.length > 0 ? (
                recentRuns.map((run) => (
                  <div
                    key={run.id}
                    className="flex items-center gap-4 p-4 rounded-lg border hover:bg-muted/50 transition-colors"
                  >
                    <StatusDot status={run.status} />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{run.name || 'Test Run'}</div>
                      <div className="text-sm text-muted-foreground">
                        {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {run.duration_ms ? formatDuration(run.duration_ms / 1000) : '-'}
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <span className="text-success flex items-center gap-1">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        {run.passed_tests}
                      </span>
                      {run.failed_tests > 0 && (
                        <span className="text-error flex items-center gap-1">
                          <XCircle className="h-3.5 w-3.5" />
                          {run.failed_tests}
                        </span>
                      )}
                    </div>
                    <div className="w-20 h-2 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full bg-success transition-all duration-300"
                        style={{
                          width: `${
                            run.total_tests > 0
                              ? ((run.passed_tests || 0) / run.total_tests) * 100
                              : 0
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No test runs yet. Run your first test to see results here.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
