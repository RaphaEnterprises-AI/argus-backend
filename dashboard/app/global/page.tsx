'use client';

import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Globe,
  MapPin,
  Activity,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  Wifi,
  Server,
  Play,
  Loader2,
} from 'lucide-react';
import { useProjects } from '@/lib/hooks/use-projects';
import { useLatestGlobalTest, useStartGlobalTest } from '@/lib/hooks/use-global';
import { Badge } from '@/components/ui/data-table';
import { cn } from '@/lib/utils';

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'success':
      return <CheckCircle className="h-4 w-4 text-success" />;
    case 'error':
    case 'timeout':
      return <AlertTriangle className="h-4 w-4 text-error" />;
    case 'slow':
      return <Clock className="h-4 w-4 text-warning" />;
    default:
      return <Loader2 className="h-4 w-4 text-info animate-spin" />;
  }
}

function LatencyBar({ value, max }: { value: number; max: number }) {
  const percentage = Math.min((value / max) * 100, 100);
  const color = value < 100 ? 'bg-success' : value < 200 ? 'bg-warning' : 'bg-error';

  return (
    <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
      <div
        className={cn('h-full rounded-full transition-all duration-500', color)}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}

export default function GlobalTestingPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [testUrl, setTestUrl] = useState('');

  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const currentProject = selectedProjectId || projects[0]?.id;

  const { data: latestTest, isLoading: testLoading } = useLatestGlobalTest(currentProject || null);
  const startTest = useStartGlobalTest();

  const isLoading = projectsLoading || testLoading;
  const test = latestTest?.test;
  const results = latestTest?.results || [];

  // Separate results by status
  const errorResults = results.filter(r => r.status === 'error' || r.status === 'timeout');
  const slowResults = results.filter(r => r.status === 'slow');

  const handleRunTest = () => {
    if (!currentProject || !testUrl) return;
    startTest.mutate({
      projectId: currentProject,
      url: testUrl,
    });
  };

  if (!projectsLoading && projects.length === 0) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Globe className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">No Projects Yet</h2>
            <p className="text-muted-foreground">Create a project to start global edge testing.</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-64">
        <header className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex h-14 items-center gap-4 px-6">
            <select
              value={currentProject || ''}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="h-9 rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {projects.map((project) => (
                <option key={project.id} value={project.id}>{project.name}</option>
              ))}
            </select>
            {test && (
              <div className="text-sm text-muted-foreground ml-4">
                Last test: {formatDistanceToNow(new Date(test.created_at), { addSuffix: true })}
              </div>
            )}
            <div className="flex-1" />
            <Input
              value={testUrl}
              onChange={(e) => setTestUrl(e.target.value)}
              placeholder="URL to test (e.g., https://example.com)"
              className="w-72"
            />
            <Button size="sm" onClick={handleRunTest} disabled={startTest.isPending || !testUrl}>
              {startTest.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Run Global Test
            </Button>
          </div>
        </header>

        <div className="p-6 space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <Wifi className="h-5 w-5 text-success" />
                <span className="font-medium">Avg Latency</span>
              </div>
              <div className="text-3xl font-bold">
                {test?.avg_latency_ms ?? '-'}
                {test?.avg_latency_ms !== null && <span className="text-lg text-muted-foreground">ms</span>}
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <Server className="h-5 w-5 text-info" />
                <span className="font-medium">Avg TTFB</span>
              </div>
              <div className="text-3xl font-bold">
                {test?.avg_ttfb_ms ?? '-'}
                {test?.avg_ttfb_ms !== null && <span className="text-lg text-muted-foreground">ms</span>}
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <Activity className="h-5 w-5 text-primary" />
                <span className="font-medium">Success Rate</span>
              </div>
              <div className="text-3xl font-bold text-success">
                {test?.success_rate !== null ? `${test?.success_rate}%` : '-'}
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <AlertTriangle className="h-5 w-5 text-warning" />
                <span className="font-medium">Problem Regions</span>
              </div>
              <div className="text-3xl font-bold text-warning">
                {(test?.slow_regions || 0) + (test?.failed_regions || 0)}
              </div>
            </div>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-3 gap-6">
            {/* Regional Results */}
            <div className="col-span-2 p-4 rounded-lg border bg-card">
              <h3 className="font-medium mb-4">Regional Performance</h3>
              <div className="space-y-3">
                {isLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : results.length > 0 ? (
                  results.map((result) => (
                    <div
                      key={result.id}
                      className={cn(
                        'p-4 rounded-lg border',
                        (result.status === 'error' || result.status === 'timeout') && 'border-error/30 bg-error/5',
                        result.status === 'slow' && 'border-warning/30 bg-warning/5'
                      )}
                    >
                      <div className="flex items-center gap-4">
                        <StatusIcon status={result.status} />
                        <div className="flex items-center gap-2 w-40">
                          <MapPin className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <div className="font-medium">{result.city}</div>
                            <div className="text-xs text-muted-foreground">{result.region_code}</div>
                          </div>
                        </div>

                        {result.status !== 'error' && result.status !== 'timeout' ? (
                          <>
                            <div className="flex-1">
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-muted-foreground">Latency</span>
                                <span className={cn(
                                  (result.latency_ms || 0) < 100 ? 'text-success' :
                                  (result.latency_ms || 0) < 200 ? 'text-warning' : 'text-error'
                                )}>{result.latency_ms}ms</span>
                              </div>
                              <LatencyBar value={result.latency_ms || 0} max={300} />
                            </div>

                            <div className="w-24 text-center">
                              <div className="text-sm font-medium">{result.ttfb_ms}ms</div>
                              <div className="text-xs text-muted-foreground">TTFB</div>
                            </div>

                            <div className="w-24 text-center">
                              <div className="text-sm font-medium">
                                {result.page_load_ms ? `${(result.page_load_ms / 1000).toFixed(1)}s` : '-'}
                              </div>
                              <div className="text-xs text-muted-foreground">Page Load</div>
                            </div>
                          </>
                        ) : (
                          <div className="flex-1 text-error text-sm">
                            {result.error_message || 'Connection failed'}
                          </div>
                        )}

                        <Badge variant={
                          result.status === 'success' ? 'success' :
                          result.status === 'slow' ? 'warning' : 'error'
                        }>
                          {result.status}
                        </Badge>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    {test ? 'No results available for this test.' : 'Run a global test to see regional performance.'}
                  </div>
                )}
              </div>
            </div>

            {/* Anomalies Sidebar */}
            <div className="p-4 rounded-lg border bg-card">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Zap className="h-5 w-5 text-warning" />
                Anomalies ({errorResults.length + slowResults.length})
              </h3>
              <div className="space-y-3">
                {errorResults.length === 0 && slowResults.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground text-sm">
                    {results.length > 0 ? 'No anomalies detected!' : 'Run a test to detect anomalies.'}
                  </div>
                ) : (
                  <>
                    {errorResults.map((result) => (
                      <div key={result.id} className="p-3 rounded-lg border border-error/30 bg-error/5">
                        <div className="flex items-center gap-2 text-error font-medium">
                          <AlertTriangle className="h-4 w-4" />
                          {result.city} Unreachable
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {result.error_message || 'Connection timeout. Possible CDN node issue.'}
                        </p>
                      </div>
                    ))}
                    {slowResults.map((result) => (
                      <div key={result.id} className="p-3 rounded-lg border border-warning/30 bg-warning/5">
                        <div className="flex items-center gap-2 text-warning font-medium">
                          <Clock className="h-4 w-4" />
                          High Latency: {result.city}
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {result.latency_ms}ms latency, {result.ttfb_ms}ms TTFB. Consider adding edge server.
                        </p>
                      </div>
                    ))}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Test URL Info */}
          {test && (
            <div className="p-4 rounded-lg border bg-card">
              <h3 className="font-medium mb-2">Test Details</h3>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">URL:</span>{' '}
                  <span className="font-mono">{test.url}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Status:</span>{' '}
                  <Badge variant={test.status === 'completed' ? 'success' : test.status === 'failed' ? 'error' : 'warning'}>
                    {test.status}
                  </Badge>
                </div>
                <div>
                  <span className="text-muted-foreground">Regions Tested:</span>{' '}
                  {results.length}
                </div>
                <div>
                  <span className="text-muted-foreground">Completed:</span>{' '}
                  {test.completed_at ? formatDistanceToNow(new Date(test.completed_at), { addSuffix: true }) : 'In progress'}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
