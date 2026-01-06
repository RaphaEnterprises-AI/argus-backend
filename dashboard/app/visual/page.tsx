'use client';

import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import {
  Eye,
  Loader2,
  Check,
  X,
  RefreshCw,
  AlertTriangle,
  Plus,
  Camera,
} from 'lucide-react';
import { useProjects } from '@/lib/hooks/use-projects';
import { useVisualBaselines, useVisualComparisons, useApproveComparison, useRunVisualTest, useUpdateBaseline } from '@/lib/hooks/use-visual';
import { Badge } from '@/components/ui/data-table';
import { cn } from '@/lib/utils';

export default function VisualPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [testUrl, setTestUrl] = useState('');
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [selectedComparison, setSelectedComparison] = useState<string | null>(null);

  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const currentProject = selectedProjectId || projects[0]?.id;

  const { data: baselines = [], isLoading: baselinesLoading } = useVisualBaselines(currentProject || null);
  const { data: comparisons = [], isLoading: comparisonsLoading } = useVisualComparisons(currentProject || null);
  const approveComparison = useApproveComparison();
  const runVisualTest = useRunVisualTest();
  const updateBaseline = useUpdateBaseline();

  const isLoading = projectsLoading || baselinesLoading || comparisonsLoading;
  const isRunning = runVisualTest.isPending;

  const handleRunVisualTest = async () => {
    if (!currentProject || !testUrl) return;

    try {
      await runVisualTest.mutateAsync({
        projectId: currentProject,
        url: testUrl,
      });
      setTestUrl('');
      setShowUrlInput(false);
    } catch (error) {
      console.error('Visual test failed:', error);
    }
  };

  const handleUpdateBaseline = async (comparisonId: string) => {
    if (!currentProject) return;
    try {
      await updateBaseline.mutateAsync({
        comparisonId,
        projectId: currentProject,
      });
    } catch (error) {
      console.error('Update baseline failed:', error);
    }
  };

  const stats = {
    total: comparisons.length,
    match: comparisons.filter((c) => c.status === 'match').length,
    mismatch: comparisons.filter((c) => c.status === 'mismatch').length,
    pending: comparisons.filter((c) => c.status === 'pending' || c.status === 'new').length,
  };

  if (!projectsLoading && projects.length === 0) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Eye className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">No Projects Yet</h2>
            <p className="text-muted-foreground">Create a project first to start visual testing.</p>
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
            <div className="flex items-center gap-3 ml-4">
              <div className="flex items-center gap-2 text-sm">
                <Check className="h-4 w-4 text-success" />
                <span className="font-medium">{stats.match}</span>
              </div>
              <div className="h-4 w-px bg-border" />
              <div className="flex items-center gap-2 text-sm">
                <X className="h-4 w-4 text-error" />
                <span className="font-medium">{stats.mismatch}</span>
              </div>
              <div className="h-4 w-px bg-border" />
              <div className="flex items-center gap-2 text-sm">
                <AlertTriangle className="h-4 w-4 text-warning" />
                <span className="font-medium">{stats.pending}</span>
              </div>
            </div>
            <div className="flex-1" />
            {showUrlInput ? (
              <div className="flex items-center gap-2">
                <input
                  type="url"
                  value={testUrl}
                  onChange={(e) => setTestUrl(e.target.value)}
                  placeholder="https://example.com"
                  className="h-9 w-64 rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  onKeyDown={(e) => e.key === 'Enter' && handleRunVisualTest()}
                  autoFocus
                />
                <Button
                  size="sm"
                  onClick={handleRunVisualTest}
                  disabled={isRunning || !testUrl}
                >
                  {isRunning ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Capturing...
                    </>
                  ) : (
                    <>
                      <Camera className="h-4 w-4 mr-2" />
                      Capture
                    </>
                  )}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setShowUrlInput(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <Button size="sm" variant="outline" onClick={() => setShowUrlInput(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Visual Test
              </Button>
            )}
          </div>
        </header>

        <div className="p-6 space-y-6">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Recent Comparisons</h3>
            <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {isLoading ? (
                Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="aspect-video rounded-lg border bg-muted animate-pulse" />
                ))
              ) : comparisons.length > 0 ? (
                comparisons.map((comparison) => (
                  <div
                    key={comparison.id}
                    className={cn(
                      'relative rounded-lg border overflow-hidden group cursor-pointer',
                      comparison.status === 'match' && 'border-success/30',
                      comparison.status === 'mismatch' && 'border-error/30',
                      comparison.status === 'new' && 'border-primary/30',
                      selectedComparison === comparison.id && 'ring-2 ring-primary'
                    )}
                    onClick={() => setSelectedComparison(selectedComparison === comparison.id ? null : comparison.id)}
                  >
                    <div className="aspect-video bg-muted relative overflow-hidden">
                      {comparison.current_url ? (
                        <img
                          src={comparison.current_url}
                          alt={comparison.name}
                          className="w-full h-full object-cover object-top"
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <Camera className="h-8 w-8 text-muted-foreground/50" />
                        </div>
                      )}
                      {comparison.status === 'mismatch' && comparison.match_percentage && (
                        <div className="absolute top-2 right-2 bg-error/90 text-error-foreground text-xs px-2 py-0.5 rounded">
                          {comparison.match_percentage.toFixed(1)}% match
                        </div>
                      )}
                    </div>
                    <div className="p-3 border-t">
                      <div className="flex items-center justify-between">
                        <div className="truncate flex-1 mr-2">
                          <div className="font-medium text-sm truncate">{comparison.name}</div>
                          <div className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(comparison.created_at), { addSuffix: true })}
                          </div>
                        </div>
                        <Badge variant={comparison.status === 'match' ? 'success' : comparison.status === 'mismatch' ? 'error' : 'warning'}>
                          {comparison.status}
                        </Badge>
                      </div>
                      {comparison.status === 'mismatch' && selectedComparison === comparison.id && (
                        <div className="mt-2 pt-2 border-t flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleUpdateBaseline(comparison.id);
                            }}
                            disabled={updateBaseline.isPending}
                          >
                            {updateBaseline.isPending ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <>
                                <RefreshCw className="h-3 w-3 mr-1" />
                                Update Baseline
                              </>
                            )}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="flex-1"
                            onClick={(e) => {
                              e.stopPropagation();
                              approveComparison.mutate({
                                comparisonId: comparison.id,
                                projectId: currentProject!,
                              });
                            }}
                          >
                            <Check className="h-3 w-3 mr-1" />
                            Approve
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-full text-center py-12 text-muted-foreground">
                  No visual comparisons yet. Run visual tests to capture screenshots.
                </div>
              )}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Baselines ({baselines.length})</h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {baselines.length > 0 ? (
                baselines.map((baseline) => (
                  <div key={baseline.id} className="rounded-lg border overflow-hidden group">
                    <div className="aspect-video bg-muted relative overflow-hidden">
                      {baseline.screenshot_url ? (
                        <img
                          src={baseline.screenshot_url}
                          alt={baseline.name}
                          className="w-full h-full object-cover object-top"
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <Camera className="h-6 w-6 text-muted-foreground/50" />
                        </div>
                      )}
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="text-white text-xs truncate">{baseline.viewport}</div>
                      </div>
                    </div>
                    <div className="p-2 border-t">
                      <div className="font-medium text-xs truncate">{baseline.name}</div>
                      <div className="text-xs text-muted-foreground truncate">{baseline.page_url}</div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-full text-center py-8 text-muted-foreground">
                  No baselines yet. Capture your first screenshot to create a baseline.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
