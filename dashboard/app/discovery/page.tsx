'use client';

import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Compass,
  Loader2,
  FileText,
  FormInput,
  MousePointer,
  ArrowRight,
  CheckCircle,
  Sparkles,
  Link2,
  Clock,
  AlertTriangle,
} from 'lucide-react';
import { useProjects } from '@/lib/hooks/use-projects';
import { useLatestDiscoveryData, useStartDiscovery } from '@/lib/hooks/use-discovery';
import { useCreateTest } from '@/lib/hooks/use-tests';
import { getSupabaseClient } from '@/lib/supabase/client';
import { Badge } from '@/components/ui/data-table';
import { useQueryClient } from '@tanstack/react-query';
import type { DiscoveredFlow } from '@/lib/supabase/types';

const WORKER_URL = process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

export default function DiscoveryPage() {
  const [appUrl, setAppUrl] = useState('');
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const currentProject = selectedProjectId || projects[0]?.id;
  const currentProjectData = projects.find((p) => p.id === currentProject);

  const { data: discoveryData, isLoading: discoveryLoading } = useLatestDiscoveryData(currentProject || null);
  const startDiscovery = useStartDiscovery();
  const createTest = useCreateTest();
  const queryClient = useQueryClient();
  const supabase = getSupabaseClient();

  // Set appUrl from project when it changes
  const effectiveAppUrl = appUrl || currentProjectData?.app_url || 'https://example.com';

  const handleGenerateTest = async (flow: DiscoveredFlow) => {
    if (!currentProject) return;

    try {
      // Create test from flow
      const test = await createTest.mutateAsync({
        project_id: currentProject,
        name: flow.name,
        description: flow.description || `Auto-generated from discovered flow: ${flow.name}`,
        steps: flow.steps || [],
        tags: ['discovered', 'auto-generated'],
        priority: flow.priority || 'medium',
        source: 'discovered',
      });

      // Update flow to mark it as converted
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await (supabase.from('discovered_flows') as any)
        .update({ converted_to_test_id: test.id })
        .eq('id', flow.id);

      // Refresh discovery data
      queryClient.invalidateQueries({ queryKey: ['latest-discovery', currentProject] });

      alert(`Test "${flow.name}" created successfully! Go to Tests page to run it.`);
    } catch (error) {
      console.error('Failed to create test:', error);
      alert('Failed to create test. Please try again.');
    }
  };

  const handleGenerateAllTests = async () => {
    if (!discoveryData?.flows?.length || !currentProject) return;

    const flowsToConvert = discoveryData.flows.filter(f => !f.converted_to_test_id);
    if (flowsToConvert.length === 0) {
      alert('All flows have already been converted to tests.');
      return;
    }

    try {
      for (const flow of flowsToConvert) {
        await handleGenerateTest(flow);
      }
      alert(`${flowsToConvert.length} tests created successfully!`);
    } catch (error) {
      console.error('Failed to generate all tests:', error);
    }
  };

  const handleStartDiscovery = async () => {
    if (!currentProject) return;
    try {
      await startDiscovery.mutateAsync({
        projectId: currentProject,
        appUrl: effectiveAppUrl,
      });
    } catch (error) {
      console.error('Failed to start discovery:', error);
    }
  };

  const isDiscovering = startDiscovery.isPending || discoveryData?.session?.status === 'running';
  const hasData = !!discoveryData?.session;

  // No project state
  if (!projectsLoading && projects.length === 0) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Compass className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">No Projects Yet</h2>
            <p className="text-muted-foreground">
              Create a project first to start discovering your application.
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
              onChange={(e) => {
                setSelectedProjectId(e.target.value);
                const project = projects.find((p) => p.id === e.target.value);
                if (project) setAppUrl(project.app_url);
              }}
              className="h-9 rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>

            <div className="flex-1" />

            <Input
              value={effectiveAppUrl}
              onChange={(e) => setAppUrl(e.target.value)}
              placeholder="App URL"
              className="w-64 h-9"
            />
            <Button size="sm" onClick={handleStartDiscovery} disabled={isDiscovering}>
              {isDiscovering ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Discovering...
                </>
              ) : (
                <>
                  <Compass className="mr-2 h-4 w-4" />
                  Start Discovery
                </>
              )}
            </Button>
          </div>
        </header>

        <div className="p-6 space-y-6">
          {/* Discovery Progress */}
          {isDiscovering && (
            <div className="p-8 rounded-lg border bg-card">
              <div className="flex flex-col items-center gap-4">
                <div className="relative">
                  <div className="absolute inset-0 animate-ping rounded-full bg-primary/20" />
                  <div className="relative p-4 rounded-full bg-primary/10">
                    <Compass className="h-8 w-8 text-primary animate-spin" />
                  </div>
                </div>
                <div className="text-center">
                  <h3 className="font-semibold">Discovering Application</h3>
                  <p className="text-sm text-muted-foreground">
                    Crawling pages, analyzing UI, identifying user flows...
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Stats */}
          {!isDiscovering && (
            <>
              <div className="grid grid-cols-4 gap-4">
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-3">
                    <FileText className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-2xl font-bold">
                        {discoveryLoading ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                          discoveryData?.pages?.length || 0
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground">Pages Found</div>
                    </div>
                  </div>
                </div>
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-3">
                    <ArrowRight className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-2xl font-bold">
                        {discoveryLoading ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                          discoveryData?.flows?.length || 0
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground">Flows Identified</div>
                    </div>
                  </div>
                </div>
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-3">
                    <FormInput className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-2xl font-bold">
                        {discoveryLoading ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                          discoveryData?.pages?.reduce((acc, p) => acc + (p.form_count || 0), 0) || 0
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground">Forms Detected</div>
                    </div>
                  </div>
                </div>
                <div className="p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-3">
                    <MousePointer className="h-5 w-5 text-primary" />
                    <div>
                      <div className="text-2xl font-bold">
                        {discoveryLoading ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                          discoveryData?.pages?.reduce((acc, p) => acc + (p.element_count || 0), 0) || 0
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground">Interactive Elements</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Last Session Info */}
              {hasData && discoveryData?.session && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Clock className="h-4 w-4" />
                  Last discovery: {formatDistanceToNow(new Date(discoveryData.session.created_at), { addSuffix: true })}
                  <span className="mx-2">•</span>
                  <Badge variant={discoveryData.session.status === 'completed' ? 'success' : 'info'}>
                    {discoveryData.session.status}
                  </Badge>
                </div>
              )}

              {/* Discovered Flows */}
              <div className="p-4 rounded-lg border bg-card">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-medium">Discovered Flows</h3>
                    <p className="text-sm text-muted-foreground">User journeys automatically identified</p>
                  </div>
                  <Button
                  size="sm"
                  disabled={!discoveryData?.flows?.length || createTest.isPending}
                  onClick={handleGenerateAllTests}
                >
                  {createTest.isPending ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Sparkles className="mr-2 h-4 w-4" />
                  )}
                  Generate All Tests
                </Button>
                </div>
                <div className="space-y-2">
                  {discoveryLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                    </div>
                  ) : discoveryData?.flows?.length ? (
                    discoveryData.flows.map((flow) => (
                      <div
                        key={flow.id}
                        className="flex items-center gap-4 p-4 rounded-lg border hover:bg-muted/50 transition-colors"
                      >
                        <CheckCircle className="h-5 w-5 text-success" />
                        <div className="flex-1">
                          <div className="font-medium">{flow.name}</div>
                          <div className="text-sm text-muted-foreground">{flow.description || 'No description'}</div>
                        </div>
                        <div className="text-sm text-muted-foreground">{flow.step_count} steps</div>
                        <Badge
                          variant={
                            flow.priority === 'critical'
                              ? 'error'
                              : flow.priority === 'high'
                              ? 'warning'
                              : 'info'
                          }
                        >
                          {flow.priority}
                        </Badge>
                        <Button
                          variant={flow.converted_to_test_id ? 'ghost' : 'outline'}
                          size="sm"
                          onClick={() => handleGenerateTest(flow)}
                          disabled={!!flow.converted_to_test_id || createTest.isPending}
                        >
                          {flow.converted_to_test_id ? (
                            <>
                              <CheckCircle className="mr-2 h-4 w-4 text-success" />
                              Test Created
                            </>
                          ) : (
                            'Generate Test'
                          )}
                        </Button>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      No flows discovered yet. Start a discovery to identify user journeys.
                    </div>
                  )}
                </div>
              </div>

              {/* Discovered Pages */}
              <div className="p-4 rounded-lg border bg-card">
                <div className="mb-4">
                  <h3 className="font-medium">Discovered Pages</h3>
                  <p className="text-sm text-muted-foreground">All pages found during discovery</p>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  {discoveryLoading ? (
                    <div className="col-span-full flex items-center justify-center py-8">
                      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                    </div>
                  ) : discoveryData?.pages?.length ? (
                    discoveryData.pages.map((page) => (
                      <div
                        key={page.id}
                        className="p-4 rounded-lg border hover:bg-muted/50 transition-colors"
                      >
                        <div className="font-medium truncate">{page.title || page.url}</div>
                        <div className="text-sm text-primary truncate">{page.url}</div>
                        <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <MousePointer className="h-3 w-3" />
                            {page.element_count} elements
                          </span>
                          <span className="flex items-center gap-1">
                            <FormInput className="h-3 w-3" />
                            {page.form_count} forms
                          </span>
                          <span className="flex items-center gap-1">
                            <Link2 className="h-3 w-3" />
                            {page.link_count} links
                          </span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="col-span-full text-center py-8 text-muted-foreground">
                      No pages discovered yet. Start a discovery to crawl your application.
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
