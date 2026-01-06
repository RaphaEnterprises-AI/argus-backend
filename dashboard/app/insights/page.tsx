'use client';

import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import {
  Brain,
  Loader2,
  AlertTriangle,
  Lightbulb,
  Target,
  TrendingUp,
  Check,
  Sparkles,
} from 'lucide-react';
import { useProjects } from '@/lib/hooks/use-projects';
import { useAIInsights, useResolveInsight, useInsightStats } from '@/lib/hooks/use-insights';
import { Badge } from '@/components/ui/data-table';
import { cn } from '@/lib/utils';

export default function InsightsPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const currentProject = selectedProjectId || projects[0]?.id;

  const { data: insights = [], isLoading: insightsLoading } = useAIInsights(currentProject || null);
  const { data: stats } = useInsightStats(currentProject || null);
  const resolveInsight = useResolveInsight();

  const isLoading = projectsLoading || insightsLoading;

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'prediction': return TrendingUp;
      case 'anomaly': return AlertTriangle;
      case 'suggestion': return Lightbulb;
      default: return Target;
    }
  };

  if (!projectsLoading && projects.length === 0) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Brain className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">No Projects Yet</h2>
            <p className="text-muted-foreground">Create a project to start getting AI insights.</p>
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
              {stats && (
                <>
                  <div className="flex items-center gap-2 text-sm">
                    <AlertTriangle className="h-4 w-4 text-error" />
                    <span className="font-medium">{stats.bySeverity.critical}</span>
                    <span className="text-muted-foreground">critical</span>
                  </div>
                  <div className="h-4 w-px bg-border" />
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-medium">{stats.unresolved}</span>
                    <span className="text-muted-foreground">active</span>
                  </div>
                </>
              )}
            </div>
            <div className="flex-1" />
            <Button size="sm" variant="outline">
              <Sparkles className="h-4 w-4 mr-2" />
              Generate Insights
            </Button>
          </div>
        </header>

        <div className="p-6 space-y-6">
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-2xl font-bold">{stats?.unresolved || 0}</div>
              <div className="text-sm text-muted-foreground">Active Insights</div>
            </div>
            <div className="p-4 rounded-lg border bg-card border-error/20 bg-error/5">
              <div className="text-2xl font-bold text-error">{stats?.bySeverity.critical || 0}</div>
              <div className="text-sm text-muted-foreground">Critical</div>
            </div>
            <div className="p-4 rounded-lg border bg-card border-warning/20 bg-warning/5">
              <div className="text-2xl font-bold text-warning">{stats?.bySeverity.high || 0}</div>
              <div className="text-sm text-muted-foreground">High Priority</div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-2xl font-bold text-success">{stats?.resolved || 0}</div>
              <div className="text-sm text-muted-foreground">Resolved</div>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Active Insights</h3>
            <div className="space-y-3">
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : insights.length > 0 ? (
                insights.map((insight) => {
                  const Icon = getInsightIcon(insight.insight_type);
                  return (
                    <div
                      key={insight.id}
                      className={cn(
                        'p-4 rounded-lg border',
                        insight.severity === 'critical' && 'border-error/30 bg-error/5',
                        insight.severity === 'high' && 'border-warning/30 bg-warning/5'
                      )}
                    >
                      <div className="flex items-start gap-4">
                        <div className={cn(
                          'p-2 rounded-lg',
                          insight.severity === 'critical' ? 'bg-error/10' : 'bg-primary/10'
                        )}>
                          <Icon className={cn(
                            'h-5 w-5',
                            insight.severity === 'critical' ? 'text-error' : 'text-primary'
                          )} />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium">{insight.title}</span>
                            <Badge variant={insight.severity === 'critical' ? 'error' : insight.severity === 'high' ? 'warning' : 'info'}>
                              {insight.severity}
                            </Badge>
                            <Badge variant="outline">{insight.insight_type}</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mb-2">{insight.description}</p>
                          {insight.suggested_action && (
                            <p className="text-sm text-primary">{insight.suggested_action}</p>
                          )}
                          <div className="text-xs text-muted-foreground mt-2">
                            {formatDistanceToNow(new Date(insight.created_at), { addSuffix: true })}
                            {insight.confidence && ` • ${(insight.confidence * 100).toFixed(0)}% confidence`}
                          </div>
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => currentProject && resolveInsight.mutate({
                            insightId: insight.id,
                            projectId: currentProject,
                          })}
                          disabled={resolveInsight.isPending}
                        >
                          <Check className="h-4 w-4 mr-2" />
                          Resolve
                        </Button>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  No active insights. Your tests are running smoothly!
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
