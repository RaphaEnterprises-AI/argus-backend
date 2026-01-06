'use client';

import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import {
  Shield,
  Loader2,
  AlertTriangle,
  CheckCircle,
  Gauge,
  Zap,
  Play,
} from 'lucide-react';
import { useProjects } from '@/lib/hooks/use-projects';
import { useLatestAudit, useQualityAudits, useStartQualityAudit } from '@/lib/hooks/use-quality';
import { Badge } from '@/components/ui/data-table';
import { cn } from '@/lib/utils';
import { Input } from '@/components/ui/input';

function ScoreBar({ label, score, color }: { label: string; score: number | null; color: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{score !== null ? `${score}%` : '-'}</span>
      </div>
      <div className="h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-500', color)}
          style={{ width: `${score || 0}%` }}
        />
      </div>
    </div>
  );
}

export default function QualityPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [auditUrl, setAuditUrl] = useState('');

  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const currentProject = selectedProjectId || projects[0]?.id;
  const currentProjectData = projects.find((p) => p.id === currentProject);

  const { data: latestAudit, isLoading: auditLoading } = useLatestAudit(currentProject || null);
  const { data: audits = [] } = useQualityAudits(currentProject || null);
  const startAudit = useStartQualityAudit();

  const effectiveUrl = auditUrl || currentProjectData?.app_url || 'https://example.com';

  const handleRunAudit = async () => {
    if (!currentProject) return;
    try {
      await startAudit.mutateAsync({
        projectId: currentProject,
        url: effectiveUrl,
      });
    } catch (error) {
      console.error('Failed to run audit:', error);
    }
  };

  const isRunning = startAudit.isPending;
  const isLoading = projectsLoading || auditLoading;
  const audit = latestAudit?.audit;
  const issues = latestAudit?.issues || [];

  if (!projectsLoading && projects.length === 0) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Shield className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">No Projects Yet</h2>
            <p className="text-muted-foreground">Create a project to start quality audits.</p>
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
            {audit && (
              <div className="text-sm text-muted-foreground ml-4">
                Last audit: {formatDistanceToNow(new Date(audit.created_at), { addSuffix: true })}
              </div>
            )}
            <div className="flex-1" />
            <Input
              value={effectiveUrl}
              onChange={(e) => setAuditUrl(e.target.value)}
              placeholder="URL to audit"
              className="w-64 h-9"
            />
            <Button size="sm" onClick={handleRunAudit} disabled={isRunning}>
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Audit
                </>
              )}
            </Button>
          </div>
        </header>

        <div className="p-6 space-y-6">
          {/* Score Cards */}
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-4">
                <Shield className="h-5 w-5 text-primary" />
                <span className="font-medium">Accessibility</span>
              </div>
              <div className="text-3xl font-bold mb-2">
                {audit?.accessibility_score ?? '-'}
                {audit?.accessibility_score !== null && <span className="text-lg text-muted-foreground">%</span>}
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-primary rounded-full" style={{ width: `${audit?.accessibility_score || 0}%` }} />
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-4">
                <Zap className="h-5 w-5 text-warning" />
                <span className="font-medium">Performance</span>
              </div>
              <div className="text-3xl font-bold mb-2">
                {audit?.performance_score ?? '-'}
                {audit?.performance_score !== null && <span className="text-lg text-muted-foreground">%</span>}
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-warning rounded-full" style={{ width: `${audit?.performance_score || 0}%` }} />
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="h-5 w-5 text-success" />
                <span className="font-medium">Best Practices</span>
              </div>
              <div className="text-3xl font-bold mb-2">
                {audit?.best_practices_score ?? '-'}
                {audit?.best_practices_score !== null && <span className="text-lg text-muted-foreground">%</span>}
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-success rounded-full" style={{ width: `${audit?.best_practices_score || 0}%` }} />
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-4">
                <Gauge className="h-5 w-5 text-info" />
                <span className="font-medium">SEO</span>
              </div>
              <div className="text-3xl font-bold mb-2">
                {audit?.seo_score ?? '-'}
                {audit?.seo_score !== null && <span className="text-lg text-muted-foreground">%</span>}
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-info rounded-full" style={{ width: `${audit?.seo_score || 0}%` }} />
              </div>
            </div>
          </div>

          {/* Core Web Vitals */}
          {audit && (
            <div className="p-4 rounded-lg border bg-card">
              <h3 className="font-medium mb-4">Core Web Vitals</h3>
              <div className="grid grid-cols-3 gap-6">
                <div>
                  <div className="text-sm text-muted-foreground mb-1">LCP (Largest Contentful Paint)</div>
                  <div className="text-2xl font-bold">{audit.lcp_ms ? `${(audit.lcp_ms / 1000).toFixed(2)}s` : '-'}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">FID (First Input Delay)</div>
                  <div className="text-2xl font-bold">{audit.fid_ms ? `${audit.fid_ms}ms` : '-'}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">CLS (Cumulative Layout Shift)</div>
                  <div className="text-2xl font-bold">{audit.cls?.toFixed(3) ?? '-'}</div>
                </div>
              </div>
            </div>
          )}

          {/* Accessibility Issues */}
          <div className="p-4 rounded-lg border bg-card">
            <h3 className="font-medium mb-4">Accessibility Issues ({issues.length})</h3>
            <div className="space-y-2">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : issues.length > 0 ? (
                issues.map((issue) => (
                  <div key={issue.id} className="p-3 rounded-lg border hover:bg-muted/50">
                    <div className="flex items-start gap-3">
                      <AlertTriangle className={cn(
                        'h-4 w-4 mt-0.5',
                        issue.severity === 'critical' ? 'text-error' : issue.severity === 'serious' ? 'text-warning' : 'text-muted-foreground'
                      )} />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">{issue.rule}</span>
                          <Badge variant={issue.severity === 'critical' ? 'error' : issue.severity === 'serious' ? 'warning' : 'info'}>
                            {issue.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{issue.description}</p>
                        {issue.suggested_fix && (
                          <p className="text-sm text-primary mt-1">{issue.suggested_fix}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  {audit ? 'No accessibility issues found!' : 'Run an audit to check for accessibility issues.'}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
