'use client';

import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  AlertCircle,
  Loader2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Sparkles,
  GitPullRequest,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
  ExternalLink,
  Code,
  FileCode,
  Shield,
  Activity,
  Zap,
  Target,
  Eye,
  Play,
  BarChart3,
  RefreshCcw,
  Brain,
} from 'lucide-react';
import { useProjects } from '@/lib/hooks/use-projects';
import {
  useProductionEvents,
  useProductionEventStats,
  useGeneratedTests,
  useRiskScores,
  useQualityIntelligenceStats,
  useGenerateTest,
  useUpdateGeneratedTestStatus,
  useCalculateRiskScores,
  useAIQualityScore,
  useAutonomousLoop,
  usePredictiveQuality,
} from '@/lib/hooks/use-intelligence';
import { Badge } from '@/components/ui/data-table';
import { cn } from '@/lib/utils';

const severityColors = {
  fatal: 'bg-red-500',
  error: 'bg-orange-500',
  warning: 'bg-yellow-500',
  info: 'bg-blue-500',
};

const statusColors = {
  new: 'info',
  analyzing: 'warning',
  test_generated: 'success',
  test_pending_review: 'warning',
  resolved: 'success',
  ignored: 'default',
  wont_fix: 'default',
} as const;

const testStatusColors = {
  pending: 'warning',
  approved: 'success',
  rejected: 'error',
  modified: 'info',
  deployed: 'success',
} as const;

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span className={cn(
      'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium text-white',
      severityColors[severity as keyof typeof severityColors] || 'bg-gray-500'
    )}>
      {severity}
    </span>
  );
}

function TrendIcon({ trend }: { trend: string | null }) {
  if (trend === 'improving') return <TrendingUp className="h-4 w-4 text-green-500" />;
  if (trend === 'degrading') return <TrendingDown className="h-4 w-4 text-red-500" />;
  return <Minus className="h-4 w-4 text-muted-foreground" />;
}

export default function IntelligencePage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'events' | 'tests' | 'risks'>('overview');
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const currentProject = selectedProjectId || projects[0]?.id;

  const { data: events = [], isLoading: eventsLoading } = useProductionEvents(currentProject || null);
  const { data: eventStats } = useProductionEventStats(currentProject || null);
  const { data: generatedTests = [], isLoading: testsLoading } = useGeneratedTests(currentProject || null);
  const { data: riskScores = [], isLoading: risksLoading } = useRiskScores(currentProject || null);
  const { data: stats } = useQualityIntelligenceStats(currentProject || null);
  const { data: qualityScore, isLoading: scoreLoading, refetch: refetchScore } = useAIQualityScore(currentProject || null);
  const { data: predictions, isLoading: predictionsLoading } = usePredictiveQuality(currentProject || null);

  const generateTest = useGenerateTest();
  const updateTestStatus = useUpdateGeneratedTestStatus();
  const calculateRiskScores = useCalculateRiskScores();
  const autonomousLoop = useAutonomousLoop();

  const isLoading = projectsLoading || eventsLoading || testsLoading || risksLoading;

  const handleGenerateTest = async (eventId: string) => {
    if (!currentProject) return;
    try {
      await generateTest.mutateAsync({
        projectId: currentProject,
        eventId,
        framework: 'playwright',
      });
    } catch (error) {
      console.error('Failed to generate test:', error);
    }
  };

  const handleApproveTest = async (testId: string) => {
    try {
      await updateTestStatus.mutateAsync({
        testId,
        status: 'approved',
      });
    } catch (error) {
      console.error('Failed to approve test:', error);
    }
  };

  const handleRejectTest = async (testId: string) => {
    try {
      await updateTestStatus.mutateAsync({
        testId,
        status: 'rejected',
      });
    } catch (error) {
      console.error('Failed to reject test:', error);
    }
  };

  const handleCalculateRiskScores = async () => {
    if (!currentProject) return;
    try {
      await calculateRiskScores.mutateAsync({
        projectId: currentProject,
        entityTypes: ['page', 'component'],
      });
    } catch (error) {
      console.error('Failed to calculate risk scores:', error);
    }
  };

  if (!projectsLoading && projects.length === 0) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <AlertCircle className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold tracking-tight mb-2">No Projects Yet</h2>
            <p className="text-muted-foreground">Create a project to start using Quality Intelligence.</p>
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
            <select
              value={currentProject || ''}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="h-9 rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {projects.map((project) => (
                <option key={project.id} value={project.id}>{project.name}</option>
              ))}
            </select>
            <div className="flex-1" />
            <div className="flex items-center gap-2">
              {stats?.last_event_received_at && (
                <span className="text-xs text-muted-foreground">
                  Last event: {formatDistanceToNow(new Date(stats.last_event_received_at), { addSuffix: true })}
                </span>
              )}
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-4 px-6 border-t">
            <button
              onClick={() => setActiveTab('overview')}
              className={cn(
                'py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'overview'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              <span className="flex items-center gap-1">
                <Brain className="h-4 w-4" />
                AI Quality Score
              </span>
            </button>
            <button
              onClick={() => setActiveTab('events')}
              className={cn(
                'py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'events'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              Production Events
              {eventStats && (
                <span className="ml-2 px-1.5 py-0.5 rounded-full text-xs bg-muted">
                  {eventStats.total}
                </span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('tests')}
              className={cn(
                'py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'tests'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              Generated Tests
              <span className="ml-2 px-1.5 py-0.5 rounded-full text-xs bg-muted">
                {generatedTests.length}
              </span>
            </button>
            <button
              onClick={() => setActiveTab('risks')}
              className={cn(
                'py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'risks'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              Risk Scores
              <span className="ml-2 px-1.5 py-0.5 rounded-full text-xs bg-muted">
                {riskScores.length}
              </span>
            </button>
          </div>
        </header>

        <div className="p-6 space-y-6">
          {/* Stats Overview */}
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <AlertTriangle className="h-5 w-5 text-warning" />
                <span className="text-sm text-muted-foreground">Production Events</span>
              </div>
              <div className="text-2xl font-bold">{stats?.total_production_events || 0}</div>
              <div className="text-xs text-muted-foreground mt-1">
                {stats?.events_last_24h || 0} in last 24h
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <Sparkles className="h-5 w-5 text-primary" />
                <span className="text-sm text-muted-foreground">Tests Generated</span>
              </div>
              <div className="text-2xl font-bold">{stats?.tests_generated || 0}</div>
              <div className="text-xs text-muted-foreground mt-1">
                {stats?.tests_approved || 0} approved, {stats?.tests_deployed || 0} deployed
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <Shield className="h-5 w-5 text-success" />
                <span className="text-sm text-muted-foreground">Incidents Prevented</span>
              </div>
              <div className="text-2xl font-bold">{stats?.incidents_prevented || 0}</div>
              <div className="text-xs text-muted-foreground mt-1">
                ~{stats?.estimated_time_saved_hours || 0}h saved
              </div>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-3 mb-2">
                <AlertCircle className="h-5 w-5 text-error" />
                <span className="text-sm text-muted-foreground">High Risk Components</span>
              </div>
              <div className="text-2xl font-bold">{stats?.high_risk_components || 0}</div>
              <div className="text-xs text-muted-foreground mt-1">
                {stats?.components_with_tests || 0} with tests
              </div>
            </div>
          </div>

          {/* AI Quality Score Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* AI Quality Score Card */}
              <div className="grid grid-cols-3 gap-6">
                {/* Main Score */}
                <div className="col-span-1 p-6 rounded-lg border bg-card">
                  <div className="flex flex-col items-center justify-center h-full">
                    {scoreLoading ? (
                      <Loader2 className="h-12 w-12 animate-spin text-muted-foreground" />
                    ) : qualityScore ? (
                      <>
                        <div className="text-sm text-muted-foreground mb-2">AI Quality Score</div>
                        <div
                          className={cn(
                            'text-6xl font-bold mb-2',
                            qualityScore.overall_score >= 80 ? 'text-green-500' :
                            qualityScore.overall_score >= 60 ? 'text-yellow-500' :
                            qualityScore.overall_score >= 40 ? 'text-orange-500' : 'text-red-500'
                          )}
                        >
                          {qualityScore.overall_score}
                        </div>
                        <div
                          className={cn(
                            'px-3 py-1 rounded-full text-lg font-bold',
                            qualityScore.grade_color === 'green' ? 'bg-green-100 text-green-700' :
                            qualityScore.grade_color === 'lime' ? 'bg-lime-100 text-lime-700' :
                            qualityScore.grade_color === 'yellow' ? 'bg-yellow-100 text-yellow-700' :
                            qualityScore.grade_color === 'orange' ? 'bg-orange-100 text-orange-700' :
                            'bg-red-100 text-red-700'
                          )}
                        >
                          Grade: {qualityScore.grade}
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => refetchScore()}
                          className="mt-4"
                        >
                          <RefreshCcw className="h-4 w-4 mr-1" />
                          Refresh
                        </Button>
                      </>
                    ) : (
                      <div className="text-center text-muted-foreground">
                        <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>No quality data yet</p>
                        <p className="text-xs">Connect monitoring tools to get started</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Component Scores */}
                <div className="col-span-2 p-6 rounded-lg border bg-card">
                  <h3 className="font-semibold mb-4">Component Scores</h3>
                  {qualityScore && (
                    <div className="space-y-4">
                      {Object.entries(qualityScore.component_scores).map(([key, value]) => (
                        <div key={key}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium">{value.label}</span>
                            <span className="text-sm font-bold">{value.score}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-muted">
                            <div
                              className={cn(
                                'h-full rounded-full transition-all',
                                value.score >= 80 ? 'bg-green-500' :
                                value.score >= 60 ? 'bg-yellow-500' :
                                value.score >= 40 ? 'bg-orange-500' : 'bg-red-500'
                              )}
                              style={{ width: `${value.score}%` }}
                            />
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">{value.description}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Insights & Metrics */}
              <div className="grid grid-cols-2 gap-6">
                {/* Insights */}
                <div className="p-6 rounded-lg border bg-card">
                  <h3 className="font-semibold mb-4 flex items-center gap-2">
                    <Zap className="h-5 w-5 text-primary" />
                    AI Insights
                  </h3>
                  {qualityScore && qualityScore.insights.length > 0 ? (
                    <ul className="space-y-2">
                      {qualityScore.insights.map((insight, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm">
                          <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                          {insight}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-muted-foreground">No insights available yet.</p>
                  )}
                </div>

                {/* Key Metrics */}
                <div className="p-6 rounded-lg border bg-card">
                  <h3 className="font-semibold mb-4 flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    Key Metrics
                  </h3>
                  {qualityScore && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="text-2xl font-bold">{qualityScore.metrics.total_events}</div>
                        <div className="text-xs text-muted-foreground">Total Events</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="text-2xl font-bold text-orange-500">{qualityScore.metrics.unresolved_events}</div>
                        <div className="text-xs text-muted-foreground">Unresolved</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="text-2xl font-bold text-primary">{qualityScore.metrics.tests_generated}</div>
                        <div className="text-xs text-muted-foreground">Tests Generated</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="text-2xl font-bold text-green-500">{qualityScore.metrics.tests_approved}</div>
                        <div className="text-xs text-muted-foreground">Tests Approved</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="text-2xl font-bold">{qualityScore.metrics.avg_confidence}%</div>
                        <div className="text-xs text-muted-foreground">Avg Confidence</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="text-2xl font-bold text-green-500">{qualityScore.metrics.incidents_prevented}</div>
                        <div className="text-xs text-muted-foreground">Incidents Prevented</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* AI Features */}
              <div className="p-6 rounded-lg border bg-card">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Brain className="h-5 w-5 text-primary" />
                  AI-Powered Features
                </h3>
                <div className="grid grid-cols-4 gap-4">
                  <div className="p-4 rounded-lg bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                    <Activity className="h-8 w-8 text-purple-500 mb-2" />
                    <h4 className="font-medium">Semantic Search</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Find similar bugs using AI embeddings (Vectorize)
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/20">
                    <Target className="h-8 w-8 text-green-500 mb-2" />
                    <h4 className="font-medium">Pattern Learning</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Cross-company error pattern intelligence
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-gradient-to-br from-orange-500/10 to-amber-500/10 border border-orange-500/20">
                    <Play className="h-8 w-8 text-orange-500 mb-2" />
                    <h4 className="font-medium">Autonomous Loop</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Auto-discover, test, verify, and PR creation
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/20">
                    <Eye className="h-8 w-8 text-blue-500 mb-2" />
                    <h4 className="font-medium">Visual AI</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      Screenshot comparison and visual regression
                    </p>
                  </div>
                </div>
              </div>

              {/* Predictive Quality Section */}
              <div className="p-6 rounded-lg border bg-card">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    Predictive Quality
                  </h3>
                  {predictions && (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-muted-foreground">Analyzed {predictions.data_quality.events_analyzed} events</span>
                    </div>
                  )}
                </div>

                {predictionsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : predictions && predictions.predictions.length > 0 ? (
                  <div className="space-y-4">
                    {/* AI Summary */}
                    {predictions.ai_summary && (
                      <div className="p-4 rounded-lg bg-primary/5 border border-primary/20">
                        <div className="flex items-start gap-2">
                          <Brain className="h-5 w-5 text-primary mt-0.5" />
                          <p className="text-sm">{predictions.ai_summary}</p>
                        </div>
                      </div>
                    )}

                    {/* Prediction Summary */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="p-3 rounded-lg bg-muted/50 text-center">
                        <div className="text-2xl font-bold">{predictions.summary.total_analyzed}</div>
                        <div className="text-xs text-muted-foreground">Components Analyzed</div>
                      </div>
                      <div className="p-3 rounded-lg bg-red-500/10 text-center">
                        <div className="text-2xl font-bold text-red-500">{predictions.summary.high_risk}</div>
                        <div className="text-xs text-muted-foreground">High Risk</div>
                      </div>
                      <div className="p-3 rounded-lg bg-orange-500/10 text-center">
                        <div className="text-2xl font-bold text-orange-500">{predictions.summary.medium_risk}</div>
                        <div className="text-xs text-muted-foreground">Medium Risk</div>
                      </div>
                      <div className="p-3 rounded-lg bg-yellow-500/10 text-center">
                        <div className="text-2xl font-bold text-yellow-500">{predictions.summary.increasing_trends}</div>
                        <div className="text-xs text-muted-foreground">Increasing Trends</div>
                      </div>
                    </div>

                    {/* Top Predictions */}
                    <div className="divide-y rounded-lg border">
                      {predictions.predictions.slice(0, 5).map((prediction, index) => (
                        <div key={index} className="p-4 hover:bg-muted/50 transition-colors">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-medium truncate max-w-[300px]">{prediction.entity}</span>
                                <Badge variant="default">{prediction.entity_type}</Badge>
                              </div>
                              <div className="flex flex-wrap gap-1 mt-2">
                                {prediction.risk_factors.slice(0, 2).map((factor, i) => (
                                  <span key={i} className="px-2 py-0.5 rounded-full text-xs bg-muted">
                                    {factor}
                                  </span>
                                ))}
                              </div>
                            </div>
                            <div className="flex flex-col items-end gap-1">
                              <div
                                className={cn(
                                  'px-3 py-1 rounded-full text-sm font-bold',
                                  prediction.prediction_score >= 70 ? 'bg-red-100 text-red-700' :
                                  prediction.prediction_score >= 40 ? 'bg-orange-100 text-orange-700' :
                                  'bg-yellow-100 text-yellow-700'
                                )}
                              >
                                {prediction.prediction_score}% Risk
                              </div>
                              <span className="text-xs text-muted-foreground">
                                {Math.round(prediction.confidence * 100)}% confidence
                              </span>
                            </div>
                          </div>
                          {prediction.recommendations.length > 0 && (
                            <div className="mt-2 text-xs text-muted-foreground">
                              <span className="font-medium">Recommended: </span>
                              {prediction.recommendations[0]}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <TrendingUp className="h-12 w-12 text-muted-foreground mb-4 opacity-50" />
                    <p className="text-muted-foreground">No predictions available yet</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Predictions require historical error data from connected monitoring tools
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Production Events Tab */}
          {activeTab === 'events' && (
            <div className="rounded-lg border bg-card">
              <div className="p-4 border-b">
                <h3 className="font-semibold">Production Events</h3>
                <p className="text-sm text-muted-foreground">
                  Errors and issues from Sentry, Datadog, and other monitoring tools
                </p>
              </div>
              <div className="divide-y">
                {isLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : events.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                    <h4 className="font-medium mb-2">No Production Events</h4>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Connect Sentry, Datadog, or other monitoring tools to automatically receive production errors.
                    </p>
                  </div>
                ) : (
                  events.map((event) => (
                    <div
                      key={event.id}
                      className="p-4 hover:bg-muted/50 transition-colors cursor-pointer"
                      onClick={() => setSelectedEventId(event.id)}
                    >
                      <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 mt-1">
                          {event.severity === 'fatal' || event.severity === 'error' ? (
                            <XCircle className="h-5 w-5 text-error" />
                          ) : event.severity === 'warning' ? (
                            <AlertTriangle className="h-5 w-5 text-warning" />
                          ) : (
                            <AlertCircle className="h-5 w-5 text-info" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium truncate">{event.title}</span>
                            <SeverityBadge severity={event.severity} />
                            <Badge variant={statusColors[event.status as keyof typeof statusColors] || 'default'}>
                              {event.status.replace('_', ' ')}
                            </Badge>
                          </div>
                          <div className="text-sm text-muted-foreground mb-2 line-clamp-2">
                            {event.message || event.title}
                          </div>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDistanceToNow(new Date(event.created_at), { addSuffix: true })}
                            </span>
                            {event.url && (
                              <span className="truncate max-w-[200px]">{event.url}</span>
                            )}
                            {event.component && (
                              <span className="flex items-center gap-1">
                                <Code className="h-3 w-3" />
                                {event.component}
                              </span>
                            )}
                            <span>{event.occurrence_count} occurrences</span>
                            <span>{event.affected_users} users</span>
                          </div>
                        </div>
                        <div className="flex-shrink-0 flex items-center gap-2">
                          {event.external_url && (
                            <Button size="sm" variant="ghost" asChild>
                              <a href={event.external_url} target="_blank" rel="noopener noreferrer">
                                <ExternalLink className="h-4 w-4" />
                              </a>
                            </Button>
                          )}
                          {event.status === 'new' && (
                            <Button
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleGenerateTest(event.id);
                              }}
                              disabled={generateTest.isPending}
                            >
                              {generateTest.isPending ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <>
                                  <Sparkles className="h-4 w-4 mr-1" />
                                  Generate Test
                                </>
                              )}
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Generated Tests Tab */}
          {activeTab === 'tests' && (
            <div className="rounded-lg border bg-card">
              <div className="p-4 border-b">
                <h3 className="font-semibold">Generated Tests</h3>
                <p className="text-sm text-muted-foreground">
                  AI-generated tests from production errors
                </p>
              </div>
              <div className="divide-y">
                {testsLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : generatedTests.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <FileCode className="h-12 w-12 text-muted-foreground mb-4" />
                    <h4 className="font-medium mb-2">No Generated Tests</h4>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Click &quot;Generate Test&quot; on a production event to create an automated test.
                    </p>
                  </div>
                ) : (
                  generatedTests.map((test) => (
                    <div key={test.id} className="p-4 hover:bg-muted/50">
                      <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 mt-1">
                          <FileCode className="h-5 w-5 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium">{test.name}</span>
                            <Badge variant={testStatusColors[test.status as keyof typeof testStatusColors] || 'default'}>
                              {test.status}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {Math.round(test.confidence_score * 100)}% confidence
                            </span>
                          </div>
                          <div className="text-sm text-muted-foreground mb-2">
                            {test.description}
                          </div>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span>{test.framework}</span>
                            <span>{test.test_file_path}</span>
                            {test.github_pr_url && (
                              <a
                                href={test.github_pr_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-1 text-primary hover:underline"
                              >
                                <GitPullRequest className="h-3 w-3" />
                                PR #{test.github_pr_number}
                              </a>
                            )}
                          </div>
                        </div>
                        <div className="flex-shrink-0 flex items-center gap-2">
                          {test.status === 'pending' && (
                            <>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleRejectTest(test.id)}
                                disabled={updateTestStatus.isPending}
                              >
                                <XCircle className="h-4 w-4 mr-1" />
                                Reject
                              </Button>
                              <Button
                                size="sm"
                                onClick={() => handleApproveTest(test.id)}
                                disabled={updateTestStatus.isPending}
                              >
                                <CheckCircle className="h-4 w-4 mr-1" />
                                Approve
                              </Button>
                            </>
                          )}
                        </div>
                      </div>
                      {/* Show test code preview */}
                      <details className="mt-3">
                        <summary className="text-sm text-muted-foreground cursor-pointer hover:text-foreground">
                          View test code
                        </summary>
                        <pre className="mt-2 p-3 rounded-lg bg-muted text-xs overflow-x-auto">
                          <code>{test.test_code}</code>
                        </pre>
                      </details>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Risk Scores Tab */}
          {activeTab === 'risks' && (
            <div className="rounded-lg border bg-card">
              <div className="p-4 border-b flex items-center justify-between">
                <div>
                  <h3 className="font-semibold">Risk Scores</h3>
                  <p className="text-sm text-muted-foreground">
                    Components and pages ranked by risk level
                  </p>
                </div>
                <Button
                  size="sm"
                  onClick={handleCalculateRiskScores}
                  disabled={calculateRiskScores.isPending}
                >
                  {calculateRiskScores.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Calculating...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="h-4 w-4 mr-2" />
                      Recalculate
                    </>
                  )}
                </Button>
              </div>
              <div className="divide-y">
                {risksLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : riskScores.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <Shield className="h-12 w-12 text-muted-foreground mb-4" />
                    <h4 className="font-medium mb-2">No Risk Data</h4>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Risk scores will be calculated as production events are received.
                    </p>
                  </div>
                ) : (
                  riskScores.map((risk) => (
                    <div key={risk.id} className="p-4 hover:bg-muted/50">
                      <div className="flex items-center gap-4">
                        <div className="flex-shrink-0">
                          <div
                            className={cn(
                              'h-12 w-12 rounded-lg flex items-center justify-center text-white font-bold',
                              risk.overall_risk_score >= 80
                                ? 'bg-red-500'
                                : risk.overall_risk_score >= 60
                                  ? 'bg-orange-500'
                                  : risk.overall_risk_score >= 40
                                    ? 'bg-yellow-500'
                                    : 'bg-green-500'
                            )}
                          >
                            {risk.overall_risk_score}
                          </div>
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium">{risk.entity_name || risk.entity_identifier}</span>
                            <Badge variant="default">{risk.entity_type}</Badge>
                            <TrendIcon trend={risk.score_trend} />
                          </div>
                          <div className="grid grid-cols-4 gap-4 mt-2">
                            <div>
                              <div className="text-xs text-muted-foreground">Error Rate</div>
                              <div className="h-1.5 rounded-full bg-muted mt-1">
                                <div
                                  className="h-full rounded-full bg-error"
                                  style={{ width: `${risk.error_frequency_score || 0}%` }}
                                />
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-muted-foreground">Severity</div>
                              <div className="h-1.5 rounded-full bg-muted mt-1">
                                <div
                                  className="h-full rounded-full bg-warning"
                                  style={{ width: `${risk.error_severity_score || 0}%` }}
                                />
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-muted-foreground">Test Coverage</div>
                              <div className="h-1.5 rounded-full bg-muted mt-1">
                                <div
                                  className="h-full rounded-full bg-success"
                                  style={{ width: `${100 - (risk.test_coverage_score || 0)}%` }}
                                />
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-muted-foreground">Change Rate</div>
                              <div className="h-1.5 rounded-full bg-muted mt-1">
                                <div
                                  className="h-full rounded-full bg-info"
                                  style={{ width: `${risk.change_frequency_score || 0}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="flex-shrink-0">
                          <Button size="sm" variant="outline">
                            View Details
                            <ArrowRight className="h-4 w-4 ml-1" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
