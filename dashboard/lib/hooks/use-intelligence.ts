'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { createClient } from '@/lib/supabase/client';
import type {
  ProductionEvent,
  GeneratedTest,
  RiskScore,
  QualityIntelligenceStats,
} from '@/lib/supabase/types';

// Helper to get supabase client with any type for new tables
// This is needed because the types file has the new table definitions
// but they haven't been picked up by the Supabase client yet
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getSupabase(): any {
  return createClient();
}

// ============================================================================
// PRODUCTION EVENTS
// ============================================================================

export function useProductionEvents(projectId: string | null) {
  return useQuery({
    queryKey: ['production-events', projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('production_events')
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false })
        .limit(100);

      if (error) throw error;
      return (data || []) as ProductionEvent[];
    },
    enabled: !!projectId,
  });
}

export function useProductionEvent(eventId: string | null) {
  return useQuery({
    queryKey: ['production-event', eventId],
    queryFn: async () => {
      if (!eventId) return null;
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('production_events')
        .select('*')
        .eq('id', eventId)
        .single();

      if (error) throw error;
      return data as ProductionEvent;
    },
    enabled: !!eventId,
  });
}

export function useProductionEventStats(projectId: string | null) {
  return useQuery({
    queryKey: ['production-event-stats', projectId],
    queryFn: async () => {
      if (!projectId) return null;

      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('production_events')
        .select('status, severity')
        .eq('project_id', projectId);

      if (error) throw error;

      const events = (data || []) as Array<{ status: string; severity: string }>;
      const stats = {
        total: events.length,
        byStatus: {} as Record<string, number>,
        bySeverity: {} as Record<string, number>,
      };

      events.forEach((e) => {
        stats.byStatus[e.status] = (stats.byStatus[e.status] || 0) + 1;
        stats.bySeverity[e.severity] = (stats.bySeverity[e.severity] || 0) + 1;
      });

      return stats;
    },
    enabled: !!projectId,
  });
}

export function useUpdateProductionEventStatus() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      eventId,
      status,
    }: {
      eventId: string;
      status: string;
    }) => {
      const supabase = getSupabase();
      const { error } = await supabase
        .from('production_events')
        .update({ status, updated_at: new Date().toISOString() })
        .eq('id', eventId);

      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['production-events'] });
      queryClient.invalidateQueries({ queryKey: ['production-event-stats'] });
    },
  });
}

// ============================================================================
// GENERATED TESTS
// ============================================================================

export function useGeneratedTests(projectId: string | null) {
  return useQuery({
    queryKey: ['generated-tests', projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('generated_tests')
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false })
        .limit(100);

      if (error) throw error;
      return (data || []) as GeneratedTest[];
    },
    enabled: !!projectId,
  });
}

export function useGeneratedTest(testId: string | null) {
  return useQuery({
    queryKey: ['generated-test', testId],
    queryFn: async () => {
      if (!testId) return null;
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('generated_tests')
        .select('*')
        .eq('id', testId)
        .single();

      if (error) throw error;
      return data as GeneratedTest;
    },
    enabled: !!testId,
  });
}

export function useUpdateGeneratedTestStatus() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      testId,
      status,
      reviewNotes,
    }: {
      testId: string;
      status: 'approved' | 'rejected' | 'modified';
      reviewNotes?: string;
    }) => {
      const supabase = getSupabase();
      const { error } = await supabase
        .from('generated_tests')
        .update({
          status,
          review_notes: reviewNotes,
          reviewed_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', testId);

      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['generated-tests'] });
      queryClient.invalidateQueries({ queryKey: ['generated-test'] });
    },
  });
}

export function useGenerateTest() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      eventId,
      framework = 'playwright',
      autoCreatePr = false,
      githubConfig,
    }: {
      projectId: string;
      eventId: string;
      framework?: 'playwright' | 'cypress' | 'jest';
      autoCreatePr?: boolean;
      githubConfig?: {
        owner: string;
        repo: string;
        base_branch?: string;
        test_directory?: string;
      };
    }) => {
      // Get Argus API URL from environment
      const apiUrl = process.env.NEXT_PUBLIC_ARGUS_API_URL || 'https://argus-api.your-domain.workers.dev';
      const apiToken = process.env.NEXT_PUBLIC_ARGUS_API_TOKEN;

      const response = await fetch(`${apiUrl}/api/generate-test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiToken ? { 'Authorization': `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({
          production_event_id: eventId,
          project_id: projectId,
          framework,
          auto_create_pr: autoCreatePr,
          github_config: githubConfig,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to generate test');
      }

      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['generated-tests'] });
      queryClient.invalidateQueries({ queryKey: ['production-events'] });
    },
  });
}

// ============================================================================
// RISK SCORES
// ============================================================================

export function useRiskScores(projectId: string | null) {
  return useQuery({
    queryKey: ['risk-scores', projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('risk_scores')
        .select('*')
        .eq('project_id', projectId)
        .order('overall_risk_score', { ascending: false })
        .limit(50);

      if (error) throw error;
      return (data || []) as RiskScore[];
    },
    enabled: !!projectId,
  });
}

export function useHighRiskComponents(projectId: string | null) {
  return useQuery({
    queryKey: ['high-risk-components', projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('risk_scores')
        .select('*')
        .eq('project_id', projectId)
        .gte('overall_risk_score', 70) // High risk = 70+
        .order('overall_risk_score', { ascending: false });

      if (error) throw error;
      return (data || []) as RiskScore[];
    },
    enabled: !!projectId,
  });
}

// ============================================================================
// QUALITY INTELLIGENCE STATS
// ============================================================================

export function useQualityIntelligenceStats(projectId: string | null) {
  return useQuery({
    queryKey: ['quality-intelligence-stats', projectId],
    queryFn: async () => {
      if (!projectId) return null;
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('quality_intelligence_stats')
        .select('*')
        .eq('project_id', projectId)
        .single();

      if (error && error.code !== 'PGRST116') throw error; // PGRST116 = no rows
      return (data || null) as QualityIntelligenceStats | null;
    },
    enabled: !!projectId,
  });
}

// ============================================================================
// RISK SCORE CALCULATION
// ============================================================================

export function useCalculateRiskScores() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      entityTypes = ['page', 'component'],
    }: {
      projectId: string;
      entityTypes?: string[];
    }) => {
      // Get Argus API URL from environment
      const apiUrl = process.env.NEXT_PUBLIC_ARGUS_API_URL || 'https://argus-api.your-domain.workers.dev';
      const apiToken = process.env.NEXT_PUBLIC_ARGUS_API_TOKEN;

      const response = await fetch(`${apiUrl}/api/calculate-risk-scores`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiToken ? { 'Authorization': `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({
          project_id: projectId,
          entity_types: entityTypes,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to calculate risk scores');
      }

      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['risk-scores'] });
      queryClient.invalidateQueries({ queryKey: ['high-risk-components'] });
      queryClient.invalidateQueries({ queryKey: ['quality-intelligence-stats'] });
    },
  });
}

// ============================================================================
// INTEGRATIONS
// ============================================================================

export function useQualityIntegrations(projectId: string | null) {
  return useQuery({
    queryKey: ['quality-integrations', projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const supabase = getSupabase();
      const { data, error } = await supabase
        .from('integrations')
        .select('*')
        .eq('project_id', projectId)
        .in('type', ['sentry', 'datadog', 'newrelic', 'bugsnag', 'rollbar']);

      if (error) throw error;
      return data || [];
    },
    enabled: !!projectId,
  });
}

// ============================================================================
// AI QUALITY SCORE
// ============================================================================

export interface AIQualityScore {
  success: boolean;
  project_id: string;
  overall_score: number;
  grade: string;
  grade_color: string;
  component_scores: {
    error_management: { score: number; label: string; description: string };
    test_coverage: { score: number; label: string; description: string };
    risk_mitigation: { score: number; label: string; description: string };
    automation: { score: number; label: string; description: string };
    prevention: { score: number; label: string; description: string };
  };
  metrics: {
    total_events: number;
    unresolved_events: number;
    tests_generated: number;
    tests_approved: number;
    avg_confidence: number;
    high_risk_components: number;
    incidents_prevented: number;
  };
  insights: string[];
  calculated_at: string;
}

export function useAIQualityScore(projectId: string | null) {
  return useQuery({
    queryKey: ['ai-quality-score', projectId],
    queryFn: async () => {
      if (!projectId) return null;

      const apiUrl = process.env.NEXT_PUBLIC_ARGUS_API_URL || 'https://argus-api.your-domain.workers.dev';
      const apiToken = process.env.NEXT_PUBLIC_ARGUS_API_TOKEN;

      const response = await fetch(`${apiUrl}/api/quality-score?project_id=${projectId}`, {
        headers: {
          ...(apiToken ? { 'Authorization': `Bearer ${apiToken}` } : {}),
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to fetch quality score');
      }

      return response.json() as Promise<AIQualityScore>;
    },
    enabled: !!projectId,
    refetchInterval: 60000, // Refetch every minute
  });
}

// ============================================================================
// SEMANTIC PATTERN SEARCH
// ============================================================================

export interface SimilarPattern {
  id: string;
  score: number;
  pattern_hash: string;
  category: string;
  example_message: string;
  known_solutions?: string[];
}

export function useSemanticSearch() {
  return useMutation({
    mutationFn: async ({
      errorText,
      limit = 5,
      minScore = 0.7,
    }: {
      errorText: string;
      limit?: number;
      minScore?: number;
    }) => {
      const apiUrl = process.env.NEXT_PUBLIC_ARGUS_API_URL || 'https://argus-api.your-domain.workers.dev';
      const apiToken = process.env.NEXT_PUBLIC_ARGUS_API_TOKEN;

      const response = await fetch(`${apiUrl}/api/semantic-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiToken ? { 'Authorization': `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({
          error_text: errorText,
          limit,
          min_score: minScore,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to search patterns');
      }

      return response.json() as Promise<{
        success: boolean;
        query: string;
        patterns: SimilarPattern[];
        count: number;
        has_solutions: boolean;
      }>;
    },
  });
}

// ============================================================================
// AUTONOMOUS QUALITY LOOP
// ============================================================================

export function useAutonomousLoop() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      url,
      stages = ['discovery', 'visual', 'generation', 'verification', 'pr', 'learning'],
      discoveryDepth = 2,
      autoCreatePr = false,
      githubConfig,
    }: {
      projectId: string;
      url: string;
      stages?: string[];
      discoveryDepth?: number;
      autoCreatePr?: boolean;
      githubConfig?: {
        owner: string;
        repo: string;
        base_branch?: string;
        test_directory?: string;
      };
    }) => {
      const apiUrl = process.env.NEXT_PUBLIC_ARGUS_API_URL || 'https://argus-api.your-domain.workers.dev';
      const apiToken = process.env.NEXT_PUBLIC_ARGUS_API_TOKEN;

      const response = await fetch(`${apiUrl}/api/autonomous-loop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiToken ? { 'Authorization': `Bearer ${apiToken}` } : {}),
        },
        body: JSON.stringify({
          project_id: projectId,
          url,
          stages,
          discovery_depth: discoveryDepth,
          auto_create_pr: autoCreatePr,
          github_config: githubConfig,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to run autonomous loop');
      }

      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['production-events'] });
      queryClient.invalidateQueries({ queryKey: ['generated-tests'] });
      queryClient.invalidateQueries({ queryKey: ['risk-scores'] });
      queryClient.invalidateQueries({ queryKey: ['ai-quality-score'] });
    },
  });
}

// ============================================================================
// PREDICTIVE QUALITY
// ============================================================================

export interface PredictedRisk {
  entity: string;
  entity_type: string;
  prediction_score: number;
  predicted_timeframe: string;
  risk_factors: string[];
  recommendations: string[];
  similar_past_failures: number;
  confidence: number;
}

export interface PredictiveQualityResponse {
  success: boolean;
  project_id: string;
  timeframe: string;
  predictions: PredictedRisk[];
  summary: {
    total_analyzed: number;
    total_predicted: number;
    high_risk: number;
    medium_risk: number;
    increasing_trends: number;
  };
  ai_summary: string | null;
  data_quality: {
    events_analyzed: number;
    risk_scores_available: number;
    patterns_learned: number;
  };
  calculated_at: string;
}

export function usePredictiveQuality(projectId: string | null, timeframe: string = '7d') {
  return useQuery({
    queryKey: ['predictive-quality', projectId, timeframe],
    queryFn: async () => {
      if (!projectId) return null;

      const apiUrl = process.env.NEXT_PUBLIC_ARGUS_API_URL || 'https://argus-api.your-domain.workers.dev';
      const apiToken = process.env.NEXT_PUBLIC_ARGUS_API_TOKEN;

      const response = await fetch(
        `${apiUrl}/api/predictive-quality?project_id=${projectId}&timeframe=${timeframe}`,
        {
          headers: {
            ...(apiToken ? { 'Authorization': `Bearer ${apiToken}` } : {}),
          },
        }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to fetch predictions');
      }

      return response.json() as Promise<PredictiveQualityResponse>;
    },
    enabled: !!projectId,
    refetchInterval: 300000, // Refetch every 5 minutes
  });
}
