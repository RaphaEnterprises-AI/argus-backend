'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getSupabaseClient } from '@/lib/supabase/client';
import type { AIInsight } from '@/lib/supabase/types';

export function useAIInsights(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['ai-insights', projectId],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('ai_insights') as any)
        .select('*')
        .eq('project_id', projectId)
        .eq('is_resolved', false)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data as AIInsight[];
    },
    enabled: !!projectId,
  });
}

export function useResolveInsight() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      insightId,
      projectId,
      resolvedBy,
    }: {
      insightId: string;
      projectId: string;
      resolvedBy?: string | null;
    }) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('ai_insights') as any)
        .update({
          is_resolved: true,
          resolved_at: new Date().toISOString(),
          resolved_by: resolvedBy || null,
        })
        .eq('id', insightId)
        .select()
        .single();

      if (error) throw error;
      return { insight: data as AIInsight, projectId };
    },
    onSuccess: ({ projectId }) => {
      queryClient.invalidateQueries({ queryKey: ['ai-insights', projectId] });
    },
  });
}

export function useInsightStats(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['insight-stats', projectId],
    queryFn: async () => {
      if (!projectId) return null;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('ai_insights') as any)
        .select('*')
        .eq('project_id', projectId);

      if (error) throw error;

      const insights = data as AIInsight[];
      const resolved = insights.filter((i) => i.is_resolved);
      const unresolved = insights.filter((i) => !i.is_resolved);

      return {
        total: insights.length,
        resolved: resolved.length,
        unresolved: unresolved.length,
        bySeverity: {
          critical: unresolved.filter((i) => i.severity === 'critical').length,
          high: unresolved.filter((i) => i.severity === 'high').length,
          medium: unresolved.filter((i) => i.severity === 'medium').length,
          low: unresolved.filter((i) => i.severity === 'low').length,
        },
      };
    },
    enabled: !!projectId,
  });
}
