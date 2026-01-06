'use client';

import { useQuery } from '@tanstack/react-query';
import { getSupabaseClient } from '@/lib/supabase/client';
import type { TestRun, DailyStats } from '@/lib/supabase/types';

export function useReportsStats(projectId: string | null, days = 7) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['reports-stats', projectId, days],
    queryFn: async () => {
      if (!projectId) return null;

      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: runs, error } = await (supabase.from('test_runs') as any)
        .select('*')
        .eq('project_id', projectId)
        .gte('created_at', startDate.toISOString())
        .order('created_at', { ascending: false });

      if (error) throw error;

      const testRuns = runs as TestRun[];

      // Calculate stats
      const totalRuns = testRuns.length;
      const totalPassed = testRuns.reduce((sum, r) => sum + (r.passed_tests || 0), 0);
      const totalFailed = testRuns.reduce((sum, r) => sum + (r.failed_tests || 0), 0);
      const totalTests = totalPassed + totalFailed;
      const avgPassRate = totalTests > 0 ? (totalPassed / totalTests) * 100 : 0;
      const avgDuration = testRuns.length > 0
        ? testRuns.reduce((sum, r) => sum + (r.duration_ms || 0), 0) / testRuns.length / 1000
        : 0;

      // Calculate daily stats
      const dailyStats: { day: string; date: string; passed: number; failed: number }[] = [];
      for (let i = days - 1; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const dayStart = new Date(date.setHours(0, 0, 0, 0));
        const dayEnd = new Date(date.setHours(23, 59, 59, 999));

        const dayRuns = testRuns.filter((r) => {
          const runDate = new Date(r.created_at);
          return runDate >= dayStart && runDate <= dayEnd;
        });

        dailyStats.push({
          day: dayStart.toLocaleDateString('en-US', { weekday: 'short' }),
          date: dayStart.toISOString().split('T')[0],
          passed: dayRuns.reduce((sum, r) => sum + (r.passed_tests || 0), 0),
          failed: dayRuns.reduce((sum, r) => sum + (r.failed_tests || 0), 0),
        });
      }

      // Get failed runs
      const failedRuns = testRuns
        .filter((r) => r.status === 'failed')
        .slice(0, 5);

      return {
        totalRuns,
        avgPassRate,
        avgDuration,
        dailyStats,
        recentRuns: testRuns.slice(0, 10),
        failedRuns,
      };
    },
    enabled: !!projectId,
  });
}

export function useRecentRuns(projectId: string | null, limit = 20) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['recent-runs', projectId, limit],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('test_runs') as any)
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data as TestRun[];
    },
    enabled: !!projectId,
  });
}
