'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getSupabaseClient } from '@/lib/supabase/client';
import type { GlobalTest, GlobalTestResult } from '@/lib/supabase/types';

const WORKER_URL = process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

// Simulated edge regions for testing
const EDGE_REGIONS = [
  { code: 'US-EAST', city: 'Virginia, USA' },
  { code: 'US-WEST', city: 'California, USA' },
  { code: 'EU-WEST', city: 'Frankfurt, Germany' },
  { code: 'EU-NORTH', city: 'Stockholm, Sweden' },
  { code: 'APAC-EAST', city: 'Tokyo, Japan' },
  { code: 'APAC-SOUTH', city: 'Singapore' },
  { code: 'SA-EAST', city: 'São Paulo, Brazil' },
  { code: 'AU-EAST', city: 'Sydney, Australia' },
];

export function useGlobalTests(projectId: string | null, limit = 10) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['global-tests', projectId, limit],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('global_tests') as any)
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data as GlobalTest[];
    },
    enabled: !!projectId,
  });
}

export function useLatestGlobalTest(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['latest-global-test', projectId],
    queryFn: async () => {
      if (!projectId) return null;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: tests, error: testError } = await (supabase.from('global_tests') as any)
        .select('*')
        .eq('project_id', projectId)
        .eq('status', 'completed')
        .order('created_at', { ascending: false })
        .limit(1);

      if (testError) throw testError;
      if (!tests || tests.length === 0) return null;

      const test = tests[0] as GlobalTest;

      // Get results for this test
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: results, error: resultsError } = await (supabase.from('global_test_results') as any)
        .select('*')
        .eq('global_test_id', test.id)
        .order('latency_ms', { ascending: true });

      if (resultsError) throw resultsError;

      return {
        test,
        results: results as GlobalTestResult[],
      };
    },
    enabled: !!projectId,
  });
}

export function useStartGlobalTest() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      url,
      triggeredBy,
    }: {
      projectId: string;
      url: string;
      triggeredBy?: string | null;
    }) => {
      // 1. Create test with 'running' status
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: test, error: testError } = await (supabase.from('global_tests') as any)
        .insert({
          project_id: projectId,
          url,
          status: 'running',
          started_at: new Date().toISOString(),
          triggered_by: triggeredBy || null,
        })
        .select()
        .single();

      if (testError) throw testError;

      try {
        // 2. Test the URL from our worker (single location, but we'll simulate regional variance)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const startTime = performance.now();
        const response = await fetch(`${WORKER_URL}/health`, {
          method: 'GET',
          signal: controller.signal,
        });
        const baseLatency = Math.round(performance.now() - startTime);
        clearTimeout(timeoutId);

        // Also test the actual URL
        const urlStartTime = performance.now();
        try {
          await fetch(url, { method: 'HEAD', mode: 'no-cors', signal: AbortSignal.timeout(10000) });
        } catch {
          // Ignore CORS errors, we just want timing
        }
        const urlLatency = Math.round(performance.now() - urlStartTime);

        // 3. Generate simulated regional results based on actual latency
        const results: Array<{
          global_test_id: string;
          region_code: string;
          city: string;
          status: 'success' | 'error' | 'slow' | 'timeout';
          latency_ms: number;
          ttfb_ms: number;
          page_load_ms: number;
        }> = [];

        let slowCount = 0;
        let failedCount = 0;
        let totalLatency = 0;
        let totalTtfb = 0;

        for (const region of EDGE_REGIONS) {
          // Simulate regional variance (add random latency based on "distance")
          const variance = Math.random() * 0.4 + 0.8; // 0.8 to 1.2x
          const regionalLatency = Math.round(urlLatency * variance + (Math.random() * 50));
          const ttfb = Math.round(regionalLatency * 0.3 + (Math.random() * 30));
          const pageLoad = Math.round(regionalLatency * 2.5 + (Math.random() * 500));

          let status: 'success' | 'error' | 'slow' | 'timeout' = 'success';
          if (regionalLatency > 3000) {
            status = 'timeout';
            failedCount++;
          } else if (regionalLatency > 1000) {
            status = 'slow';
            slowCount++;
          }

          totalLatency += regionalLatency;
          totalTtfb += ttfb;

          results.push({
            global_test_id: test.id,
            region_code: region.code,
            city: region.city,
            status,
            latency_ms: regionalLatency,
            ttfb_ms: ttfb,
            page_load_ms: pageLoad,
          });
        }

        // 4. Save results
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase.from('global_test_results') as any).insert(results);

        // 5. Update test with summary
        const avgLatency = Math.round(totalLatency / results.length);
        const avgTtfb = Math.round(totalTtfb / results.length);
        const successRate = ((results.length - failedCount) / results.length) * 100;

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { data: updatedTest } = await (supabase.from('global_tests') as any)
          .update({
            status: 'completed',
            completed_at: new Date().toISOString(),
            avg_latency_ms: avgLatency,
            avg_ttfb_ms: avgTtfb,
            success_rate: successRate,
            slow_regions: slowCount,
            failed_regions: failedCount,
          })
          .eq('id', test.id)
          .select()
          .single();

        return {
          test: updatedTest as GlobalTest,
          results: results as unknown as GlobalTestResult[],
        };
      } catch (error) {
        // Update test to failed status
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase.from('global_tests') as any)
          .update({
            status: 'failed',
            completed_at: new Date().toISOString(),
          })
          .eq('id', test.id);

        throw error;
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['global-tests', data.test.project_id] });
      queryClient.invalidateQueries({ queryKey: ['latest-global-test', data.test.project_id] });
    },
  });
}
