'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { getSupabaseClient } from '@/lib/supabase/client';
import type { Test, TestRun, TestResult, InsertTables } from '@/lib/supabase/types';

const WORKER_URL = process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

// ============================================
// TESTS
// ============================================

export function useTests(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['tests', projectId],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('tests') as any)
        .select('*')
        .eq('project_id', projectId)
        .eq('is_active', true)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data as Test[];
    },
    enabled: !!projectId,
    staleTime: 2 * 60 * 1000, // 2 minutes - tests change occasionally
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    placeholderData: [], // Prevent loading state flash
  });
}

export function useCreateTest() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (test: InsertTables<'tests'>) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('tests') as any)
        .insert(test)
        .select()
        .single();

      if (error) throw error;
      return data as Test;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['tests', data.project_id] });
    },
  });
}

export function useUpdateTest() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ id, ...updates }: { id: string } & Partial<Test>) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('tests') as any)
        .update(updates)
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;
      return data as Test;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['tests', data.project_id] });
    },
  });
}

export function useDeleteTest() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ testId, projectId }: { testId: string; projectId: string }) => {
      // Soft delete by setting is_active to false
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { error } = await (supabase.from('tests') as any)
        .update({ is_active: false })
        .eq('id', testId);

      if (error) throw error;
      return projectId;
    },
    onSuccess: (projectId) => {
      queryClient.invalidateQueries({ queryKey: ['tests', projectId] });
    },
  });
}

// ============================================
// TEST RUNS
// ============================================

export function useTestRuns(projectId: string | null, limit = 50) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['test-runs', projectId, limit],
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
    staleTime: 30 * 1000, // 30 seconds - runs update more frequently
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
    placeholderData: [], // Prevent loading state flash
  });
}

export function useTestRun(runId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['test-run', runId],
    queryFn: async () => {
      if (!runId) return null;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('test_runs') as any)
        .select('*')
        .eq('id', runId)
        .single();

      if (error) throw error;
      return data as TestRun;
    },
    enabled: !!runId,
  });
}

export function useTestResults(runId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['test-results', runId],
    queryFn: async () => {
      if (!runId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('test_results') as any)
        .select('*')
        .eq('test_run_id', runId)
        .order('created_at', { ascending: true });

      if (error) throw error;
      return data as TestResult[];
    },
    enabled: !!runId,
  });
}

// Real-time subscription for test runs
export function useTestRunSubscription(projectId: string | null) {
  const queryClient = useQueryClient();
  const supabase = getSupabaseClient();

  useEffect(() => {
    if (!projectId) return;

    const channel = supabase
      .channel(`test-runs-${projectId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'test_runs',
          filter: `project_id=eq.${projectId}`,
        },
        () => {
          queryClient.invalidateQueries({ queryKey: ['test-runs', projectId] });
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [projectId, queryClient, supabase]);
}

// ============================================
// RUN TEST
// ============================================

export function useRunTest() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      appUrl,
      tests,
      browser = 'chromium',
    }: {
      projectId: string;
      appUrl: string;
      tests: Test[];
      browser?: string;
    }) => {
      // 1. Create test run in Supabase
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: testRun, error: runError } = await (supabase.from('test_runs') as any)
        .insert({
          project_id: projectId,
          app_url: appUrl,
          browser,
          status: 'running',
          total_tests: tests.length,
          started_at: new Date().toISOString(),
        })
        .select()
        .single();

      if (runError) throw runError;

      // 2. Execute tests via Worker
      const results: TestResult[] = [];
      let passedCount = 0;
      let failedCount = 0;

      for (const test of tests) {
        const steps = Array.isArray(test.steps)
          ? (test.steps as { instruction: string }[]).map((s) => s.instruction)
          : [];

        try {
          const response = await fetch(`${WORKER_URL}/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              url: appUrl,
              steps,
              browser,
              screenshot: true,
            }),
          });

          const workerResult = await response.json();
          const passed = workerResult.success;

          // 3. Save result to Supabase
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const { data: result, error: resultError } = await (supabase.from('test_results') as any)
            .insert({
              test_run_id: testRun.id,
              test_id: test.id,
              name: test.name,
              status: passed ? 'passed' : 'failed',
              duration_ms: workerResult.duration || 0,
              steps_total: steps.length,
              steps_completed: passed ? steps.length : 0,
              error_message: workerResult.error || null,
              step_results: workerResult.steps || [],
              completed_at: new Date().toISOString(),
            })
            .select()
            .single();

          if (resultError) throw resultError;
          results.push(result as TestResult);

          if (passed) passedCount++;
          else failedCount++;
        } catch (error) {
          // Record failed test
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const { data: result } = await (supabase.from('test_results') as any)
            .insert({
              test_run_id: testRun.id,
              test_id: test.id,
              name: test.name,
              status: 'failed',
              steps_total: steps.length,
              error_message: String(error),
              completed_at: new Date().toISOString(),
            })
            .select()
            .single();

          if (result) results.push(result as TestResult);
          failedCount++;
        }
      }

      // 4. Update test run with final status
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: finalRun, error: updateError } = await (supabase.from('test_runs') as any)
        .update({
          status: failedCount === 0 ? 'passed' : 'failed',
          passed_tests: passedCount,
          failed_tests: failedCount,
          completed_at: new Date().toISOString(),
          duration_ms: Date.now() - new Date(testRun.started_at!).getTime(),
        })
        .eq('id', testRun.id)
        .select()
        .single();

      if (updateError) throw updateError;

      return { testRun: finalRun as TestRun, results };
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['test-runs'] });
      queryClient.invalidateQueries({ queryKey: ['test-run', data.testRun.id] });
    },
  });
}

export function useRunSingleTest() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      appUrl,
      test,
      browser = 'chromium',
    }: {
      projectId: string;
      appUrl: string;
      test: Test;
      browser?: string;
    }) => {
      const steps = Array.isArray(test.steps)
        ? (test.steps as { instruction: string }[]).map((s) => s.instruction)
        : [];

      // Create test run
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: testRun, error: runError } = await (supabase.from('test_runs') as any)
        .insert({
          project_id: projectId,
          app_url: appUrl,
          browser,
          name: test.name,
          status: 'running',
          total_tests: 1,
          started_at: new Date().toISOString(),
        })
        .select()
        .single();

      if (runError) throw runError;

      // Execute via Worker
      const response = await fetch(`${WORKER_URL}/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: appUrl,
          steps,
          browser,
          screenshot: true,
        }),
      });

      const workerResult = await response.json();
      const passed = workerResult.success;

      // Save result
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await (supabase.from('test_results') as any).insert({
        test_run_id: testRun.id,
        test_id: test.id,
        name: test.name,
        status: passed ? 'passed' : 'failed',
        duration_ms: workerResult.duration || 0,
        steps_total: steps.length,
        steps_completed: passed ? steps.length : 0,
        error_message: workerResult.error || null,
        completed_at: new Date().toISOString(),
      });

      // Update run status
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: finalRun } = await (supabase.from('test_runs') as any)
        .update({
          status: passed ? 'passed' : 'failed',
          passed_tests: passed ? 1 : 0,
          failed_tests: passed ? 0 : 1,
          completed_at: new Date().toISOString(),
          duration_ms: Date.now() - new Date(testRun.started_at!).getTime(),
        })
        .eq('id', testRun.id)
        .select()
        .single();

      return finalRun as TestRun;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['test-runs'] });
    },
  });
}
