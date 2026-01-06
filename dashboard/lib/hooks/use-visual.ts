'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getSupabaseClient } from '@/lib/supabase/client';
import type { VisualBaseline, VisualComparison } from '@/lib/supabase/types';

const WORKER_URL = process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

// Simple pixel diffing using canvas
async function compareScreenshots(
  baselineBase64: string,
  currentBase64: string
): Promise<{ matchPercentage: number; diffBase64: string }> {
  return new Promise((resolve) => {
    const baseImg = new Image();
    const currImg = new Image();
    let loaded = 0;

    const onBothLoaded = () => {
      // Create canvases
      const baseCanvas = document.createElement('canvas');
      const currCanvas = document.createElement('canvas');
      const diffCanvas = document.createElement('canvas');

      const width = Math.max(baseImg.width, currImg.width);
      const height = Math.max(baseImg.height, currImg.height);

      baseCanvas.width = currCanvas.width = diffCanvas.width = width;
      baseCanvas.height = currCanvas.height = diffCanvas.height = height;

      const baseCtx = baseCanvas.getContext('2d')!;
      const currCtx = currCanvas.getContext('2d')!;
      const diffCtx = diffCanvas.getContext('2d')!;

      baseCtx.drawImage(baseImg, 0, 0);
      currCtx.drawImage(currImg, 0, 0);

      const baseData = baseCtx.getImageData(0, 0, width, height);
      const currData = currCtx.getImageData(0, 0, width, height);
      const diffData = diffCtx.createImageData(width, height);

      let matchingPixels = 0;
      const totalPixels = width * height;

      for (let i = 0; i < baseData.data.length; i += 4) {
        const rDiff = Math.abs(baseData.data[i] - currData.data[i]);
        const gDiff = Math.abs(baseData.data[i + 1] - currData.data[i + 1]);
        const bDiff = Math.abs(baseData.data[i + 2] - currData.data[i + 2]);

        const threshold = 10; // Allow small color variations
        if (rDiff <= threshold && gDiff <= threshold && bDiff <= threshold) {
          matchingPixels++;
          // Keep original pixel in diff
          diffData.data[i] = currData.data[i];
          diffData.data[i + 1] = currData.data[i + 1];
          diffData.data[i + 2] = currData.data[i + 2];
          diffData.data[i + 3] = currData.data[i + 3];
        } else {
          // Highlight difference in red
          diffData.data[i] = 255;
          diffData.data[i + 1] = 0;
          diffData.data[i + 2] = 0;
          diffData.data[i + 3] = 255;
        }
      }

      diffCtx.putImageData(diffData, 0, 0);
      const matchPercentage = (matchingPixels / totalPixels) * 100;
      const diffBase64 = diffCanvas.toDataURL('image/png');

      resolve({ matchPercentage, diffBase64 });
    };

    const checkLoaded = () => {
      loaded++;
      if (loaded === 2) onBothLoaded();
    };

    baseImg.onload = checkLoaded;
    currImg.onload = checkLoaded;
    baseImg.onerror = () => resolve({ matchPercentage: 0, diffBase64: '' });
    currImg.onerror = () => resolve({ matchPercentage: 0, diffBase64: '' });

    baseImg.src = baselineBase64.startsWith('data:') ? baselineBase64 : `data:image/png;base64,${baselineBase64}`;
    currImg.src = currentBase64.startsWith('data:') ? currentBase64 : `data:image/png;base64,${currentBase64}`;
  });
}

export function useVisualBaselines(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['visual-baselines', projectId],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('visual_baselines') as any)
        .select('*')
        .eq('project_id', projectId)
        .eq('is_active', true)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data as VisualBaseline[];
    },
    enabled: !!projectId,
  });
}

export function useVisualComparisons(projectId: string | null, limit = 20) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['visual-comparisons', projectId, limit],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('visual_comparisons') as any)
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data as VisualComparison[];
    },
    enabled: !!projectId,
  });
}

export function useApproveComparison() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      comparisonId,
      projectId,
      approvedBy,
    }: {
      comparisonId: string;
      projectId: string;
      approvedBy?: string | null;
    }) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('visual_comparisons') as any)
        .update({
          status: 'match',
          approved_by: approvedBy || null,
          approved_at: new Date().toISOString(),
        })
        .eq('id', comparisonId)
        .select()
        .single();

      if (error) throw error;
      return { comparison: data as VisualComparison, projectId };
    },
    onSuccess: ({ projectId }) => {
      queryClient.invalidateQueries({ queryKey: ['visual-comparisons', projectId] });
    },
  });
}

interface VisualTestResult {
  comparison: VisualComparison;
  baseline: VisualBaseline | null;
  isNew: boolean;
}

export function useRunVisualTest() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      url,
      name,
      viewport = '1920x1080',
      threshold = 0.1,
    }: {
      projectId: string;
      url: string;
      name?: string;
      viewport?: string;
      threshold?: number;
    }): Promise<VisualTestResult> => {
      // Parse viewport
      const [width, height] = viewport.split('x').map(Number);
      const device = width <= 768 ? 'mobile' : width <= 1024 ? 'tablet' : 'desktop';

      // 1. Capture screenshot using worker /test endpoint
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      const response = await fetch(`${WORKER_URL}/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url,
          steps: ['Wait for page to load'],
          screenshot: true,
          device,
          timeout: 30000,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Worker returned ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();

      // Get screenshot from first browser result or root screenshot
      const screenshotBase64 = result.browsers?.[0]?.screenshot || result.screenshot;
      if (!screenshotBase64) {
        throw new Error('No screenshot captured');
      }

      // Create data URL for storage
      const screenshotDataUrl = screenshotBase64.startsWith('data:')
        ? screenshotBase64
        : `data:image/png;base64,${screenshotBase64}`;

      // Generate name from URL if not provided
      const testName = name || new URL(url).pathname.replace(/\//g, '-').replace(/^-/, '') || 'homepage';

      // 2. Check for existing baseline
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: existingBaselines } = await (supabase.from('visual_baselines') as any)
        .select('*')
        .eq('project_id', projectId)
        .eq('page_url', url)
        .eq('viewport', viewport)
        .eq('is_active', true)
        .limit(1);

      const existingBaseline = existingBaselines?.[0] as VisualBaseline | undefined;

      if (!existingBaseline) {
        // 3a. No baseline - create new baseline and comparison marked as "new"
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { data: newBaseline, error: baselineError } = await (supabase.from('visual_baselines') as any)
          .insert({
            project_id: projectId,
            name: testName,
            page_url: url,
            viewport,
            screenshot_url: screenshotDataUrl,
            is_active: true,
          })
          .select()
          .single();

        if (baselineError) throw baselineError;

        // Create comparison marked as new
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { data: comparison, error: compError } = await (supabase.from('visual_comparisons') as any)
          .insert({
            project_id: projectId,
            baseline_id: newBaseline.id,
            name: testName,
            status: 'new',
            match_percentage: 100,
            difference_count: 0,
            baseline_url: screenshotDataUrl,
            current_url: screenshotDataUrl,
            threshold,
          })
          .select()
          .single();

        if (compError) throw compError;

        return {
          comparison: comparison as VisualComparison,
          baseline: newBaseline as VisualBaseline,
          isNew: true,
        };
      } else {
        // 3b. Baseline exists - compare screenshots
        const { matchPercentage, diffBase64 } = await compareScreenshots(
          existingBaseline.screenshot_url,
          screenshotDataUrl
        );

        const status = matchPercentage >= (100 - threshold * 100) ? 'match' : 'mismatch';
        const differenceCount = Math.round((100 - matchPercentage) * 100);

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { data: comparison, error: compError } = await (supabase.from('visual_comparisons') as any)
          .insert({
            project_id: projectId,
            baseline_id: existingBaseline.id,
            name: testName,
            status,
            match_percentage: matchPercentage,
            difference_count: differenceCount,
            baseline_url: existingBaseline.screenshot_url,
            current_url: screenshotDataUrl,
            diff_url: diffBase64 || null,
            threshold,
          })
          .select()
          .single();

        if (compError) throw compError;

        return {
          comparison: comparison as VisualComparison,
          baseline: existingBaseline,
          isNew: false,
        };
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['visual-baselines', data.comparison.project_id] });
      queryClient.invalidateQueries({ queryKey: ['visual-comparisons', data.comparison.project_id] });
    },
  });
}

export function useUpdateBaseline() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      comparisonId,
      projectId,
    }: {
      comparisonId: string;
      projectId: string;
    }) => {
      // Get the comparison
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: comparison, error: compError } = await (supabase.from('visual_comparisons') as any)
        .select('*')
        .eq('id', comparisonId)
        .single();

      if (compError || !comparison) throw new Error('Comparison not found');

      // Update the baseline with the current screenshot
      if (comparison.baseline_id) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase.from('visual_baselines') as any)
          .update({
            screenshot_url: comparison.current_url,
          })
          .eq('id', comparison.baseline_id);
      }

      // Mark comparison as approved
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: updated, error: updateError } = await (supabase.from('visual_comparisons') as any)
        .update({
          status: 'match',
          approved_at: new Date().toISOString(),
        })
        .eq('id', comparisonId)
        .select()
        .single();

      if (updateError) throw updateError;

      return { comparison: updated as VisualComparison, projectId };
    },
    onSuccess: ({ projectId }) => {
      queryClient.invalidateQueries({ queryKey: ['visual-baselines', projectId] });
      queryClient.invalidateQueries({ queryKey: ['visual-comparisons', projectId] });
    },
  });
}
