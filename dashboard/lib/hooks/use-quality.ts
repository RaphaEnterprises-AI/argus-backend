'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getSupabaseClient } from '@/lib/supabase/client';
import type { QualityAudit, AccessibilityIssue } from '@/lib/supabase/types';

const WORKER_URL = process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

export function useQualityAudits(projectId: string | null, limit = 10) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['quality-audits', projectId, limit],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('quality_audits') as any)
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data as QualityAudit[];
    },
    enabled: !!projectId,
  });
}

export function useLatestAudit(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['latest-audit', projectId],
    queryFn: async () => {
      if (!projectId) return null;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: audits, error: auditError } = await (supabase.from('quality_audits') as any)
        .select('*')
        .eq('project_id', projectId)
        .eq('status', 'completed')
        .order('created_at', { ascending: false })
        .limit(1);

      if (auditError) throw auditError;
      if (!audits || audits.length === 0) return null;

      const audit = audits[0] as QualityAudit;

      // Get accessibility issues for this audit
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: issues, error: issuesError } = await (supabase.from('accessibility_issues') as any)
        .select('*')
        .eq('audit_id', audit.id)
        .order('severity', { ascending: true });

      if (issuesError) throw issuesError;

      return {
        audit,
        issues: issues as AccessibilityIssue[],
      };
    },
    enabled: !!projectId,
  });
}

export function useAccessibilityIssues(auditId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['accessibility-issues', auditId],
    queryFn: async () => {
      if (!auditId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('accessibility_issues') as any)
        .select('*')
        .eq('audit_id', auditId)
        .order('severity', { ascending: true });

      if (error) throw error;
      return data as AccessibilityIssue[];
    },
    enabled: !!auditId,
  });
}

export function useStartQualityAudit() {
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
      // 1. Create audit with 'running' status
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: audit, error: auditError } = await (supabase.from('quality_audits') as any)
        .insert({
          project_id: projectId,
          url,
          status: 'running',
          started_at: new Date().toISOString(),
          triggered_by: triggeredBy || null,
        })
        .select()
        .single();

      if (auditError) throw auditError;

      try {
        // 2. Call worker to observe the page
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const response = await fetch(`${WORKER_URL}/observe`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url,
            instruction: 'Analyze this page for accessibility issues, missing alt text, form labels, and semantic HTML problems',
          }),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`Worker returned ${response.status}`);
        }

        const observeResult = await response.json();

        // Worker returns 'actions' not 'elements'
        const actions = observeResult.actions || [];

        // 3. Analyze actions for accessibility patterns
        const issues: Array<{
          rule: string;
          severity: 'critical' | 'serious' | 'moderate' | 'minor';
          element_selector: string;
          description: string;
          suggested_fix: string;
          wcag_criteria: string[];
        }> = [];

        // Check for elements that might have accessibility issues based on selectors/descriptions
        const imageActions = actions.filter((a: any) =>
          a.selector?.includes('img') || a.description?.toLowerCase().includes('image')
        );
        const inputActions = actions.filter((a: any) =>
          a.selector?.includes('input') || a.description?.toLowerCase().includes('input')
        );
        const buttonActions = actions.filter((a: any) =>
          a.selector?.includes('button') || a.description?.toLowerCase().includes('button')
        );
        const linkActions = actions.filter((a: any) =>
          a.selector?.includes('a.') || a.selector?.includes('a[') || a.description?.toLowerCase().includes('link')
        );

        // For images without descriptive alt in the description
        imageActions.forEach((img: any) => {
          if (!img.description || img.description === 'Image' || img.description.length < 10) {
            issues.push({
              rule: 'image-alt',
              severity: 'critical',
              element_selector: img.selector || 'img',
              description: 'Image may be missing descriptive alt text',
              suggested_fix: 'Add an alt attribute describing the image content',
              wcag_criteria: ['1.1.1'],
            });
          }
        });

        // Check for generic input descriptions
        inputActions.forEach((input: any) => {
          if (!input.description || input.description === 'Input' || input.description.length < 5) {
            issues.push({
              rule: 'label',
              severity: 'serious',
              element_selector: input.selector || 'input',
              description: 'Form input may be missing an associated label',
              suggested_fix: 'Add a <label> element or aria-label attribute',
              wcag_criteria: ['1.3.1', '3.3.2'],
            });
          }
        });

        // Check for generic button descriptions
        buttonActions.forEach((btn: any) => {
          if (!btn.description || btn.description === 'Button' || btn.description.length < 5) {
            issues.push({
              rule: 'button-name',
              severity: 'critical',
              element_selector: btn.selector || 'button',
              description: 'Button may not have accessible name',
              suggested_fix: 'Add text content or aria-label to the button',
              wcag_criteria: ['4.1.2'],
            });
          }
        });

        // Check for generic link descriptions
        linkActions.forEach((link: any) => {
          if (!link.description || link.description === 'Link' || link.description.length < 5) {
            issues.push({
              rule: 'link-name',
              severity: 'serious',
              element_selector: link.selector || 'a',
              description: 'Link may not have accessible name',
              suggested_fix: 'Add text content or aria-label to the link',
              wcag_criteria: ['2.4.4'],
            });
          }
        });

        // Use actions as elements for counting
        const elements = actions;

        // 4. Save accessibility issues
        if (issues.length > 0) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          await (supabase.from('accessibility_issues') as any)
            .insert(issues.map(issue => ({
              ...issue,
              audit_id: audit.id,
            })));
        }

        // 5. Calculate scores based on issues
        const criticalCount = issues.filter(i => i.severity === 'critical').length;
        const seriousCount = issues.filter(i => i.severity === 'serious').length;
        const moderateCount = issues.filter(i => i.severity === 'moderate').length;

        // Simple scoring - 100 minus penalties
        const accessibilityScore = Math.max(0, 100 - (criticalCount * 20) - (seriousCount * 10) - (moderateCount * 5));

        // Other scores are simulated based on page analysis
        const hasSemanticHTML = elements.some((el: any) =>
          ['HEADER', 'NAV', 'MAIN', 'FOOTER', 'ARTICLE', 'SECTION'].includes(el.tagName)
        );
        const bestPracticesScore = hasSemanticHTML ? 85 : 65;
        const seoScore = elements.some((el: any) => el.tagName === 'H1') ? 90 : 70;

        // 6. Update audit with results
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { data: updatedAudit } = await (supabase.from('quality_audits') as any)
          .update({
            status: 'completed',
            completed_at: new Date().toISOString(),
            accessibility_score: accessibilityScore,
            performance_score: 75, // Placeholder - would need real measurement
            best_practices_score: bestPracticesScore,
            seo_score: seoScore,
            lcp_ms: 2500, // Placeholder metrics
            fcp_ms: 1200,
            cls: 0.1,
            ttfb_ms: 200,
          })
          .eq('id', audit.id)
          .select()
          .single();

        return {
          audit: updatedAudit as QualityAudit,
          issues: issues as unknown as AccessibilityIssue[],
        };
      } catch (error) {
        // Update audit to failed status
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase.from('quality_audits') as any)
          .update({
            status: 'failed',
            completed_at: new Date().toISOString(),
          })
          .eq('id', audit.id);

        throw error;
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['quality-audits', data.audit.project_id] });
      queryClient.invalidateQueries({ queryKey: ['latest-audit', data.audit.project_id] });
    },
  });
}
