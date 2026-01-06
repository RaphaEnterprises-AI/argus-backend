'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getSupabaseClient } from '@/lib/supabase/client';
import type { DiscoverySession, DiscoveredPage, DiscoveredFlow, InsertTables } from '@/lib/supabase/types';

const WORKER_URL = process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev';

export function useDiscoverySessions(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['discovery-sessions', projectId],
    queryFn: async () => {
      if (!projectId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('discovery_sessions') as any)
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data as DiscoverySession[];
    },
    enabled: !!projectId,
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    placeholderData: [], // Prevent loading state flash
  });
}

export function useDiscoveredPages(sessionId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['discovered-pages', sessionId],
    queryFn: async () => {
      if (!sessionId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('discovered_pages') as any)
        .select('*')
        .eq('discovery_session_id', sessionId)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data as DiscoveredPage[];
    },
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes - discovery results don't change
    gcTime: 15 * 60 * 1000, // Keep in cache for 15 minutes
    placeholderData: [], // Prevent loading state flash
  });
}

export function useDiscoveredFlows(sessionId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['discovered-flows', sessionId],
    queryFn: async () => {
      if (!sessionId) return [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase.from('discovered_flows') as any)
        .select('*')
        .eq('discovery_session_id', sessionId)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data as DiscoveredFlow[];
    },
    enabled: !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes - discovery results don't change
    gcTime: 15 * 60 * 1000, // Keep in cache for 15 minutes
    placeholderData: [], // Prevent loading state flash
  });
}

export function useLatestDiscoveryData(projectId: string | null) {
  const supabase = getSupabaseClient();

  return useQuery({
    queryKey: ['latest-discovery', projectId],
    queryFn: async () => {
      if (!projectId) return null;

      // Get latest session
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: sessions, error: sessionError } = await (supabase.from('discovery_sessions') as any)
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false })
        .limit(1);

      if (sessionError) throw sessionError;
      if (!sessions || sessions.length === 0) return null;

      const latestSession = sessions[0] as DiscoverySession;

      // Fetch pages and flows in parallel instead of sequentially
      const [pagesResult, flowsResult] = await Promise.all([
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (supabase.from('discovered_pages') as any)
          .select('*')
          .eq('discovery_session_id', latestSession.id)
          .order('created_at', { ascending: false }),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (supabase.from('discovered_flows') as any)
          .select('*')
          .eq('discovery_session_id', latestSession.id)
          .order('created_at', { ascending: false }),
      ]);

      if (pagesResult.error) throw pagesResult.error;
      if (flowsResult.error) throw flowsResult.error;

      return {
        session: latestSession,
        pages: pagesResult.data as DiscoveredPage[],
        flows: flowsResult.data as DiscoveredFlow[],
      };
    },
    enabled: !!projectId,
    staleTime: 2 * 60 * 1000, // 2 minutes - discovery data changes less frequently
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    placeholderData: null, // Prevent loading state flash
  });
}

interface DiscoveryResult {
  session: DiscoverySession;
  pages: DiscoveredPage[];
  flows: DiscoveredFlow[];
}

export function useStartDiscovery() {
  const supabase = getSupabaseClient();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      projectId,
      appUrl,
      triggeredBy,
    }: {
      projectId: string;
      appUrl: string;
      triggeredBy?: string | null;
    }): Promise<DiscoveryResult> => {
      // 1. Create session with 'running' status
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: session, error: sessionError } = await (supabase.from('discovery_sessions') as any)
        .insert({
          project_id: projectId,
          app_url: appUrl,
          status: 'running',
          started_at: new Date().toISOString(),
          triggered_by: triggeredBy || null,
        })
        .select()
        .single();

      if (sessionError) throw sessionError;

      try {
        // 2. Call worker to observe the page
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const response = await fetch(`${WORKER_URL}/observe`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: appUrl,
            instruction: 'Analyze this page and identify all interactive elements, forms, links, and possible user flows',
          }),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`Worker returned ${response.status}: ${await response.text()}`);
        }

        const observeResult = await response.json();

        // 3. Parse the observation results - worker returns 'actions' not 'elements'
        const actions = observeResult.actions || [];

        // Categorize actions by their description/selector
        const links = actions.filter((a: any) =>
          a.selector?.includes('a') ||
          a.description?.toLowerCase().includes('link') ||
          a.description?.toLowerCase().includes('navigate')
        );
        const buttons = actions.filter((a: any) =>
          a.selector?.includes('button') ||
          a.description?.toLowerCase().includes('button') ||
          a.description?.toLowerCase().includes('click') ||
          a.description?.toLowerCase().includes('submit')
        );
        const inputs = actions.filter((a: any) =>
          a.selector?.includes('input') ||
          a.description?.toLowerCase().includes('input') ||
          a.description?.toLowerCase().includes('enter') ||
          a.description?.toLowerCase().includes('type')
        );
        const forms = actions.filter((a: any) =>
          a.selector?.includes('form') ||
          a.description?.toLowerCase().includes('form')
        );

        // Use actions as our elements list
        const elements = actions;

        // 4. Save discovered page (upsert to handle re-discovery)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { data: page, error: pageError } = await (supabase.from('discovered_pages') as any)
          .upsert({
            discovery_session_id: session.id,
            project_id: projectId,
            url: appUrl,
            title: observeResult.pageTitle || appUrl,
            page_type: 'landing',
            element_count: buttons.length + inputs.length,
            form_count: forms.length,
            link_count: links.length,
            metadata: {
              elements: elements.slice(0, 50), // Store first 50 elements
              screenshot: observeResult.screenshot,
            },
          }, {
            onConflict: 'project_id,url',
          })
          .select()
          .single();

        if (pageError) throw pageError;

        // 5. Delete old flows for this session (to avoid duplicates on re-discovery)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase.from('discovered_flows') as any)
          .delete()
          .eq('discovery_session_id', session.id);

        // 6. Generate flows from discovered elements
        const flows: DiscoveredFlow[] = [];

        // Check for login-related elements
        const hasLoginElements = actions.some((a: any) =>
          a.description?.toLowerCase().includes('login') ||
          a.description?.toLowerCase().includes('sign in') ||
          a.description?.toLowerCase().includes('password') ||
          a.selector?.includes('password')
        );

        if (hasLoginElements) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const { data: loginFlow } = await (supabase.from('discovered_flows') as any)
            .insert({
              discovery_session_id: session.id,
              project_id: projectId,
              name: 'User Login Flow',
              description: 'Authenticate user with credentials',
              steps: [
                { instruction: 'Navigate to login page' },
                { instruction: 'Enter username/email' },
                { instruction: 'Enter password' },
                { instruction: 'Click login button' },
                { instruction: 'Verify successful login' },
              ],
              step_count: 5,
              priority: 'critical',
            })
            .select()
            .single();
          if (loginFlow) flows.push(loginFlow);
        }

        // Create navigation flow if multiple links exist
        if (links.length >= 3) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const { data: navFlow } = await (supabase.from('discovered_flows') as any)
            .insert({
              discovery_session_id: session.id,
              project_id: projectId,
              name: 'Navigation Flow',
              description: 'Verify main navigation links work correctly',
              steps: links.slice(0, 5).map((link: any) => ({
                instruction: link.description || `Click on link`,
              })),
              step_count: Math.min(links.length, 5),
              priority: 'high',
            })
            .select()
            .single();
          if (navFlow) flows.push(navFlow);
        }

        // Create form submission flow if forms exist
        if (forms.length > 0) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const { data: formFlow } = await (supabase.from('discovered_flows') as any)
            .insert({
              discovery_session_id: session.id,
              project_id: projectId,
              name: 'Form Submission Flow',
              description: 'Submit form with valid data',
              steps: [
                { instruction: 'Locate the form' },
                { instruction: 'Fill in required fields' },
                { instruction: 'Submit the form' },
                { instruction: 'Verify submission success' },
              ],
              step_count: 4,
              priority: 'high',
            })
            .select()
            .single();
          if (formFlow) flows.push(formFlow);
        }

        // Create button interaction flow if buttons exist
        if (buttons.length > 0) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const { data: buttonFlow } = await (supabase.from('discovered_flows') as any)
            .insert({
              discovery_session_id: session.id,
              project_id: projectId,
              name: 'Button Interactions',
              description: 'Test interactive button elements',
              steps: buttons.slice(0, 3).map((btn: any) => ({
                instruction: btn.description || `Click the button`,
              })),
              step_count: Math.min(buttons.length, 3),
              priority: 'medium',
            })
            .select()
            .single();
          if (buttonFlow) flows.push(buttonFlow);
        }

        // 7. Update session with final counts
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase.from('discovery_sessions') as any)
          .update({
            status: 'completed',
            completed_at: new Date().toISOString(),
            pages_found: 1,
            flows_found: flows.length,
            forms_found: forms.length,
            elements_found: elements.length,
          })
          .eq('id', session.id);

        return {
          session: { ...session, status: 'completed' },
          pages: [page],
          flows,
        };
      } catch (error) {
        // Update session to failed status
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase.from('discovery_sessions') as any)
          .update({
            status: 'failed',
            completed_at: new Date().toISOString(),
          })
          .eq('id', session.id);

        throw error;
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['discovery-sessions', data.session.project_id] });
      queryClient.invalidateQueries({ queryKey: ['latest-discovery', data.session.project_id] });
    },
  });
}
