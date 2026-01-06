import { anthropic } from '@ai-sdk/anthropic';
import { streamText, tool } from 'ai';
import { z } from 'zod';

// Use Edge runtime to avoid serverless function size limits
export const runtime = 'edge';

// Allow streaming responses up to 5 minutes for long-running tests
export const maxDuration = 300;

// Helper to create fetch with timeout
function fetchWithTimeout(url: string, options: RequestInit, timeoutMs: number = 90000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  return fetch(url, {
    ...options,
    signal: controller.signal,
  }).finally(() => clearTimeout(timeoutId));
}

// Argus Core Backend (Python - LangGraph Orchestrator)
const BACKEND_URL = process.env.ARGUS_BACKEND_URL || 'http://localhost:8000';

// Cloudflare Worker API URL (Browser Automation)
const WORKER_URL = process.env.E2E_WORKER_URL || 'https://argus-api.samuelvinay-kumar.workers.dev';

// Check if backend is available
async function isBackendAvailable(): Promise<boolean> {
  try {
    const response = await fetchWithTimeout(`${BACKEND_URL}/health`, {
      method: 'GET',
    }, 5000);
    return response.ok;
  } catch {
    return false;
  }
}

export async function POST(req: Request) {
  try {
    // Check for API key
    if (!process.env.ANTHROPIC_API_KEY) {
      console.error('ANTHROPIC_API_KEY is not set');
      return new Response(
        JSON.stringify({ error: 'ANTHROPIC_API_KEY environment variable is not configured' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const { messages } = await req.json();

    // Check backend availability for system prompt
    const backendAvailable = await isBackendAvailable();

    const result = streamText({
    model: anthropic('claude-sonnet-4-20250514'),
    system: `You are Argus, an AI-powered E2E Testing Agent with ${backendAvailable ? 'FULL orchestration capabilities' : 'basic browser automation'}. You help users:
1. Create tests from natural language descriptions
2. Run tests and report results with self-healing capabilities
3. Discover application flows automatically
4. Detect visual regressions
5. ${backendAvailable ? 'Analyze codebases and generate comprehensive test plans' : 'Execute browser actions'}

${backendAvailable ? `
ADVANCED CAPABILITIES (Argus Brain Connected):
- Use runOrchestratedTest for complex multi-step tests with full LangGraph orchestration
- Use createTestFromNLP to create tests from plain English descriptions
- Use analyzeCodebase to understand an application and suggest tests
- Use autoDiscoverTests to automatically find test scenarios
- Use visualCompare to detect visual regressions
` : `
NOTE: Argus Brain (Python backend) is not connected. Using direct browser automation mode.
For full orchestration capabilities, deploy the Python backend.
`}

QUICK ACTIONS (Always Available):
- Use executeAction for single browser actions (click, type, navigate)
- Use runTest for simple multi-step tests
- Use discoverElements to find interactive elements
- Use extractData to scrape structured data
- Use runAgent for autonomous task completion

Be helpful, concise, and proactive. Format test results clearly with pass/fail status.
When showing test steps, use numbered lists. When showing code or selectors, use code blocks.
Report any self-healing that occurred during tests.`,
    messages,
    tools: {
      // ============================================================================
      // ORCHESTRATED TOOLS (Python Backend - Full LangGraph Power)
      // ============================================================================

      // Run orchestrated test with full LangGraph pipeline
      runOrchestratedTest: tool({
        description: 'Run a comprehensive test suite with full LangGraph orchestration, code analysis, self-healing, and detailed reporting. Use this for complex testing scenarios.',
        parameters: z.object({
          codebasePath: z.string().describe('Path to the codebase to analyze (can be a GitHub URL or local path)'),
          appUrl: z.string().describe('URL of the application to test'),
          focusAreas: z.array(z.string()).optional().describe('Specific areas to focus testing on'),
          maxTests: z.number().optional().describe('Maximum number of tests to run'),
        }),
        execute: async ({ codebasePath, appUrl, focusAreas, maxTests }) => {
          try {
            // Check if backend is available
            const available = await isBackendAvailable();
            if (!available) {
              return {
                success: false,
                error: 'Argus Brain (Python backend) is not available. Please deploy it to Railway/Render or run locally with: python -m uvicorn src.api.server:app --port 8000',
                fallback: 'Use runTest for basic browser automation instead.',
              };
            }

            const response = await fetchWithTimeout(`${BACKEND_URL}/api/v1/tests/run`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                codebase_path: codebasePath,
                app_url: appUrl,
                focus_areas: focusAreas,
                max_tests: maxTests,
              }),
            }, 30000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Orchestrated test failed: ${error}` };
            }

            const data = await response.json();
            return {
              success: true,
              jobId: data.job_id,
              status: data.status,
              message: data.message,
              checkStatusUrl: `${BACKEND_URL}/api/v1/jobs/${data.job_id}`,
            };
          } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
              return { success: false, error: 'Request timed out' };
            }
            return { success: false, error: String(error) };
          }
        },
      }),

      // Create test from natural language
      createTestFromNLP: tool({
        description: 'Create a test specification from a plain English description. Uses AI to generate proper test steps.',
        parameters: z.object({
          description: z.string().describe('Plain English test description (e.g., "Login as admin and verify dashboard shows 5 widgets")'),
          appUrl: z.string().describe('Application URL to test'),
          context: z.string().optional().describe('Additional context about the application'),
        }),
        execute: async ({ description, appUrl, context }) => {
          try {
            const available = await isBackendAvailable();
            if (!available) {
              return {
                success: false,
                error: 'Argus Brain not available. Deploy Python backend first.',
              };
            }

            const response = await fetchWithTimeout(`${BACKEND_URL}/api/v1/tests/create`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                description,
                app_url: appUrl,
                context,
              }),
            }, 60000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Test creation failed: ${error}` };
            }

            const data = await response.json();
            return {
              success: true,
              test: data.test,
              spec: data.spec,
            };
          } catch (error) {
            return { success: false, error: String(error) };
          }
        },
      }),

      // Visual AI comparison
      visualCompare: tool({
        description: 'Compare two screenshots using Visual AI to detect regressions',
        parameters: z.object({
          baselineBase64: z.string().describe('Base64 encoded baseline screenshot'),
          currentBase64: z.string().describe('Base64 encoded current screenshot'),
          context: z.string().optional().describe('Context about what is being compared'),
        }),
        execute: async ({ baselineBase64, currentBase64, context }) => {
          try {
            const available = await isBackendAvailable();
            if (!available) {
              return { success: false, error: 'Argus Brain not available for visual comparison.' };
            }

            const response = await fetchWithTimeout(`${BACKEND_URL}/api/v1/visual/compare`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                baseline_b64: baselineBase64,
                current_b64: currentBase64,
                context,
              }),
            }, 60000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Visual comparison failed: ${error}` };
            }

            const data = await response.json();
            return { success: true, result: data.result };
          } catch (error) {
            return { success: false, error: String(error) };
          }
        },
      }),

      // Auto-discover test scenarios
      autoDiscoverTests: tool({
        description: 'Automatically discover test scenarios by crawling the application',
        parameters: z.object({
          appUrl: z.string().describe('Application URL to crawl'),
          focusAreas: z.array(z.string()).optional().describe('Areas to focus on'),
          maxPages: z.number().optional().describe('Maximum pages to crawl'),
        }),
        execute: async ({ appUrl, focusAreas, maxPages }) => {
          try {
            const available = await isBackendAvailable();
            if (!available) {
              return { success: false, error: 'Argus Brain not available for auto-discovery.' };
            }

            const params = new URLSearchParams({ app_url: appUrl });
            if (maxPages) params.append('max_pages', maxPages.toString());
            if (focusAreas) focusAreas.forEach(area => params.append('focus_areas', area));

            const response = await fetchWithTimeout(`${BACKEND_URL}/api/v1/discover?${params}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
            }, 120000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Auto-discovery failed: ${error}` };
            }

            const data = await response.json();
            return { success: true, result: data.result };
          } catch (error) {
            return { success: false, error: String(error) };
          }
        },
      }),

      // Check job status
      checkJobStatus: tool({
        description: 'Check the status of a running test job',
        parameters: z.object({
          jobId: z.string().describe('The job ID to check'),
        }),
        execute: async ({ jobId }) => {
          try {
            const available = await isBackendAvailable();
            if (!available) {
              return { success: false, error: 'Argus Brain not available.' };
            }

            const response = await fetchWithTimeout(`${BACKEND_URL}/api/v1/jobs/${jobId}`, {
              method: 'GET',
            }, 10000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Status check failed: ${error}` };
            }

            const data = await response.json();
            return {
              success: true,
              jobId: data.job_id,
              status: data.status,
              progress: data.progress,
              result: data.result,
              error: data.error,
            };
          } catch (error) {
            return { success: false, error: String(error) };
          }
        },
      }),

      // ============================================================================
      // QUICK BROWSER TOOLS (Cloudflare Worker - Direct Browser Automation)
      // ============================================================================

      // Execute a browser action using natural language
      executeAction: tool({
        description: 'Execute a browser action like clicking, typing, or navigating. Use for quick single actions.',
        parameters: z.object({
          url: z.string().describe('URL of the page to test'),
          instruction: z.string().describe('Natural language instruction (e.g., "Click the login button")'),
        }),
        execute: async ({ url, instruction }) => {
          try {
            const response = await fetchWithTimeout(`${WORKER_URL}/act`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                url,
                instruction,
                selfHeal: true,
                screenshot: true,
              }),
            }, 60000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Action failed: ${error}` };
            }

            const data = await response.json();
            return {
              success: data.success,
              message: data.message,
              actions: data.actions,
              screenshot: data.screenshot,
              healed: data.healed,
              healingMethod: data.healingMethod,
            };
          } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
              return { success: false, error: 'Action timed out after 60 seconds' };
            }
            return { success: false, error: String(error) };
          }
        },
      }),

      // Run a multi-step test
      runTest: tool({
        description: 'Run a multi-step E2E test with self-healing and screenshots. Use for quick test execution.',
        parameters: z.object({
          url: z.string().describe('Application URL to test'),
          steps: z.array(z.string()).describe('Array of test step instructions'),
          browser: z.string().optional().describe('Browser to use (chrome, firefox, safari)'),
        }),
        execute: async ({ url, steps, browser }) => {
          try {
            const timeout = Math.max(120000, steps.length * 30000 + 30000);

            const response = await fetchWithTimeout(`${WORKER_URL}/test`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                url,
                steps,
                browser: browser || 'chrome',
                screenshot: true,
                captureScreenshots: true,
              }),
            }, timeout);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Test failed: ${error}` };
            }

            const data = await response.json();
            return {
              success: data.success,
              steps: data.steps,
              browsers: data.browsers,
              screenshots: data.screenshots,
              finalScreenshot: data.finalScreenshot,
              healingReport: data.healingReport,
            };
          } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
              return { success: false, error: `Test timed out. Try reducing the number of steps.` };
            }
            return { success: false, error: String(error) };
          }
        },
      }),

      // Discover interactive elements on a page
      discoverElements: tool({
        description: 'Discover interactive elements and possible actions on a page',
        parameters: z.object({
          url: z.string().describe('URL to analyze'),
          instruction: z.string().optional().describe('What to look for (e.g., "Find all buttons")'),
        }),
        execute: async ({ url, instruction }) => {
          try {
            const response = await fetchWithTimeout(`${WORKER_URL}/observe`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                url,
                instruction: instruction || 'What actions can I take on this page?',
              }),
            }, 60000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Discovery failed: ${error}` };
            }

            const data = await response.json();
            return { success: true, actions: data.actions };
          } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
              return { success: false, error: 'Discovery timed out after 60 seconds' };
            }
            return { success: false, error: String(error) };
          }
        },
      }),

      // Extract data from a page
      extractData: tool({
        description: 'Extract structured data from a web page',
        parameters: z.object({
          url: z.string().describe('URL to extract data from'),
          instruction: z.string().describe('What data to extract'),
          schema: z.record(z.string()).optional().describe('Expected data schema'),
        }),
        execute: async ({ url, instruction, schema }) => {
          try {
            const response = await fetchWithTimeout(`${WORKER_URL}/extract`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                url,
                instruction,
                schema: schema || {},
              }),
            }, 60000);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Extraction failed: ${error}` };
            }

            const data = await response.json();
            return { success: true, data };
          } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
              return { success: false, error: 'Extraction timed out after 60 seconds' };
            }
            return { success: false, error: String(error) };
          }
        },
      }),

      // Run an autonomous agent task
      runAgent: tool({
        description: 'Run an autonomous agent to complete a complex task with screenshots',
        parameters: z.object({
          url: z.string().describe('Starting URL'),
          instruction: z.string().describe('Task to complete (e.g., "Sign up for a new account")'),
          maxSteps: z.number().optional().describe('Maximum steps to take'),
        }),
        execute: async ({ url, instruction, maxSteps }) => {
          try {
            const steps = maxSteps || 10;
            const timeout = Math.max(120000, steps * 30000 + 60000);

            const response = await fetchWithTimeout(`${WORKER_URL}/agent`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                url,
                instruction,
                maxSteps: steps,
                captureScreenshots: true,
              }),
            }, timeout);

            if (!response.ok) {
              const error = await response.text();
              return { success: false, error: `Agent failed: ${error}` };
            }

            const data = await response.json();
            return {
              success: data.success,
              completed: data.completed,
              message: data.message,
              actions: data.actions,
              usage: data.usage,
              screenshots: data.screenshots,
            };
          } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
              return { success: false, error: `Agent timed out. Try reducing maxSteps.` };
            }
            return { success: false, error: String(error) };
          }
        },
      }),

      // ============================================================================
      // SYSTEM TOOLS
      // ============================================================================

      // Check system status
      checkSystemStatus: tool({
        description: 'Check the status of Argus components (Brain and Browser Worker)',
        parameters: z.object({}),
        execute: async () => {
          const [brainStatus, workerStatus] = await Promise.all([
            isBackendAvailable().then(ok => ({
              name: 'Argus Brain (Python/LangGraph)',
              url: BACKEND_URL,
              status: ok ? 'connected' : 'disconnected',
              capabilities: ok ? ['orchestration', 'code-analysis', 'nlp-tests', 'visual-ai', 'auto-discovery'] : [],
            })),
            fetchWithTimeout(`${WORKER_URL}/health`, { method: 'GET' }, 5000)
              .then(r => r.ok)
              .catch(() => false)
              .then(ok => ({
                name: 'Argus Worker (Cloudflare/Browser)',
                url: WORKER_URL,
                status: ok ? 'connected' : 'disconnected',
                capabilities: ok ? ['browser-automation', 'self-healing', 'screenshots', 'extraction'] : [],
              })),
          ]);

          return {
            success: true,
            components: [brainStatus, workerStatus],
            fullCapabilities: brainStatus.status === 'connected' && workerStatus.status === 'connected',
          };
        },
      }),
    },
  });

    return result.toDataStreamResponse();
  } catch (error) {
    console.error('Chat API error:', error);
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : 'An unexpected error occurred',
        details: String(error)
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
