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

// Cloudflare Worker API URL
const WORKER_URL = process.env.E2E_WORKER_URL || 'https://argus-api.samuelvinay-kumar.workers.dev';

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

    const result = streamText({
    model: anthropic('claude-sonnet-4-20250514'),
    system: `You are an AI-powered E2E Testing Agent. You help users:
1. Create tests from natural language descriptions
2. Run tests and report results
3. Discover application flows automatically
4. Detect visual regressions

When users ask you to create tests, use the createTest tool.
When users want to run tests, use the runTests tool.
When users want to discover their app, use the discoverApp tool.
When users want visual comparison, use the compareVisual tool.

Be helpful, concise, and proactive. Format test results clearly with pass/fail status.
When showing test steps, use numbered lists. When showing code or selectors, use code blocks.`,
    messages,
    tools: {
      // Execute a browser action using natural language
      executeAction: tool({
        description: 'Execute a browser action like clicking, typing, or navigating',
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
            }, 60000); // 60 second timeout

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
        description: 'Run a multi-step E2E test on an application with screenshots',
        parameters: z.object({
          url: z.string().describe('Application URL to test'),
          steps: z.array(z.string()).describe('Array of test step instructions'),
          browser: z.string().optional().describe('Browser to use (chrome, firefox, safari)'),
        }),
        execute: async ({ url, steps, browser }) => {
          try {
            // Calculate timeout based on number of steps (30s per step + 30s buffer)
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
              backend: data.backend,
              screenshots: data.screenshots,
              finalScreenshot: data.finalScreenshot,
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
            }, 60000); // 60 second timeout

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
            }, 60000); // 60 second timeout

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
            // Calculate timeout based on max steps (30s per step + 60s buffer)
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
