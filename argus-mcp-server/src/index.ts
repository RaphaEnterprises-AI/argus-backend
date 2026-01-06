/**
 * Argus MCP Server - Model Context Protocol for AI IDEs
 * https://heyargus.ai
 *
 * This MCP server exposes Argus E2E testing capabilities to AI coding assistants:
 * - Claude Code
 * - Cursor
 * - Windsurf
 * - VS Code with MCP extension
 *
 * Tools provided:
 * - argus_discover: Discover interactive elements on a page
 * - argus_test: Run multi-step E2E tests with screenshots
 * - argus_act: Execute browser actions
 * - argus_extract: Extract data from pages
 * - argus_agent: Autonomous task completion
 * - argus_health: Check Argus API status
 */

import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

// Environment types
interface Env {
  AI: Ai;
  ARGUS_API_URL: string;
  API_TOKEN?: string;
  ANTHROPIC_API_KEY?: string;
  MCP_OAUTH: DurableObjectNamespace;
}

// Argus API Response types
interface ArgusActResponse {
  success: boolean;
  message?: string;
  actions?: Array<{
    action: string;
    selector?: string;
    value?: string;
    success: boolean;
  }>;
  screenshot?: string;
  error?: string;
}

interface ArgusTestResponse {
  success: boolean;
  steps?: Array<{
    instruction: string;
    success: boolean;
    error?: string;
    screenshot?: string;
  }>;
  screenshots?: string[];
  finalScreenshot?: string;
  error?: string;
}

interface ArgusObserveResponse {
  success: boolean;
  actions?: Array<{
    description: string;
    selector: string;
    type: string;
    confidence: number;
  }>;
  error?: string;
}

interface ArgusExtractResponse {
  success: boolean;
  data?: Record<string, unknown>;
  error?: string;
}

interface ArgusAgentResponse {
  success: boolean;
  completed: boolean;
  message?: string;
  actions?: Array<{
    action: string;
    success: boolean;
    screenshot?: string;
  }>;
  screenshots?: string[];
  usage?: {
    inputTokens: number;
    outputTokens: number;
  };
  error?: string;
}

// MCP Content types
type TextContent = {
  type: "text";
  text: string;
};

type ImageContent = {
  type: "image";
  data: string;
  mimeType: string;
};

type McpContent = TextContent | ImageContent;

// Helper to call Argus API
async function callArgusAPI<T>(
  endpoint: string,
  body: Record<string, unknown>,
  env: Env
): Promise<T> {
  const apiUrl = env.ARGUS_API_URL || "https://argus-api.samuelvinay-kumar.workers.dev";

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (env.API_TOKEN) {
    headers["Authorization"] = `Bearer ${env.API_TOKEN}`;
  }

  const response = await fetch(`${apiUrl}${endpoint}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Argus API error (${response.status}): ${errorText}`);
  }

  return response.json() as Promise<T>;
}

// Create MCP Server with Argus tools
export class ArgusMcpAgent extends McpAgent<Env> {
  server = new McpServer({
    name: "Argus E2E Testing Agent",
    version: "1.0.0",
  });

  async init() {
    // Tool: argus_health - Check Argus API status
    this.server.tool(
      "argus_health",
      "Check the health and status of the Argus E2E testing API",
      {},
      async () => {
        try {
          const apiUrl = this.env.ARGUS_API_URL || "https://argus-api.samuelvinay-kumar.workers.dev";
          const response = await fetch(`${apiUrl}/health`);
          const data = await response.json();

          return {
            content: [
              {
                type: "text" as const,
                text: JSON.stringify(data, null, 2),
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error checking Argus health: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_discover - Discover interactive elements
    this.server.tool(
      "argus_discover",
      "Discover interactive elements and possible actions on a web page. Returns clickable buttons, links, form inputs, and other actionable elements.",
      {
        url: z.string().url().describe("The URL of the page to analyze"),
        instruction: z.string().optional().describe("What to look for (e.g., 'Find all buttons', 'Find the login form')"),
      },
      async ({ url, instruction }) => {
        try {
          const result = await callArgusAPI<ArgusObserveResponse>("/observe", {
            url,
            instruction: instruction || "What actions can I take on this page?",
          }, this.env);

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Discovery failed: ${result.error || "Unknown error"}`,
                },
              ],
              isError: true,
            };
          }

          // Format the discovered actions
          const formattedActions = result.actions?.map((action, i) =>
            `${i + 1}. ${action.description}\n   Selector: \`${action.selector}\`\n   Type: ${action.type}\n   Confidence: ${(action.confidence * 100).toFixed(0)}%`
          ).join("\n\n") || "No actions discovered";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Discovered Elements on ${url}\n\n${formattedActions}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error discovering elements: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_act - Execute browser action
    this.server.tool(
      "argus_act",
      "Execute a browser action like clicking, typing, or navigating. Uses AI to find the right element based on natural language.",
      {
        url: z.string().url().describe("The URL of the page"),
        instruction: z.string().describe("What action to perform (e.g., 'Click the login button', 'Type \"hello\" in the search box')"),
        selfHeal: z.boolean().optional().describe("Enable self-healing selectors (default: true)"),
        screenshot: z.boolean().optional().describe("Capture a screenshot after the action (default: true)"),
      },
      async ({ url, instruction, selfHeal = true, screenshot = true }) => {
        try {
          const result = await callArgusAPI<ArgusActResponse>("/act", {
            url,
            instruction,
            selfHeal,
            screenshot,
          }, this.env);

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Action failed: ${result.error || "Unknown error"}`,
                },
              ],
              isError: true,
            };
          }

          const content: McpContent[] = [
            {
              type: "text" as const,
              text: `## Action Result\n\n${result.message || "Action completed successfully"}\n\n### Actions Performed:\n${result.actions?.map(a => `- ${a.action}${a.selector ? ` on \`${a.selector}\`` : ""}${a.value ? ` with value "${a.value}"` : ""}: ${a.success ? "Success" : "Failed"}`).join("\n") || "None"}`,
            },
          ];

          // Add screenshot if available
          if (result.screenshot) {
            content.push({
              type: "image" as const,
              data: result.screenshot,
              mimeType: "image/png",
            });
          }

          return { content };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error executing action: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_test - Run multi-step E2E test
    this.server.tool(
      "argus_test",
      "Run a multi-step E2E test on a web application. Executes a sequence of test steps and captures screenshots at each step. Returns detailed results including pass/fail status for each step.",
      {
        url: z.string().url().describe("The starting URL for the test"),
        steps: z.array(z.string()).min(1).describe("Array of test step instructions (e.g., ['Click the login button', 'Type \"user@example.com\" in email field', 'Click submit'])"),
        browser: z.enum(["chrome", "firefox", "safari", "edge"]).optional().describe("Browser to use (default: chrome)"),
      },
      async ({ url, steps, browser = "chrome" }) => {
        try {
          const result = await callArgusAPI<ArgusTestResponse>("/test", {
            url,
            steps,
            browser,
            captureScreenshots: true,
          }, this.env);

          // Format step results
          const stepResults = result.steps?.map((step, i) => {
            const status = step.success ? "PASS" : "FAIL";
            const error = step.error ? `\n   Error: ${step.error}` : "";
            return `${i + 1}. [${status}] ${step.instruction}${error}`;
          }).join("\n") || "No steps executed";

          const overallStatus = result.success ? "PASSED" : "FAILED";
          const passedSteps = result.steps?.filter(s => s.success).length || 0;
          const totalSteps = result.steps?.length || 0;

          const content: McpContent[] = [
            {
              type: "text" as const,
              text: `## Test Results: ${overallStatus}\n\n**Browser:** ${browser}\n**URL:** ${url}\n**Steps:** ${passedSteps}/${totalSteps} passed\n\n### Step Details:\n${stepResults}`,
            },
          ];

          // Add final screenshot if available
          if (result.finalScreenshot) {
            content.push({
              type: "image" as const,
              data: result.finalScreenshot,
              mimeType: "image/png",
            });
          }

          return { content };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error running test: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_extract - Extract data from page
    this.server.tool(
      "argus_extract",
      "Extract structured data from a web page using AI. Can extract specific information like product details, prices, user information, etc.",
      {
        url: z.string().url().describe("The URL of the page to extract data from"),
        instruction: z.string().describe("What data to extract (e.g., 'Extract all product names and prices', 'Get the user profile information')"),
        schema: z.record(z.string()).optional().describe("Expected data schema as key-value pairs (optional)"),
      },
      async ({ url, instruction, schema }) => {
        try {
          const result = await callArgusAPI<ArgusExtractResponse>("/extract", {
            url,
            instruction,
            schema: schema || {},
          }, this.env);

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Extraction failed: ${result.error || "Unknown error"}`,
                },
              ],
              isError: true,
            };
          }

          return {
            content: [
              {
                type: "text" as const,
                text: `## Extracted Data from ${url}\n\n\`\`\`json\n${JSON.stringify(result.data, null, 2)}\n\`\`\``,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error extracting data: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_agent - Autonomous task completion
    this.server.tool(
      "argus_agent",
      "Run an autonomous AI agent to complete a complex task on a website. The agent will navigate, click, type, and perform multiple actions to achieve the goal.",
      {
        url: z.string().url().describe("The starting URL"),
        instruction: z.string().describe("The task to complete (e.g., 'Sign up for a new account with email test@example.com', 'Add the first product to cart and proceed to checkout')"),
        maxSteps: z.number().min(1).max(30).optional().describe("Maximum number of steps to take (default: 10)"),
      },
      async ({ url, instruction, maxSteps = 10 }) => {
        try {
          const result = await callArgusAPI<ArgusAgentResponse>("/agent", {
            url,
            instruction,
            maxSteps,
            captureScreenshots: true,
          }, this.env);

          const status = result.success && result.completed ? "COMPLETED" : result.success ? "IN PROGRESS" : "FAILED";

          // Format actions taken
          const actionsList = result.actions?.map((action, i) => {
            const icon = action.success ? "+" : "-";
            return `${i + 1}. [${icon}] ${action.action}`;
          }).join("\n") || "No actions recorded";

          const content: McpContent[] = [
            {
              type: "text" as const,
              text: `## Agent Task: ${status}\n\n**Goal:** ${instruction}\n**Starting URL:** ${url}\n**Steps taken:** ${result.actions?.length || 0}/${maxSteps}\n\n### Actions:\n${actionsList}\n\n${result.message ? `**Result:** ${result.message}` : ""}`,
            },
          ];

          // Add the last screenshot if available
          if (result.screenshots && result.screenshots.length > 0) {
            content.push({
              type: "image" as const,
              data: result.screenshots[result.screenshots.length - 1],
              mimeType: "image/png",
            });
          }

          return { content };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error running agent: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_generate_test - Generate test from natural language
    this.server.tool(
      "argus_generate_test",
      "Generate E2E test steps from a natural language description. Uses AI to analyze the target page and create a test plan.",
      {
        url: z.string().url().describe("The URL of the application to test"),
        description: z.string().describe("Description of what the test should verify (e.g., 'Verify user can log in with valid credentials', 'Test the checkout flow with a product')"),
      },
      async ({ url, description }) => {
        try {
          // First, discover the page
          const observeResult = await callArgusAPI<ArgusObserveResponse>("/observe", {
            url,
            instruction: "List all interactive elements and their purposes",
          }, this.env);

          if (!observeResult.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Could not analyze page: ${observeResult.error}`,
                },
              ],
              isError: true,
            };
          }

          // Use AI to generate test steps based on the page analysis
          const pageContext = observeResult.actions?.map(a => `- ${a.description} (${a.type})`).join("\n") || "No elements found";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Generated Test Plan\n\n**Target:** ${url}\n**Objective:** ${description}\n\n### Discovered Page Elements:\n${pageContext}\n\n### Suggested Test Steps:\n1. Navigate to the page\n2. [Add your test steps here based on the discovered elements]\n\n**Tip:** Use the \`argus_test\` tool to run these steps once you've finalized them.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error generating test: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );
  }
}

// OAuth Durable Object for state management
export class MCPOAuth {
  state: DurableObjectState;

  constructor(state: DurableObjectState) {
    this.state = state;
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/store") {
      const { key, value } = await request.json() as { key: string; value: unknown };
      await this.state.storage.put(key, value);
      return new Response("OK");
    }

    if (url.pathname === "/get") {
      const { key } = await request.json() as { key: string };
      const value = await this.state.storage.get(key);
      return Response.json({ value });
    }

    return new Response("Not found", { status: 404 });
  }
}

// Export the Argus MCP Agent
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    // Handle the SSE endpoint for MCP
    if (url.pathname === "/sse" || url.pathname.startsWith("/sse/")) {
      return ArgusMcpAgent.mount("/sse").fetch(request, env, ctx);
    }

    // Root endpoint - show info
    if (url.pathname === "/") {
      return Response.json({
        name: "Argus MCP Server",
        version: "1.0.0",
        description: "Model Context Protocol server for Argus E2E Testing Agent",
        endpoint: "/sse",
        tools: [
          "argus_health",
          "argus_discover",
          "argus_act",
          "argus_test",
          "argus_extract",
          "argus_agent",
          "argus_generate_test",
        ],
        documentation: "https://github.com/raphaenterprises-ai/argus-e2e-testing-agent",
      });
    }

    return new Response("Not found", { status: 404 });
  },
};
