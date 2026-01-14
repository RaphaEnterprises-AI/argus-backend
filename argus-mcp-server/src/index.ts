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
  ARGUS_API_URL: string;  // Worker - browser automation
  ARGUS_BRAIN_URL: string;  // Brain - intelligence
  API_TOKEN?: string;
  ANTHROPIC_API_KEY?: string;
  MCP_OAUTH: DurableObjectNamespace;
}

// Brain API Response types
interface BrainHealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

interface BrainTestCreateResponse {
  success: boolean;
  test: {
    id: string;
    name: string;
    description: string;
    steps: Array<{
      action: string;
      target?: string;
      value?: string;
    }>;
    assertions: Array<{
      type: string;
      target?: string;
      expected?: string;
    }>;
  };
  spec?: string;
}

interface BrainQualityStatsResponse {
  stats: {
    total_events: number;
    events_by_status: Record<string, number>;
    events_by_severity: Record<string, number>;
    total_generated_tests: number;
    tests_by_status: Record<string, number>;
    coverage_rate: number;
  };
}

interface BrainQualityScoreResponse {
  quality_score: number;
  risk_level: string;
  test_coverage: number;
  total_events: number;
  total_tests: number;
  approved_tests: number;
}

interface BrainRiskScoresResponse {
  success: boolean;
  risk_scores: Array<{
    entity_type: string;
    entity_identifier: string;
    overall_score: number;
    factors: Record<string, number>;
    error_count: number;
    affected_users: number;
    trend: string;
  }>;
  total_entities: number;
}

// Sync API Response types
interface SyncPushResponse {
  success: boolean;
  events_pushed: number;
  new_version?: number;
  conflicts?: Array<{
    id: string;
    test_id: string;
    path: string[];
    local_value: unknown;
    remote_value: unknown;
  }>;
  error?: string;
}

interface SyncPullResponse {
  success: boolean;
  events: Array<{
    id: string;
    type: string;
    test_id: string;
    content?: Record<string, unknown>;
    timestamp: string;
  }>;
  new_version?: number;
  error?: string;
}

interface SyncStatusResponse {
  success: boolean;
  project_id: string;
  status: string;
  tests: Record<string, {
    test_id: string;
    status: string;
    local_version: number;
    remote_version: number;
    pending_changes: number;
    conflicts: number;
  }>;
  total_pending: number;
  total_conflicts: number;
}

interface SyncResolveResponse {
  success: boolean;
  resolved: boolean;
  conflict_id: string;
  resolved_value?: unknown;
  error?: string;
}

// Export API Response types
interface ExportResponse {
  success: boolean;
  language: string;
  framework: string;
  code: string;
  filename: string;
  imports?: string[];
  error?: string;
}

interface ExportLanguagesResponse {
  languages: Array<{
    id: string;
    name: string;
    frameworks: string[];
  }>;
}

// Recording API Response types
interface RecordingConvertResponse {
  success: boolean;
  test: {
    id: string;
    name: string;
    source: string;
    steps: Array<{
      action: string;
      target?: string;
      value?: string;
    }>;
    assertions: Array<{
      type: string;
      target?: string;
      expected?: string;
    }>;
  };
  recording_id: string;
  duration_ms: number;
  events_processed: number;
  error?: string;
}

// Collaboration API Response types
interface PresenceResponse {
  success: boolean;
  users: Array<{
    user_id: string;
    user_name: string;
    status: string;
    test_id?: string;
    cursor?: {
      step_index?: number;
      field?: string;
    };
    color: string;
    last_active: string;
  }>;
}

// Production Events Response types
interface ProductionEventsResponse {
  events: Array<{
    id: string;
    title: string;
    message?: string;
    severity: string;
    status: string;
    url?: string;
    component?: string;
    occurrence_count: number;
    affected_users: number;
    source: string;
    first_seen_at: string;
    last_seen_at: string;
    ai_analysis?: {
      generated_test_id?: string;
      confidence_score?: number;
    };
  }>;
}

// Generated Tests Response types
interface GeneratedTestsResponse {
  tests: Array<{
    id: string;
    name: string;
    description?: string;
    status: string;
    framework: string;
    test_code?: string;
    test_file_path?: string;
    confidence_score: number;
    production_event_id?: string;
    created_at: string;
    reviewed_at?: string;
  }>;
}

// Healing Config Response types
interface HealingConfigResponse {
  id: string;
  organization_id: string;
  project_id?: string;
  enabled: boolean;
  auto_apply: boolean;
  min_confidence_auto: number;
  min_confidence_suggest: number;
  heal_selectors: boolean;
  heal_timeouts: boolean;
  heal_text_content: boolean;
  learn_from_success: boolean;
  notify_on_heal: boolean;
  require_approval: boolean;
}

// Healing Patterns Response types
interface HealingPatternsResponse {
  patterns: Array<{
    id: string;
    fingerprint: string;
    original_selector: string;
    healed_selector: string;
    error_type: string;
    success_count: number;
    failure_count: number;
    confidence: number;
    project_id?: string;
    created_at: string;
  }>;
}

// Healing Stats Response types
interface HealingStatsResponse {
  total_patterns: number;
  total_heals_applied: number;
  total_heals_suggested: number;
  success_rate: number;
  top_error_types: Record<string, number>;
  heals_last_24h: number;
  heals_last_7d: number;
  avg_confidence: number;
  recent_heals: Array<{
    id: string;
    original: string;
    healed: string;
    error_type: string;
    confidence: number;
  }>;
}

// Projects Response types
interface ProjectsResponse {
  projects: Array<{
    id: string;
    name: string;
    description?: string;
    app_url?: string;
    organization_id: string;
    created_at: string;
    test_count?: number;
    event_count?: number;
  }>;
}

// Test Run History Response types
interface TestRunHistoryResponse {
  runs: Array<{
    id: string;
    test_id: string;
    status: string;
    duration_ms: number;
    passed_steps: number;
    total_steps: number;
    error_message?: string;
    screenshot_url?: string;
    started_at: string;
    completed_at?: string;
  }>;
}

// What To Test Response types
interface WhatToTestResponse {
  recommendations: Array<{
    priority: string;
    entity: string;
    entity_type: string;
    reason: string;
    risk_score: number;
    affected_users: number;
    suggested_test_description: string;
  }>;
  summary: string;
}

// Coverage Gaps Response types
interface CoverageGapsResponse {
  gaps: Array<{
    entity: string;
    entity_type: string;
    error_count: number;
    has_test: boolean;
    risk_score: number;
    suggestion: string;
  }>;
  coverage_percentage: number;
  total_entities: number;
  tested_entities: number;
}

// Flaky Tests Response types
interface FlakyTestsResponse {
  flaky_tests: Array<{
    test_id: string;
    test_name: string;
    flakiness_score: number;
    pass_rate: number;
    total_runs: number;
    recent_failures: number;
    common_failure_reason?: string;
  }>;
  total_flaky: number;
}

// Schedule Response types
interface ScheduleResponse {
  schedules: Array<{
    id: string;
    name: string;
    cron_expression: string;
    test_ids: string[];
    enabled: boolean;
    last_run_at?: string;
    next_run_at?: string;
    created_at: string;
  }>;
}

// AI Ask Response types
interface AskResponse {
  answer: string;
  sources: Array<{
    type: string;
    id: string;
    relevance: number;
  }>;
  suggestions?: string[];
}

interface CommentsResponse {
  success: boolean;
  comments: Array<{
    id: string;
    test_id: string;
    step_index?: number;
    author_id: string;
    author_name: string;
    content: string;
    mentions: string[];
    resolved: boolean;
    created_at: string;
    replies?: Array<{
      id: string;
      author_id: string;
      author_name: string;
      content: string;
      created_at: string;
    }>;
  }>;
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

// Helper to call Worker API (browser automation)
async function callWorkerAPI<T>(
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
    throw new Error(`Worker API error (${response.status}): ${errorText}`);
  }

  return response.json() as Promise<T>;
}

// Helper to call Brain API (intelligence)
async function callBrainAPI<T>(
  endpoint: string,
  method: "GET" | "POST" | "PUT" | "DELETE" = "POST",
  body?: Record<string, unknown>,
  env?: Env
): Promise<T> {
  const brainUrl = env?.ARGUS_BRAIN_URL || "https://argus-brain-production.up.railway.app";

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  const fetchOptions: RequestInit = {
    method,
    headers,
  };

  if (body && (method === "POST" || method === "PUT")) {
    fetchOptions.body = JSON.stringify(body);
  }

  const response = await fetch(`${brainUrl}${endpoint}`, fetchOptions);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Brain API error (${response.status}): ${errorText}`);
  }

  return response.json() as Promise<T>;
}

// Alias for backward compatibility
const callArgusAPI = callWorkerAPI;

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

    // Tool: argus_generate_test - Generate test from natural language (Brain)
    this.server.tool(
      "argus_generate_test",
      "Generate E2E test steps from a natural language description. Uses AI Brain service to create a comprehensive test plan with steps and assertions.",
      {
        url: z.string().url().describe("The URL of the application to test"),
        description: z.string().describe("Description of what the test should verify (e.g., 'Verify user can log in with valid credentials', 'Test the checkout flow with a product')"),
      },
      async ({ url, description }) => {
        try {
          // Call Brain to generate test from natural language
          const result = await callBrainAPI<BrainTestCreateResponse>(
            "/api/v1/tests/create",
            "POST",
            {
              description,
              app_url: url,
            },
            this.env
          );

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Could not generate test. Try using argus_discover first to understand the page.`,
                },
              ],
              isError: true,
            };
          }

          // Format the generated test
          const stepsText = result.test.steps.map((s, i) =>
            `${i + 1}. ${s.action}${s.target ? ` on "${s.target}"` : ""}${s.value ? ` with "${s.value}"` : ""}`
          ).join("\n");

          const assertionsText = result.test.assertions.map((a, i) =>
            `${i + 1}. ${a.type}${a.target ? ` "${a.target}"` : ""}${a.expected ? ` = "${a.expected}"` : ""}`
          ).join("\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Generated Test: ${result.test.name}\n\n**Target:** ${url}\n**Description:** ${result.test.description}\n\n### Test Steps:\n${stepsText}\n\n### Assertions:\n${assertionsText}\n\n**Tip:** Use \`argus_test\` with these steps to run the test.`,
              },
            ],
          };
        } catch (error) {
          // Fall back to Worker-based discovery if Brain fails
          try {
            const observeResult = await callArgusAPI<ArgusObserveResponse>("/observe", {
              url,
              instruction: "List all interactive elements and their purposes",
            }, this.env);

            const pageContext = observeResult.actions?.map(a => `- ${a.description} (${a.type})`).join("\n") || "No elements found";

            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Test Plan (Discovery Mode)\n\n**Target:** ${url}\n**Objective:** ${description}\n\n### Discovered Page Elements:\n${pageContext}\n\n### Suggested Test Steps:\n1. Navigate to the page\n2. [Create steps based on the discovered elements]\n\n**Note:** Brain service unavailable. Use \`argus_test\` to run manually created steps.`,
                },
              ],
            };
          } catch (fallbackError) {
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
      }
    );

    // Tool: argus_quality_score - Get quality score for a project (Brain)
    this.server.tool(
      "argus_quality_score",
      "Get the overall quality score and metrics for a project. Shows test coverage, risk level, and production error statistics.",
      {
        project_id: z.string().describe("The project UUID to get quality score for"),
      },
      async ({ project_id }) => {
        try {
          const result = await callBrainAPI<BrainQualityScoreResponse>(
            `/api/v1/quality/score?project_id=${project_id}`,
            "GET",
            undefined,
            this.env
          );

          const riskEmoji = result.risk_level === "high" ? "🔴" : result.risk_level === "medium" ? "🟡" : "🟢";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Quality Score: ${result.quality_score}/100\n\n**Risk Level:** ${riskEmoji} ${result.risk_level.toUpperCase()}\n**Test Coverage:** ${result.test_coverage}%\n\n### Metrics:\n- Total Production Events: ${result.total_events}\n- Generated Tests: ${result.total_tests}\n- Approved Tests: ${result.approved_tests}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error getting quality score: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_quality_stats - Get detailed quality statistics (Brain)
    this.server.tool(
      "argus_quality_stats",
      "Get detailed quality intelligence statistics including event counts by status, severity breakdown, and test coverage metrics.",
      {
        project_id: z.string().describe("The project UUID to get statistics for"),
      },
      async ({ project_id }) => {
        try {
          const result = await callBrainAPI<BrainQualityStatsResponse>(
            `/api/v1/quality/stats?project_id=${project_id}`,
            "GET",
            undefined,
            this.env
          );

          const stats = result.stats;

          const statusBreakdown = Object.entries(stats.events_by_status)
            .map(([status, count]) => `- ${status}: ${count}`)
            .join("\n");

          const severityBreakdown = Object.entries(stats.events_by_severity)
            .map(([sev, count]) => `- ${sev}: ${count}`)
            .join("\n");

          const testsBreakdown = Object.entries(stats.tests_by_status)
            .map(([status, count]) => `- ${status}: ${count}`)
            .join("\n") || "No tests generated yet";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Quality Intelligence Statistics\n\n### Production Events: ${stats.total_events}\n\n**By Status:**\n${statusBreakdown}\n\n**By Severity:**\n${severityBreakdown}\n\n### Generated Tests: ${stats.total_generated_tests}\n\n**By Status:**\n${testsBreakdown}\n\n**Coverage Rate:** ${stats.coverage_rate}%`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error getting quality stats: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_risk_scores - Calculate and get risk scores (Brain)
    this.server.tool(
      "argus_risk_scores",
      "Calculate risk scores for components and pages based on production errors. Identifies high-risk areas that need more testing.",
      {
        project_id: z.string().describe("The project UUID to calculate risk scores for"),
        calculate: z.boolean().optional().describe("If true, recalculate scores (default: false, just retrieve)"),
      },
      async ({ project_id, calculate = false }) => {
        try {
          let result: BrainRiskScoresResponse;

          if (calculate) {
            // Recalculate risk scores
            result = await callBrainAPI<BrainRiskScoresResponse>(
              "/api/v1/quality/calculate-risk",
              "POST",
              { project_id },
              this.env
            );
          } else {
            // Just retrieve existing scores
            const response = await callBrainAPI<{ risk_scores: BrainRiskScoresResponse["risk_scores"] }>(
              `/api/v1/quality/risk-scores?project_id=${project_id}`,
              "GET",
              undefined,
              this.env
            );
            result = { success: true, risk_scores: response.risk_scores, total_entities: response.risk_scores.length };
          }

          if (!result.risk_scores || result.risk_scores.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Risk Scores\n\nNo risk data available yet. Production events need to be ingested via webhooks first.`,
                },
              ],
            };
          }

          const scoresText = result.risk_scores.slice(0, 10).map((score, i) => {
            const emoji = score.overall_score > 70 ? "🔴" : score.overall_score > 40 ? "🟡" : "🟢";
            return `${i + 1}. ${emoji} **${score.entity_identifier}** (${score.entity_type})\n   Score: ${score.overall_score}/100 | Errors: ${score.error_count} | Users affected: ${score.affected_users}`;
          }).join("\n\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Risk Scores (Top ${Math.min(10, result.risk_scores.length)} of ${result.total_entities})\n\n${scoresText}\n\n**Legend:** 🔴 High Risk (>70) | 🟡 Medium Risk (40-70) | 🟢 Low Risk (<40)`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error getting risk scores: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // =========================================================================
    // SYNC TOOLS - Two-way IDE synchronization
    // =========================================================================

    // Tool: argus_sync_push - Push local test changes to Argus
    this.server.tool(
      "argus_sync_push",
      "Push local test changes to Argus cloud. Syncs test specifications from your IDE to the Argus platform for team collaboration and cloud execution.",
      {
        project_id: z.string().describe("The project UUID"),
        test_id: z.string().describe("The test UUID to push"),
        content: z.object({
          id: z.string(),
          name: z.string(),
          description: z.string().optional(),
          steps: z.array(z.object({
            action: z.string(),
            target: z.string().optional(),
            value: z.string().optional(),
          })),
          assertions: z.array(z.object({
            type: z.string(),
            target: z.string().optional(),
            expected: z.string().optional(),
          })).optional(),
          metadata: z.record(z.unknown()).optional(),
        }).describe("The test specification to push"),
        local_version: z.number().describe("Local version number for conflict detection"),
      },
      async ({ project_id, test_id, content, local_version }) => {
        try {
          const result = await callBrainAPI<SyncPushResponse>(
            "/api/v1/sync/push",
            "POST",
            {
              project_id,
              test_id,
              content,
              local_version,
              source: "mcp",
            },
            this.env
          );

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Push failed: ${result.error || "Unknown error"}`,
                },
              ],
              isError: true,
            };
          }

          // Check for conflicts
          if (result.conflicts && result.conflicts.length > 0) {
            const conflictsList = result.conflicts.map((c, i) =>
              `${i + 1}. Test: ${c.test_id}\n   Path: ${c.path.join(".")}\n   Local: ${JSON.stringify(c.local_value)}\n   Remote: ${JSON.stringify(c.remote_value)}`
            ).join("\n\n");

            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Sync Conflicts Detected\n\nPushed ${result.events_pushed} events but ${result.conflicts.length} conflicts need resolution:\n\n${conflictsList}\n\n**Use \`argus_sync_resolve\` to resolve conflicts.**`,
                },
              ],
            };
          }

          return {
            content: [
              {
                type: "text" as const,
                text: `## Push Successful\n\n**Test:** ${test_id}\n**Events pushed:** ${result.events_pushed}\n**New version:** ${result.new_version || "N/A"}\n\nTest is now synced with Argus cloud.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error pushing changes: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_sync_pull - Pull remote test changes
    this.server.tool(
      "argus_sync_pull",
      "Pull test changes from Argus cloud to your local IDE. Fetches the latest test specifications and updates from team members.",
      {
        project_id: z.string().describe("The project UUID to pull tests from"),
        since_version: z.number().optional().describe("Only pull changes since this version (default: 0 for all)"),
        test_id: z.string().optional().describe("Pull specific test only (optional)"),
      },
      async ({ project_id, since_version = 0, test_id }) => {
        try {
          const queryParams = new URLSearchParams({
            project_id,
            since_version: since_version.toString(),
          });
          if (test_id) {
            queryParams.set("test_id", test_id);
          }

          const result = await callBrainAPI<SyncPullResponse>(
            `/api/v1/sync/pull?${queryParams}`,
            "GET",
            undefined,
            this.env
          );

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Pull failed: ${result.error || "Unknown error"}`,
                },
              ],
              isError: true,
            };
          }

          if (result.events.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## No New Changes\n\nYour local tests are up to date with Argus cloud.`,
                },
              ],
            };
          }

          // Format events
          const eventsList = result.events.map((e, i) => {
            const icon = e.type.includes("created") ? "+" : e.type.includes("deleted") ? "-" : "~";
            return `${i + 1}. [${icon}] ${e.type} - Test: ${e.test_id}\n   Time: ${e.timestamp}`;
          }).join("\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Pulled ${result.events.length} Changes\n\n**New version:** ${result.new_version || "N/A"}\n\n### Changes:\n${eventsList}\n\n\`\`\`json\n${JSON.stringify(result.events, null, 2)}\n\`\`\``,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error pulling changes: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_sync_status - Get sync status
    this.server.tool(
      "argus_sync_status",
      "Get the synchronization status for a project. Shows pending changes, conflicts, and sync state for all tests.",
      {
        project_id: z.string().describe("The project UUID to check status for"),
      },
      async ({ project_id }) => {
        try {
          const result = await callBrainAPI<SyncStatusResponse>(
            `/api/v1/sync/status/${project_id}`,
            "GET",
            undefined,
            this.env
          );

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Could not get sync status. The project may not exist or sync is not initialized.`,
                },
              ],
              isError: true,
            };
          }

          const statusEmoji = result.status === "synced" ? "✅" : result.status === "pending" ? "🔄" : result.status === "conflict" ? "⚠️" : "❌";

          // Format test statuses
          const testsStatus = Object.values(result.tests).map(t => {
            const icon = t.status === "synced" ? "✅" : t.status === "pending" ? "🔄" : "⚠️";
            return `- ${icon} ${t.test_id}: v${t.local_version} (local) / v${t.remote_version} (remote)${t.pending_changes > 0 ? ` - ${t.pending_changes} pending` : ""}${t.conflicts > 0 ? ` - ${t.conflicts} conflicts` : ""}`;
          }).join("\n") || "No tests tracked";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Sync Status: ${statusEmoji} ${result.status.toUpperCase()}\n\n**Project:** ${project_id}\n**Pending changes:** ${result.total_pending}\n**Conflicts:** ${result.total_conflicts}\n\n### Tests:\n${testsStatus}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error getting sync status: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_sync_resolve - Resolve sync conflicts
    this.server.tool(
      "argus_sync_resolve",
      "Resolve a synchronization conflict between local and remote test versions. Choose to keep local, keep remote, or provide a custom resolution.",
      {
        project_id: z.string().describe("The project UUID"),
        conflict_id: z.string().describe("The conflict ID to resolve"),
        strategy: z.enum(["keep_local", "keep_remote", "merge", "manual"]).describe("Resolution strategy"),
        manual_value: z.unknown().optional().describe("Custom value for manual resolution"),
      },
      async ({ project_id, conflict_id, strategy, manual_value }) => {
        try {
          const result = await callBrainAPI<SyncResolveResponse>(
            "/api/v1/sync/resolve",
            "POST",
            {
              project_id,
              conflict_id,
              strategy,
              manual_value,
            },
            this.env
          );

          if (!result.success || !result.resolved) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Resolution failed: ${result.error || "Could not resolve conflict"}`,
                },
              ],
              isError: true,
            };
          }

          return {
            content: [
              {
                type: "text" as const,
                text: `## Conflict Resolved ✅\n\n**Conflict ID:** ${result.conflict_id}\n**Strategy:** ${strategy}\n**Resolved Value:**\n\`\`\`json\n${JSON.stringify(result.resolved_value, null, 2)}\n\`\`\`\n\nRun \`argus_sync_push\` to sync the resolved changes.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error resolving conflict: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // =========================================================================
    // EXPORT TOOLS - Multi-language test export
    // =========================================================================

    // Tool: argus_export - Export test to multiple languages
    this.server.tool(
      "argus_export",
      "Export an Argus test to executable code in multiple programming languages and frameworks. Supports Python, TypeScript, Java, C#, Ruby, and Go with various testing frameworks.",
      {
        test_id: z.string().describe("The test UUID to export"),
        language: z.enum(["python", "typescript", "java", "csharp", "ruby", "go"]).describe("Target programming language"),
        framework: z.string().describe("Testing framework (e.g., 'playwright', 'selenium', 'cypress', 'puppeteer', 'capybara', 'rod')"),
        options: z.object({
          include_comments: z.boolean().optional().describe("Include explanatory comments"),
          include_assertions: z.boolean().optional().describe("Include assertion code"),
          base_url_variable: z.string().optional().describe("Variable name for base URL"),
          class_name: z.string().optional().describe("Custom test class name"),
        }).optional().describe("Export options"),
      },
      async ({ test_id, language, framework, options = {} }) => {
        try {
          const result = await callBrainAPI<ExportResponse>(
            "/api/v1/export/generate",
            "POST",
            {
              test_id,
              language,
              framework,
              options: {
                include_comments: options.include_comments ?? true,
                include_assertions: options.include_assertions ?? true,
                base_url_variable: options.base_url_variable ?? "BASE_URL",
                class_name: options.class_name,
              },
            },
            this.env
          );

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Export failed: ${result.error || "Unknown error"}`,
                },
              ],
              isError: true,
            };
          }

          const importsSection = result.imports && result.imports.length > 0
            ? `### Required Imports/Dependencies:\n\`\`\`\n${result.imports.join("\n")}\n\`\`\`\n\n`
            : "";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Exported Test: ${result.filename}\n\n**Language:** ${result.language}\n**Framework:** ${result.framework}\n\n${importsSection}### Generated Code:\n\`\`\`${result.language}\n${result.code}\n\`\`\``,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error exporting test: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_export_languages - List supported export languages
    this.server.tool(
      "argus_export_languages",
      "List all supported programming languages and testing frameworks for test export.",
      {},
      async () => {
        try {
          const result = await callBrainAPI<ExportLanguagesResponse>(
            "/api/v1/export/languages",
            "GET",
            undefined,
            this.env
          );

          const languagesList = result.languages.map(l =>
            `### ${l.name} (\`${l.id}\`)\nFrameworks: ${l.frameworks.join(", ")}`
          ).join("\n\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Supported Export Languages\n\n${languagesList}\n\n**Usage:** \`argus_export(test_id, language, framework)\``,
              },
            ],
          };
        } catch (error) {
          // Return static list if API fails
          return {
            content: [
              {
                type: "text" as const,
                text: `## Supported Export Languages\n\n### Python (\`python\`)\nFrameworks: playwright, selenium\n\n### TypeScript (\`typescript\`)\nFrameworks: playwright, puppeteer, cypress\n\n### Java (\`java\`)\nFrameworks: selenium\n\n### C# (\`csharp\`)\nFrameworks: selenium, playwright\n\n### Ruby (\`ruby\`)\nFrameworks: capybara, selenium\n\n### Go (\`go\`)\nFrameworks: rod\n\n**Usage:** \`argus_export(test_id, language, framework)\``,
              },
            ],
          };
        }
      }
    );

    // =========================================================================
    // RECORDING TOOLS - Browser recording to test conversion
    // =========================================================================

    // Tool: argus_recording_to_test - Convert browser recording to test
    this.server.tool(
      "argus_recording_to_test",
      "Convert a browser recording (rrweb format) to an Argus test. Analyzes DOM events from recorded sessions and generates executable test steps. Zero AI cost - pure DOM event parsing.",
      {
        recording: z.object({
          events: z.array(z.object({
            type: z.number(),
            data: z.record(z.unknown()),
            timestamp: z.number(),
          })).describe("rrweb event array"),
          metadata: z.object({
            duration: z.number().optional(),
            startTime: z.string().optional(),
            url: z.string().optional(),
          }).optional(),
        }).describe("The rrweb recording data"),
        options: z.object({
          name: z.string().optional().describe("Test name (auto-generated if not provided)"),
          filter_actions: z.array(z.string()).optional().describe("Only include these action types"),
          min_confidence: z.number().optional().describe("Minimum selector confidence (0-1)"),
          deduplicate: z.boolean().optional().describe("Remove duplicate consecutive actions"),
        }).optional(),
      },
      async ({ recording, options = {} }) => {
        try {
          const result = await callBrainAPI<RecordingConvertResponse>(
            "/api/v1/recording/convert",
            "POST",
            {
              recording,
              options: {
                name: options.name,
                filter_actions: options.filter_actions,
                min_confidence: options.min_confidence ?? 0.7,
                deduplicate: options.deduplicate ?? true,
              },
            },
            this.env
          );

          if (!result.success) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Conversion failed: ${result.error || "Unknown error"}`,
                },
              ],
              isError: true,
            };
          }

          // Format steps
          const stepsText = result.test.steps.map((s, i) =>
            `${i + 1}. ${s.action}${s.target ? ` on "${s.target}"` : ""}${s.value ? ` with "${s.value}"` : ""}`
          ).join("\n");

          const assertionsText = result.test.assertions?.map((a, i) =>
            `${i + 1}. ${a.type}${a.target ? ` "${a.target}"` : ""}${a.expected ? ` = "${a.expected}"` : ""}`
          ).join("\n") || "None generated";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Test Generated from Recording ✨\n\n**Test ID:** ${result.test.id}\n**Name:** ${result.test.name}\n**Source:** ${result.test.source}\n**Recording ID:** ${result.recording_id}\n**Duration:** ${(result.duration_ms / 1000).toFixed(1)}s\n**Events processed:** ${result.events_processed}\n\n### Test Steps (${result.test.steps.length}):\n${stepsText}\n\n### Auto-Generated Assertions:\n${assertionsText}\n\n**Tip:** Use \`argus_test\` to run this test or \`argus_export\` to convert to code.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error converting recording: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_recording_snippet - Get recorder snippet for websites
    this.server.tool(
      "argus_recording_snippet",
      "Generate a JavaScript snippet to record user sessions on any website. The snippet uses rrweb for DOM-based recording that can be converted to tests.",
      {
        project_id: z.string().describe("The project UUID to associate recordings with"),
        options: z.object({
          mask_inputs: z.boolean().optional().describe("Mask sensitive input fields"),
          record_canvas: z.boolean().optional().describe("Record canvas elements"),
          sample_rate: z.number().optional().describe("Sampling rate for mouse movements"),
        }).optional(),
      },
      async ({ project_id, options = {} }) => {
        const snippet = `<!-- Argus Session Recorder -->
<script src="https://cdn.jsdelivr.net/npm/rrweb@latest/dist/rrweb.min.js"></script>
<script>
(function() {
  const events = [];
  const projectId = "${project_id}";

  // Start recording
  rrweb.record({
    emit(event) {
      events.push(event);
    },
    maskAllInputs: ${options.mask_inputs ?? true},
    recordCanvas: ${options.record_canvas ?? false},
    sampling: {
      mousemove: ${options.sample_rate ?? 50}
    }
  });

  // Upload recording on page unload or after 5 minutes
  const uploadRecording = () => {
    if (events.length > 0) {
      navigator.sendBeacon(
        "https://argus-brain-production.up.railway.app/api/v1/recording/upload",
        JSON.stringify({
          project_id: projectId,
          recording: { events, metadata: { url: window.location.href } }
        })
      );
    }
  };

  window.addEventListener("beforeunload", uploadRecording);
  setTimeout(uploadRecording, 300000); // 5 min max

  // Export for manual control
  window.ArgusRecorder = {
    stop: uploadRecording,
    getEvents: () => events
  };
})();
</script>`;

        return {
          content: [
            {
              type: "text" as const,
              text: `## Argus Recording Snippet\n\nAdd this snippet to your website to record user sessions:\n\n\`\`\`html\n${snippet}\n\`\`\`\n\n### Usage:\n1. Add the snippet before \`</body>\`\n2. User sessions are auto-recorded\n3. Use \`argus_recording_to_test\` to convert to tests\n\n### Manual Control:\n- \`window.ArgusRecorder.stop()\` - Stop and upload\n- \`window.ArgusRecorder.getEvents()\` - Get events array`,
            },
          ],
        };
      }
    );

    // =========================================================================
    // COLLABORATION TOOLS - Real-time team collaboration
    // =========================================================================

    // Tool: argus_presence - Get/update user presence
    this.server.tool(
      "argus_presence",
      "Get or update user presence information for real-time collaboration. See who else is viewing or editing tests in your workspace.",
      {
        workspace_id: z.string().describe("The workspace UUID"),
        action: z.enum(["get", "join", "leave", "update"]).describe("Presence action"),
        user_id: z.string().optional().describe("User ID (required for join/leave/update)"),
        user_name: z.string().optional().describe("User display name (required for join)"),
        test_id: z.string().optional().describe("Currently viewed test ID"),
        cursor: z.object({
          step_index: z.number().optional(),
          field: z.string().optional(),
        }).optional().describe("Current cursor position"),
      },
      async ({ workspace_id, action, user_id, user_name, test_id, cursor }) => {
        try {
          if (action === "get") {
            const result = await callBrainAPI<PresenceResponse>(
              `/api/v1/collaboration/presence/${workspace_id}`,
              "GET",
              undefined,
              this.env
            );

            if (result.users.length === 0) {
              return {
                content: [
                  {
                    type: "text" as const,
                    text: `## Workspace Presence\n\nNo other users currently online in this workspace.`,
                  },
                ],
              };
            }

            const usersList = result.users.map(u => {
              const statusIcon = u.status === "online" ? "🟢" : u.status === "idle" ? "🟡" : "⚫";
              const location = u.test_id ? `viewing ${u.test_id}` : "in workspace";
              return `${statusIcon} **${u.user_name}** - ${location}`;
            }).join("\n");

            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Workspace Presence (${result.users.length} online)\n\n${usersList}`,
                },
              ],
            };
          }

          // Join/Leave/Update actions
          const result = await callBrainAPI<{ success: boolean }>(
            "/api/v1/collaboration/presence",
            "POST",
            {
              workspace_id,
              action,
              user_id,
              user_name,
              test_id,
              cursor,
            },
            this.env
          );

          const actionText = action === "join" ? "joined" : action === "leave" ? "left" : "updated presence in";

          return {
            content: [
              {
                type: "text" as const,
                text: `Successfully ${actionText} workspace.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error with presence: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_comments - Manage test comments
    this.server.tool(
      "argus_comments",
      "Get or add comments on tests for team collaboration. Support for threaded discussions, @mentions, and resolution tracking.",
      {
        test_id: z.string().describe("The test UUID"),
        action: z.enum(["get", "add", "reply", "resolve"]).describe("Comment action"),
        comment_id: z.string().optional().describe("Comment ID (for reply/resolve)"),
        content: z.string().optional().describe("Comment content (for add/reply)"),
        step_index: z.number().optional().describe("Step index to attach comment to"),
        mentions: z.array(z.string()).optional().describe("User IDs to mention"),
      },
      async ({ test_id, action, comment_id, content, step_index, mentions }) => {
        try {
          if (action === "get") {
            const result = await callBrainAPI<CommentsResponse>(
              `/api/v1/collaboration/comments/${test_id}`,
              "GET",
              undefined,
              this.env
            );

            if (result.comments.length === 0) {
              return {
                content: [
                  {
                    type: "text" as const,
                    text: `## Test Comments\n\nNo comments on this test yet. Use \`argus_comments\` with action "add" to start a discussion.`,
                  },
                ],
              };
            }

            const commentsList = result.comments.map(c => {
              const resolved = c.resolved ? " ✅" : "";
              const stepRef = c.step_index !== undefined ? ` (Step ${c.step_index + 1})` : "";
              const replies = c.replies && c.replies.length > 0
                ? `\n  └─ ${c.replies.length} replies`
                : "";
              return `- **${c.author_name}**${stepRef}${resolved}: ${c.content}${replies}`;
            }).join("\n");

            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Test Comments (${result.comments.length})\n\n${commentsList}`,
                },
              ],
            };
          }

          // Add/Reply/Resolve actions
          const result = await callBrainAPI<{ success: boolean; comment_id?: string }>(
            "/api/v1/collaboration/comments",
            "POST",
            {
              test_id,
              action,
              comment_id,
              content,
              step_index,
              mentions,
            },
            this.env
          );

          const actionText = action === "add" ? "Comment added" : action === "reply" ? "Reply added" : "Comment resolved";

          return {
            content: [
              {
                type: "text" as const,
                text: `${actionText} successfully.${result.comment_id ? ` (ID: ${result.comment_id})` : ""}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error with comments: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // =========================================================================
    // PRODUCTION EVENTS - Query and manage production errors
    // =========================================================================

    // Tool: argus_events - List and filter production events
    this.server.tool(
      "argus_events",
      "List production errors and events that need test coverage. Filter by severity, status, or time range to find what needs attention. This is your window into what's breaking in production.",
      {
        project_id: z.string().describe("The project UUID"),
        status: z.enum(["new", "analyzing", "test_pending_review", "test_generated", "ignored"]).optional().describe("Filter by status"),
        severity: z.enum(["fatal", "error", "warning"]).optional().describe("Filter by severity"),
        source: z.string().optional().describe("Filter by source (sentry, datadog, etc.)"),
        limit: z.number().min(1).max(100).optional().describe("Max results (default: 20)"),
      },
      async ({ project_id, status, severity, source, limit = 20 }) => {
        try {
          const params = new URLSearchParams({ project_id, limit: limit.toString() });
          if (status) params.set("status", status);
          if (severity) params.set("severity", severity);
          if (source) params.set("source", source);

          const result = await callBrainAPI<ProductionEventsResponse>(
            `/api/v1/quality/events?${params}`,
            "GET",
            undefined,
            this.env
          );

          if (!result.events || result.events.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Production Events\n\nNo events found matching your criteria. Your app is running clean! 🎉\n\n**Tip:** Set up webhooks from Sentry/Datadog to start capturing production errors.`,
                },
              ],
            };
          }

          const eventsList = result.events.map((e, i) => {
            const severityIcon = e.severity === "fatal" ? "🔴" : e.severity === "error" ? "🟠" : "🟡";
            const statusIcon = e.status === "test_generated" ? "✅" : e.status === "test_pending_review" ? "⏳" : "❌";
            return `${i + 1}. ${severityIcon} **${e.title}**\n   Status: ${statusIcon} ${e.status} | Occurrences: ${e.occurrence_count} | Users: ${e.affected_users}\n   Component: ${e.component || "Unknown"} | Source: ${e.source}\n   ID: \`${e.id}\``;
          }).join("\n\n");

          const newCount = result.events.filter(e => e.status === "new").length;
          const summary = newCount > 0 
            ? `⚠️ **${newCount} events need tests!** Use \`argus_test_from_event\` to generate tests.`
            : "All events have been processed.";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Production Events (${result.events.length} found)\n\n${summary}\n\n${eventsList}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error fetching events: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_test_from_event - Generate test from production error
    this.server.tool(
      "argus_test_from_event",
      "Generate an E2E test from a specific production error. The AI analyzes the error context, stack trace, and user action to create a comprehensive test that prevents regression.",
      {
        event_id: z.string().describe("The production event UUID to generate test from"),
        project_id: z.string().describe("The project UUID"),
        framework: z.enum(["playwright", "cypress", "jest"]).optional().describe("Test framework (default: playwright)"),
      },
      async ({ event_id, project_id, framework = "playwright" }) => {
        try {
          const result = await callBrainAPI<{
            success: boolean;
            message: string;
            generated_test?: {
              id: string;
              name: string;
              file_path: string;
              confidence_score: number;
            };
            test_code?: string;
          }>(
            "/api/v1/quality/generate-test",
            "POST",
            {
              production_event_id: event_id,
              project_id,
              framework,
            },
            this.env
          );

          if (!result.success || !result.generated_test) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Test Generation Failed\n\n${result.message || "Could not generate test from this event."}\n\n**Tip:** Make sure the event has enough context (URL, stack trace, component).`,
                },
              ],
              isError: true,
            };
          }

          const confidenceEmoji = result.generated_test.confidence_score > 0.8 ? "🟢" : result.generated_test.confidence_score > 0.6 ? "🟡" : "🔴";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Test Generated Successfully! 🎉\n\n**Name:** ${result.generated_test.name}\n**File:** \`${result.generated_test.file_path}\`\n**Confidence:** ${confidenceEmoji} ${(result.generated_test.confidence_score * 100).toFixed(0)}%\n**Framework:** ${framework}\n\n### Generated Code:\n\`\`\`typescript\n${result.test_code || "// Code available in the dashboard"}\n\`\`\`\n\n**Next Steps:**\n1. Review the test with \`argus_tests\`\n2. Approve with \`argus_test_review\`\n3. Export to your repo with \`argus_export\``,
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

    // Tool: argus_event_triage - AI-powered event triage
    this.server.tool(
      "argus_event_triage",
      "Get AI-powered triage recommendations for production events. Identifies which errors are most critical, suggests groupings, and recommends test priorities.",
      {
        project_id: z.string().describe("The project UUID"),
        limit: z.number().min(1).max(50).optional().describe("Max events to analyze (default: 10)"),
      },
      async ({ project_id, limit = 10 }) => {
        try {
          // Get events and risk scores to provide triage recommendations
          const [eventsResult, riskResult] = await Promise.all([
            callBrainAPI<ProductionEventsResponse>(
              `/api/v1/quality/events?project_id=${project_id}&status=new&limit=${limit}`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<BrainRiskScoresResponse>(
              `/api/v1/quality/risk-scores?project_id=${project_id}&limit=10`,
              "GET",
              undefined,
              this.env
            ),
          ]);

          const events = eventsResult.events || [];
          const riskScores = riskResult.risk_scores || [];

          if (events.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Event Triage\n\n✅ **No untriaged events!** All production errors have been processed.\n\nYour test coverage is keeping production stable.`,
                },
              ],
            };
          }

          // Group events by severity and component
          const fatalEvents = events.filter(e => e.severity === "fatal");
          const errorEvents = events.filter(e => e.severity === "error");
          const warningEvents = events.filter(e => e.severity === "warning");

          // Create triage recommendations
          let triageText = `## Event Triage Report\n\n`;
          triageText += `**${events.length} events need attention**\n\n`;

          if (fatalEvents.length > 0) {
            triageText += `### 🔴 CRITICAL (${fatalEvents.length})\nThese are crashing your app for users!\n`;
            fatalEvents.forEach(e => {
              triageText += `- **${e.title}** - ${e.affected_users} users affected\n  \`${e.id}\`\n`;
            });
            triageText += "\n";
          }

          if (errorEvents.length > 0) {
            triageText += `### 🟠 HIGH PRIORITY (${errorEvents.length})\nThese are causing errors but not crashes.\n`;
            errorEvents.slice(0, 5).forEach(e => {
              triageText += `- **${e.title}** - ${e.occurrence_count} occurrences\n`;
            });
            triageText += "\n";
          }

          if (warningEvents.length > 0) {
            triageText += `### 🟡 MEDIUM PRIORITY (${warningEvents.length})\nWarnings that might become errors.\n`;
          }

          // Add risk context
          if (riskScores.length > 0) {
            const topRisk = riskScores[0];
            triageText += `\n### 🎯 Recommended Focus\nHighest risk area: **${topRisk.entity_identifier}** (Risk: ${topRisk.overall_score}/100)\n`;
          }

          triageText += `\n**Quick Actions:**\n`;
          triageText += `- Generate tests: \`argus_test_from_event(event_id, project_id)\`\n`;
          triageText += `- Batch generate: \`argus_batch_generate(project_id)\`\n`;

          return {
            content: [
              {
                type: "text" as const,
                text: triageText,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error triaging events: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // =========================================================================
    // SELF-HEALING - Configuration and pattern management
    // =========================================================================

    // Tool: argus_healing_config - Get/update self-healing configuration
    this.server.tool(
      "argus_healing_config",
      "View or update self-healing configuration. Control how Argus automatically fixes broken selectors, handles timeouts, and learns from patterns.",
      {
        organization_id: z.string().describe("The organization UUID"),
        project_id: z.string().optional().describe("Project-specific config (optional)"),
        action: z.enum(["get", "update"]).describe("Get or update configuration"),
        config: z.object({
          enabled: z.boolean().optional(),
          auto_apply: z.boolean().optional(),
          min_confidence_auto: z.number().optional(),
          heal_selectors: z.boolean().optional(),
          heal_timeouts: z.boolean().optional(),
          learn_from_success: z.boolean().optional(),
          notify_on_heal: z.boolean().optional(),
          require_approval: z.boolean().optional(),
        }).optional().describe("Configuration updates (for update action)"),
      },
      async ({ organization_id, project_id, action, config }) => {
        try {
          if (action === "get") {
            const params = project_id ? `?project_id=${project_id}` : "";
            const result = await callBrainAPI<HealingConfigResponse>(
              `/api/v1/healing/organizations/${organization_id}/config${params}`,
              "GET",
              undefined,
              this.env
            );

            const statusEmoji = result.enabled ? "🟢 Enabled" : "🔴 Disabled";
            const autoApplyEmoji = result.auto_apply ? "✅ Auto" : "👤 Manual";

            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Self-Healing Configuration\n\n**Status:** ${statusEmoji}\n**Application:** ${autoApplyEmoji}\n\n### Settings:\n- Min confidence (auto): ${(result.min_confidence_auto * 100).toFixed(0)}%\n- Min confidence (suggest): ${(result.min_confidence_suggest * 100).toFixed(0)}%\n- Heal selectors: ${result.heal_selectors ? "Yes" : "No"}\n- Heal timeouts: ${result.heal_timeouts ? "Yes" : "No"}\n- Heal text content: ${result.heal_text_content ? "Yes" : "No"}\n- Learn from success: ${result.learn_from_success ? "Yes" : "No"}\n\n### Notifications:\n- Notify on heal: ${result.notify_on_heal ? "Yes" : "No"}\n- Require approval: ${result.require_approval ? "Yes" : "No"}`,
                },
              ],
            };
          }

          // Update config - Backend uses PUT method
          const result = await callBrainAPI<HealingConfigResponse>(
            `/api/v1/healing/organizations/${organization_id}/config${project_id ? `?project_id=${project_id}` : ""}`,
            "PUT",
            config || {},
            this.env
          );

          return {
            content: [
              {
                type: "text" as const,
                text: `## Configuration Updated ✅\n\nSelf-healing is now ${result.enabled ? "enabled" : "disabled"}.\n\nChanges will take effect immediately for new test runs.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error with healing config: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_healing_patterns - List learned healing patterns
    this.server.tool(
      "argus_healing_patterns",
      "View learned self-healing patterns. These are selector fixes that Argus has learned from past test runs and can apply automatically.",
      {
        organization_id: z.string().describe("The organization UUID"),
        project_id: z.string().optional().describe("Filter by project"),
        min_confidence: z.number().min(0).max(1).optional().describe("Minimum confidence filter (default: 0.5)"),
        limit: z.number().min(1).max(100).optional().describe("Max patterns (default: 20)"),
      },
      async ({ organization_id, project_id, min_confidence = 0.5, limit = 20 }) => {
        try {
          const params = new URLSearchParams({
            min_confidence: min_confidence.toString(),
            limit: limit.toString(),
          });
          if (project_id) params.set("project_id", project_id);

          // Backend returns array directly, not wrapped
          const result = await callBrainAPI<HealingPatternsResponse["patterns"]>(
            `/api/v1/healing/organizations/${organization_id}/patterns?${params}`,
            "GET",
            undefined,
            this.env
          );

          // Handle both array response and wrapped response
          const patterns = Array.isArray(result) ? result : (result as unknown as { patterns: HealingPatternsResponse["patterns"] }).patterns || [];

          if (patterns.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Healing Patterns\n\nNo patterns learned yet. As tests run and selectors break, Argus will learn how to fix them automatically.\n\n**Tip:** Run tests with self-healing enabled to start building patterns.`,
                },
              ],
            };
          }

          const patternsList = patterns.map((p, i) => {
            const confidenceEmoji = p.confidence > 0.9 ? "🟢" : p.confidence > 0.7 ? "🟡" : "🔴";
            const successRate = p.success_count / (p.success_count + p.failure_count) * 100;
            return `${i + 1}. ${confidenceEmoji} **${p.error_type}** (${(p.confidence * 100).toFixed(0)}% confident)\n   From: \`${p.original_selector.slice(0, 40)}...\`\n   To: \`${p.healed_selector.slice(0, 40)}...\`\n   Success: ${successRate.toFixed(0)}% (${p.success_count}/${p.success_count + p.failure_count})`;
          }).join("\n\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Healing Patterns (${patterns.length} learned)\n\n${patternsList}\n\n**Legend:** 🟢 High confidence | 🟡 Medium | 🔴 Low`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error fetching patterns: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_healing_stats - Get healing statistics
    this.server.tool(
      "argus_healing_stats",
      "Get comprehensive self-healing statistics. See how many tests have been healed, success rates, and trends over time.",
      {
        organization_id: z.string().describe("The organization UUID"),
        project_id: z.string().optional().describe("Filter by project"),
      },
      async ({ organization_id, project_id }) => {
        try {
          const params = project_id ? `?project_id=${project_id}` : "";
          const result = await callBrainAPI<HealingStatsResponse>(
            `/api/v1/healing/organizations/${organization_id}/stats${params}`,
            "GET",
            undefined,
            this.env
          );

          const successEmoji = result.success_rate > 80 ? "🟢" : result.success_rate > 50 ? "🟡" : "🔴";

          let errorTypesText = "";
          if (Object.keys(result.top_error_types).length > 0) {
            errorTypesText = "\n### Top Error Types:\n" +
              Object.entries(result.top_error_types)
                .map(([type, count]) => `- ${type}: ${count}`)
                .join("\n");
          }

          let recentHealsText = "";
          if (result.recent_heals && result.recent_heals.length > 0) {
            recentHealsText = "\n### Recent Heals:\n" +
              result.recent_heals.slice(0, 5)
                .map(h => `- ${h.error_type}: \`${h.original.slice(0, 30)}...\` → \`${h.healed.slice(0, 30)}...\``)
                .join("\n");
          }

          return {
            content: [
              {
                type: "text" as const,
                text: `## Self-Healing Statistics\n\n### Overview:\n- **Total Patterns:** ${result.total_patterns}\n- **Total Heals Applied:** ${result.total_heals_applied}\n- **Success Rate:** ${successEmoji} ${result.success_rate.toFixed(1)}%\n- **Avg Confidence:** ${(result.avg_confidence * 100).toFixed(0)}%\n\n### Activity:\n- Last 24 hours: ${result.heals_last_24h} heals\n- Last 7 days: ${result.heals_last_7d} heals${errorTypesText}${recentHealsText}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error fetching stats: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_healing_review - Approve or reject healing suggestions
    this.server.tool(
      "argus_healing_review",
      "Review and approve/reject healing suggestions. When require_approval is enabled, heals wait for human review before being applied.",
      {
        organization_id: z.string().describe("The organization UUID"),
        pattern_id: z.string().describe("The pattern UUID to review"),
        action: z.enum(["approve", "reject"]).describe("Approve or reject the healing"),
      },
      async ({ organization_id, pattern_id, action }) => {
        try {
          const endpoint = action === "approve" ? "approve" : "reject";
          const result = await callBrainAPI<{ success: boolean; message: string }>(
            `/api/v1/healing/organizations/${organization_id}/${endpoint}/${pattern_id}`,
            "POST",
            {},
            this.env
          );

          const emoji = action === "approve" ? "✅" : "❌";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Healing ${action === "approve" ? "Approved" : "Rejected"} ${emoji}\n\n${result.message}\n\n${action === "approve" 
                  ? "This pattern will now be applied automatically in future test runs."
                  : "This pattern has been marked as unreliable and won't be used."}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error reviewing healing: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // =========================================================================
    // PROJECT MANAGEMENT - List and switch contexts
    // =========================================================================

    // Tool: argus_projects - List all projects
    this.server.tool(
      "argus_projects",
      "List all projects you have access to. Projects organize tests, events, and configurations for different applications.",
      {
        organization_id: z.string().optional().describe("Filter by organization (optional)"),
      },
      async ({ organization_id }) => {
        try {
          const params = organization_id ? `?organization_id=${organization_id}` : "";
          // Backend returns array directly, not wrapped in { projects: [...] }
          const result = await callBrainAPI<ProjectsResponse["projects"]>(
            `/api/v1/projects${params}`,
            "GET",
            undefined,
            this.env
          );

          // Handle both array response and wrapped response
          const projects = Array.isArray(result) ? result : (result as unknown as ProjectsResponse).projects || [];

          if (projects.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Projects\n\nNo projects found. Create a project in the Argus dashboard to get started.\n\n**Tip:** Each project typically maps to one application or microservice.`,
                },
              ],
            };
          }

          const projectsList = projects.map((p, i) => {
            return `${i + 1}. **${p.name}**\n   ${p.description || "No description"}\n   URL: ${p.app_url || "Not set"}\n   Tests: ${p.test_count || 0} | Events: ${p.event_count || 0}\n   ID: \`${p.id}\``;
          }).join("\n\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Your Projects (${projects.length})\n\n${projectsList}\n\n**Tip:** Use the project ID with other commands like \`argus_events\`, \`argus_tests\`, etc.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error fetching projects: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // =========================================================================
    // TEST MANAGEMENT - List, review, and manage tests
    // =========================================================================

    // Tool: argus_tests - List tests with filters
    this.server.tool(
      "argus_tests",
      "List generated tests for a project. Filter by status to find tests pending review, approved tests, or rejected ones.",
      {
        project_id: z.string().describe("The project UUID"),
        status: z.enum(["pending", "approved", "rejected", "modified"]).optional().describe("Filter by review status"),
        limit: z.number().min(1).max(100).optional().describe("Max results (default: 20)"),
      },
      async ({ project_id, status, limit = 20 }) => {
        try {
          const params = new URLSearchParams({ project_id, limit: limit.toString() });
          if (status) params.set("status", status);

          const result = await callBrainAPI<GeneratedTestsResponse>(
            `/api/v1/quality/generated-tests?${params}`,
            "GET",
            undefined,
            this.env
          );

          const tests = result.tests || [];

          if (tests.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Generated Tests\n\nNo tests found${status ? ` with status "${status}"` : ""}.\n\n**Tip:** Use \`argus_test_from_event\` or \`argus_generate_test\` to create tests from production errors.`,
                },
              ],
            };
          }

          const pendingCount = tests.filter(t => t.status === "pending").length;
          const statusBanner = pendingCount > 0 
            ? `⚠️ **${pendingCount} tests need review!**\n\n`
            : "";

          const testsList = tests.map((t, i) => {
            const statusIcon = t.status === "approved" ? "✅" : t.status === "pending" ? "⏳" : t.status === "rejected" ? "❌" : "✏️";
            const confidenceEmoji = t.confidence_score > 0.8 ? "🟢" : t.confidence_score > 0.6 ? "🟡" : "🔴";
            return `${i + 1}. ${statusIcon} **${t.name}**\n   Confidence: ${confidenceEmoji} ${(t.confidence_score * 100).toFixed(0)}% | Framework: ${t.framework}\n   File: \`${t.test_file_path || "Not generated"}\`\n   ID: \`${t.id}\``;
          }).join("\n\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Generated Tests (${tests.length})\n\n${statusBanner}${testsList}\n\n**Actions:**\n- Review: \`argus_test_review(test_id, action)\`\n- Export: \`argus_export(test_id, language, framework)\``,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error fetching tests: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_test_review - Approve/reject/modify generated tests
    this.server.tool(
      "argus_test_review",
      "Review a generated test - approve it for use, reject it, or modify the code. Approved tests can be exported and run in CI.",
      {
        test_id: z.string().describe("The generated test UUID"),
        action: z.enum(["approve", "reject", "modify"]).describe("Review action"),
        review_notes: z.string().optional().describe("Notes about your review decision"),
        modified_code: z.string().optional().describe("Modified test code (for modify action)"),
      },
      async ({ test_id, action, review_notes, modified_code }) => {
        try {
          const result = await callBrainAPI<{ success: boolean; message: string }>(
            "/api/v1/quality/update-test",
            "POST",
            {
              test_id,
              action,
              review_notes,
              modified_code,
            },
            this.env
          );

          const emoji = action === "approve" ? "✅" : action === "reject" ? "❌" : "✏️";
          const actionText = action === "approve" ? "approved" : action === "reject" ? "rejected" : "modified";

          return {
            content: [
              {
                type: "text" as const,
                text: `## Test ${actionText.charAt(0).toUpperCase() + actionText.slice(1)} ${emoji}\n\n${result.message}\n\n${action === "approve" 
                  ? "**Next:** Export this test with `argus_export` to add it to your test suite."
                  : action === "modify"
                  ? "Your changes have been saved. Review again when ready to approve."
                  : "This test won't be used. Consider regenerating with more context."}`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error reviewing test: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // =========================================================================
    // SMART INSIGHTS - AI-powered recommendations
    // =========================================================================

    // Tool: argus_what_to_test - AI recommendations on what to test
    this.server.tool(
      "argus_what_to_test",
      "Get AI-powered recommendations on what to test next. Analyzes risk scores, recent errors, and coverage gaps to prioritize testing efforts.",
      {
        project_id: z.string().describe("The project UUID"),
      },
      async ({ project_id }) => {
        try {
          // Gather data for recommendations
          const [eventsResult, riskResult, statsResult] = await Promise.all([
            callBrainAPI<ProductionEventsResponse>(
              `/api/v1/quality/events?project_id=${project_id}&status=new&limit=10`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<BrainRiskScoresResponse>(
              `/api/v1/quality/risk-scores?project_id=${project_id}&limit=5`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<BrainQualityStatsResponse>(
              `/api/v1/quality/stats?project_id=${project_id}`,
              "GET",
              undefined,
              this.env
            ),
          ]);

          const events = eventsResult.events || [];
          const riskScores = riskResult.risk_scores || [];
          const stats = statsResult.stats;

          let recommendations = `## What to Test Next 🎯\n\n`;

          // Priority 1: High-risk areas without tests
          if (riskScores.length > 0) {
            const highRisk = riskScores.filter(r => r.overall_score > 70);
            if (highRisk.length > 0) {
              recommendations += `### 🔴 Critical Priority\nHigh-risk areas that need immediate test coverage:\n\n`;
              highRisk.forEach(r => {
                recommendations += `- **${r.entity_identifier}** (${r.entity_type})\n  Risk: ${r.overall_score}/100 | Errors: ${r.error_count} | Users: ${r.affected_users}\n`;
              });
              recommendations += "\n";
            }
          }

          // Priority 2: Recent production errors
          if (events.length > 0) {
            const fatalErrors = events.filter(e => e.severity === "fatal");
            const recentErrors = events.slice(0, 5);
            
            if (fatalErrors.length > 0) {
              recommendations += `### 🟠 Fatal Errors (${fatalErrors.length})\nThese are crashing your app:\n\n`;
              fatalErrors.forEach(e => {
                recommendations += `- **${e.title}**\n  ${e.affected_users} users affected | \`${e.id}\`\n`;
              });
              recommendations += "\n";
            } else if (recentErrors.length > 0) {
              recommendations += `### 🟡 Recent Errors (${recentErrors.length})\nNew errors that need test coverage:\n\n`;
              recentErrors.forEach(e => {
                recommendations += `- **${e.title}** (${e.severity})\n`;
              });
              recommendations += "\n";
            }
          }

          // Summary with action items
          recommendations += `### 📊 Coverage Summary\n`;
          recommendations += `- Total events: ${stats.total_events}\n`;
          recommendations += `- Tests generated: ${stats.total_generated_tests}\n`;
          recommendations += `- Coverage rate: ${stats.coverage_rate}%\n\n`;

          recommendations += `### 💡 Quick Actions\n`;
          if (events.length > 0) {
            recommendations += `- Generate tests for all new events: Use \`argus_batch_generate\`\n`;
          }
          if (stats.coverage_rate < 50) {
            recommendations += `- Low coverage! Focus on high-risk areas first\n`;
          }
          if (stats.tests_by_status && stats.tests_by_status["pending"] > 0) {
            recommendations += `- Review ${stats.tests_by_status["pending"]} pending tests with \`argus_tests\`\n`;
          }

          return {
            content: [
              {
                type: "text" as const,
                text: recommendations,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error getting recommendations: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_coverage_gaps - Find untested areas
    this.server.tool(
      "argus_coverage_gaps",
      "Identify areas of your application that have production errors but no test coverage. These are your blind spots.",
      {
        project_id: z.string().describe("The project UUID"),
      },
      async ({ project_id }) => {
        try {
          const [eventsResult, testsResult] = await Promise.all([
            callBrainAPI<ProductionEventsResponse>(
              `/api/v1/quality/events?project_id=${project_id}&limit=100`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<GeneratedTestsResponse>(
              `/api/v1/quality/generated-tests?project_id=${project_id}&status=approved&limit=100`,
              "GET",
              undefined,
              this.env
            ),
          ]);

          const events = eventsResult.events || [];
          const tests = testsResult.tests || [];

          // Find components with errors but no tests
          const testedEventIds = new Set(tests.map(t => t.production_event_id).filter(Boolean));
          const untestedEvents = events.filter(e => !testedEventIds.has(e.id));

          // Group by component
          const componentGaps: Record<string, { errors: number; severity: string }> = {};
          untestedEvents.forEach(e => {
            const component = e.component || e.url || "Unknown";
            if (!componentGaps[component]) {
              componentGaps[component] = { errors: 0, severity: "warning" };
            }
            componentGaps[component].errors++;
            if (e.severity === "fatal" || (e.severity === "error" && componentGaps[component].severity === "warning")) {
              componentGaps[component].severity = e.severity;
            }
          });

          const coveragePercent = events.length > 0 
            ? ((events.length - untestedEvents.length) / events.length * 100).toFixed(1)
            : "100";

          if (Object.keys(componentGaps).length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `## Coverage Gaps\n\n✅ **No gaps found!** All known production errors have test coverage.\n\nCoverage: ${coveragePercent}%`,
                },
              ],
            };
          }

          const gapsList = Object.entries(componentGaps)
            .sort((a, b) => b[1].errors - a[1].errors)
            .map(([component, data], i) => {
              const icon = data.severity === "fatal" ? "🔴" : data.severity === "error" ? "🟠" : "🟡";
              return `${i + 1}. ${icon} **${component}**\n   ${data.errors} untested error${data.errors > 1 ? "s" : ""} (${data.severity})`;
            })
            .join("\n\n");

          return {
            content: [
              {
                type: "text" as const,
                text: `## Coverage Gaps Found! ⚠️\n\n**Coverage:** ${coveragePercent}% (${events.length - untestedEvents.length}/${events.length} events)\n\n### Untested Areas:\n\n${gapsList}\n\n**Action:** Generate tests with \`argus_test_from_event\` for critical gaps.`,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error finding gaps: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_batch_generate - Generate tests for multiple events
    this.server.tool(
      "argus_batch_generate",
      "Generate tests for multiple production events at once. Perfect for catching up on test coverage after connecting a new error source.",
      {
        project_id: z.string().describe("The project UUID"),
        status: z.enum(["new", "analyzing"]).optional().describe("Process events with this status (default: new)"),
        limit: z.number().min(1).max(50).optional().describe("Max events to process (default: 10)"),
        framework: z.enum(["playwright", "cypress", "jest"]).optional().describe("Test framework (default: playwright)"),
      },
      async ({ project_id, status = "new", limit = 10, framework = "playwright" }) => {
        try {
          const result = await callBrainAPI<{
            success: boolean;
            message: string;
            job_id: string;
            results: Array<{
              event_id: string;
              success: boolean;
              test_id?: string;
              error?: string;
            }>;
          }>(
            "/api/v1/quality/batch-generate",
            "POST",
            {
              project_id,
              status,
              limit,
              framework,
            },
            this.env
          );

          const successCount = result.results.filter(r => r.success).length;
          const failCount = result.results.filter(r => !r.success).length;

          let resultText = `## Batch Generation Complete\n\n`;
          resultText += `**Generated:** ${successCount}/${result.results.length} tests\n`;
          resultText += `**Framework:** ${framework}\n`;
          resultText += `**Job ID:** \`${result.job_id}\`\n\n`;

          if (successCount > 0) {
            resultText += `### ✅ Successful (${successCount})\n`;
            result.results.filter(r => r.success).slice(0, 5).forEach(r => {
              resultText += `- Event \`${r.event_id.slice(0, 8)}...\` → Test \`${r.test_id?.slice(0, 8)}...\`\n`;
            });
            if (successCount > 5) resultText += `- ... and ${successCount - 5} more\n`;
            resultText += "\n";
          }

          if (failCount > 0) {
            resultText += `### ❌ Failed (${failCount})\n`;
            result.results.filter(r => !r.success).slice(0, 3).forEach(r => {
              resultText += `- Event \`${r.event_id.slice(0, 8)}...\`: ${r.error}\n`;
            });
            resultText += "\n";
          }

          resultText += `**Next:** Review generated tests with \`argus_tests(project_id, status="pending")\``;

          return {
            content: [
              {
                type: "text" as const,
                text: resultText,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error in batch generation: ${error instanceof Error ? error.message : "Unknown error"}`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_ask - Conversational AI for testing questions
    this.server.tool(
      "argus_ask",
      "Ask any question about your tests, errors, or testing strategy. The AI will analyze your data and provide insights.",
      {
        question: z.string().describe("Your question about testing"),
        project_id: z.string().optional().describe("Project context (optional)"),
      },
      async ({ question, project_id }) => {
        try {
          // Gather context if project is specified
          let context = "";
          if (project_id) {
            const [statsResult, eventsResult] = await Promise.all([
              callBrainAPI<BrainQualityStatsResponse>(
                `/api/v1/quality/stats?project_id=${project_id}`,
                "GET",
                undefined,
                this.env
              ),
              callBrainAPI<ProductionEventsResponse>(
                `/api/v1/quality/events?project_id=${project_id}&limit=5`,
                "GET",
                undefined,
                this.env
              ),
            ]);

            context = `
Project Stats:
- Total events: ${statsResult.stats.total_events}
- Coverage rate: ${statsResult.stats.coverage_rate}%
- Tests generated: ${statsResult.stats.total_generated_tests}

Recent events: ${eventsResult.events?.map(e => e.title).join(", ") || "None"}
`;
          }

          // For now, provide helpful responses based on common questions
          const lowerQuestion = question.toLowerCase();
          let answer = "";

          if (lowerQuestion.includes("start") || lowerQuestion.includes("begin") || lowerQuestion.includes("how to")) {
            answer = `## Getting Started with Argus\n\n1. **Connect error sources** - Set up webhooks from Sentry, Datadog, etc.\n2. **View events** - Use \`argus_events(project_id)\` to see production errors\n3. **Generate tests** - Use \`argus_test_from_event\` or \`argus_batch_generate\`\n4. **Review & approve** - Use \`argus_tests\` and \`argus_test_review\`\n5. **Export** - Use \`argus_export\` to add tests to your repo`;
          } else if (lowerQuestion.includes("coverage") || lowerQuestion.includes("gap")) {
            answer = `## Test Coverage\n\nUse \`argus_coverage_gaps(project_id)\` to find untested areas.\n\n${context ? `Your current coverage: ${context}` : "Specify a project_id for specific coverage data."}`;
          } else if (lowerQuestion.includes("heal") || lowerQuestion.includes("self-healing") || lowerQuestion.includes("selector")) {
            answer = `## Self-Healing\n\nArgus automatically learns to fix broken selectors.\n\n- View config: \`argus_healing_config(org_id, "get")\`\n- See patterns: \`argus_healing_patterns(org_id)\`\n- View stats: \`argus_healing_stats(org_id)\``;
          } else if (lowerQuestion.includes("export") || lowerQuestion.includes("playwright") || lowerQuestion.includes("cypress")) {
            answer = `## Exporting Tests\n\nExport generated tests to code:\n\n\`\`\`\nargus_export(test_id, "typescript", "playwright")\nargus_export(test_id, "python", "selenium")\n\`\`\`\n\nSupported: Python, TypeScript, Java, C#, Ruby, Go`;
          } else if (lowerQuestion.includes("risk") || lowerQuestion.includes("priorit")) {
            answer = `## Risk Prioritization\n\nArgus calculates risk scores based on:\n- Error frequency & severity\n- User impact\n- Test coverage\n- Recency\n\nUse \`argus_risk_scores(project_id)\` to see high-risk areas.\nUse \`argus_what_to_test(project_id)\` for prioritized recommendations.`;
          } else {
            answer = `## Argus Help\n\nHere are some things I can help with:\n\n**Production Events:**\n- \`argus_events\` - View errors\n- \`argus_event_triage\` - AI triage\n- \`argus_test_from_event\` - Generate test\n\n**Test Management:**\n- \`argus_tests\` - List tests\n- \`argus_test_review\` - Approve/reject\n- \`argus_export\` - Export to code\n\n**Insights:**\n- \`argus_what_to_test\` - Recommendations\n- \`argus_coverage_gaps\` - Find gaps\n- \`argus_risk_scores\` - Risk analysis\n\n**Self-Healing:**\n- \`argus_healing_config\` - Configuration\n- \`argus_healing_patterns\` - Learned fixes\n- \`argus_healing_stats\` - Statistics`;
          }

          return {
            content: [
              {
                type: "text" as const,
                text: answer,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `I encountered an error processing your question. Please try rephrasing or use specific tool commands.`,
              },
            ],
            isError: true,
          };
        }
      }
    );

    // Tool: argus_dashboard - Get a quick overview of everything
    this.server.tool(
      "argus_dashboard",
      "Get a comprehensive dashboard view of your testing status. Shows quality score, recent events, pending tests, and actionable insights all in one place.",
      {
        project_id: z.string().describe("The project UUID"),
      },
      async ({ project_id }) => {
        try {
          // Fetch all relevant data in parallel
          const [scoreResult, statsResult, eventsResult, testsResult, riskResult] = await Promise.all([
            callBrainAPI<BrainQualityScoreResponse>(
              `/api/v1/quality/score?project_id=${project_id}`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<BrainQualityStatsResponse>(
              `/api/v1/quality/stats?project_id=${project_id}`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<ProductionEventsResponse>(
              `/api/v1/quality/events?project_id=${project_id}&limit=5`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<GeneratedTestsResponse>(
              `/api/v1/quality/generated-tests?project_id=${project_id}&status=pending&limit=5`,
              "GET",
              undefined,
              this.env
            ),
            callBrainAPI<BrainRiskScoresResponse>(
              `/api/v1/quality/risk-scores?project_id=${project_id}&limit=3`,
              "GET",
              undefined,
              this.env
            ),
          ]);

          const stats = statsResult.stats;
          const events = eventsResult.events || [];
          const tests = testsResult.tests || [];
          const risks = riskResult.risk_scores || [];

          // Build dashboard
          const scoreEmoji = scoreResult.quality_score >= 80 ? "🟢" : scoreResult.quality_score >= 50 ? "🟡" : "🔴";
          const riskEmoji = scoreResult.risk_level === "low" ? "🟢" : scoreResult.risk_level === "medium" ? "🟡" : "🔴";

          let dashboard = `# Argus Dashboard 📊\n\n`;
          
          // Quality Score Section
          dashboard += `## Quality Score: ${scoreEmoji} ${scoreResult.quality_score}/100\n\n`;
          dashboard += `| Metric | Value |\n|--------|-------|\n`;
          dashboard += `| Risk Level | ${riskEmoji} ${scoreResult.risk_level.toUpperCase()} |\n`;
          dashboard += `| Test Coverage | ${scoreResult.test_coverage}% |\n`;
          dashboard += `| Production Events | ${scoreResult.total_events} |\n`;
          dashboard += `| Generated Tests | ${scoreResult.total_tests} |\n`;
          dashboard += `| Approved Tests | ${scoreResult.approved_tests} |\n\n`;

          // Alerts Section
          const newEvents = events.filter(e => e.status === "new").length;
          const pendingTests = tests.length;
          
          if (newEvents > 0 || pendingTests > 0) {
            dashboard += `## ⚠️ Needs Attention\n\n`;
            if (newEvents > 0) {
              dashboard += `- **${newEvents} new production errors** need tests\n`;
            }
            if (pendingTests > 0) {
              dashboard += `- **${pendingTests} tests pending review**\n`;
            }
            dashboard += "\n";
          }

          // High Risk Areas
          if (risks.length > 0) {
            dashboard += `## 🎯 High Risk Areas\n\n`;
            risks.forEach(r => {
              const emoji = r.overall_score > 70 ? "🔴" : r.overall_score > 40 ? "🟡" : "🟢";
              dashboard += `- ${emoji} **${r.entity_identifier}**: ${r.overall_score}/100 risk\n`;
            });
            dashboard += "\n";
          }

          // Recent Events
          if (events.length > 0) {
            dashboard += `## 📋 Recent Events\n\n`;
            events.slice(0, 3).forEach(e => {
              const icon = e.severity === "fatal" ? "🔴" : e.severity === "error" ? "🟠" : "🟡";
              dashboard += `- ${icon} ${e.title}\n`;
            });
            dashboard += "\n";
          }

          // Quick Actions
          dashboard += `## ⚡ Quick Actions\n\n`;
          dashboard += `\`\`\`\n`;
          dashboard += `argus_events("${project_id}")           # View all events\n`;
          dashboard += `argus_what_to_test("${project_id}")     # Get recommendations\n`;
          dashboard += `argus_batch_generate("${project_id}")   # Generate tests\n`;
          dashboard += `argus_tests("${project_id}")            # Review tests\n`;
          dashboard += `\`\`\``;

          return {
            content: [
              {
                type: "text" as const,
                text: dashboard,
              },
            ],
          };
        } catch (error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Error loading dashboard: ${error instanceof Error ? error.message : "Unknown error"}`,
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
        version: "3.0.0",
        description: "Model Context Protocol server for Argus E2E Testing Agent - Full IDE Integration with Next-Gen AI Testing Intelligence",
        endpoint: "/sse",
        tools: {
          core: [
            "argus_health",
            "argus_discover",
            "argus_act",
            "argus_test",
            "argus_extract",
            "argus_agent",
            "argus_generate_test",
          ],
          quality_intelligence: [
            "argus_quality_score",
            "argus_quality_stats",
            "argus_risk_scores",
          ],
          production_events: [
            "argus_events",
            "argus_event_triage",
            "argus_test_from_event",
            "argus_batch_generate",
          ],
          test_management: [
            "argus_tests",
            "argus_test_review",
          ],
          self_healing: [
            "argus_healing_config",
            "argus_healing_patterns",
            "argus_healing_stats",
            "argus_healing_review",
          ],
          smart_insights: [
            "argus_what_to_test",
            "argus_coverage_gaps",
            "argus_dashboard",
            "argus_ask",
          ],
          projects: [
            "argus_projects",
          ],
          sync: [
            "argus_sync_push",
            "argus_sync_pull",
            "argus_sync_status",
            "argus_sync_resolve",
          ],
          export: [
            "argus_export",
            "argus_export_languages",
          ],
          recording: [
            "argus_recording_to_test",
            "argus_recording_snippet",
          ],
          collaboration: [
            "argus_presence",
            "argus_comments",
          ],
        },
        total_tools: 32,
        documentation: "https://github.com/raphaenterprises-ai/argus-e2e-testing-agent",
      });
    }

    return new Response("Not found", { status: 404 });
  },
};
