import { z } from "zod";
import type { ArgusClient } from "../argus-client.js";
import type { ToolDef } from "../server.js";

const NO_KEY_MSG =
  "Configure ARGUS_API_KEY to enable AI features. Set it in your MCP server env config.";

export function intelligenceTools(client: ArgusClient): ToolDef[] {
  return [
    {
      name: "argus_heal",
      description:
        "Get AI-powered healing suggestions for a broken test selector or assertion. Requires Argus API key.",
      inputSchema: z.object({
        orgId: z.string().describe("Organization ID"),
        errorType: z
          .enum([
            "selector_not_found",
            "assertion_failed",
            "timeout",
            "navigation_error",
            "api_error",
          ])
          .describe("Type of test failure"),
        errorMessage: z.string().describe("The error message from the failure"),
        failedSelector: z
          .string()
          .optional()
          .describe("The CSS selector that failed"),
        pageUrl: z.string().optional().describe("The page URL where the failure occurred"),
        pageHtml: z
          .string()
          .optional()
          .describe("Relevant HTML snippet from the page (optional, improves suggestions)"),
      }),
      handler: async ({ orgId, errorType, errorMessage, failedSelector, pageUrl, pageHtml }) => {
        if (!client.available) return { error: NO_KEY_MSG };

        return client.post(`/api/v1/healing/organizations/${orgId}/suggest`, {
          error_type: errorType,
          error_message: errorMessage,
          failed_selector: failedSelector,
          page_url: pageUrl,
          page_html: pageHtml,
        });
      },
    },
    {
      name: "argus_generate_test",
      description:
        "AI-generate a test from a natural language description or page context. Requires Argus API key.",
      inputSchema: z.object({
        projectId: z.string().describe("Project ID"),
        description: z
          .string()
          .describe("Natural language description of what to test (e.g. 'test login form validation')"),
        pageUrl: z
          .string()
          .optional()
          .describe("URL of the page to test"),
        framework: z
          .enum(["playwright", "cypress", "selenium", "puppeteer"])
          .default("playwright")
          .describe("Target test framework"),
      }),
      handler: async ({ projectId, description, pageUrl, framework }) => {
        if (!client.available) return { error: NO_KEY_MSG };

        return client.post("/api/v1/quality/generate-test", {
          project_id: projectId,
          description,
          page_url: pageUrl,
          framework,
        });
      },
    },
    {
      name: "argus_quality_score",
      description:
        "Get the AI-computed quality score for a project. Includes test coverage, flakiness, and reliability metrics. Requires Argus API key.",
      inputSchema: z.object({
        projectId: z.string().describe("Project ID"),
      }),
      handler: async ({ projectId }) => {
        if (!client.available) return { error: NO_KEY_MSG };

        return client.get(
          `/api/v1/quality/score?project_id=${encodeURIComponent(projectId)}`
        );
      },
    },
    {
      name: "argus_similar_failures",
      description:
        "Find similar past test failures using semantic search. Helps understand if a failure is known and how it was fixed before. Requires Argus API key.",
      inputSchema: z.object({
        orgId: z.string().describe("Organization ID"),
        errorMessage: z
          .string()
          .describe("Error message to search for similar failures"),
        limit: z
          .number()
          .default(5)
          .describe("Max number of similar failures to return"),
      }),
      handler: async ({ orgId, errorMessage, limit }) => {
        if (!client.available) return { error: NO_KEY_MSG };

        return client.post(
          `/api/v1/healing/organizations/${orgId}/similar-errors`,
          {
            error_message: errorMessage,
            limit,
          }
        );
      },
    },
    {
      name: "argus_ask",
      description:
        "Ask Argus AI about testing strategy, best practices, or how to test a specific feature. Requires Argus API key.",
      inputSchema: z.object({
        message: z.string().describe("Your question about testing"),
        projectId: z
          .string()
          .optional()
          .describe("Project ID for context-aware answers"),
        sessionId: z
          .string()
          .optional()
          .describe("Chat session ID for conversation continuity"),
      }),
      handler: async ({ message, projectId, sessionId }) => {
        if (!client.available) return { error: NO_KEY_MSG };

        return client.post("/api/v1/chat/message", {
          message,
          project_id: projectId,
          session_id: sessionId,
        });
      },
    },
    {
      name: "argus_report",
      description:
        "Generate a test report for a project or test run. Requires Argus API key.",
      inputSchema: z.object({
        projectId: z.string().describe("Project ID"),
        reportType: z
          .enum(["summary", "detailed", "regression", "coverage"])
          .default("summary")
          .describe("Type of report to generate"),
        testRunId: z
          .string()
          .optional()
          .describe("Specific test run ID (optional)"),
      }),
      handler: async ({ projectId, reportType, testRunId }) => {
        if (!client.available) return { error: NO_KEY_MSG };

        if (testRunId) {
          return client.post(`/api/v1/reports/test-runs/${testRunId}/report`, {
            report_type: reportType,
          });
        }

        return client.post("/api/v1/reports", {
          project_id: projectId,
          report_type: reportType,
        });
      },
    },
  ];
}
