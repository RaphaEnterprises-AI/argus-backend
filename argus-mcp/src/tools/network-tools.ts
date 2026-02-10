import { z } from "zod";
import type { BrowserEngine } from "../browser/engine.js";
import type { ToolDef } from "../server.js";

export function networkTools(engine: BrowserEngine): ToolDef[] {
  return [
    {
      name: "network_get_requests",
      description:
        "List all network requests captured since page load. Includes URL, method, status, and size.",
      inputSchema: z.object({
        urlFilter: z
          .string()
          .optional()
          .describe("Filter requests by URL substring"),
        methodFilter: z
          .string()
          .optional()
          .describe("Filter by HTTP method (GET, POST, etc.)"),
      }),
      handler: async ({ urlFilter, methodFilter }) => {
        const cdp = await engine.ensureCDP();
        let requests = cdp.getRequests();

        if (urlFilter) {
          requests = requests.filter((r) => r.url.includes(urlFilter));
        }
        if (methodFilter) {
          const m = methodFilter.toUpperCase();
          requests = requests.filter((r) => r.method === m);
        }

        return {
          count: requests.length,
          requests: requests.map((r) => ({
            requestId: r.requestId,
            method: r.method,
            url: r.url,
            status: r.status ?? null,
            mimeType: r.mimeType ?? null,
            size: r.responseSize ?? null,
            failed: r.failed ?? false,
          })),
        };
      },
    },
    {
      name: "network_get_response",
      description:
        "Get the response body for a specific network request by its requestId (from network_get_requests).",
      inputSchema: z.object({
        requestId: z
          .string()
          .describe("The requestId from network_get_requests"),
      }),
      handler: async ({ requestId }) => {
        const cdp = await engine.ensureCDP();
        const body = await cdp.getResponseBody(requestId);
        return { requestId, body };
      },
    },
    {
      name: "network_get_failures",
      description:
        "List all failed network requests (4xx, 5xx, or connection errors).",
      inputSchema: z.object({}),
      handler: async () => {
        const cdp = await engine.ensureCDP();
        const failures = cdp.getFailedRequests();
        return {
          count: failures.length,
          failures: failures.map((r) => ({
            requestId: r.requestId,
            method: r.method,
            url: r.url,
            status: r.status ?? null,
            failureReason: r.failureReason ?? null,
          })),
        };
      },
    },
    {
      name: "network_intercept",
      description:
        "Intercept and mock network requests matching a URL pattern. Useful for testing error states or stubbing APIs.",
      inputSchema: z.object({
        urlPattern: z
          .string()
          .describe("URL glob pattern to intercept (e.g. '**/api/users*')"),
        status: z
          .number()
          .default(200)
          .describe("HTTP status code for the mock response"),
        body: z
          .string()
          .default("{}")
          .describe("Response body (JSON string)"),
        contentType: z
          .string()
          .default("application/json")
          .describe("Content-Type header"),
      }),
      handler: async ({ urlPattern, status, body, contentType }) => {
        const page = await engine.ensurePage();
        await page.route(urlPattern, (route) => {
          route.fulfill({
            status,
            body,
            contentType,
          });
        });
        return { intercepting: urlPattern, status, contentType };
      },
    },
  ];
}
