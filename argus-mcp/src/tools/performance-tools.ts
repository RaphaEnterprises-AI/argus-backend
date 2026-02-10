import { z } from "zod";
import type { BrowserEngine } from "../browser/engine.js";
import type { ToolDef } from "../server.js";

export function performanceTools(engine: BrowserEngine): ToolDef[] {
  return [
    {
      name: "perf_get_metrics",
      description:
        "Get Chrome performance metrics including timestamps, layout counts, JS heap size, and navigation timings. Use browser_evaluate with the Performance API for Core Web Vitals (LCP, CLS, FCP).",
      inputSchema: z.object({}),
      handler: async () => {
        const cdp = await engine.ensureCDP();
        const metrics = await cdp.getPerformanceMetrics();

        // Also grab Navigation Timing via the page
        const page = await engine.ensurePage();
        const navTiming = await page.evaluate(() => {
          const nav = performance.getEntriesByType(
            "navigation"
          )[0] as PerformanceNavigationTiming | undefined;
          if (!nav) return null;
          return {
            domContentLoaded: nav.domContentLoadedEventEnd - nav.startTime,
            loadComplete: nav.loadEventEnd - nav.startTime,
            domInteractive: nav.domInteractive - nav.startTime,
            ttfb: nav.responseStart - nav.requestStart,
            dnsLookup: nav.domainLookupEnd - nav.domainLookupStart,
            tlsHandshake: nav.connectEnd - nav.secureConnectionStart,
          };
        });

        return { chrome: metrics, navigation: navTiming };
      },
    },
    {
      name: "perf_start_trace",
      description:
        "Start a Chrome trace for detailed performance profiling. Call perf_stop_trace to get results.",
      inputSchema: z.object({}),
      handler: async () => {
        const cdp = await engine.ensureCDP();
        await cdp.startTrace();
        return { status: "tracing_started" };
      },
    },
    {
      name: "perf_stop_trace",
      description:
        "Stop the running Chrome trace and return profiling events. Must call perf_start_trace first.",
      inputSchema: z.object({}),
      handler: async () => {
        const cdp = await engine.ensureCDP();
        const { events } = await cdp.stopTrace();
        return {
          eventCount: events.length,
          summary:
            events.length > 100
              ? `${events.length} trace events captured (showing first 100)`
              : `${events.length} trace events captured`,
          events: events.slice(0, 100),
        };
      },
    },
  ];
}
