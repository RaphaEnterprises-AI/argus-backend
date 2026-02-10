/**
 * End-to-end test against real websites.
 *
 * Tests the full MCP tool suite against:
 * 1. Playwright's TodoMVC demo — form interaction, assertions, discovery
 * 2. httpbin.org — network capture, API responses, failures
 *
 * Usage: node test/test-e2e-real-site.mjs
 */

import { BrowserEngine } from "../dist/browser/engine.js";
import { ScreenshotManager } from "../dist/browser/screenshots.js";
import { ArgusClient } from "../dist/argus-client.js";
import { loadConfig } from "../dist/config.js";
import { browserTools } from "../dist/tools/browser-tools.js";
import { networkTools } from "../dist/tools/network-tools.js";
import { consoleTools } from "../dist/tools/console-tools.js";
import { performanceTools } from "../dist/tools/performance-tools.js";
import { testTools } from "../dist/tools/test-tools.js";
import * as fs from "node:fs";

// ── Harness ───────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    failed++;
    failures.push({ name, error: err.message });
    console.log(`  ✗ ${name}: ${err.message}`);
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || "Assertion failed");
}

// ── Main ──────────────────────────────────────────────────────

async function main() {
  const config = loadConfig();
  const engine = new BrowserEngine(config);
  const screenshots = new ScreenshotManager(config);

  const b = Object.fromEntries(browserTools(engine, screenshots).map((t) => [t.name, t]));
  const n = Object.fromEntries(networkTools(engine).map((t) => [t.name, t]));
  const c = Object.fromEntries(consoleTools(engine).map((t) => [t.name, t]));
  const p = Object.fromEntries(performanceTools(engine).map((t) => [t.name, t]));
  const t = Object.fromEntries(testTools(engine, screenshots).map((t) => [t.name, t]));

  try {
    // ══════════════════════════════════════════════════════════
    console.log("\n── 1. TodoMVC (https://demo.playwright.dev/todomvc) ──");
    // ══════════════════════════════════════════════════════════

    await test("navigate to TodoMVC", async () => {
      const r = await b.browser_navigate.handler({
        url: "https://demo.playwright.dev/todomvc/#/",
        waitUntil: "networkidle",
      });
      assert(r.title.toLowerCase().includes("todo"), `Title: ${r.title}`);
      assert(r.url.includes("todomvc"), `URL: ${r.url}`);
    });

    await test("screenshot the landing page", async () => {
      const r = await b.browser_screenshot.handler({ name: "todomvc_landing.png" });
      assert(fs.existsSync(r.path), "Screenshot file should exist");
      assert(r.base64.length > 1000, "Screenshot should have content");
    });

    await test("discover interactive elements on TodoMVC", async () => {
      const r = await t.test_discover.handler({ includeHidden: false });
      assert(r.elementCount > 0, `Found ${r.elementCount} elements`);
      // Should find at least one input element
      const hasInput = r.elements.some((el) => el.tag === "input");
      assert(hasInput, `Should discover an input. Elements: ${r.elements.map(e => e.tag).join(", ")}`);
    });

    await test("add first todo item", async () => {
      await b.browser_fill.handler({ selector: ".new-todo", value: "Buy groceries" });
      await b.browser_press_key.handler({ key: "Enter", selector: ".new-todo" });
      // Wait for item to appear
      await b.browser_wait.handler({ selector: ".todo-list li", timeout: 3000, state: "visible" });
    });

    await test("add second todo item", async () => {
      await b.browser_fill.handler({ selector: ".new-todo", value: "Write tests" });
      await b.browser_press_key.handler({ key: "Enter", selector: ".new-todo" });
      await new Promise((r) => setTimeout(r, 300));
    });

    await test("add third todo item", async () => {
      await b.browser_fill.handler({ selector: ".new-todo", value: "Deploy to production" });
      await b.browser_press_key.handler({ key: "Enter", selector: ".new-todo" });
      await new Promise((r) => setTimeout(r, 300));
    });

    await test("assert 3 todos exist", async () => {
      const r = await t.test_assert.handler({
        assertion: "element_count",
        selector: ".todo-list li",
        expected: "3",
      });
      assert(r.pass, `Expected 3 todos, got ${r.actual}`);
    });

    await test("assert 'Buy groceries' text visible", async () => {
      const r = await t.test_assert.handler({
        assertion: "text_contains",
        selector: ".todo-list",
        expected: "Buy groceries",
      });
      assert(r.pass, "Should contain 'Buy groceries'");
    });

    await test("complete first todo by clicking checkbox", async () => {
      await b.browser_click.handler({
        selector: ".todo-list li:first-child .toggle",
        button: "left",
        clickCount: 1,
      });
      await new Promise((r) => setTimeout(r, 300));
    });

    await test("assert todo count shows '2 items left'", async () => {
      const r = await t.test_assert.handler({
        assertion: "text_contains",
        selector: ".todo-count",
        expected: "2",
      });
      assert(r.pass, `Todo count: ${r.actual}`);
    });

    await test("screenshot after completing todo", async () => {
      const r = await b.browser_screenshot.handler({ name: "todomvc_completed.png" });
      assert(fs.existsSync(r.path));
    });

    await test("filter to show active todos only", async () => {
      await b.browser_click.handler({
        selector: 'a[href="#/active"]',
        button: "left",
        clickCount: 1,
      });
      await new Promise((r) => setTimeout(r, 300));
    });

    await test("assert only 2 active todos visible", async () => {
      const r = await t.test_assert.handler({
        assertion: "element_count",
        selector: ".todo-list li",
        expected: "2",
      });
      assert(r.pass, `Expected 2 active, got ${r.actual}`);
    });

    await test("assert URL changed to #/active", async () => {
      const r = await t.test_assert.handler({
        assertion: "url_matches",
        expected: "#/active",
      });
      assert(r.pass, `URL: ${r.actual}`);
    });

    await test("filter to show completed todos", async () => {
      await b.browser_click.handler({
        selector: 'a[href="#/completed"]',
        button: "left",
        clickCount: 1,
      });
      await new Promise((r) => setTimeout(r, 300));
    });

    await test("assert 1 completed todo visible", async () => {
      const r = await t.test_assert.handler({
        assertion: "element_count",
        selector: ".todo-list li",
        expected: "1",
      });
      assert(r.pass, `Expected 1 completed, got ${r.actual}`);
    });

    await test("get accessibility snapshot of todo list", async () => {
      // Go back to all
      await b.browser_click.handler({
        selector: 'a[href="#/"]',
        button: "left",
        clickCount: 1,
      });
      await new Promise((r) => setTimeout(r, 300));
      const r = await b.browser_snapshot.handler({ selector: ".todo-list" });
      assert(r.snapshot, "Should return snapshot");
      assert(r.snapshot.length > 10, "Snapshot should have content");
    });

    await test("evaluate JS to count todos via DOM", async () => {
      const r = await b.browser_evaluate.handler({
        script: "document.querySelectorAll('.todo-list li').length",
      });
      assert(r.result === 3, `Expected 3, got ${r.result}`);
    });

    await test("full-page screenshot of TodoMVC", async () => {
      const r = await b.browser_screenshot.handler({
        fullPage: true,
        name: "todomvc_full.png",
      });
      assert(fs.existsSync(r.path));
      assert(r.base64.length > 1000);
    });

    // ══════════════════════════════════════════════════════════
    console.log("\n── 2. Network & Console (httpbin.org) ────────");
    // ══════════════════════════════════════════════════════════

    // Clear CDP captures for fresh network data
    const cdp = await engine.ensureCDP();
    cdp.clearCaptures();

    await test("navigate to httpbin", async () => {
      const r = await b.browser_navigate.handler({
        url: "https://httpbin.org/",
        waitUntil: "load",
      });
      assert(r.url.includes("httpbin"), `URL: ${r.url}`);
    });

    await test("capture network requests from page load", async () => {
      const r = await n.network_get_requests.handler({});
      assert(r.count > 0, `Expected requests, got ${r.count}`);
      // Should have the initial page request
      const pageReq = r.requests.find((req) => req.url.includes("httpbin.org") && req.method === "GET");
      assert(pageReq, "Should have GET to httpbin.org");
      assert(pageReq.status === 200, `Status: ${pageReq.status}`);
    });

    await test("filter network requests by method", async () => {
      const r = await n.network_get_requests.handler({ methodFilter: "GET" });
      assert(r.count > 0, "Should have GET requests");
      assert(r.requests.every((req) => req.method === "GET"), "All should be GET");
    });

    await test("get response body for a request", async () => {
      const all = await n.network_get_requests.handler({ urlFilter: "httpbin.org" });
      const htmlReq = all.requests.find((req) => req.mimeType?.includes("html"));
      if (htmlReq) {
        const r = await n.network_get_response.handler({ requestId: htmlReq.requestId });
        assert(r.body.length > 0, "Body should not be empty");
        assert(r.body.includes("httpbin") || r.body.includes("html"), "Body should be HTML");
      }
    });

    await test("trigger a fetch and capture it", async () => {
      // Make a fetch call from the page
      cdp.clearCaptures();
      await b.browser_evaluate.handler({
        script: `fetch('https://httpbin.org/get?test=argus').then(r => r.json())`,
      });
      await new Promise((r) => setTimeout(r, 2000));

      const r = await n.network_get_requests.handler({ urlFilter: "test=argus" });
      assert(r.count >= 1, `Expected fetch request, got ${r.count}`);
      assert(r.requests[0].url.includes("test=argus"), "Should contain query param");
    });

    await test("capture a 404 failure", async () => {
      cdp.clearCaptures();
      await b.browser_evaluate.handler({
        script: `fetch('https://httpbin.org/status/404').catch(() => {})`,
      });
      await new Promise((r) => setTimeout(r, 2000));

      const r = await n.network_get_failures.handler({});
      const has404 = r.failures.some((f) => f.status === 404);
      assert(has404, `Expected 404 in failures: ${JSON.stringify(r.failures.map(f => f.status))}`);
    });

    await test("intercept a request with mock response", async () => {
      await n.network_intercept.handler({
        urlPattern: "**/api/mocked-endpoint*",
        status: 200,
        body: '{"mocked": true, "source": "argus-mcp"}',
        contentType: "application/json",
      });

      const result = await b.browser_evaluate.handler({
        script: `fetch('/api/mocked-endpoint').then(r => r.json())`,
      });
      assert(result.result?.mocked === true, `Mock response: ${JSON.stringify(result.result)}`);
    });

    await test("get console messages", async () => {
      const r = await c.console_get_messages.handler({});
      // Just verify the structure is correct
      assert(Array.isArray(r.messages), "Should return messages array");
      assert(typeof r.count === "number", "Should return count");
    });

    await test("get JS errors (should be empty or have errors)", async () => {
      const r = await c.console_get_errors.handler({});
      assert(Array.isArray(r.errors), "Should return errors array");
    });

    await test("trigger and capture a JS error", async () => {
      // Force a JS error
      await b.browser_evaluate.handler({
        script: `setTimeout(() => { throw new Error("Argus test error"); }, 0)`,
      });
      await new Promise((r) => setTimeout(r, 500));

      const r = await c.console_get_errors.handler({});
      const hasArgusError = r.errors.some((e) => e.message.includes("Argus test error"));
      assert(hasArgusError, `Expected 'Argus test error' in: ${JSON.stringify(r.errors.map(e => e.message))}`);
    });

    // ══════════════════════════════════════════════════════════
    console.log("\n── 3. Performance Profiling ───────────────────");
    // ══════════════════════════════════════════════════════════

    await test("get performance metrics from httpbin", async () => {
      const r = await p.perf_get_metrics.handler({});
      assert(r.chrome, "Should have chrome metrics");
      assert(typeof r.chrome.Timestamp === "number", "Should have Timestamp");
      assert(typeof r.chrome.Documents === "number", "Should have Documents count");
      // Navigation timing may be null on cross-origin navigations
    });

    await test("performance trace on a navigation", async () => {
      const start = await p.perf_start_trace.handler({});
      assert(start.status === "tracing_started");

      // Navigate to generate trace events
      await b.browser_navigate.handler({
        url: "https://httpbin.org/html",
        waitUntil: "load",
      });
      await new Promise((r) => setTimeout(r, 500));

      const stop = await p.perf_stop_trace.handler({});
      assert(stop.eventCount > 0, `Expected trace events, got ${stop.eventCount}`);
    });

    await test("navigation timing on real page", async () => {
      await b.browser_navigate.handler({
        url: "https://httpbin.org/html",
        waitUntil: "load",
      });
      await new Promise((r) => setTimeout(r, 500));

      const r = await p.perf_get_metrics.handler({});
      if (r.navigation) {
        assert(
          r.navigation.ttfb > 0 || r.navigation.domContentLoaded > 0,
          "Should have timing data"
        );
      }
    });

    // ══════════════════════════════════════════════════════════
    console.log("\n── 4. Multi-Step Test Runner ──────────────────");
    // ══════════════════════════════════════════════════════════

    await test("test_run: TodoMVC add-and-verify flow", async () => {
      const r = await t.test_run.handler({
        name: "todomvc_e2e",
        steps: [
          { action: "navigate", value: "https://demo.playwright.dev/todomvc/#/" },
          { action: "wait", selector: ".new-todo" },
          { action: "fill", selector: ".new-todo", value: "E2E test item" },
          { action: "press_key", value: "Enter" },
          { action: "wait", selector: ".todo-list li" },
          { action: "assert_text", selector: ".todo-list", value: "E2E test item" },
          { action: "assert_visible", selector: ".todo-count" },
          { action: "screenshot" },
        ],
      });
      assert(r.status === "passed", `Test failed: ${JSON.stringify(r.results.filter(s => s.status === "failed"))}`);
      assert(r.stepsRun === 8, `Steps run: ${r.stepsRun}`);
    });

    await test("test_run: detect failure on wrong assertion", async () => {
      const r = await t.test_run.handler({
        name: "todomvc_fail",
        steps: [
          { action: "navigate", value: "https://demo.playwright.dev/todomvc/#/" },
          { action: "wait", selector: ".new-todo" },
          { action: "assert_text", selector: "body", value: "THIS TEXT DOES NOT EXIST ANYWHERE" },
        ],
      });
      assert(r.status === "failed", "Should detect failure");
      assert(r.results.some((s) => s.status === "failed"), "Should have a failed step");
      assert(r.results.some((s) => s.error), "Failed step should have error message");
    });

    // ══════════════════════════════════════════════════════════
    console.log("\n── 5. Scroll & Hover on Real Page ────────────");
    // ══════════════════════════════════════════════════════════

    await test("scroll down on httpbin", async () => {
      await b.browser_navigate.handler({ url: "https://httpbin.org/", waitUntil: "load" });
      const r = await b.browser_scroll.handler({ direction: "down", amount: 500 });
      assert(r.scrolled === "down");
    });

    await test("scroll back up", async () => {
      const r = await b.browser_scroll.handler({ direction: "up", amount: 500 });
      assert(r.scrolled === "up");
    });

    await test("hover over a visible link on httpbin", async () => {
      // First visible link in the page content (not the hidden GitHub corner)
      const r = await b.browser_hover.handler({ selector: ".swagger-ui a, #manpage a, main a, .description a" });
      assert(r.hovered);
    });

  } finally {
    await engine.close();
    // Clean up screenshots
    const dir = config.screenshotDir;
    if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true });
  }

  // ── Summary ────────────────────────────────────────────────
  console.log("\n══════════════════════════════════════════════════");
  console.log(`  PASSED: ${passed}  FAILED: ${failed}  TOTAL: ${passed + failed}`);
  if (failures.length > 0) {
    console.log("\n  Failures:");
    for (const f of failures) {
      console.log(`    ✗ ${f.name}: ${f.error}`);
    }
  }
  console.log("══════════════════════════════════════════════════\n");

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(2);
});
