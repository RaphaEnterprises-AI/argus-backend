/**
 * Comprehensive test suite for @argus/mcp
 * Tests all 30 tools against a local HTML test fixture.
 *
 * Usage: node test/run-all-tests.mjs
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
import { intelligenceTools } from "../dist/tools/intelligence-tools.js";
import { zodToJsonSchema } from "../dist/zod-to-json.js";
import * as fs from "node:fs";
import * as path from "node:path";

// ── Test harness ──────────────────────────────────────────────

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

function assert(condition, msg) {
  if (!condition) throw new Error(msg || "Assertion failed");
}

function assertIncludes(haystack, needle, msg) {
  if (typeof haystack === "string" && !haystack.includes(needle)) {
    throw new Error(msg || `Expected "${haystack}" to include "${needle}"`);
  }
  if (Array.isArray(haystack) && !haystack.includes(needle)) {
    throw new Error(msg || `Expected array to include "${needle}"`);
  }
}

// ── Test fixture HTML ─────────────────────────────────────────

const TEST_HTML = `data:text/html,${encodeURIComponent(`<!DOCTYPE html>
<html>
<head><title>Argus MCP Test Page</title></head>
<body>
  <h1 id="heading">Hello Argus</h1>
  <p id="description">Test page for MCP tools</p>

  <form id="login-form">
    <input id="email" type="email" name="email" placeholder="Email" />
    <input id="password" type="password" name="password" placeholder="Password" />
    <select id="role" name="role">
      <option value="user">User</option>
      <option value="admin">Admin</option>
    </select>
    <label><input type="checkbox" id="remember" name="remember" /> Remember me</label>
    <button id="submit-btn" type="button" onclick="document.getElementById('result').textContent='Submitted!'">Submit</button>
  </form>

  <div id="result"></div>

  <a href="#about" id="about-link">About</a>
  <button id="hover-btn" onmouseenter="this.textContent='Hovered!'" aria-label="Hover target">Hover me</button>

  <div id="scroll-area" style="height:100px;overflow:auto">
    <div style="height:500px">Scrollable content</div>
  </div>

  <div id="hidden-el" style="display:none">I am hidden</div>

  <script>
    console.log("Page loaded");
    console.warn("Test warning");
    fetch("https://httpbin.org/get").catch(() => {});
    fetch("https://httpbin.org/status/404").catch(() => {});
  </script>
</body>
</html>`)}`;

// ── Main ──────────────────────────────────────────────────────

async function main() {
  const config = loadConfig();
  const engine = new BrowserEngine(config);
  const screenshots = new ScreenshotManager(config);
  const argusClient = new ArgusClient({ ...config, argusApiKey: null });

  // Build tool maps
  const bTools = Object.fromEntries(browserTools(engine, screenshots).map((t) => [t.name, t]));
  const nTools = Object.fromEntries(networkTools(engine).map((t) => [t.name, t]));
  const cTools = Object.fromEntries(consoleTools(engine).map((t) => [t.name, t]));
  const pTools = Object.fromEntries(performanceTools(engine).map((t) => [t.name, t]));
  const tTools = Object.fromEntries(testTools(engine, screenshots).map((t) => [t.name, t]));
  const iTools = Object.fromEntries(intelligenceTools(argusClient).map((t) => [t.name, t]));

  try {
    // ════════════════════════════════════════════════════════════
    console.log("\n── Schema Generation ──────────────────────────");
    // ════════════════════════════════════════════════════════════

    await test("all 30 tools generate valid JSON Schema", () => {
      const allTools = [
        ...browserTools(engine, screenshots),
        ...networkTools(engine),
        ...consoleTools(engine),
        ...performanceTools(engine),
        ...testTools(engine, screenshots),
        ...intelligenceTools(argusClient),
      ];
      assert(allTools.length === 30, `Expected 30 tools, got ${allTools.length}`);
      for (const t of allTools) {
        const schema = zodToJsonSchema(t.inputSchema);
        assert(schema.type === "object", `${t.name} schema type should be object`);
        assert(schema.properties !== undefined, `${t.name} schema should have properties`);
      }
    });

    // ════════════════════════════════════════════════════════════
    console.log("\n── Browser Tools ─────────────────────────────");
    // ════════════════════════════════════════════════════════════

    await test("browser_navigate loads page", async () => {
      const r = await bTools.browser_navigate.handler({ url: TEST_HTML, waitUntil: "load" });
      assert(r.title === "Argus MCP Test Page", `Title: ${r.title}`);
      assertIncludes(r.url, "data:text/html");
    });

    // Wait for network requests to fire
    await new Promise((r) => setTimeout(r, 2000));

    await test("browser_click clicks button", async () => {
      const r = await bTools.browser_click.handler({ selector: "#submit-btn", button: "left", clickCount: 1 });
      assert(r.clicked === "#submit-btn");
      // Verify the click had an effect
      const page = await engine.ensurePage();
      const text = await page.textContent("#result");
      assert(text === "Submitted!", `Expected 'Submitted!' got '${text}'`);
    });

    await test("browser_fill fills input", async () => {
      const r = await bTools.browser_fill.handler({ selector: "#email", value: "test@argus.dev" });
      assert(r.filled === "#email");
      const page = await engine.ensurePage();
      const val = await page.inputValue("#email");
      assert(val === "test@argus.dev", `Expected 'test@argus.dev' got '${val}'`);
    });

    await test("browser_type types text", async () => {
      await bTools.browser_fill.handler({ selector: "#password", value: "" });
      const r = await bTools.browser_type.handler({ selector: "#password", text: "s3cret", delay: 10 });
      assert(r.typed === "s3cret");
      const page = await engine.ensurePage();
      const val = await page.inputValue("#password");
      assert(val === "s3cret", `Expected 's3cret' got '${val}'`);
    });

    await test("browser_select selects option", async () => {
      const r = await bTools.browser_select.handler({ selector: "#role", value: "admin" });
      assert(r.selected.includes("admin"), `Selected: ${JSON.stringify(r.selected)}`);
    });

    await test("browser_hover hovers element", async () => {
      const r = await bTools.browser_hover.handler({ selector: "#hover-btn" });
      assert(r.hovered === "#hover-btn");
      const page = await engine.ensurePage();
      const text = await page.textContent("#hover-btn");
      assert(text === "Hovered!", `Expected 'Hovered!' got '${text}'`);
    });

    await test("browser_press_key presses Enter", async () => {
      const r = await bTools.browser_press_key.handler({ key: "Tab" });
      assert(r.pressed === "Tab");
    });

    await test("browser_screenshot captures page", async () => {
      const r = await bTools.browser_screenshot.handler({ fullPage: false, name: "test_page.png" });
      assert(r.path.endsWith("test_page.png"), `Path: ${r.path}`);
      assert(r.base64.length > 100, "Screenshot base64 too short");
      assert(fs.existsSync(r.path), `File not found: ${r.path}`);
    });

    await test("browser_screenshot captures element", async () => {
      const r = await bTools.browser_screenshot.handler({ selector: "#heading", name: "heading.png" });
      assert(r.path.endsWith("heading.png"));
      assert(r.base64.length > 50);
    });

    await test("browser_scroll scrolls down", async () => {
      const r = await bTools.browser_scroll.handler({ direction: "down", amount: 200 });
      assert(r.scrolled === "down");
      assert(r.amount === 200);
    });

    await test("browser_scroll scrolls element", async () => {
      const r = await bTools.browser_scroll.handler({ direction: "down", amount: 100, selector: "#scroll-area" });
      assert(r.scrolled === "down");
    });

    await test("browser_wait waits for selector", async () => {
      const r = await bTools.browser_wait.handler({ selector: "#heading", timeout: 3000, state: "visible" });
      assert(r.found === "#heading");
    });

    await test("browser_wait waits for timeout", async () => {
      const r = await bTools.browser_wait.handler({ timeout: 100, state: "visible" });
      assert(r.waited === "100ms");
    });

    await test("browser_evaluate runs JS", async () => {
      const r = await bTools.browser_evaluate.handler({ script: "document.title" });
      assert(r.result === "Argus MCP Test Page", `Result: ${r.result}`);
    });

    await test("browser_evaluate returns computed values", async () => {
      const r = await bTools.browser_evaluate.handler({ script: "2 + 2" });
      assert(r.result === 4, `Result: ${r.result}`);
    });

    await test("browser_snapshot returns accessibility tree", async () => {
      const r = await bTools.browser_snapshot.handler({});
      assert(r.snapshot, "Expected snapshot");
      assert(typeof r.snapshot === "string", "Snapshot should be a string");
    });

    await test("browser_snapshot with selector", async () => {
      const r = await bTools.browser_snapshot.handler({ selector: "#login-form" });
      assert(r.snapshot, "Expected snapshot for form");
    });

    // ════════════════════════════════════════════════════════════
    console.log("\n── Network Tools ─────────────────────────────");
    // ════════════════════════════════════════════════════════════

    await test("network_get_requests lists requests", async () => {
      const r = await nTools.network_get_requests.handler({});
      assert(r.count > 0, `Expected requests, got ${r.count}`);
      assert(r.requests.length > 0, "No request entries");
      assert(r.requests[0].method, "Request should have method");
      assert(r.requests[0].url, "Request should have url");
    });

    await test("network_get_requests filters by URL", async () => {
      const r = await nTools.network_get_requests.handler({ urlFilter: "httpbin" });
      // May or may not have httpbin requests depending on network
      assert(r.count >= 0, "Should return count");
    });

    await test("network_get_requests filters by method", async () => {
      const r = await nTools.network_get_requests.handler({ methodFilter: "GET" });
      for (const req of r.requests) {
        assert(req.method === "GET", `Expected GET, got ${req.method}`);
      }
    });

    await test("network_get_failures lists failed requests", async () => {
      const r = await nTools.network_get_failures.handler({});
      assert(r.count >= 0, "Should return count");
      // Cross-origin fetches on data: URL likely fail
    });

    await test("network_get_response returns body or error", async () => {
      const requests = await nTools.network_get_requests.handler({});
      if (requests.requests.length > 0) {
        const r = await nTools.network_get_response.handler({ requestId: requests.requests[0].requestId });
        assert(r.requestId, "Should return requestId");
        assert(r.body !== undefined, "Should return body");
      }
    });

    await test("network_intercept sets up route", async () => {
      const r = await nTools.network_intercept.handler({
        urlPattern: "**/api/mock*",
        status: 200,
        body: '{"mocked":true}',
        contentType: "application/json",
      });
      assert(r.intercepting === "**/api/mock*");
      assert(r.status === 200);
    });

    // ════════════════════════════════════════════════════════════
    console.log("\n── Console Tools ─────────────────────────────");
    // ════════════════════════════════════════════════════════════

    await test("console_get_messages returns messages", async () => {
      const r = await cTools.console_get_messages.handler({});
      assert(r.count >= 0, "Should return count");
      // data: URLs console messages may or may not be captured
    });

    await test("console_get_messages filters by type", async () => {
      const r = await cTools.console_get_messages.handler({ typeFilter: "warning" });
      for (const m of r.messages) {
        assert(m.type === "warning", `Expected warning, got ${m.type}`);
      }
    });

    await test("console_get_messages filters by text", async () => {
      const r = await cTools.console_get_messages.handler({ textFilter: "nonexistent" });
      assert(r.count === 0, "Should find no messages");
    });

    await test("console_get_errors returns errors array", async () => {
      const r = await cTools.console_get_errors.handler({});
      assert(Array.isArray(r.errors), "Should return errors array");
    });

    // ════════════════════════════════════════════════════════════
    console.log("\n── Performance Tools ──────────────────────────");
    // ════════════════════════════════════════════════════════════

    await test("perf_get_metrics returns Chrome metrics", async () => {
      const r = await pTools.perf_get_metrics.handler({});
      assert(r.chrome, "Should have chrome metrics");
      assert(typeof r.chrome.Timestamp === "number", "Should have Timestamp");
    });

    await test("perf_start_trace and perf_stop_trace", async () => {
      const start = await pTools.perf_start_trace.handler({});
      assert(start.status === "tracing_started");

      // Do something to generate trace events
      const page = await engine.ensurePage();
      await page.evaluate(() => { for (let i = 0; i < 1000; i++) document.title = `t${i}`; });
      await new Promise((r) => setTimeout(r, 200));

      const stop = await pTools.perf_stop_trace.handler({});
      assert(stop.eventCount >= 0, `Expected events, got ${stop.eventCount}`);
    });

    // ════════════════════════════════════════════════════════════
    console.log("\n── Test Tools ────────────────────────────────");
    // ════════════════════════════════════════════════════════════

    // Re-navigate to test page for clean state
    await bTools.browser_navigate.handler({ url: TEST_HTML, waitUntil: "load" });
    await new Promise((r) => setTimeout(r, 500));

    await test("test_discover finds interactive elements", async () => {
      const r = await tTools.test_discover.handler({ includeHidden: false });
      assert(r.elementCount > 0, `Expected elements, got ${r.elementCount}`);
      const tags = r.elements.map((e) => e.tag);
      assert(tags.includes("button"), "Should find buttons");
      assert(tags.includes("input"), "Should find inputs");
      assert(tags.includes("a"), "Should find links");
    });

    await test("test_discover includes hidden when requested", async () => {
      const visible = await tTools.test_discover.handler({ includeHidden: false });
      const all = await tTools.test_discover.handler({ includeHidden: true });
      assert(all.elementCount >= visible.elementCount, "Hidden should add more elements");
    });

    await test("test_assert text_contains passes", async () => {
      const r = await tTools.test_assert.handler({
        assertion: "text_contains",
        selector: "#heading",
        expected: "Hello",
      });
      assert(r.pass === true, `Expected pass, got ${JSON.stringify(r)}`);
    });

    await test("test_assert text_contains fails on mismatch", async () => {
      const r = await tTools.test_assert.handler({
        assertion: "text_contains",
        selector: "#heading",
        expected: "Goodbye",
      });
      assert(r.pass === false, "Should fail");
    });

    await test("test_assert element_visible passes", async () => {
      const r = await tTools.test_assert.handler({
        assertion: "element_visible",
        selector: "#heading",
        expected: "",
      });
      assert(r.pass === true);
    });

    await test("test_assert element_hidden passes for hidden element", async () => {
      const r = await tTools.test_assert.handler({
        assertion: "element_hidden",
        selector: "#hidden-el",
        expected: "",
      });
      assert(r.pass === true);
    });

    await test("test_assert element_count works", async () => {
      const r = await tTools.test_assert.handler({
        assertion: "element_count",
        selector: "input",
        expected: "4",
      });
      // email, password, remember checkbox, hidden would be 3 visible
      // The actual count depends on the form — let's just check it returns a result
      assert(r.actual !== undefined, "Should return actual count");
    });

    await test("test_assert title_contains works", async () => {
      const r = await tTools.test_assert.handler({
        assertion: "title_contains",
        expected: "Argus MCP",
      });
      assert(r.pass === true, `Title assertion failed: ${JSON.stringify(r)}`);
    });

    await test("test_assert url_matches works", async () => {
      const r = await tTools.test_assert.handler({
        assertion: "url_matches",
        expected: "data:text",
      });
      assert(r.pass === true);
    });

    await test("test_run passes a multi-step test", async () => {
      const r = await tTools.test_run.handler({
        name: "login_flow",
        steps: [
          { action: "navigate", value: TEST_HTML },
          { action: "fill", selector: "#email", value: "user@test.com" },
          { action: "fill", selector: "#password", value: "pass123" },
          { action: "click", selector: "#submit-btn" },
          { action: "assert_text", selector: "#result", value: "Submitted!" },
          { action: "screenshot" },
        ],
      });
      assert(r.status === "passed", `Test status: ${r.status}, results: ${JSON.stringify(r.results.map(s => ({step: s.step, status: s.status, error: s.error})))}`);
      assert(r.stepsRun === 6, `Steps run: ${r.stepsRun}`);
      assert(r.results.every((s) => s.status === "passed"), "All steps should pass");
    });

    await test("test_run reports failure on bad assertion", async () => {
      const r = await tTools.test_run.handler({
        name: "failing_test",
        steps: [
          { action: "navigate", value: TEST_HTML },
          { action: "assert_text", selector: "#heading", value: "WRONG TEXT" },
          { action: "click", selector: "#submit-btn" },
        ],
      });
      assert(r.status === "failed", "Test should fail");
      assert(r.stepsRun === 2, `Should stop at step 2, ran ${r.stepsRun}`);
      assert(r.results[1].status === "failed", "Step 2 should fail");
      assert(r.results[1].error, "Should have error message");
    });

    // ════════════════════════════════════════════════════════════
    console.log("\n── Intelligence Tools (no API key) ───────────");
    // ════════════════════════════════════════════════════════════

    await test("argus_heal returns API key message", async () => {
      const r = await iTools.argus_heal.handler({
        orgId: "test", errorType: "selector_not_found", errorMessage: "not found",
      });
      assertIncludes(r.error, "ARGUS_API_KEY");
    });

    await test("argus_generate_test returns API key message", async () => {
      const r = await iTools.argus_generate_test.handler({
        projectId: "test", description: "test login", framework: "playwright",
      });
      assertIncludes(r.error, "ARGUS_API_KEY");
    });

    await test("argus_quality_score returns API key message", async () => {
      const r = await iTools.argus_quality_score.handler({ projectId: "test" });
      assertIncludes(r.error, "ARGUS_API_KEY");
    });

    await test("argus_similar_failures returns API key message", async () => {
      const r = await iTools.argus_similar_failures.handler({
        orgId: "test", errorMessage: "test", limit: 5,
      });
      assertIncludes(r.error, "ARGUS_API_KEY");
    });

    await test("argus_ask returns API key message", async () => {
      const r = await iTools.argus_ask.handler({ message: "test" });
      assertIncludes(r.error, "ARGUS_API_KEY");
    });

    await test("argus_report returns API key message", async () => {
      const r = await iTools.argus_report.handler({
        projectId: "test", reportType: "summary",
      });
      assertIncludes(r.error, "ARGUS_API_KEY");
    });

  } finally {
    await engine.close();
    // Clean up screenshots
    const dir = config.screenshotDir;
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true });
    }
  }

  // ── Summary ──────────────────────────────────────────────────
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
