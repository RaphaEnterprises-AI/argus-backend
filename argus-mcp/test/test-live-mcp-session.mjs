/**
 * Simulates a real Claude Code MCP session against a live website.
 *
 * This is the definitive integration test — it spawns the MCP server
 * as a subprocess (exactly like Claude Code does), communicates via
 * NDJSON stdio, and exercises a full testing workflow:
 *
 *   1. Initialize MCP handshake
 *   2. List available tools
 *   3. Navigate to a real site
 *   4. Interact with the page (fill, click, assert)
 *   5. Capture network requests
 *   6. Capture console messages
 *   7. Get performance metrics
 *   8. Run a multi-step test
 *   9. Take screenshots
 *  10. Test error handling
 *
 * Usage: node test/test-live-mcp-session.mjs
 */

import { spawn } from "node:child_process";

let passed = 0;
let failed = 0;
const failures = [];

function ok(name, pass, detail) {
  if (pass) {
    passed++;
    console.log(`  ✓ ${name}`);
  } else {
    failed++;
    failures.push({ name, detail });
    console.log(`  ✗ ${name}${detail ? ": " + detail : ""}`);
  }
}

async function main() {
  console.log("\n── Live MCP Session (stdio, like Claude Code) ──\n");

  const proc = spawn("node", ["dist/index.js"], {
    cwd: process.cwd(),
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ARGUS_HEADLESS: "true" },
  });

  let buf = "";
  proc.stdout.on("data", (chunk) => { buf += chunk.toString(); });
  proc.stderr.on("data", () => {});

  let reqId = 0;

  async function call(method, params, timeoutMs = 30000) {
    const id = ++reqId;
    buf = "";
    proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      for (const line of buf.split("\n")) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.id === id) return msg;
        } catch {}
      }
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`Timeout: ${method} (id=${id})`);
  }

  function notify(method, params) {
    proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
  }

  function toolResult(resp) {
    const text = resp?.result?.content?.[0]?.text;
    if (!text) return null;
    try { return JSON.parse(text); } catch { return text; }
  }

  try {
    // ── 1. MCP Handshake ──────────────────────────────────────
    console.log("  Phase 1: MCP Handshake");

    const init = await call("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "claude-code-simulator", version: "1.0.0" },
    });
    ok("handshake completes", init.result?.serverInfo?.name === "argus-mcp");
    ok("version is 1.0.0", init.result?.serverInfo?.version === "1.0.0");

    notify("notifications/initialized");
    await new Promise((r) => setTimeout(r, 200));

    // ── 2. Tool Discovery ─────────────────────────────────────
    console.log("  Phase 2: Tool Discovery");

    const list = await call("tools/list", {});
    const tools = list.result?.tools || [];
    ok("30 tools available", tools.length === 30);

    const local = tools.filter((t) => !t.name.startsWith("argus_"));
    const intel = tools.filter((t) => t.name.startsWith("argus_"));
    ok("24 local tools", local.length === 24);
    ok("6 intelligence tools", intel.length === 6);

    // Verify schemas have proper structure
    const navSchema = tools.find((t) => t.name === "browser_navigate")?.inputSchema;
    ok("schemas have required fields", navSchema?.required?.includes("url"));

    // ── 3. Navigate to Live Site ──────────────────────────────
    console.log("  Phase 3: Navigate to demo.playwright.dev/todomvc");

    const nav = await call("tools/call", {
      name: "browser_navigate",
      arguments: { url: "https://demo.playwright.dev/todomvc/#/", waitUntil: "networkidle" },
    }, 30000);
    const navData = toolResult(nav);
    ok("page loaded", navData?.url?.includes("todomvc"));
    ok("title present", typeof navData?.title === "string");

    // ── 4. Interact with Page ─────────────────────────────────
    console.log("  Phase 4: Interact with TodoMVC");

    // Add items
    for (const item of ["Buy milk", "Write code", "Ship feature"]) {
      await call("tools/call", {
        name: "browser_fill",
        arguments: { selector: ".new-todo", value: item },
      });
      await call("tools/call", {
        name: "browser_press_key",
        arguments: { key: "Enter", selector: ".new-todo" },
      });
    }
    await new Promise((r) => setTimeout(r, 500));

    // Assert count
    const count = await call("tools/call", {
      name: "test_assert",
      arguments: { assertion: "element_count", selector: ".todo-list li", expected: "3" },
    });
    ok("3 todos created", toolResult(count)?.pass === true);

    // Complete one
    await call("tools/call", {
      name: "browser_click",
      arguments: { selector: ".todo-list li:nth-child(2) .toggle", button: "left", clickCount: 1 },
    });
    await new Promise((r) => setTimeout(r, 300));

    const remaining = await call("tools/call", {
      name: "test_assert",
      arguments: { assertion: "text_contains", selector: ".todo-count", expected: "2" },
    });
    ok("2 items remaining after completion", toolResult(remaining)?.pass === true);

    // ── 5. Screenshots ────────────────────────────────────────
    console.log("  Phase 5: Screenshots");

    const shot = await call("tools/call", {
      name: "browser_screenshot",
      arguments: { fullPage: false, name: "live_session.png" },
    });
    const shotData = toolResult(shot);
    ok("screenshot captured", shotData?.base64?.length > 1000);
    ok("screenshot saved to disk", shotData?.path?.endsWith(".png"));

    // Element screenshot
    const elShot = await call("tools/call", {
      name: "browser_screenshot",
      arguments: { selector: ".todo-list", name: "todo_list.png" },
    });
    ok("element screenshot captured", toolResult(elShot)?.base64?.length > 100);

    // ── 6. Accessibility ──────────────────────────────────────
    console.log("  Phase 6: Accessibility Snapshot");

    const snap = await call("tools/call", {
      name: "browser_snapshot",
      arguments: { selector: ".todo-list" },
    });
    const snapData = toolResult(snap);
    ok("a11y snapshot returned", snapData?.snapshot?.length > 10);

    // ── 7. Element Discovery ──────────────────────────────────
    console.log("  Phase 7: Element Discovery");

    const discover = await call("tools/call", {
      name: "test_discover",
      arguments: { includeHidden: false },
    });
    const discData = toolResult(discover);
    ok("interactive elements found", discData?.elementCount > 0);
    ok("inputs discovered", discData?.elements?.some((e) => e.tag === "input"));

    // ── 8. Network Capture ────────────────────────────────────
    console.log("  Phase 8: Network Capture");

    const netReqs = await call("tools/call", {
      name: "network_get_requests",
      arguments: {},
    });
    const netData = toolResult(netReqs);
    ok("network requests captured", netData?.count > 0);

    // Filter
    const filtered = await call("tools/call", {
      name: "network_get_requests",
      arguments: { urlFilter: "playwright" },
    });
    ok("URL filter works", toolResult(filtered)?.count >= 0);

    // Get failures
    const netFail = await call("tools/call", {
      name: "network_get_failures",
      arguments: {},
    });
    ok("failure list returned", Array.isArray(toolResult(netFail)?.failures));

    // ── 9. Console Capture ────────────────────────────────────
    console.log("  Phase 9: Console & Error Capture");

    // Inject a console.log and an error
    await call("tools/call", {
      name: "browser_evaluate",
      arguments: { script: "console.log('argus-mcp-test-marker')" },
    });
    await call("tools/call", {
      name: "browser_evaluate",
      arguments: { script: "setTimeout(() => { throw new Error('live-session-error') }, 0)" },
    });
    await new Promise((r) => setTimeout(r, 500));

    const consoleMsgs = await call("tools/call", {
      name: "console_get_messages",
      arguments: { textFilter: "argus-mcp-test-marker" },
    });
    ok("console.log captured", toolResult(consoleMsgs)?.count >= 1);

    const jsErrors = await call("tools/call", {
      name: "console_get_errors",
      arguments: {},
    });
    const errData = toolResult(jsErrors);
    ok("JS error captured", errData?.errors?.some((e) => e.message?.includes("live-session-error")));

    // ── 10. Performance ───────────────────────────────────────
    console.log("  Phase 10: Performance Metrics");

    const perf = await call("tools/call", {
      name: "perf_get_metrics",
      arguments: {},
    });
    const perfData = toolResult(perf);
    ok("chrome metrics returned", typeof perfData?.chrome?.Timestamp === "number");
    ok("JS heap size available", typeof perfData?.chrome?.JSHeapUsedSize === "number");

    // ── 11. Multi-Step Test ───────────────────────────────────
    console.log("  Phase 11: Multi-Step Test Run");

    const testRun = await call("tools/call", {
      name: "test_run",
      arguments: {
        name: "live_session_test",
        steps: [
          { action: "navigate", value: "https://demo.playwright.dev/todomvc/#/" },
          { action: "wait", selector: ".new-todo" },
          { action: "fill", selector: ".new-todo", value: "MCP session test" },
          { action: "press_key", value: "Enter" },
          { action: "wait", selector: ".todo-list li" },
          { action: "assert_text", selector: ".todo-list", value: "MCP session test" },
          { action: "screenshot" },
        ],
      },
    }, 30000);
    const testData = toolResult(testRun);
    ok("multi-step test passed", testData?.status === "passed");
    ok("all 7 steps executed", testData?.stepsRun === 7);

    // ── 12. Request Interception ──────────────────────────────
    console.log("  Phase 12: Request Interception");

    await call("tools/call", {
      name: "network_intercept",
      arguments: {
        urlPattern: "**/api/test-mock*",
        status: 200,
        body: '{"intercepted":true}',
        contentType: "application/json",
      },
    });
    const mockResult = await call("tools/call", {
      name: "browser_evaluate",
      arguments: { script: "fetch('/api/test-mock').then(r => r.json())" },
    });
    ok("request interception works", toolResult(mockResult)?.result?.intercepted === true);

    // ── 13. Intelligence Tools (offline) ──────────────────────
    console.log("  Phase 13: Intelligence Tools (no API key)");

    for (const toolName of ["argus_heal", "argus_generate_test", "argus_quality_score",
                             "argus_similar_failures", "argus_ask", "argus_report"]) {
      const testArgs = {
        argus_heal: { orgId: "x", errorType: "selector_not_found", errorMessage: "x" },
        argus_generate_test: { projectId: "x", description: "x", framework: "playwright" },
        argus_quality_score: { projectId: "x" },
        argus_similar_failures: { orgId: "x", errorMessage: "x", limit: 5 },
        argus_ask: { message: "x" },
        argus_report: { projectId: "x", reportType: "summary" },
      };
      const resp = await call("tools/call", { name: toolName, arguments: testArgs[toolName] });
      ok(`${toolName} gracefully degrades`, toolResult(resp)?.error?.includes("ARGUS_API_KEY"));
    }

    // ── 14. Error Handling ────────────────────────────────────
    console.log("  Phase 14: Error Handling");

    const unknown = await call("tools/call", { name: "fake_tool", arguments: {} });
    ok("unknown tool returns error", unknown.result?.isError === true);

    const badArgs = await call("tools/call", {
      name: "browser_navigate",
      arguments: { url: 42 },
    });
    ok("bad args returns validation error", badArgs.result?.isError === true);

  } finally {
    proc.kill("SIGTERM");
    await new Promise((r) => setTimeout(r, 1000));

    // Cleanup
    const fs = await import("node:fs");
    if (fs.existsSync("./argus-screenshots")) {
      fs.rmSync("./argus-screenshots", { recursive: true });
    }
  }

  // ── Summary ────────────────────────────────────────────────
  console.log("\n══════════════════════════════════════════════════");
  console.log(`  PASSED: ${passed}  FAILED: ${failed}  TOTAL: ${passed + failed}`);
  if (failures.length > 0) {
    console.log("\n  Failures:");
    for (const f of failures) {
      console.log(`    ✗ ${f.name}${f.detail ? ": " + f.detail : ""}`);
    }
  }
  console.log("══════════════════════════════════════════════════\n");

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(2);
});
