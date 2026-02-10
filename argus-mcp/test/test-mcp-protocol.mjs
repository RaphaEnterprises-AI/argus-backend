/**
 * Test the MCP server via stdio protocol — simulates how Claude Code
 * communicates with MCP servers.
 *
 * The MCP SDK v1.3.0 uses newline-delimited JSON (NDJSON) for stdio:
 *   send:    JSON.stringify(message) + '\n'
 *   receive: split on '\n', JSON.parse each line
 */

import { spawn } from "node:child_process";

let passed = 0;
let failed = 0;

function test(name, pass) {
  if (pass) {
    passed++;
    console.log(`  ✓ ${name}`);
  } else {
    failed++;
    console.log(`  ✗ ${name}`);
  }
}

async function main() {
  console.log("\n── MCP Protocol Integration Tests ────────────");

  const proc = spawn("node", ["dist/index.js"], {
    cwd: process.cwd(),
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ARGUS_HEADLESS: "true" },
  });

  let stdout = "";
  proc.stdout.on("data", (chunk) => { stdout += chunk.toString(); });

  let stderr = "";
  proc.stderr.on("data", (chunk) => { stderr += chunk.toString(); });

  // Helper: send NDJSON message and wait for response line
  async function sendAndWait(msg, timeoutMs = 15000) {
    stdout = "";
    proc.stdin.write(JSON.stringify(msg) + "\n");
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const lines = stdout.split("\n").filter((l) => l.trim());
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          // Match response by id (skip notifications)
          if (msg.id !== undefined && parsed.id === msg.id) return parsed;
          // For notifications (no id), return first JSON-RPC response
          if (msg.id === undefined && parsed.jsonrpc) return parsed;
        } catch { /* incomplete line, keep waiting */ }
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error(`Timeout waiting for response to ${msg.method} (id=${msg.id}). stderr: ${stderr.slice(-200)}`);
  }

  try {
    // 1. Initialize
    const initResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "test-client", version: "1.0.0" },
      },
    });

    test("initialize returns server info", initResp.result?.serverInfo?.name === "argus-mcp");
    test("initialize returns capabilities", initResp.result?.capabilities?.tools !== undefined);
    test("initialize returns protocol version", !!initResp.result?.protocolVersion);

    // Send initialized notification (no id = notification, no response expected)
    proc.stdin.write(JSON.stringify({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    }) + "\n");
    await new Promise((r) => setTimeout(r, 300));

    // 2. List tools
    const listResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });

    const tools = listResp.result?.tools || [];
    test("tools/list returns 30 tools", tools.length === 30);
    test("tools have name property", tools.every((t) => t.name));
    test("tools have description property", tools.every((t) => t.description));
    test("tools have inputSchema property", tools.every((t) => t.inputSchema));

    const toolNames = tools.map((t) => t.name);
    test("includes browser_navigate", toolNames.includes("browser_navigate"));
    test("includes network_get_requests", toolNames.includes("network_get_requests"));
    test("includes console_get_errors", toolNames.includes("console_get_errors"));
    test("includes perf_get_metrics", toolNames.includes("perf_get_metrics"));
    test("includes test_run", toolNames.includes("test_run"));
    test("includes argus_heal", toolNames.includes("argus_heal"));

    // 3. Call a tool — browser_navigate
    const navResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "browser_navigate",
        arguments: { url: "data:text/html,<h1>MCP Protocol Test</h1>" },
      },
    });

    const navContent = navResp.result?.content?.[0]?.text;
    test("tools/call returns content", !!navContent);
    const navData = JSON.parse(navContent);
    test("browser_navigate returns title", navData.title !== undefined);
    test("browser_navigate returns url", navData.url?.startsWith("data:"));

    // 4. Call browser_evaluate
    const evalResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 4,
      method: "tools/call",
      params: {
        name: "browser_evaluate",
        arguments: { script: "document.querySelector('h1').textContent" },
      },
    });

    const evalData = JSON.parse(evalResp.result?.content?.[0]?.text);
    test("browser_evaluate returns result", evalData.result === "MCP Protocol Test");

    // 5. Call unknown tool — should return error
    const unknownResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 5,
      method: "tools/call",
      params: {
        name: "nonexistent_tool",
        arguments: {},
      },
    });

    test("unknown tool returns isError", unknownResp.result?.isError === true);
    test("unknown tool returns error message", unknownResp.result?.content?.[0]?.text?.includes("Unknown tool"));

    // 6. Call with invalid arguments — should return validation error
    const badArgsResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 6,
      method: "tools/call",
      params: {
        name: "browser_navigate",
        arguments: { url: 12345 },  // url should be string
      },
    });

    test("invalid args returns isError", badArgsResp.result?.isError === true);
    test("invalid args mentions validation", badArgsResp.result?.content?.[0]?.text?.includes("Validation") ||
      badArgsResp.result?.content?.[0]?.text?.includes("Expected string"));

    // 7. Intelligence tool without API key
    const healResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 7,
      method: "tools/call",
      params: {
        name: "argus_heal",
        arguments: {
          orgId: "test",
          errorType: "selector_not_found",
          errorMessage: "Element not found",
        },
      },
    });

    const healData = JSON.parse(healResp.result?.content?.[0]?.text);
    test("argus_heal without key returns config message", healData.error?.includes("ARGUS_API_KEY"));

    // 8. Call browser_screenshot via protocol
    const shotResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 8,
      method: "tools/call",
      params: {
        name: "browser_screenshot",
        arguments: { fullPage: false },
      },
    });

    const shotData = JSON.parse(shotResp.result?.content?.[0]?.text);
    test("browser_screenshot returns path", !!shotData.path);
    test("browser_screenshot returns base64", shotData.base64?.length > 100);

    // 9. Call test_assert via protocol (text_contains on body)
    const assertResp = await sendAndWait({
      jsonrpc: "2.0",
      id: 9,
      method: "tools/call",
      params: {
        name: "test_assert",
        arguments: {
          assertion: "text_contains",
          selector: "h1",
          expected: "MCP Protocol Test",
        },
      },
    });

    const assertData = JSON.parse(assertResp.result?.content?.[0]?.text);
    test("test_assert via protocol works", assertData.pass === true);

  } finally {
    proc.kill("SIGTERM");
    await new Promise((r) => setTimeout(r, 1000));
  }

  // Clean up screenshots
  const fs = await import("node:fs");
  const dir = "./argus-screenshots";
  if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true });

  // Summary
  console.log(`\n══════════════════════════════════════════════════`);
  console.log(`  PASSED: ${passed}  FAILED: ${failed}  TOTAL: ${passed + failed}`);
  console.log(`══════════════════════════════════════════════════\n`);

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(2);
});
