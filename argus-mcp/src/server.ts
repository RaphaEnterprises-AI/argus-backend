import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z, type ZodObject, type ZodRawShape } from "zod";
import { zodToJsonSchema } from "./zod-to-json.js";
import { BrowserEngine } from "./browser/engine.js";
import { ScreenshotManager } from "./browser/screenshots.js";
import { ArgusClient } from "./argus-client.js";
import { loadConfig } from "./config.js";
import { browserTools } from "./tools/browser-tools.js";
import { networkTools } from "./tools/network-tools.js";
import { consoleTools } from "./tools/console-tools.js";
import { performanceTools } from "./tools/performance-tools.js";
import { testTools } from "./tools/test-tools.js";
import { intelligenceTools } from "./tools/intelligence-tools.js";

export interface ToolDef {
  name: string;
  description: string;
  inputSchema: ZodObject<ZodRawShape>;
  handler: (args: any) => Promise<any>;
}

export function createServer(): Server {
  const config = loadConfig();
  const engine = new BrowserEngine(config);
  const screenshots = new ScreenshotManager(config);
  const argusClient = new ArgusClient(config);

  // Collect all tools
  const tools: ToolDef[] = [
    ...browserTools(engine, screenshots),
    ...networkTools(engine),
    ...consoleTools(engine),
    ...performanceTools(engine),
    ...testTools(engine, screenshots),
    ...intelligenceTools(argusClient),
  ];

  const toolMap = new Map<string, ToolDef>();
  for (const tool of tools) {
    toolMap.set(tool.name, tool);
  }

  const server = new Server(
    { name: "argus-mcp", version: "1.0.0" },
    { capabilities: { tools: {} } }
  );

  // List tools
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: tools.map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: zodToJsonSchema(t.inputSchema),
      })),
    };
  });

  // Call tool
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: rawArgs } = request.params;
    const tool = toolMap.get(name);

    if (!tool) {
      return {
        content: [{ type: "text", text: `Unknown tool: ${name}` }],
        isError: true,
      };
    }

    try {
      // Validate and parse input with Zod
      const args = tool.inputSchema.parse(rawArgs ?? {});
      const result = await tool.handler(args);

      // Return result as text content
      const text =
        typeof result === "string" ? result : JSON.stringify(result, null, 2);

      return {
        content: [{ type: "text", text }],
      };
    } catch (err: any) {
      const message = err instanceof z.ZodError
        ? `Validation error: ${err.errors.map((e) => `${e.path.join(".")}: ${e.message}`).join(", ")}`
        : err.message || String(err);

      return {
        content: [{ type: "text", text: message }],
        isError: true,
      };
    }
  });

  // Cleanup on server close
  server.onclose = async () => {
    await engine.close();
  };

  return server;
}
