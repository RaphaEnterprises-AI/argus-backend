import { z } from "zod";
import type { BrowserEngine } from "../browser/engine.js";
import type { ToolDef } from "../server.js";

export function consoleTools(engine: BrowserEngine): ToolDef[] {
  return [
    {
      name: "console_get_messages",
      description:
        "Get all console messages (log, warn, info, debug) captured since page load.",
      inputSchema: z.object({
        typeFilter: z
          .enum(["log", "warning", "error", "info", "debug"])
          .optional()
          .describe("Filter by console message type"),
        textFilter: z
          .string()
          .optional()
          .describe("Filter messages containing this text"),
      }),
      handler: async ({ typeFilter, textFilter }) => {
        const cdp = await engine.ensureCDP();
        let messages = cdp.getConsoleMessages();

        if (typeFilter) {
          messages = messages.filter((m) => m.type === typeFilter);
        }
        if (textFilter) {
          messages = messages.filter((m) => m.text.includes(textFilter));
        }

        return { count: messages.length, messages };
      },
    },
    {
      name: "console_get_errors",
      description:
        "Get all JavaScript errors (uncaught exceptions) with stack traces, captured since page load.",
      inputSchema: z.object({}),
      handler: async () => {
        const cdp = await engine.ensureCDP();
        const errors = cdp.getJSErrors();
        return { count: errors.length, errors };
      },
    },
  ];
}
