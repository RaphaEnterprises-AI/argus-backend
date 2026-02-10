import { z } from "zod";
import type { BrowserEngine } from "../browser/engine.js";
import type { ScreenshotManager } from "../browser/screenshots.js";
import type { ToolDef } from "../server.js";

export function browserTools(
  engine: BrowserEngine,
  screenshots: ScreenshotManager
): ToolDef[] {
  return [
    {
      name: "browser_navigate",
      description:
        "Navigate to a URL. Returns the page title and current URL after navigation.",
      inputSchema: z.object({
        url: z.string().describe("The URL to navigate to"),
        waitUntil: z
          .enum(["load", "domcontentloaded", "networkidle", "commit"])
          .default("load")
          .describe("When to consider navigation complete"),
      }),
      handler: async ({ url, waitUntil }) => {
        const page = await engine.ensurePage();
        await page.goto(url, { waitUntil });
        return {
          title: await page.title(),
          url: page.url(),
        };
      },
    },
    {
      name: "browser_click",
      description:
        "Click an element matching the given CSS selector or text.",
      inputSchema: z.object({
        selector: z.string().describe("CSS selector or text= selector"),
        button: z
          .enum(["left", "right", "middle"])
          .default("left")
          .describe("Mouse button"),
        clickCount: z.number().default(1).describe("Number of clicks"),
      }),
      handler: async ({ selector, button, clickCount }) => {
        const page = await engine.ensurePage();
        await page.click(selector, { button, clickCount });
        return { clicked: selector };
      },
    },
    {
      name: "browser_fill",
      description: "Clear an input and fill it with the given value.",
      inputSchema: z.object({
        selector: z.string().describe("CSS selector for the input element"),
        value: z.string().describe("Value to fill"),
      }),
      handler: async ({ selector, value }) => {
        const page = await engine.ensurePage();
        await page.fill(selector, value);
        return { filled: selector, value };
      },
    },
    {
      name: "browser_type",
      description:
        "Type text character by character (with key events). Use browser_fill for bulk input.",
      inputSchema: z.object({
        selector: z.string().describe("CSS selector for the element"),
        text: z.string().describe("Text to type"),
        delay: z
          .number()
          .default(50)
          .describe("Delay between keystrokes in ms"),
      }),
      handler: async ({ selector, text, delay }) => {
        const page = await engine.ensurePage();
        await page.type(selector, text, { delay });
        return { typed: text, into: selector };
      },
    },
    {
      name: "browser_select",
      description: "Select an option from a <select> dropdown.",
      inputSchema: z.object({
        selector: z.string().describe("CSS selector for the select element"),
        value: z.string().describe("Value or label of the option to select"),
      }),
      handler: async ({ selector, value }) => {
        const page = await engine.ensurePage();
        const selected = await page.selectOption(selector, value);
        return { selected };
      },
    },
    {
      name: "browser_hover",
      description: "Hover over an element.",
      inputSchema: z.object({
        selector: z.string().describe("CSS selector for the element"),
      }),
      handler: async ({ selector }) => {
        const page = await engine.ensurePage();
        await page.hover(selector);
        return { hovered: selector };
      },
    },
    {
      name: "browser_press_key",
      description:
        'Press a keyboard key (e.g. "Enter", "Tab", "Escape", "ArrowDown").',
      inputSchema: z.object({
        key: z.string().describe("Key to press (Playwright key name)"),
        selector: z
          .string()
          .optional()
          .describe("Optional selector to focus first"),
      }),
      handler: async ({ key, selector }) => {
        const page = await engine.ensurePage();
        if (selector) {
          await page.press(selector, key);
        } else {
          await page.keyboard.press(key);
        }
        return { pressed: key };
      },
    },
    {
      name: "browser_screenshot",
      description:
        "Capture a screenshot of the current page. Returns the file path and base64 image data.",
      inputSchema: z.object({
        fullPage: z
          .boolean()
          .default(false)
          .describe("Capture the full scrollable page"),
        name: z.string().optional().describe("Custom filename for the screenshot"),
        selector: z
          .string()
          .optional()
          .describe("Capture only this element instead of the full page"),
      }),
      handler: async ({ fullPage, name, selector }) => {
        const page = await engine.ensurePage();
        if (selector) {
          return await screenshots.captureElement(page, selector, name);
        }
        return await screenshots.capture(page, { fullPage, name });
      },
    },
    {
      name: "browser_scroll",
      description: "Scroll the page or an element.",
      inputSchema: z.object({
        direction: z.enum(["up", "down"]).describe("Scroll direction"),
        amount: z
          .number()
          .default(500)
          .describe("Pixels to scroll"),
        selector: z
          .string()
          .optional()
          .describe("Scroll within this element instead of the page"),
      }),
      handler: async ({ direction, amount, selector }) => {
        const page = await engine.ensurePage();
        const delta = direction === "down" ? amount : -amount;
        if (selector) {
          await page.evaluate(
            ({ sel, d }) => {
              const el = document.querySelector(sel);
              if (el) el.scrollBy(0, d);
            },
            { sel: selector, d: delta }
          );
        } else {
          await page.evaluate((d) => window.scrollBy(0, d), delta);
        }
        return { scrolled: direction, amount };
      },
    },
    {
      name: "browser_wait",
      description: "Wait for a selector to appear, or wait a fixed time.",
      inputSchema: z.object({
        selector: z
          .string()
          .optional()
          .describe("CSS selector to wait for"),
        timeout: z
          .number()
          .default(5000)
          .describe("Max wait time in ms"),
        state: z
          .enum(["attached", "detached", "visible", "hidden"])
          .default("visible")
          .describe("Element state to wait for"),
      }),
      handler: async ({ selector, timeout, state }) => {
        const page = await engine.ensurePage();
        if (selector) {
          await page.waitForSelector(selector, { timeout, state });
          return { found: selector, state };
        }
        await page.waitForTimeout(timeout);
        return { waited: `${timeout}ms` };
      },
    },
    {
      name: "browser_evaluate",
      description:
        "Execute JavaScript in the browser page context. Returns the result.",
      inputSchema: z.object({
        script: z
          .string()
          .describe("JavaScript code to execute in the page"),
      }),
      handler: async ({ script }) => {
        const page = await engine.ensurePage();
        const result = await page.evaluate(script);
        return { result };
      },
    },
    {
      name: "browser_snapshot",
      description:
        "Get the accessibility tree of the page (useful for understanding page structure without screenshots).",
      inputSchema: z.object({
        selector: z
          .string()
          .optional()
          .describe("Root element for the snapshot (default: full page)"),
      }),
      handler: async ({ selector }) => {
        const page = await engine.ensurePage();
        const locator = selector ? page.locator(selector) : page.locator("body");
        const snapshot = await locator.ariaSnapshot();
        return { snapshot };
      },
    },
  ];
}
