import { z } from "zod";
import type { BrowserEngine } from "../browser/engine.js";
import type { ScreenshotManager } from "../browser/screenshots.js";
import type { ToolDef } from "../server.js";

export function testTools(
  engine: BrowserEngine,
  screenshots: ScreenshotManager
): ToolDef[] {
  return [
    {
      name: "test_run",
      description:
        "Run a multi-step test with automatic screenshots at each step. Each step is an action (navigate, click, fill, assert) with a selector/value.",
      inputSchema: z.object({
        name: z.string().describe("Test name for the report"),
        steps: z
          .array(
            z.object({
              action: z
                .enum([
                  "navigate",
                  "click",
                  "fill",
                  "type",
                  "press_key",
                  "select",
                  "wait",
                  "assert_text",
                  "assert_visible",
                  "assert_url",
                  "screenshot",
                ])
                .describe("Action to perform"),
              selector: z
                .string()
                .optional()
                .describe("CSS selector (for click, fill, type, wait, assert_visible)"),
              value: z
                .string()
                .optional()
                .describe("Value (URL for navigate, text for fill/type/assert_text, key for press_key, pattern for assert_url)"),
            })
          )
          .describe("Ordered list of test steps"),
      }),
      handler: async ({ name, steps }) => {
        const page = await engine.ensurePage();
        const results: Array<{
          step: number;
          action: string;
          status: "passed" | "failed";
          error?: string;
          screenshot?: string;
        }> = [];

        for (let i = 0; i < steps.length; i++) {
          const step = steps[i];
          try {
            switch (step.action) {
              case "navigate":
                await page.goto(step.value!, { waitUntil: "load" });
                break;
              case "click":
                await page.click(step.selector!);
                break;
              case "fill":
                await page.fill(step.selector!, step.value!);
                break;
              case "type":
                await page.type(step.selector!, step.value!);
                break;
              case "press_key":
                await page.keyboard.press(step.value!);
                break;
              case "select":
                await page.selectOption(step.selector!, step.value!);
                break;
              case "wait":
                if (step.selector) {
                  await page.waitForSelector(step.selector);
                } else {
                  await page.waitForTimeout(parseInt(step.value || "1000", 10));
                }
                break;
              case "assert_text": {
                const text = await page.textContent(step.selector || "body");
                if (!text?.includes(step.value!)) {
                  throw new Error(
                    `Expected text "${step.value}" not found in "${step.selector || "body"}"`
                  );
                }
                break;
              }
              case "assert_visible":
                await page.waitForSelector(step.selector!, {
                  state: "visible",
                  timeout: 5000,
                });
                break;
              case "assert_url": {
                const url = page.url();
                if (!url.match(new RegExp(step.value!))) {
                  throw new Error(
                    `URL "${url}" does not match pattern "${step.value}"`
                  );
                }
                break;
              }
              case "screenshot":
                break; // Screenshot is taken below for every step
            }

            // Capture screenshot after each step
            const shot = await screenshots.capture(page, {
              name: `${name}_step${i + 1}_${step.action}.png`,
            });

            results.push({
              step: i + 1,
              action: step.action,
              status: "passed",
              screenshot: shot.path,
            });
          } catch (err: any) {
            // Capture failure screenshot
            let screenshotPath: string | undefined;
            try {
              const shot = await screenshots.capture(page, {
                name: `${name}_step${i + 1}_FAIL.png`,
              });
              screenshotPath = shot.path;
            } catch {
              /* screenshot itself may fail */
            }

            results.push({
              step: i + 1,
              action: step.action,
              status: "failed",
              error: err.message,
              screenshot: screenshotPath,
            });
            break; // Stop on first failure
          }
        }

        const passed = results.every((r) => r.status === "passed");
        return {
          name,
          status: passed ? "passed" : "failed",
          stepsRun: results.length,
          totalSteps: steps.length,
          results,
        };
      },
    },
    {
      name: "test_discover",
      description:
        "Crawl the current page and discover all interactive elements with their selectors. Useful for understanding what can be tested.",
      inputSchema: z.object({
        includeHidden: z
          .boolean()
          .default(false)
          .describe("Include hidden elements"),
      }),
      handler: async ({ includeHidden }) => {
        const page = await engine.ensurePage();
        const elements: Array<{
          tag: string;
          type?: string;
          text?: string;
          selector: string;
          role?: string;
          name?: string;
          visible: boolean;
        }> = await page.evaluate((showHidden: boolean) => {
          const interactiveSelectors = [
            "a[href]",
            "button",
            'input:not([type="hidden"])',
            "textarea",
            "select",
            '[role="button"]',
            '[role="link"]',
            '[role="checkbox"]',
            '[role="radio"]',
            '[role="tab"]',
            '[role="menuitem"]',
            "[onclick]",
            "[contenteditable]",
          ];

          const found: Array<{
            tag: string;
            type?: string;
            text?: string;
            selector: string;
            role?: string;
            name?: string;
            visible: boolean;
          }> = [];
          const seen = new Set<Element>();

          for (const sel of interactiveSelectors) {
            for (const el of document.querySelectorAll(sel)) {
              if (seen.has(el)) continue;
              seen.add(el);

              const rect = el.getBoundingClientRect();
              const isVisible = rect.width > 0 && rect.height > 0;
              if (!showHidden && !isVisible) continue;

              // Build a useful selector
              let cssSelector = el.tagName.toLowerCase();
              if (el.id) {
                cssSelector = `#${el.id}`;
              } else if (el.getAttribute("data-testid")) {
                cssSelector = `[data-testid="${el.getAttribute("data-testid")}"]`;
              } else if (
                el.getAttribute("aria-label")
              ) {
                cssSelector = `[aria-label="${el.getAttribute("aria-label")}"]`;
              } else if (el.getAttribute("name")) {
                cssSelector = `${el.tagName.toLowerCase()}[name="${el.getAttribute("name")}"]`;
              }

              found.push({
                tag: el.tagName.toLowerCase(),
                type: (el as HTMLInputElement).type || undefined,
                text: (el.textContent || "").trim().slice(0, 100) || undefined,
                selector: cssSelector,
                role: el.getAttribute("role") || undefined,
                name:
                  el.getAttribute("aria-label") ||
                  el.getAttribute("name") ||
                  undefined,
                visible: isVisible,
              });
            }
          }
          return found;
        }, includeHidden);

        return {
          url: page.url(),
          elementCount: elements.length,
          elements,
        };
      },
    },
    {
      name: "test_assert",
      description:
        "Assert a condition on the current page. Returns pass/fail with details.",
      inputSchema: z.object({
        assertion: z
          .enum([
            "text_contains",
            "text_equals",
            "element_visible",
            "element_hidden",
            "element_count",
            "url_matches",
            "title_contains",
          ])
          .describe("Type of assertion"),
        selector: z
          .string()
          .optional()
          .describe("CSS selector (for element assertions)"),
        expected: z
          .string()
          .describe("Expected value (text, count, URL pattern, or title text)"),
      }),
      handler: async ({ assertion, selector, expected }) => {
        const page = await engine.ensurePage();

        switch (assertion) {
          case "text_contains": {
            const text = await page.textContent(selector || "body");
            const pass = text?.includes(expected) ?? false;
            return {
              assertion,
              pass,
              actual: text?.slice(0, 200),
              expected,
            };
          }
          case "text_equals": {
            const text = (
              await page.textContent(selector || "body")
            )?.trim();
            const pass = text === expected;
            return { assertion, pass, actual: text?.slice(0, 200), expected };
          }
          case "element_visible": {
            const visible = await page
              .locator(selector!)
              .isVisible()
              .catch(() => false);
            return { assertion, pass: visible, selector };
          }
          case "element_hidden": {
            const visible = await page
              .locator(selector!)
              .isVisible()
              .catch(() => false);
            return { assertion, pass: !visible, selector };
          }
          case "element_count": {
            const count = await page.locator(selector!).count();
            const expectedCount = parseInt(expected, 10);
            return {
              assertion,
              pass: count === expectedCount,
              actual: count,
              expected: expectedCount,
            };
          }
          case "url_matches": {
            const url = page.url();
            const pass = !!url.match(new RegExp(expected));
            return { assertion, pass, actual: url, expected };
          }
          case "title_contains": {
            const title = await page.title();
            const pass = title.includes(expected);
            return { assertion, pass, actual: title, expected };
          }
        }
      },
    },
  ];
}
