/**
 * Argus Worker - Browser Automation API (Simplified)
 * https://heyargus.ai
 *
 * Supports multiple browser backends:
 * - Cloudflare Browser Rendering (free, Chromium only)
 * - TestingBot (paid, all browsers + real devices)
 *
 * Endpoints:
 *   POST /act      - Execute browser actions
 *   POST /extract  - Extract structured data
 *   POST /observe  - Discover available actions
 *   POST /agent    - Run autonomous multi-step workflows
 *   POST /test     - Run tests with multiple steps
 *   GET  /health   - Health check
 *
 * NOTE: Intelligence features (webhooks, quality intelligence, test generation)
 * have been moved to the Brain service (Python/Railway).
 */

import { z } from "zod";
import { chromium as cfChromium } from "@cloudflare/playwright";
import { createCache, createStorage } from "./utils";
import { RealtimeSession, broadcastEvent } from "./realtime";

// ============================================================================
// ENVIRONMENT & TYPES
// ============================================================================

interface Env {
  // Browser Rendering
  BROWSER: BrowserBinding;

  // Workers AI
  AI: Ai;

  // KV Namespace for caching
  CACHE?: KVNamespace;

  // R2 Bucket for artifact storage
  ARTIFACTS?: R2Bucket;

  // Vectorize for semantic pattern matching
  VECTOR_INDEX?: VectorizeIndex;

  // Queue for async event processing
  EVENT_QUEUE?: Queue;
  DLQ?: Queue;

  // Hyperdrive for database connection pooling
  DB?: Hyperdrive;

  // Durable Object for realtime WebSocket
  REALTIME?: DurableObjectNamespace;

  // External service credentials
  TESTINGBOT_KEY?: string;
  TESTINGBOT_SECRET?: string;
  OPENAI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
  API_TOKEN?: string;

  // Supabase credentials (for Brain communication)
  SUPABASE_URL?: string;
  SUPABASE_SERVICE_KEY?: string;

  // Railway Brain URL
  BRAIN_URL?: string;

  // Configuration
  DEFAULT_MODEL_PROVIDER: string;
  DEFAULT_BACKEND: string;
  ENABLE_CACHING: string;
  ENABLE_SELF_HEALING: string;
}

type Backend = "cloudflare" | "testingbot" | "auto";
type BrowserType = "chrome" | "firefox" | "safari" | "edge" | "webkit";
type Platform = "windows" | "macos" | "linux";

// ============================================================================
// DEVICE & BROWSER CONFIGURATIONS
// ============================================================================

const DEVICE_PRESETS = {
  // Desktop
  "desktop": { viewport: { width: 1920, height: 1080 }, userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0", isMobile: false, hasTouch: false },
  "desktop-hd": { viewport: { width: 2560, height: 1440 }, userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0", isMobile: false, hasTouch: false },
  "desktop-mac": { viewport: { width: 1920, height: 1080 }, userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/120.0.0.0", isMobile: false, hasTouch: false },

  // Tablets
  "tablet": { viewport: { width: 768, height: 1024 }, userAgent: "Mozilla/5.0 (iPad; CPU OS 17_0) AppleWebKit/605.1.15 Mobile/15E148", isMobile: true, hasTouch: true },
  "tablet-landscape": { viewport: { width: 1024, height: 768 }, userAgent: "Mozilla/5.0 (iPad; CPU OS 17_0) AppleWebKit/605.1.15 Mobile/15E148", isMobile: true, hasTouch: true },
  "tablet-android": { viewport: { width: 800, height: 1280 }, userAgent: "Mozilla/5.0 (Linux; Android 14; Tab) AppleWebKit/537.36 Chrome/120.0.0.0", isMobile: true, hasTouch: true },

  // Mobile
  "mobile": { viewport: { width: 375, height: 812 }, userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605.1.15 Mobile/15E148", isMobile: true, hasTouch: true },
  "mobile-android": { viewport: { width: 412, height: 915 }, userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/120.0.0.0 Mobile", isMobile: true, hasTouch: true },
  "mobile-landscape": { viewport: { width: 812, height: 375 }, userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605.1.15 Mobile/15E148", isMobile: true, hasTouch: true },

  // Real devices (TestingBot only)
  "iphone-15": { viewport: { width: 393, height: 852 }, userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605.1.15", isMobile: true, hasTouch: true, realDevice: true, deviceName: "iPhone 15" },
  "iphone-14": { viewport: { width: 390, height: 844 }, userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0) AppleWebKit/605.1.15", isMobile: true, hasTouch: true, realDevice: true, deviceName: "iPhone 14" },
  "pixel-8": { viewport: { width: 412, height: 915 }, userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 8)", isMobile: true, hasTouch: true, realDevice: true, deviceName: "Pixel 8" },
  "pixel-7": { viewport: { width: 412, height: 915 }, userAgent: "Mozilla/5.0 (Linux; Android 13; Pixel 7)", isMobile: true, hasTouch: true, realDevice: true, deviceName: "Pixel 7" },
  "samsung-s24": { viewport: { width: 412, height: 915 }, userAgent: "Mozilla/5.0 (Linux; Android 14; Samsung S24)", isMobile: true, hasTouch: true, realDevice: true, deviceName: "Samsung Galaxy S24" },
};

type DeviceType = keyof typeof DEVICE_PRESETS;

// TestingBot browser configurations
const TESTINGBOT_BROWSERS: Record<BrowserType, { browserName: string; versions: string[] }> = {
  chrome: { browserName: "chrome", versions: ["latest", "latest-1", "120", "119", "118"] },
  firefox: { browserName: "firefox", versions: ["latest", "latest-1", "120", "119", "118"] },
  safari: { browserName: "safari", versions: ["latest", "17", "16"] },
  edge: { browserName: "MicrosoftEdge", versions: ["latest", "latest-1", "120", "119"] },
  webkit: { browserName: "webkit", versions: ["latest"] },
};

const TESTINGBOT_PLATFORMS: Record<Platform, string> = {
  windows: "WIN11",
  macos: "SONOMA",
  linux: "LINUX",
};

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

interface Action {
  selector: string;
  description: string;
  method?: string;
  arguments?: string[];
}

interface ActResult {
  success: boolean;
  message: string;
  actionDescription: string;
  actions: Action[];
  backend?: string;
  browser?: string;
}

interface AgentResult {
  success: boolean;
  message: string;
  actions: Action[];
  completed: boolean;
  usage: { inputTokens: number; outputTokens: number; totalSteps: number };
  backend?: string;
}

interface TestResult {
  success: boolean;
  steps: Array<{ instruction: string; success: boolean; error?: string }>;
  browsers?: BrowserResult[];
  screenshot?: string;
  backend?: string;
}

interface BrowserResult {
  browser: string;
  platform: string;
  device?: string;
  success: boolean;
  error?: string;
  screenshot?: string;
}

// ============================================================================
// REQUEST SCHEMAS
// ============================================================================

const BackendSchema = z.enum(["cloudflare", "testingbot", "auto"]).optional().default("auto");
const BrowserSchema = z.enum(["chrome", "firefox", "safari", "edge", "webkit"]).optional().default("chrome");
const DeviceSchema = z.enum([
  "desktop", "desktop-hd", "desktop-mac",
  "tablet", "tablet-landscape", "tablet-android",
  "mobile", "mobile-android", "mobile-landscape",
  "iphone-15", "iphone-14", "pixel-8", "pixel-7", "samsung-s24"
]).optional().default("desktop");

const ActRequestSchema = z.object({
  url: z.string().url(),
  instruction: z.string().optional(),
  action: z.object({
    selector: z.string(),
    description: z.string(),
    method: z.string(),
    arguments: z.array(z.string()).optional().default([]),
  }).optional(),
  variables: z.record(z.string()).optional(),
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  browser: BrowserSchema,
  device: DeviceSchema,
  platform: z.enum(["windows", "macos", "linux"]).optional().default("windows"),
  selfHeal: z.boolean().optional().default(true),
  screenshot: z.boolean().optional().default(false),
});

const ExtractRequestSchema = z.object({
  url: z.string().url(),
  instruction: z.string().optional(),
  schema: z.record(z.any()).optional(),
  selector: z.string().optional(),
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  browser: BrowserSchema,
  device: DeviceSchema,
});

const ObserveRequestSchema = z.object({
  url: z.string().url(),
  instruction: z.string().optional().default("Find all interactive elements"),
  selector: z.string().optional(),
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  browser: BrowserSchema,
  device: DeviceSchema,
});

const AgentRequestSchema = z.object({
  url: z.string().url(),
  instruction: z.string(),
  systemPrompt: z.string().optional(),
  maxSteps: z.number().optional().default(20),
  timeout: z.number().optional().default(60000),
  backend: BackendSchema,
  browser: BrowserSchema,
  device: DeviceSchema,
  captureScreenshots: z.boolean().optional().default(false),
});

const TestRequestSchema = z.object({
  url: z.string().url(),
  steps: z.array(z.string()),
  screenshot: z.boolean().optional().default(false),
  captureScreenshots: z.boolean().optional().default(false),
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  browsers: z.array(BrowserSchema).optional(),
  device: DeviceSchema,
  devices: z.array(DeviceSchema).optional(),
});

// ============================================================================
// TESTINGBOT WEBDRIVER CLIENT (W3C WebDriver Protocol)
// ============================================================================

class TestingBotSession {
  private sessionId: string;
  private baseUrl: string;
  private auth: string;

  constructor(sessionId: string, key: string, secret: string) {
    this.sessionId = sessionId;
    this.baseUrl = "https://hub.testingbot.com/wd/hub";
    this.auth = btoa(`${key}:${secret}`);
  }

  private async request(method: string, path: string, body?: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}/session/${this.sessionId}${path}`, {
      method,
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Basic ${this.auth}`,
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json();
    if (data.value?.error) throw new Error(data.value.message || data.value.error);
    return data.value;
  }

  async navigate(url: string): Promise<void> {
    await this.request("POST", "/url", { url });
  }

  async getUrl(): Promise<string> {
    return await this.request("GET", "/url");
  }

  async getPageSource(): Promise<string> {
    return await this.request("GET", "/source");
  }

  async findElement(selector: string): Promise<string> {
    const result = await this.request("POST", "/element", {
      using: selector.startsWith("#") || selector.startsWith(".") || selector.includes("[") ? "css selector" : "xpath",
      value: selector,
    });
    return Object.values(result)[0] as string;
  }

  async findElements(selector: string): Promise<string[]> {
    const result = await this.request("POST", "/elements", {
      using: selector.startsWith("#") || selector.startsWith(".") || selector.includes("[") ? "css selector" : "xpath",
      value: selector,
    });
    return result.map((r: any) => Object.values(r)[0] as string);
  }

  async clickElement(elementId: string): Promise<void> {
    await this.request("POST", `/element/${elementId}/click`, {});
  }

  async sendKeys(elementId: string, text: string): Promise<void> {
    await this.request("POST", `/element/${elementId}/value`, { text });
  }

  async clearElement(elementId: string): Promise<void> {
    await this.request("POST", `/element/${elementId}/clear`, {});
  }

  async getElementText(elementId: string): Promise<string> {
    return await this.request("GET", `/element/${elementId}/text`);
  }

  async getElementAttribute(elementId: string, name: string): Promise<string> {
    return await this.request("GET", `/element/${elementId}/attribute/${name}`);
  }

  async executeScript(script: string, args: any[] = []): Promise<any> {
    return await this.request("POST", "/execute/sync", { script, args });
  }

  async screenshot(): Promise<string> {
    return await this.request("GET", "/screenshot");
  }

  async close(): Promise<void> {
    await fetch(`${this.baseUrl}/session/${this.sessionId}`, {
      method: "DELETE",
      headers: { "Authorization": `Basic ${this.auth}` },
    }).catch(() => {});
  }
}

async function createTestingBotSession(
  env: Env,
  browserType: BrowserType,
  platform: Platform,
  deviceConfig: any,
  device: DeviceType
): Promise<TestingBotSession> {
  const tbConfig = TESTINGBOT_BROWSERS[browserType] || TESTINGBOT_BROWSERS.chrome;
  const tbPlatform = TESTINGBOT_PLATFORMS[platform] || "WIN11";

  const capabilities: Record<string, any> = {
    browserName: tbConfig.browserName,
    browserVersion: "latest",
    platformName: tbPlatform,
    "tb:options": {
      name: "Argus Browser Automation",
      screenrecorder: false,
    },
  };

  if ((deviceConfig as any).realDevice && (deviceConfig as any).deviceName) {
    capabilities.platformName = device.startsWith("iphone") ? "iOS" : "Android";
    capabilities["appium:deviceName"] = (deviceConfig as any).deviceName;
  }

  const response = await fetch("https://hub.testingbot.com/wd/hub/session", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Basic ${btoa(`${env.TESTINGBOT_KEY}:${env.TESTINGBOT_SECRET}`)}`,
    },
    body: JSON.stringify({ capabilities: { alwaysMatch: capabilities } }),
  });

  const data = await response.json() as any;
  if (!data.value?.sessionId) {
    throw new Error(`Failed to create TestingBot session: ${JSON.stringify(data)}`);
  }

  return new TestingBotSession(data.value.sessionId, env.TESTINGBOT_KEY!, env.TESTINGBOT_SECRET!);
}

// ============================================================================
// UNIFIED BROWSER INTERFACE
// ============================================================================

interface BrowserSession {
  navigate(url: string): Promise<void>;
  getUrl(): Promise<string>;
  getContent(): Promise<string>;
  click(selector: string): Promise<void>;
  fill(selector: string, value: string): Promise<void>;
  type(selector: string, value: string): Promise<void>;
  hover(selector: string): Promise<void>;
  screenshot(): Promise<string>;
  evaluate<T>(fn: string | (() => T)): Promise<T>;
  getInteractiveElements(): Promise<Array<{ tag: string; text: string; selector: string; attributes: Record<string, string> }>>;
  close(): Promise<void>;
  backendUsed: string;
}

// Cloudflare Playwright wrapper
class CloudflareBrowserSession implements BrowserSession {
  private browser: any;
  private page: any;
  backendUsed = "cloudflare";

  constructor(browser: any, page: any) {
    this.browser = browser;
    this.page = page;
  }

  async navigate(url: string): Promise<void> {
    await this.page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
  }

  async getUrl(): Promise<string> {
    return this.page.url();
  }

  async getContent(): Promise<string> {
    return await this.page.content();
  }

  async click(selector: string): Promise<void> {
    await this.page.click(selector, { timeout: 5000 });
  }

  async fill(selector: string, value: string): Promise<void> {
    await this.page.fill(selector, value, { timeout: 5000 });
  }

  async type(selector: string, value: string): Promise<void> {
    await this.page.type(selector, value, { timeout: 5000 });
  }

  async hover(selector: string): Promise<void> {
    await this.page.hover(selector, { timeout: 5000 });
  }

  async screenshot(): Promise<string> {
    const buf = await this.page.screenshot({ type: "png" });
    return Buffer.from(buf).toString("base64");
  }

  async evaluate<T>(fn: string | (() => T)): Promise<T> {
    return await this.page.evaluate(fn);
  }

  async getInteractiveElements() {
    return await this.page.evaluate(() => {
      const elements: Array<{ tag: string; text: string; selector: string; attributes: Record<string, string> }> = [];
      const getSelector = (el: Element): string => {
        if (el.id) return `#${el.id}`;
        if (el.className && typeof el.className === "string") {
          const cls = el.className.trim().split(/\s+/).slice(0, 2).join(".");
          if (cls) return `${el.tagName.toLowerCase()}.${cls}`;
        }
        return el.tagName.toLowerCase();
      };
      const getAttrs = (el: Element): Record<string, string> => {
        const attrs: Record<string, string> = {};
        ["href", "type", "name", "placeholder", "aria-label", "role"].forEach((a) => {
          const v = el.getAttribute(a);
          if (v) attrs[a] = v;
        });
        return attrs;
      };
      document.querySelectorAll('button, a[href], input:not([type="hidden"]), textarea, select, [role="button"], [onclick]').forEach((el) => {
        elements.push({
          tag: el.tagName.toLowerCase(),
          text: (el.textContent || (el as HTMLInputElement).placeholder || "").trim().substring(0, 80),
          selector: getSelector(el),
          attributes: getAttrs(el),
        });
      });
      return elements.slice(0, 80);
    });
  }

  async close(): Promise<void> {
    await this.browser.close();
  }
}

// TestingBot WebDriver wrapper
class TestingBotBrowserSession implements BrowserSession {
  private session: TestingBotSession;
  backendUsed: string;

  constructor(session: TestingBotSession, browserType: BrowserType, platform: Platform) {
    this.session = session;
    this.backendUsed = `testingbot:${browserType}:${platform}`;
  }

  async navigate(url: string): Promise<void> {
    await this.session.navigate(url);
  }

  async getUrl(): Promise<string> {
    return await this.session.getUrl();
  }

  async getContent(): Promise<string> {
    return await this.session.getPageSource();
  }

  async click(selector: string): Promise<void> {
    const elementId = await this.session.findElement(selector);
    await this.session.clickElement(elementId);
  }

  async fill(selector: string, value: string): Promise<void> {
    const elementId = await this.session.findElement(selector);
    await this.session.clearElement(elementId);
    await this.session.sendKeys(elementId, value);
  }

  async type(selector: string, value: string): Promise<void> {
    const elementId = await this.session.findElement(selector);
    await this.session.sendKeys(elementId, value);
  }

  async hover(selector: string): Promise<void> {
    await this.session.executeScript(`
      const el = document.querySelector('${selector}');
      if (el) el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
    `);
  }

  async screenshot(): Promise<string> {
    return await this.session.screenshot();
  }

  async evaluate<T>(fn: string | (() => T)): Promise<T> {
    const script = typeof fn === "string" ? fn : `return (${fn.toString()})()`;
    return await this.session.executeScript(script);
  }

  async getInteractiveElements() {
    return await this.session.executeScript(`
      const elements = [];
      const getSelector = (el) => {
        if (el.id) return '#' + el.id;
        if (el.className && typeof el.className === 'string') {
          const cls = el.className.trim().split(/\\s+/).slice(0, 2).join('.');
          if (cls) return el.tagName.toLowerCase() + '.' + cls;
        }
        return el.tagName.toLowerCase();
      };
      const getAttrs = (el) => {
        const attrs = {};
        ['href', 'type', 'name', 'placeholder', 'aria-label', 'role'].forEach(a => {
          const v = el.getAttribute(a);
          if (v) attrs[a] = v;
        });
        return attrs;
      };
      document.querySelectorAll('button, a[href], input:not([type="hidden"]), textarea, select, [role="button"], [onclick]').forEach(el => {
        elements.push({
          tag: el.tagName.toLowerCase(),
          text: (el.textContent || el.placeholder || '').trim().substring(0, 80),
          selector: getSelector(el),
          attributes: getAttrs(el),
        });
      });
      return elements.slice(0, 80);
    `) as any;
  }

  async close(): Promise<void> {
    await this.session.close();
  }
}

// ============================================================================
// BROWSER CONNECTION
// ============================================================================

async function connectToCloudflare(env: Env, deviceConfig: any, timeout: number): Promise<CloudflareBrowserSession> {
  const { endpointURLString } = await import("@cloudflare/playwright");
  const cdpUrl = endpointURLString(env.BROWSER);

  const browser = await cfChromium.connectOverCDP(cdpUrl);
  const context = await browser.newContext({
    viewport: deviceConfig.viewport,
    userAgent: deviceConfig.userAgent,
    isMobile: deviceConfig.isMobile,
    hasTouch: deviceConfig.hasTouch,
  });
  const page = await context.newPage();

  return new CloudflareBrowserSession(browser, page);
}

async function connectToBrowser(
  env: Env,
  options: {
    backend: Backend;
    browser: BrowserType;
    device: DeviceType;
    platform: Platform;
    timeout: number;
  }
): Promise<BrowserSession> {
  const { backend, browser: browserType, device, timeout } = options;
  let { platform } = options;
  const deviceConfig = DEVICE_PRESETS[device] || DEVICE_PRESETS.desktop;

  // Auto-select platform based on browser (Safari requires macOS)
  if (browserType === "safari" || browserType === "webkit") {
    platform = "macos";
  }

  // Determine which backend to use
  let useTestingBot = backend === "testingbot";

  if (backend === "auto") {
    const needsTestingBot =
      browserType !== "chrome" ||
      (deviceConfig as any).realDevice ||
      device.startsWith("iphone") ||
      device.startsWith("pixel") ||
      device.startsWith("samsung");

    useTestingBot = needsTestingBot && !!env.TESTINGBOT_KEY && !!env.TESTINGBOT_SECRET;
  }

  if (useTestingBot && env.TESTINGBOT_KEY && env.TESTINGBOT_SECRET) {
    const session = await createTestingBotSession(env, browserType, platform, deviceConfig, device);
    return new TestingBotBrowserSession(session, browserType, platform);
  } else {
    // Try Cloudflare first, with automatic failover to TestingBot on rate limit
    try {
      return await connectToCloudflare(env, deviceConfig, timeout);
    } catch (error) {
      const errorMessage = String(error);
      const isRateLimited = errorMessage.includes("429") || errorMessage.includes("Rate limit") || errorMessage.includes("rate limit");

      if (isRateLimited && env.TESTINGBOT_KEY && env.TESTINGBOT_SECRET) {
        console.log("Cloudflare rate limited, failing over to TestingBot...");
        const session = await createTestingBotSession(env, browserType, platform, deviceConfig, device);
        return new TestingBotBrowserSession(session, browserType, platform);
      }

      throw error;
    }
  }
}

// ============================================================================
// TIMEOUT UTILITIES
// ============================================================================

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, operation: string = "Operation"): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error(`${operation} timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    promise
      .then((result) => {
        clearTimeout(timeoutId);
        resolve(result);
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}

function createTimeoutController(timeoutMs: number): { controller: AbortController; cleanup: () => void } {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  return {
    controller,
    cleanup: () => clearTimeout(timeoutId),
  };
}

// ============================================================================
// AI PROVIDER (for browser automation interpretation)
// ============================================================================

interface AIMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface AICallOptions {
  messages: AIMessage[];
  maxTokens?: number;
  temperature?: number;
  timeout?: number;
}

async function callAI(env: Env, options: AICallOptions): Promise<string> {
  const defaults = getEnvDefaults(env);
  const provider = defaults.modelProvider;

  const { messages, maxTokens = 500, temperature = 0.3, timeout = 30000 } = options;
  const { controller, cleanup } = createTimeoutController(timeout);

  try {
    if (provider === "openai" && env.OPENAI_API_KEY) {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: messages,
          max_tokens: maxTokens,
          temperature: temperature,
        }),
        signal: controller.signal,
      });

      cleanup();

      if (!response.ok) {
        console.error("OpenAI API error:", await response.text());
        throw new Error(`OpenAI API error: ${response.status}`);
      }

      const data = await response.json() as { choices: Array<{ message: { content: string } }> };
      return data.choices[0]?.message?.content || "";

    } else if (provider === "anthropic" && env.ANTHROPIC_API_KEY) {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": env.ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: "claude-3-haiku-20240307",
          max_tokens: maxTokens,
          messages: messages.filter(m => m.role !== "system").map(m => ({
            role: m.role,
            content: m.content,
          })),
          system: messages.find(m => m.role === "system")?.content,
        }),
        signal: controller.signal,
      });

      cleanup();

      if (!response.ok) {
        console.error("Anthropic API error:", await response.text());
        throw new Error(`Anthropic API error: ${response.status}`);
      }

      const data = await response.json() as { content: Array<{ text: string }> };
      return data.content[0]?.text || "";

    } else {
      // Default: Workers AI (free)
      cleanup();
      const response = await withTimeout(
        env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
          messages: messages.map(m => ({ role: m.role, content: m.content })),
          max_tokens: maxTokens,
        }),
        timeout,
        "Workers AI call"
      ) as { response?: string } | string;

      return typeof response === "string" ? response : response.response || "";
    }
  } catch (e) {
    cleanup();
    console.error("AI call failed:", e);
    // Fallback to Workers AI if external provider fails
    if (provider !== "workers-ai") {
      console.log("Falling back to Workers AI...");
      const response = await withTimeout(
        env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
          messages: messages.map(m => ({ role: m.role, content: m.content })),
          max_tokens: maxTokens,
        }),
        timeout,
        "Workers AI fallback"
      ) as { response?: string } | string;
      return typeof response === "string" ? response : response.response || "";
    }
    throw e;
  }
}

// ============================================================================
// AI HELPERS
// ============================================================================

async function interpretAction(
  env: Env,
  instruction: string,
  pageHtml: string,
  elements: string
): Promise<{ action: string; selector: string; value?: string; description: string } | null> {
  try {
    const prompt = `You are a browser automation expert. Convert the instruction to a browser action.

INSTRUCTION: "${instruction}"

AVAILABLE ELEMENTS:
${elements}

PAGE HTML (excerpt):
${pageHtml.substring(0, 2000)}

RULES FOR SELECTORS (IMPORTANT - follow exactly):
1. Use VALID CSS selectors only: #id, .class, tag, [attr="value"], tag.class
2. For links by text: use a[href="/login"] or the actual selector from AVAILABLE ELEMENTS
3. For buttons by text: button[type="submit"], input[type="submit"]
4. For inputs by label: input[name="email"], input[placeholder*="email"], #email
5. NEVER invent selectors - only use selectors that appear in AVAILABLE ELEMENTS
6. Prefer: id (#login-btn) > name ([name="email"]) > class (.submit-btn) > tag+attr (button[type="submit"])

RESPOND WITH ONLY VALID JSON (no explanation):
{"action": "click|fill|type|hover|scroll", "selector": "valid CSS selector from elements", "value": "text to type if fill/type", "description": "brief action description"}`;

    const text = await callAI(env, {
      messages: [{ role: "user", content: prompt }],
      maxTokens: 300,
    });

    const match = text.match(/\{[\s\S]*?\}/);
    if (match) {
      return JSON.parse(match[0]);
    }
  } catch (e) {
    console.error("interpretAction failed:", e);
  }
  return null;
}

async function extractWithAI(
  env: Env,
  instruction: string | undefined,
  schema: Record<string, any> | undefined,
  pageContent: string
): Promise<any> {
  const prompt = instruction
    ? `Extract from page: ${instruction}\n\nPAGE:\n${pageContent.substring(0, 3000)}\n\nRESPOND WITH JSON.`
    : schema
      ? `Extract data matching schema:\n${JSON.stringify(schema)}\n\nPAGE:\n${pageContent.substring(0, 3000)}\n\nRESPOND WITH JSON.`
      : `Extract key information from:\n${pageContent.substring(0, 3000)}\n\nRESPOND WITH JSON.`;

  const text = await callAI(env, {
    messages: [{ role: "user", content: prompt }],
    maxTokens: 1000,
  });

  const match = text.match(/\{[\s\S]*\}/);
  if (match) {
    try { return JSON.parse(match[0]); } catch { return { extraction: text }; }
  }
  return { extraction: text };
}

async function observeWithAI(
  env: Env,
  instruction: string,
  elements: Array<{ tag: string; text: string; selector: string }>
): Promise<Action[]> {
  const prompt = `Find elements matching: "${instruction}"

ELEMENTS:
${JSON.stringify(elements.slice(0, 40), null, 2)}

RESPOND WITH JSON ARRAY:
[{"selector": "CSS selector", "description": "what it does", "method": "click|fill|etc", "arguments": []}]`;

  const text = await callAI(env, {
    messages: [{ role: "user", content: prompt }],
    maxTokens: 1000,
  });

  const match = text.match(/\[[\s\S]*\]/);
  if (match) {
    try { return JSON.parse(match[0]); } catch { return []; }
  }
  return [];
}

async function planNextStep(
  env: Env,
  goal: string,
  systemPrompt: string | undefined,
  currentUrl: string,
  pageContent: string,
  history: Action[],
  elements: string
): Promise<{ action: Action | null; completed: boolean; message: string }> {
  const historyText = history.length > 0
    ? history.map((a, i) => `${i + 1}. ${a.description}`).join("\n")
    : "No actions yet";

  const prompt = `${systemPrompt || "Complete the task by taking browser actions."}

GOAL: ${goal}
URL: ${currentUrl}
HISTORY:\n${historyText}
ELEMENTS:\n${elements}
PAGE:\n${pageContent.substring(0, 1500)}

If COMPLETE: {"completed": true, "message": "why complete"}
If MORE ACTIONS: {"completed": false, "action": {"selector": "...", "description": "...", "method": "click|fill|etc", "arguments": []}, "message": "why"}

RESPOND WITH ONLY JSON.`;

  const text = await callAI(env, {
    messages: [{ role: "user", content: prompt }],
    maxTokens: 500,
  });

  const match = text.match(/\{[\s\S]*\}/);
  if (match) {
    try {
      const result = JSON.parse(match[0]);
      return { action: result.action || null, completed: result.completed || false, message: result.message || "" };
    } catch { return { action: null, completed: false, message: "Parse error" }; }
  }
  return { action: null, completed: false, message: "No response" };
}

// ============================================================================
// SELF-HEALING (Simplified - uses fallback selectors)
// ============================================================================

function generateSelectorFallbacks(originalSelector: string, description?: string): string[] {
  const selectors: string[] = [originalSelector];

  // Extract parts from original selector
  const idMatch = originalSelector.match(/#([\w-]+)/);
  const classMatch = originalSelector.match(/\.([\w-]+)/);
  const attrMatch = originalSelector.match(/\[([^\]]+)\]/);

  // ID variations
  if (idMatch) {
    selectors.push(`#${idMatch[1]}`);
    selectors.push(`[id="${idMatch[1]}"]`);
    selectors.push(`[id*="${idMatch[1]}"]`);
  }

  // Class variations
  if (classMatch) {
    selectors.push(`.${classMatch[1]}`);
    selectors.push(`[class*="${classMatch[1]}"]`);
  }

  // Attribute variations
  if (attrMatch) {
    selectors.push(`[${attrMatch[1]}]`);
  }

  // Text-based selectors from description
  if (description) {
    const cleanDesc = description.replace(/['"]/g, '').trim();
    selectors.push(`text="${cleanDesc}"`);
    selectors.push(`button:has-text("${cleanDesc}")`);
    selectors.push(`a:has-text("${cleanDesc}")`);
    selectors.push(`[aria-label="${cleanDesc}"]`);
    selectors.push(`[aria-label*="${cleanDesc.toLowerCase()}"]`);
    selectors.push(`[title="${cleanDesc}"]`);
  }

  // Common button/input selectors
  if (description?.toLowerCase().includes('submit') || description?.toLowerCase().includes('login')) {
    selectors.push('button[type="submit"]');
    selectors.push('input[type="submit"]');
    selectors.push('[role="button"]');
  }

  if (description?.toLowerCase().includes('email')) {
    selectors.push('input[type="email"]');
    selectors.push('input[name*="email"]');
    selectors.push('input[placeholder*="email"]');
  }

  if (description?.toLowerCase().includes('password')) {
    selectors.push('input[type="password"]');
    selectors.push('input[name*="password"]');
  }

  return [...new Set(selectors)]; // Remove duplicates
}

async function executeActionWithSelector(
  session: BrowserSession,
  actionType: string,
  selector: string,
  value?: string
): Promise<void> {
  switch (actionType.toLowerCase()) {
    case "click":
      await session.click(selector);
      break;
    case "fill":
      await session.fill(selector, value || "");
      break;
    case "type":
      await session.type(selector, value || "");
      break;
    case "hover":
      await session.hover(selector);
      break;
    default:
      await session.click(selector);
  }
}

async function executeAction(
  session: BrowserSession,
  action: { action: string; selector: string; value?: string; description: string },
  selfHeal: boolean = true,
  env?: Env
): Promise<{ success: boolean; healed?: boolean; usedSelector?: string; healingMethod?: string; error?: string }> {
  const errors: string[] = [];
  const val = action.value;

  // Phase 1: Try original selector
  try {
    await executeActionWithSelector(session, action.action, action.selector, val);
    return { success: true, healed: false, usedSelector: action.selector, healingMethod: "original" };
  } catch (e) {
    errors.push(`${action.selector}: ${e instanceof Error ? e.message : "Unknown"}`);
  }

  if (!selfHeal) {
    return { success: false, healed: false, error: errors[0] };
  }

  // Phase 2: Try static fallback selectors
  const fallbackSelectors = generateSelectorFallbacks(action.selector, action.description);

  for (let i = 1; i < fallbackSelectors.length; i++) {
    const sel = fallbackSelectors[i];
    try {
      await executeActionWithSelector(session, action.action, sel, val);
      console.log(`Self-healed via fallback: "${action.selector}" → "${sel}"`);
      return {
        success: true,
        healed: true,
        usedSelector: sel,
        healingMethod: "static-fallback"
      };
    } catch (e) {
      errors.push(`${sel}: ${e instanceof Error ? e.message : "Unknown"}`);
    }
  }

  // All healing attempts failed
  return {
    success: false,
    healed: false,
    error: `All ${errors.length} healing attempts failed. Tried: ${errors.slice(0, 5).join("; ")}${errors.length > 5 ? `... (+${errors.length - 5} more)` : ""}`
  };
}

function substituteVariables(instruction: string, variables?: Record<string, string>): string {
  if (!variables) return instruction;
  let result = instruction;
  for (const [key, value] of Object.entries(variables)) {
    result = result.replace(new RegExp(`%${key}%`, 'g'), value);
  }
  return result;
}

// ============================================================================
// ENVIRONMENT DEFAULTS
// ============================================================================

function getEnvDefaults(env: Env) {
  return {
    backend: (env.DEFAULT_BACKEND || "auto") as Backend,
    selfHeal: env.ENABLE_SELF_HEALING !== "false",
    caching: env.ENABLE_CACHING === "true",
    modelProvider: (env.DEFAULT_MODEL_PROVIDER || "workers-ai") as "workers-ai" | "openai" | "anthropic",
  };
}

function applyEnvDefaults<T extends { backend?: string; selfHeal?: boolean }>(
  parsed: T,
  env: Env
): T & { backend: Backend; selfHeal: boolean } {
  const defaults = getEnvDefaults(env);
  return {
    ...parsed,
    backend: (parsed.backend || defaults.backend) as Backend,
    selfHeal: parsed.selfHeal ?? defaults.selfHeal,
  };
}

// ============================================================================
// AUTHENTICATION
// ============================================================================

function authenticate(request: Request, env: Env): boolean {
  if (!env.API_TOKEN) return true;
  const auth = request.headers.get("Authorization");
  return auth?.replace("Bearer ", "") === env.API_TOKEN;
}

// ============================================================================
// REQUEST HANDLERS
// ============================================================================

async function handleAct(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = ActRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const data = applyEnvDefaults(parsed.data, env);
  const { url, instruction, action, variables, timeout, backend, browser: browserType, device, platform, selfHeal, screenshot } = data;
  if (!instruction && !action) {
    return Response.json({ error: "Either 'instruction' or 'action' required" }, { status: 400, headers: corsHeaders });
  }

  let session: BrowserSession | null = null;

  const executeWithTimeout = async (): Promise<Response> => {
    session = await connectToBrowser(env, { backend: backend as Backend, browser: browserType as BrowserType, device: device as DeviceType, platform: platform as Platform, timeout });

    await session.navigate(url);

    let actionToExecute: { action: string; selector: string; value?: string; description: string };

    if (action) {
      actionToExecute = { action: action.method, selector: action.selector, value: action.arguments?.[0], description: action.description };
    } else {
      const processed = substituteVariables(instruction!, variables);
      const pageHtml = await session.getContent();
      const elements = await session.getInteractiveElements();
      const elementsStr = elements.map(e => `${e.tag}: "${e.text}" [${e.selector}]`).join("\n");

      const interpreted = await interpretAction(env, processed, pageHtml, elementsStr);
      if (!interpreted) {
        await session.close();
        return Response.json({ success: false, message: "Could not interpret instruction", actionDescription: processed, actions: [], backend: session.backendUsed } as ActResult, { headers: corsHeaders });
      }
      actionToExecute = interpreted;
    }

    const result = await executeAction(session, actionToExecute, selfHeal, env);

    let screenshotBase64: string | undefined;
    if (screenshot) {
      try {
        screenshotBase64 = await session.screenshot();
      } catch {}
    }

    const backendUsed = session.backendUsed;
    await session.close();

    return Response.json({
      success: result.success,
      message: result.success ? `Executed: ${actionToExecute.description}` : `Failed: ${result.error}`,
      actionDescription: actionToExecute.description,
      actions: [{ selector: actionToExecute.selector, description: actionToExecute.description, method: actionToExecute.action }],
      backend: backendUsed,
      browser: browserType,
      screenshot: screenshotBase64,
    } as ActResult, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Act operation");
  } catch (error) {
    if (session) await session.close().catch(() => {});
    const message = error instanceof Error ? error.message : "Unknown error";
    const isTimeout = message.includes("timed out");
    return Response.json(
      { success: false, message, actions: [], timedOut: isTimeout },
      { status: isTimeout ? 408 : 500, headers: corsHeaders }
    );
  }
}

async function handleExtract(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = ExtractRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const { url, instruction, schema, selector, timeout, backend, browser: browserType, device } = parsed.data;
  let session: BrowserSession | null = null;

  const executeWithTimeout = async (): Promise<Response> => {
    session = await connectToBrowser(env, { backend: backend as Backend, browser: browserType as BrowserType, device: device as DeviceType, platform: "windows", timeout });

    await session.navigate(url);

    let content: string;
    if (selector) {
      try {
        content = await session.evaluate(`document.querySelector('${selector}')?.innerText || document.body.innerText || ''`);
      } catch {
        content = await session.evaluate(`document.body.innerText || ''`);
      }
    } else {
      content = await session.evaluate(`document.body.innerText || ''`);
    }

    content = content || '';

    const data = await extractWithAI(env, instruction, schema, content);
    await session.close();

    return Response.json({ ...data, _backend: session.backendUsed }, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Extract operation");
  } catch (error) {
    if (session) await session.close().catch(() => {});
    const message = error instanceof Error ? error.message : "Unknown";
    const isTimeout = message.includes("timed out");
    return Response.json(
      { error: "Extract failed", details: message, timedOut: isTimeout },
      { status: isTimeout ? 408 : 500, headers: corsHeaders }
    );
  }
}

async function handleObserve(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = ObserveRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const { url, instruction, timeout, backend, browser: browserType, device } = parsed.data;
  let session: BrowserSession | null = null;

  const executeWithTimeout = async (): Promise<Response> => {
    session = await connectToBrowser(env, { backend: backend as Backend, browser: browserType as BrowserType, device: device as DeviceType, platform: "windows", timeout });

    await session.navigate(url);
    const elements = await session.getInteractiveElements();

    let actions = await observeWithAI(env, instruction, elements);

    if (!actions || actions.length === 0) {
      actions = elements.map(el => ({
        selector: el.selector,
        description: el.text
          ? `${el.tag === 'a' ? 'Link to' : el.tag === 'button' ? 'Button:' : el.tag === 'input' ? 'Input:' : ''} ${el.text}`.trim()
          : `${el.tag} element`,
        method: el.tag === 'input' || el.tag === 'textarea' ? 'fill' : 'click',
        arguments: [],
      }));
    }

    await session.close();

    return Response.json({ actions, elements, _backend: session.backendUsed }, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Observe operation");
  } catch (error) {
    if (session) await session.close().catch(() => {});
    const message = error instanceof Error ? error.message : "Unknown";
    const isTimeout = message.includes("timed out");
    return Response.json(
      { error: "Observe failed", details: message, timedOut: isTimeout },
      { status: isTimeout ? 408 : 500, headers: corsHeaders }
    );
  }
}

async function handleAgent(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = AgentRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const { url, instruction, systemPrompt, maxSteps, timeout, backend, browser: browserType, device, captureScreenshots } = parsed.data;
  let session: BrowserSession | null = null;

  const executeWithTimeout = async (): Promise<Response> => {
    session = await connectToBrowser(env, { backend: backend as Backend, browser: browserType as BrowserType, device: device as DeviceType, platform: "windows", timeout });

    await session.navigate(url);

    const history: Action[] = [];
    const screenshots: string[] = [];
    let completed = false;
    let finalMessage = "";
    let tokens = { input: 0, output: 0 };

    for (let step = 0; step < maxSteps && !completed; step++) {
      const currentUrl = await session.getUrl();
      const rawPageContent = await session.evaluate(`document.body.innerText || ''`);
      const pageContent = (rawPageContent || '') as string;
      const elements = await session.getInteractiveElements();
      const elementsStr = elements.map(e => `${e.tag}: "${e.text}" [${e.selector}]`).join("\n");

      if (captureScreenshots) {
        try {
          const screenshot = await session.screenshot();
          screenshots.push(screenshot);
        } catch {}
      }

      const plan = await planNextStep(env, instruction, systemPrompt, currentUrl, pageContent, history, elementsStr);
      tokens.input += Math.floor((pageContent.length + elementsStr.length) / 4);
      tokens.output += 100;

      if (plan.completed) { completed = true; finalMessage = plan.message; break; }
      if (!plan.action) { finalMessage = "Could not determine action"; break; }

      const result = await executeAction(session, { action: plan.action.method || "click", selector: plan.action.selector, value: plan.action.arguments?.[0], description: plan.action.description }, true, env);
      history.push({ ...plan.action, selector: plan.action.selector });

      if (!result.success) { finalMessage = `Action failed: ${result.error}`; break; }

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    const backendUsed = session.backendUsed;
    await session.close();

    return Response.json({
      success: completed,
      message: finalMessage || (completed ? "Completed" : "Max steps reached"),
      actions: history,
      completed,
      usage: { inputTokens: tokens.input, outputTokens: tokens.output, totalSteps: history.length },
      backend: backendUsed,
      screenshots: captureScreenshots ? screenshots : undefined,
    } as AgentResult & { screenshots?: string[] }, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Agent operation");
  } catch (error) {
    if (session) await session.close().catch(() => {});
    const message = error instanceof Error ? error.message : "Unknown";
    const isTimeout = message.includes("timed out");
    return Response.json(
      { success: false, message, actions: [], completed: false, usage: { inputTokens: 0, outputTokens: 0, totalSteps: 0 }, timedOut: isTimeout },
      { status: isTimeout ? 408 : 500, headers: corsHeaders }
    );
  }
}

async function handleTest(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = TestRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const { url, steps, screenshot, captureScreenshots, timeout, backend, browsers, device, devices } = parsed.data;

  const browserList = browsers || ["chrome"];
  const deviceList = devices || [device];

  const executeWithTimeout = async (): Promise<Response> => {
    const results: BrowserResult[] = [];
    const allScreenshots: string[] = [];
    const stepResults: Array<{ instruction: string; success: boolean; error?: string; screenshot?: string }> = [];

    for (const browserType of browserList) {
      for (const deviceType of deviceList) {
        let session: BrowserSession | null = null;
        try {
          session = await connectToBrowser(env, {
            backend: backend as Backend,
            browser: browserType as BrowserType,
            device: deviceType as DeviceType,
            platform: "windows",
            timeout
          });

          await session.navigate(url);

          if (captureScreenshots) {
            try {
              const initialScreenshot = await session.screenshot();
              allScreenshots.push(initialScreenshot);
            } catch {}
          }

          let allSuccess = true;
          for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const pageHtml = await session.getContent();
            const elements = await session.getInteractiveElements();
            const elementsStr = elements.map(e => `${e.tag}: "${e.text}" [${e.selector}]`).join("\n");

            const interpreted = await interpretAction(env, step, pageHtml, elementsStr);
            if (!interpreted) {
              stepResults.push({ instruction: step, success: false, error: "Could not interpret instruction" });
              allSuccess = false;
              break;
            }

            const result = await executeAction(session, interpreted, true, env);

            let stepScreenshot: string | undefined;
            if (captureScreenshots) {
              try {
                stepScreenshot = await session.screenshot();
                allScreenshots.push(stepScreenshot);
              } catch {}
            }

            stepResults.push({
              instruction: step,
              success: result.success,
              error: result.success ? undefined : result.error,
              screenshot: stepScreenshot,
            });

            if (!result.success) {
              allSuccess = false;
              break;
            }
          }

          let screenshotBase64: string | undefined;
          if (screenshot && !captureScreenshots) {
            try {
              screenshotBase64 = await session.screenshot();
            } catch {}
          }

          const backendUsed = session.backendUsed;
          await session.close();

          results.push({
            browser: browserType,
            platform: backendUsed.includes("testingbot") ? "testingbot" : "cloudflare",
            device: deviceType,
            success: allSuccess,
            screenshot: screenshotBase64,
          });
        } catch (error) {
          if (session) await session.close().catch(() => {});
          results.push({
            browser: browserType,
            platform: "error",
            device: deviceType,
            success: false,
            error: error instanceof Error ? error.message : "Unknown error",
          });
        }
      }
    }

    return Response.json({
      success: results.every(r => r.success),
      steps: stepResults.length > 0 ? stepResults : steps.map(s => ({ instruction: s, success: results.some(r => r.success) })),
      browsers: results,
      backend: backend,
      screenshots: allScreenshots.length > 0 ? allScreenshots : undefined,
      finalScreenshot: results[0]?.screenshot,
    } as TestResult, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Test operation");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown";
    const isTimeout = message.includes("timed out");
    return Response.json(
      { success: false, steps: [], browsers: [], error: message, timedOut: isTimeout },
      { status: isTimeout ? 408 : 500, headers: corsHeaders }
    );
  }
}

// ============================================================================
// QUEUE MESSAGE TYPES
// ============================================================================

interface QueueMessage {
  type: 'webhook_event' | 'test_request' | 'error_event';
  payload: Record<string, unknown>;
  timestamp: number;
  projectId?: string;
}

// ============================================================================
// MAIN WORKER
// ============================================================================

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // Health check
    if (path === "/health") {
      const hasTestingBot = !!env.TESTINGBOT_KEY && !!env.TESTINGBOT_SECRET;
      const hasOpenAI = !!env.OPENAI_API_KEY;
      const hasAnthropic = !!env.ANTHROPIC_API_KEY;
      const defaults = getEnvDefaults(env);

      // Check Cloudflare service availability
      const cache = createCache(env.CACHE);
      const storage = createStorage(env.ARTIFACTS);

      return Response.json({
        status: "healthy",
        version: "3.1.0",
        type: "browser-automation",
        backends: {
          cloudflare: true,
          testingbot: hasTestingBot,
        },
        features: ["act", "extract", "observe", "agent", "test"],
        browsers: hasTestingBot ? ["chrome", "firefox", "safari", "edge", "webkit"] : ["chrome"],
        devices: Object.keys(DEVICE_PRESETS),
        services: {
          kv_cache: cache.isAvailable(),
          r2_storage: storage.isAvailable(),
          vectorize: !!env.VECTOR_INDEX,
          queues: !!env.EVENT_QUEUE,
          hyperdrive: !!env.DB,
          realtime: !!env.REALTIME,
          brain: !!env.BRAIN_URL,
        },
        config: {
          defaultBackend: defaults.backend,
          defaultModelProvider: defaults.modelProvider,
          selfHealingEnabled: defaults.selfHeal,
          cachingEnabled: defaults.caching,
          autoFailover: hasTestingBot,
          aiProviders: {
            "workers-ai": true,
            "openai": hasOpenAI,
            "anthropic": hasAnthropic,
          },
        },
        note: "Intelligence features (webhooks, quality intelligence) have moved to Brain service",
      }, { headers: corsHeaders });
    }

    // API docs
    if (path === "/") {
      return Response.json({
        name: "Argus Worker API",
        version: "3.0.0",
        description: "Browser Automation API - heyargus.ai",
        endpoints: {
          "POST /act": "Execute browser action",
          "POST /extract": "Extract structured data",
          "POST /observe": "Discover available actions",
          "POST /agent": "Run autonomous workflow",
          "POST /test": "Run cross-browser tests",
          "GET /health": "Health check",
        },
        backends: ["cloudflare (free, Chromium)", "testingbot (paid, all browsers + devices)"],
        devices: Object.keys(DEVICE_PRESETS),
        note: "For intelligence features (webhooks, test generation, quality scoring), use the Brain service.",
      }, { headers: corsHeaders });
    }

    // Auth for all other endpoints
    if (!authenticate(request, env)) {
      return Response.json({ error: "Unauthorized" }, { status: 401, headers: corsHeaders });
    }

    try {
      if (request.method === "POST") {
        switch (path) {
          case "/act": return await handleAct(request, env, corsHeaders);
          case "/extract": return await handleExtract(request, env, corsHeaders);
          case "/observe": return await handleObserve(request, env, corsHeaders);
          case "/agent": return await handleAgent(request, env, corsHeaders);
          case "/test": return await handleTest(request, env, corsHeaders);
        }
      }

      return Response.json({ error: "Not found" }, { status: 404, headers: corsHeaders });
    } catch (error) {
      console.error("Request failed:", error);
      return Response.json({ error: "Internal error", details: error instanceof Error ? error.message : "Unknown" }, { status: 500, headers: corsHeaders });
    }
  },

  // Queue consumer handler for async event processing
  async queue(batch: MessageBatch<QueueMessage>, env: Env): Promise<void> {
    const cache = createCache(env.CACHE);
    const storage = createStorage(env.ARTIFACTS);

    for (const message of batch.messages) {
      try {
        const { type, payload, timestamp, projectId } = message.body;
        console.log(`Processing ${type} event for project ${projectId}`);

        switch (type) {
          case 'webhook_event':
            // Process webhook events - deduplicate and forward to Brain
            if (payload.fingerprint) {
              const isDupe = await cache.isDuplicate(String(payload.fingerprint));
              if (isDupe) {
                console.log(`Duplicate event skipped: ${payload.fingerprint}`);
                message.ack();
                continue;
              }
              await cache.markAsSeen(String(payload.fingerprint), { projectId, timestamp });
            }

            // Forward to Brain service for processing
            if (env.BRAIN_URL) {
              try {
                await fetch(`${env.BRAIN_URL}/api/v1/webhooks/process`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ type, payload, projectId, timestamp }),
                });
              } catch (e) {
                console.error('Failed to forward to Brain:', e);
                // Re-queue for retry
                message.retry();
                continue;
              }
            }
            break;

          case 'test_request':
            // Process test generation requests
            console.log(`Test request for project ${projectId}:`, payload);
            // Store request and trigger processing
            if (projectId && storage.isAvailable()) {
              await storage.storeTestResults(
                projectId,
                `request-${timestamp}`,
                { type: 'pending', payload, requestedAt: new Date(timestamp).toISOString() }
              );
            }
            break;

          case 'error_event':
            // Process error events for correlation
            console.log(`Error event for project ${projectId}:`, payload);
            break;

          default:
            console.warn(`Unknown event type: ${type}`);
        }

        message.ack();
      } catch (error) {
        console.error('Queue message processing failed:', error);
        // Move to DLQ after max retries
        message.retry();
      }
    }
  },
};

// Export Durable Object for Realtime WebSocket handling
export { RealtimeSession };
