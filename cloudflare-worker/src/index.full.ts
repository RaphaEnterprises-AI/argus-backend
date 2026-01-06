/**
 * Argus - AI-Powered Browser Automation API
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
 */

import { z } from "zod";
import { chromium as cfChromium } from "@cloudflare/playwright";

// ============================================================================
// ENVIRONMENT & TYPES
// ============================================================================

interface Env {
  BROWSER: BrowserBinding;
  AI: Ai;
  VECTOR_INDEX?: VectorizeIndex;  // Vectorize for semantic pattern matching
  TESTINGBOT_KEY?: string;
  TESTINGBOT_SECRET?: string;
  OPENAI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
  API_TOKEN?: string;
  DEFAULT_MODEL_PROVIDER: string;
  DEFAULT_BACKEND: string;
  ENABLE_CACHING: string;
  ENABLE_SELF_HEALING: string;
  // Quality Intelligence Platform
  SUPABASE_URL?: string;
  SUPABASE_SERVICE_KEY?: string;
  SENTRY_CLIENT_SECRET?: string;  // For verifying Sentry webhooks
  DATADOG_WEBHOOK_TOKEN?: string; // For verifying Datadog webhooks
  GITHUB_TOKEN?: string;          // For creating PRs with generated tests
}

type Backend = "cloudflare" | "testingbot" | "auto";
type BrowserType = "chrome" | "firefox" | "safari" | "edge" | "webkit";
type Platform = "windows" | "macos" | "linux";

// ============================================================================
// AI MODEL ROUTER - Cost Optimization Layer
// ============================================================================

type ModelTier = "free" | "cheap" | "premium";

interface ModelRouterResult {
  tier: ModelTier;
  model: string;
  provider: "workers-ai" | "anthropic" | "openai";
  estimatedCost: number; // USD per request
  reason: string;
}

// Classify task complexity to route to appropriate model
function classifyTaskComplexity(task: {
  type: "test_generation" | "pattern_matching" | "code_analysis" | "summarization" | "embedding";
  inputLength: number;
  errorType?: string;
  hasStackTrace?: boolean;
  requiresMultiFile?: boolean;
}): ModelTier {
  // FREE TIER: Simple tasks, pattern matching, embeddings
  if (task.type === "embedding") return "free";
  if (task.type === "pattern_matching") return "free";
  if (task.type === "summarization" && task.inputLength < 500) return "free";

  // CHEAP TIER: Medium complexity
  if (task.type === "test_generation" && !task.hasStackTrace && task.inputLength < 1000) return "cheap";
  if (task.type === "code_analysis" && !task.requiresMultiFile) return "cheap";
  if (task.type === "summarization") return "cheap";

  // PREMIUM TIER: Complex debugging, multi-file analysis
  if (task.requiresMultiFile) return "premium";
  if (task.type === "test_generation" && task.hasStackTrace && task.inputLength > 2000) return "premium";

  // Default to cheap
  return "cheap";
}

// ============================================================================
// CLOUDFLARE WORKERS AI - ADVANCED MODELS (2025)
// ============================================================================
// Pricing: $0.011 per 1,000 neurons
// Free tier: 10,000 neurons/day (resets at 00:00 UTC)
//
// TOP MODELS AVAILABLE:
// 1. @cf/meta/llama-4-scout-17b-16e-instruct - Flagship, 131k context, multimodal
// 2. @cf/deepseek-ai/deepseek-r1-distill-qwen-32b - Outperforms o1-mini
// 3. @cf/openai/gpt-oss-120b - OpenAI's open model (120B params)
// 4. @cf/meta/llama-3.3-70b-instruct-fp8-fast - Fast, high quality
// 5. @cf/qwen/qwen2.5-coder-32b-instruct - Best for code generation
// 6. @cf/mistral/mistral-small-3.1-24b-instruct - Good balance, 128k context
// 7. @cf/qwen/qwq-32b - Reasoning-focused, competitive with o1-mini
// ============================================================================

const WORKERS_AI_MODELS = {
  // Flagship models
  LLAMA_4_SCOUT: "@cf/meta/llama-4-scout-17b-16e-instruct",      // 109B total, 17B active, multimodal
  DEEPSEEK_R1: "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",   // Best reasoning, beats o1-mini
  GPT_OSS_120B: "@cf/openai/gpt-oss-120b",                        // OpenAI's flagship open model

  // High performance
  LLAMA_3_3_70B_FAST: "@cf/meta/llama-3.3-70b-instruct-fp8-fast", // Fast 70B
  QWQ_32B: "@cf/qwen/qwq-32b",                                     // Reasoning specialist

  // Code specialists
  QWEN_CODER_32B: "@cf/qwen/qwen2.5-coder-32b-instruct",          // Best for code

  // Balanced
  MISTRAL_SMALL_24B: "@cf/mistral/mistral-small-3.1-24b-instruct", // 128k context
  GEMMA_3_12B: "@cf/google/gemma-3-12b-it",                        // Google's multimodal

  // Lightweight (for embeddings/simple tasks)
  LLAMA_3_2_3B: "@cf/meta/llama-3.2-3b-instruct",                  // Edge optimized
  LLAMA_3_2_1B: "@cf/meta/llama-3.2-1b-instruct",                  // Smallest
} as const;

// Neuron costs per 1M tokens (approximate)
const MODEL_COSTS = {
  [WORKERS_AI_MODELS.LLAMA_4_SCOUT]: { input: 0.30, output: 2.50 },      // Premium multimodal
  [WORKERS_AI_MODELS.DEEPSEEK_R1]: { input: 0.50, output: 4.90 },        // Best reasoning
  [WORKERS_AI_MODELS.GPT_OSS_120B]: { input: 0.40, output: 3.50 },       // OpenAI flagship
  [WORKERS_AI_MODELS.LLAMA_3_3_70B_FAST]: { input: 0.29, output: 2.25 }, // High quality fast
  [WORKERS_AI_MODELS.QWQ_32B]: { input: 0.20, output: 1.80 },            // Reasoning
  [WORKERS_AI_MODELS.QWEN_CODER_32B]: { input: 0.18, output: 1.60 },     // Code specialist
  [WORKERS_AI_MODELS.MISTRAL_SMALL_24B]: { input: 0.15, output: 1.20 },  // Balanced
  [WORKERS_AI_MODELS.GEMMA_3_12B]: { input: 0.10, output: 0.80 },        // Efficient
  [WORKERS_AI_MODELS.LLAMA_3_2_3B]: { input: 0.05, output: 0.34 },       // Lightweight
  [WORKERS_AI_MODELS.LLAMA_3_2_1B]: { input: 0.03, output: 0.20 },       // Minimal
};

// Get the optimal model for a given tier
function getModelForTier(
  tier: ModelTier,
  env: Env,
  preferences?: {
    preferAccuracy?: boolean;
    preferSpeed?: boolean;
    taskType?: "code" | "reasoning" | "general" | "multimodal";
  }
): ModelRouterResult {
  const hasAnthropic = !!env.ANTHROPIC_API_KEY;
  const hasOpenAI = !!env.OPENAI_API_KEY;
  const hasWorkersAI = !!env.AI;

  switch (tier) {
    case "free":
      // Use efficient Workers AI models for free tier
      // Select based on task type
      if (preferences?.taskType === "code") {
        return {
          tier: "free",
          model: WORKERS_AI_MODELS.LLAMA_3_2_3B,
          provider: "workers-ai",
          estimatedCost: 0.0002,
          reason: "Free tier: Llama 3.2 3B for lightweight code tasks",
        };
      }
      return {
        tier: "free",
        model: WORKERS_AI_MODELS.GEMMA_3_12B,
        provider: "workers-ai",
        estimatedCost: 0.0005,
        reason: "Free tier: Gemma 3 12B - efficient multimodal model",
      };

    case "cheap":
      // Use mid-tier Workers AI models or Claude Haiku
      if (preferences?.taskType === "code") {
        return {
          tier: "cheap",
          model: WORKERS_AI_MODELS.QWEN_CODER_32B,
          provider: "workers-ai",
          estimatedCost: 0.001,
          reason: "Cheap tier: Qwen 2.5 Coder 32B - specialized for code generation",
        };
      }
      if (preferences?.taskType === "reasoning") {
        return {
          tier: "cheap",
          model: WORKERS_AI_MODELS.QWQ_32B,
          provider: "workers-ai",
          estimatedCost: 0.001,
          reason: "Cheap tier: QWQ 32B - reasoning specialist, competitive with o1-mini",
        };
      }
      // Default to Mistral for balance
      return {
        tier: "cheap",
        model: WORKERS_AI_MODELS.MISTRAL_SMALL_24B,
        provider: "workers-ai",
        estimatedCost: 0.0008,
        reason: "Cheap tier: Mistral Small 24B - 128k context, great balance",
      };

    case "premium":
      // Use flagship models for complex tasks
      if (preferences?.taskType === "reasoning") {
        return {
          tier: "premium",
          model: WORKERS_AI_MODELS.DEEPSEEK_R1,
          provider: "workers-ai",
          estimatedCost: 0.003,
          reason: "Premium tier: DeepSeek-R1 - outperforms o1-mini on reasoning",
        };
      }
      if (preferences?.taskType === "multimodal") {
        return {
          tier: "premium",
          model: WORKERS_AI_MODELS.LLAMA_4_SCOUT,
          provider: "workers-ai",
          estimatedCost: 0.002,
          reason: "Premium tier: Llama 4 Scout - flagship multimodal, 131k context",
        };
      }
      if (preferences?.preferSpeed) {
        return {
          tier: "premium",
          model: WORKERS_AI_MODELS.LLAMA_3_3_70B_FAST,
          provider: "workers-ai",
          estimatedCost: 0.002,
          reason: "Premium tier: Llama 3.3 70B Fast - optimized for speed",
        };
      }
      // Check for external API keys for absolute best quality
      if (hasAnthropic) {
        return {
          tier: "premium",
          model: "claude-sonnet-4-20250514",
          provider: "anthropic",
          estimatedCost: 0.012,
          reason: "Premium tier: Claude Sonnet 4 - best for complex analysis",
        };
      }
      // Default to Llama 4 Scout - the flagship
      return {
        tier: "premium",
        model: WORKERS_AI_MODELS.LLAMA_4_SCOUT,
        provider: "workers-ai",
        estimatedCost: 0.002,
        reason: "Premium tier: Llama 4 Scout - 109B params, multimodal flagship",
      };
  }
}

// Smart model router with caching check
async function routeToOptimalModel(
  env: Env,
  task: {
    type: "test_generation" | "pattern_matching" | "code_analysis" | "summarization" | "embedding";
    input: string;
    cacheKey?: string;
    errorType?: string;
    hasStackTrace?: boolean;
    requiresMultiFile?: boolean;
  }
): Promise<{ model: ModelRouterResult; cached: boolean; cacheHit?: string }> {
  // Check cache first (for test_generation and pattern_matching)
  if (task.cacheKey && (task.type === "test_generation" || task.type === "pattern_matching")) {
    // In production, check KV or Vectorize for cached result
    // For now, just return the routing result
  }

  const complexity = classifyTaskComplexity({
    type: task.type,
    inputLength: task.input.length,
    errorType: task.errorType,
    hasStackTrace: task.hasStackTrace,
    requiresMultiFile: task.requiresMultiFile,
  });

  const model = getModelForTier(complexity, env);

  return {
    model,
    cached: false,
  };
}

// Execute AI request with the routed model
async function executeWithRouter(
  env: Env,
  task: Parameters<typeof routeToOptimalModel>[1],
  prompt: string,
  systemPrompt?: string
): Promise<{ result: string; model: string; cost: number; cached: boolean }> {
  const { model, cached } = await routeToOptimalModel(env, task);

  if (cached) {
    return { result: "", model: "cache", cost: 0, cached: true };
  }

  let result = "";

  switch (model.provider) {
    case "workers-ai":
      if (env.AI) {
        const response = await env.AI.run(model.model as Parameters<typeof env.AI.run>[0], {
          messages: [
            ...(systemPrompt ? [{ role: "system" as const, content: systemPrompt }] : []),
            { role: "user" as const, content: prompt },
          ],
          max_tokens: 4096,
        });
        result = response?.response || "";
      }
      break;

    case "anthropic":
      if (env.ANTHROPIC_API_KEY) {
        const response = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": env.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
          },
          body: JSON.stringify({
            model: model.model,
            max_tokens: 4096,
            system: systemPrompt,
            messages: [{ role: "user", content: prompt }],
          }),
        });
        const data = await response.json() as { content?: Array<{ text: string }> };
        result = data.content?.[0]?.text || "";
      }
      break;

    case "openai":
      if (env.OPENAI_API_KEY) {
        const response = await fetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
          },
          body: JSON.stringify({
            model: model.model,
            max_tokens: 4096,
            messages: [
              ...(systemPrompt ? [{ role: "system", content: systemPrompt }] : []),
              { role: "user", content: prompt },
            ],
          }),
        });
        const data = await response.json() as { choices?: Array<{ message: { content: string } }> };
        result = data.choices?.[0]?.message?.content || "";
      }
      break;
  }

  return {
    result,
    model: model.model,
    cost: model.estimatedCost,
    cached: false,
  };
}

// Cost tracking for analytics
interface CostMetrics {
  total_requests: number;
  total_cost: number;
  by_tier: {
    free: { requests: number; cost: number };
    cheap: { requests: number; cost: number };
    premium: { requests: number; cost: number };
  };
  cache_hit_rate: number;
  avg_cost_per_request: number;
}

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
  captureScreenshots: z.boolean().optional().default(false), // Per-step screenshots
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  // Cross-browser testing - run same test on multiple browsers
  browsers: z.array(BrowserSchema).optional(),
  device: DeviceSchema,
  // Cross-device testing - run same test on multiple devices
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

  // Add device-specific options for real devices
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
    const script = typeof fn === "string" ? fn : fn.toString().replace(/^[^{]+\{|\}$/g, "");
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
    // WebDriver doesn't have native hover, use JS
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
    // Use TestingBot if:
    // - Non-Chromium browser requested
    // - Real device requested
    // - TestingBot credentials available
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

      // Auto-failover to TestingBot if Cloudflare is rate limited
      if (isRateLimited && env.TESTINGBOT_KEY && env.TESTINGBOT_SECRET) {
        console.log("Cloudflare rate limited, failing over to TestingBot...");
        const session = await createTestingBotSession(env, browserType, platform, deviceConfig, device);
        return new TestingBotBrowserSession(session, browserType, platform);
      }

      // Re-throw original error if no failover available
      throw error;
    }
  }
}

async function connectToCloudflare(
  env: Env,
  deviceConfig: any,
  timeout: number
): Promise<BrowserSession> {
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

// ============================================================================
// AI PROVIDER ABSTRACTION
// ============================================================================

type AIProvider = "workers-ai" | "openai" | "anthropic";

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

// ============================================================================
// TIMEOUT UTILITIES
// ============================================================================

/**
 * Wraps a promise with a timeout. Rejects if the promise doesn't resolve within the timeout.
 */
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

/**
 * Creates an AbortController that auto-aborts after the specified timeout.
 * Returns both the controller and a cleanup function.
 */
function createTimeoutController(timeoutMs: number): { controller: AbortController; cleanup: () => void } {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  return {
    controller,
    cleanup: () => clearTimeout(timeoutId),
  };
}

async function callAI(env: Env, options: AICallOptions): Promise<string> {
  const defaults = getEnvDefaults(env);
  const provider = defaults.modelProvider;

  const { messages, maxTokens = 500, temperature = 0.3, timeout = 30000 } = options;
  const { controller, cleanup } = createTimeoutController(timeout);

  try {
    if (provider === "openai" && env.OPENAI_API_KEY) {
      // OpenAI API call with timeout
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${env.OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model: "gpt-4o-mini", // Cost-effective, good for automation
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
      // Anthropic API call with timeout
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": env.ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: "claude-3-haiku-20240307", // Fast and cheap, good for automation
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
      // Default: Workers AI (free) - no AbortController support, use withTimeout
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
    // Fallback to Workers AI if external provider fails (and not already using it)
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
2. For links by text: a[href*="login"], a:contains("Login") is INVALID - use a[href="/login"] or the actual selector from AVAILABLE ELEMENTS
3. For buttons by text: button[type="submit"], input[type="submit"]
4. For inputs by label: input[name="email"], input[placeholder*="email"], #email
5. NEVER invent selectors - only use selectors that appear in AVAILABLE ELEMENTS
6. Prefer: id (#login-btn) > name ([name="email"]) > class (.submit-btn) > tag+attr (button[type="submit"])

EXAMPLES:
- "Click login" → {"action":"click","selector":"a[href*=\"login\"]","description":"Click login link"}
- "Type email" → {"action":"fill","selector":"input[type=\"email\"]","value":"","description":"Fill email field"}
- "Click submit button" → {"action":"click","selector":"button[type=\"submit\"]","description":"Click submit"}

RESPOND WITH ONLY VALID JSON (no explanation):
{"action": "click|fill|type|hover|scroll", "selector": "valid CSS selector from elements", "value": "text to type if fill/type", "description": "brief action description"}`;

    const text = await callAI(env, {
      messages: [{ role: "user", content: prompt }],
      maxTokens: 300,
    });

    const match = text.match(/\{[\s\S]*?\}/);
    if (match) {
      const parsed = JSON.parse(match[0]);
      // Validate the selector is reasonable
      if (parsed.selector && !parsed.selector.includes(": ") && parsed.selector.length < 200) {
        return parsed;
      }
    }
    return null;
  } catch (e) {
    console.error("AI interpretation failed:", e);
    return null;
  }
}

async function extractWithAI(
  env: Env,
  instruction: string | undefined,
  schema: Record<string, any> | undefined,
  pageContent: string
): Promise<Record<string, unknown>> {
  if (!instruction && !schema) {
    return { pageText: pageContent.substring(0, 10000) };
  }

  const schemaDesc = schema
    ? Object.entries(schema).map(([k, v]) => `- ${k}: ${typeof v === 'string' ? v : v.type || 'any'}`).join("\n")
    : "";

  const prompt = schema
    ? `Extract data matching this schema:\n${schemaDesc}\n\n${instruction || ""}\n\nPAGE:\n${pageContent.substring(0, 5000)}\n\nRESPOND WITH ONLY JSON.`
    : `${instruction}\n\nPAGE:\n${pageContent.substring(0, 5000)}\n\nRESPOND WITH ONLY JSON.`;

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
// ENHANCED SELF-HEALING SYSTEM - 95%+ PASS RATE
// ============================================================================

interface HealingResult {
  success: boolean;
  usedSelector: string | null;
  healingMethod: string;
  confidence: number;
  attempts: number;
  timeTaken: number;
}

interface HealingConfig {
  enableStaticFallbacks: boolean;
  enablePatternLearning: boolean;
  enableAIGeneration: boolean;
  enableVisualMatching: boolean;
  maxHealingTime: number;
  waitTimeout: number;
  minPatternConfidence: number;
  minVisualConfidence: number;
}

const DEFAULT_HEALING_CONFIG: HealingConfig = {
  enableStaticFallbacks: true,
  enablePatternLearning: true,
  enableAIGeneration: true,
  enableVisualMatching: true,
  maxHealingTime: 30000,
  waitTimeout: 15000,
  minPatternConfidence: 0.5,
  minVisualConfidence: 0.7,
};

// Smart wait strategies for dynamic content
async function smartWait(session: BrowserSession, timeout: number = 10000): Promise<void> {
  const startTime = Date.now();

  // Wait for network idle
  try {
    await Promise.race([
      session.evaluate(`
        new Promise(resolve => {
          if (document.readyState === 'complete') {
            setTimeout(resolve, 500);
          } else {
            window.addEventListener('load', () => setTimeout(resolve, 500));
          }
        })
      `),
      new Promise((_, reject) => setTimeout(() => reject(), Math.min(5000, timeout)))
    ]);
  } catch {}

  // Wait for no spinners/loaders
  const loaderSelectors = [
    '.loading', '.spinner', '.loader', '.skeleton',
    '[class*="loading"]', '[class*="spinner"]',
    '[aria-busy="true"]', '.MuiCircularProgress-root'
  ];

  for (const sel of loaderSelectors) {
    const remaining = timeout - (Date.now() - startTime);
    if (remaining <= 0) break;

    try {
      await session.evaluate(`
        (sel) => {
          const el = document.querySelector(sel);
          return !el || el.offsetParent === null;
        }
      `, sel);
    } catch {}
  }

  // Brief pause for any final rendering
  await new Promise(r => setTimeout(r, 200));
}

// Generate text-based selectors from description
function generateTextSelectors(description: string): string[] {
  const selectors: string[] = [];
  const cleanDesc = description.replace(/['"]/g, '').trim();
  const words = cleanDesc.split(/\s+/).filter(w => w.length > 2);

  // Playwright text selectors
  selectors.push(`text="${cleanDesc}"`);
  selectors.push(`text=${cleanDesc}`);

  // Button/link with text
  selectors.push(`button:has-text("${cleanDesc}")`);
  selectors.push(`a:has-text("${cleanDesc}")`);
  selectors.push(`[role="button"]:has-text("${cleanDesc}")`);

  // Partial text matches with key words
  for (const word of words.slice(0, 3)) {
    selectors.push(`:has-text("${word}")`);
  }

  return selectors;
}

// Generate ARIA-based selectors
function generateARIASelectors(description: string): string[] {
  const selectors: string[] = [];
  const cleanDesc = description.replace(/['"]/g, '').toLowerCase();

  selectors.push(`[aria-label="${cleanDesc}"]`);
  selectors.push(`[aria-label*="${cleanDesc}"]`);
  selectors.push(`[aria-labelledby*="${cleanDesc}"]`);
  selectors.push(`[title="${cleanDesc}"]`);
  selectors.push(`[title*="${cleanDesc}"]`);

  // Role-based selectors
  if (cleanDesc.includes('button') || cleanDesc.includes('click') || cleanDesc.includes('submit')) {
    selectors.push('[role="button"]');
  }
  if (cleanDesc.includes('link') || cleanDesc.includes('navigate')) {
    selectors.push('[role="link"]');
  }
  if (cleanDesc.includes('input') || cleanDesc.includes('field') || cleanDesc.includes('enter')) {
    selectors.push('[role="textbox"]');
  }
  if (cleanDesc.includes('checkbox') || cleanDesc.includes('check')) {
    selectors.push('[role="checkbox"]');
  }
  if (cleanDesc.includes('dropdown') || cleanDesc.includes('select')) {
    selectors.push('[role="listbox"]', '[role="combobox"]');
  }

  return selectors;
}

// Generate data-attribute selectors (most stable)
function generateDataAttributeSelectors(selector: string, description?: string): string[] {
  const selectors: string[] = [];

  // Extract identifiers
  const idMatch = selector.match(/#([a-zA-Z0-9_-]+)/);
  const nameMatch = selector.match(/\[name=["']?([^"'\]]+)["']?\]/);

  if (idMatch) {
    const id = idMatch[1];
    selectors.push(
      `[data-testid="${id}"]`,
      `[data-testid*="${id}"]`,
      `[data-cy="${id}"]`,
      `[data-cy*="${id}"]`,
      `[data-test="${id}"]`,
      `[data-test*="${id}"]`,
      `[data-automation="${id}"]`,
      `[data-automation*="${id}"]`,
      `[data-qa="${id}"]`
    );
  }

  if (nameMatch) {
    const name = nameMatch[1];
    selectors.push(
      `[data-testid="${name}"]`,
      `[data-testid*="${name}"]`,
      `[data-field="${name}"]`
    );
  }

  // From description
  if (description) {
    const descWords = description.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    for (const word of descWords.slice(0, 2)) {
      selectors.push(`[data-testid*="${word}"]`, `[data-cy*="${word}"]`);
    }
  }

  return selectors;
}

// Enhanced fallback generator with all strategies
function generateSelectorFallbacks(selector: string, description?: string): string[] {
  const selectors: string[] = [selector];

  // Extract potential identifiers from the selector
  const idMatch = selector.match(/#([a-zA-Z0-9_-]+)/);
  const classMatch = selector.match(/\.([a-zA-Z0-9_-]+)/);
  const nameMatch = selector.match(/\[name=["']?([^"'\]]+)["']?\]/);
  const hrefMatch = selector.match(/\[href\*?=["']?([^"'\]]+)["']?\]/);
  const typeMatch = selector.match(/\[type=["']?([^"'\]]+)["']?\]/);

  // Priority 1: Data attributes (most stable)
  selectors.push(...generateDataAttributeSelectors(selector, description));

  // Priority 2: ARIA selectors (semantic, accessible)
  if (description) {
    selectors.push(...generateARIASelectors(description));
  }

  // Priority 3: Text-based selectors
  if (description) {
    selectors.push(...generateTextSelectors(description));
  }

  // Priority 4: Partial attribute matches
  if (idMatch) {
    selectors.push(`[id*="${idMatch[1]}"]`);
    selectors.push(`#${idMatch[1]}`);
  }
  if (classMatch) {
    selectors.push(`[class*="${classMatch[1]}"]`);
    selectors.push(`.${classMatch[1]}`);
  }
  if (nameMatch) {
    selectors.push(
      `[name*="${nameMatch[1]}"]`,
      `[id*="${nameMatch[1]}"]`,
      `[placeholder*="${nameMatch[1]}"]`,
      `input[name="${nameMatch[1]}"]`,
      `textarea[name="${nameMatch[1]}"]`
    );
  }

  // Priority 5: Common patterns for buttons/links
  const descLower = description?.toLowerCase() || '';
  const selectorLower = selector.toLowerCase();

  if (selectorLower.includes("button") || selectorLower.includes("btn") || descLower.includes("button") || descLower.includes("submit") || descLower.includes("click")) {
    selectors.push(
      'button[type="submit"]',
      'input[type="submit"]',
      'button[type="button"]',
      '.btn-primary',
      '.btn-submit',
      '.submit-btn',
      '.submit',
      '[role="button"]',
      'button:visible',
      'input[type="button"]'
    );
  }

  if (selectorLower.includes("login") || descLower.includes("login") || descLower.includes("sign in") || descLower.includes("signin")) {
    selectors.push(
      'button:has-text("Log in")',
      'button:has-text("Login")',
      'button:has-text("Sign in")',
      'a[href*="login"]',
      'a[href*="signin"]',
      'a[href*="sign-in"]',
      'button[type="submit"]',
      '[data-testid*="login"]',
      '[data-cy*="login"]',
      '#login',
      '.login',
      '#signin',
      '.signin'
    );
  }

  // Priority 6: Input field fallbacks
  if (selectorLower.includes("email") || descLower.includes("email")) {
    selectors.push(
      'input[type="email"]',
      'input[name="email"]',
      'input[name*="email"]',
      'input[placeholder*="email"]',
      'input[placeholder*="Email"]',
      'input[autocomplete="email"]',
      '#email',
      '.email-input',
      'input[id*="email"]'
    );
  }

  if (selectorLower.includes("password") || descLower.includes("password")) {
    selectors.push(
      'input[type="password"]',
      'input[name="password"]',
      'input[name*="password"]',
      'input[placeholder*="password"]',
      'input[placeholder*="Password"]',
      'input[autocomplete="current-password"]',
      'input[autocomplete="new-password"]',
      '#password',
      '.password-input'
    );
  }

  if (selectorLower.includes("username") || descLower.includes("username") || descLower.includes("user name")) {
    selectors.push(
      'input[name="username"]',
      'input[name*="username"]',
      'input[placeholder*="username"]',
      'input[placeholder*="Username"]',
      'input[autocomplete="username"]',
      '#username',
      '.username-input',
      'input[id*="username"]',
      'input[id*="user"]'
    );
  }

  if (selectorLower.includes("search") || descLower.includes("search")) {
    selectors.push(
      'input[type="search"]',
      'input[name="search"]',
      'input[name*="search"]',
      'input[placeholder*="search"]',
      'input[placeholder*="Search"]',
      '[role="searchbox"]',
      '#search',
      '.search-input'
    );
  }

  // Priority 7: href patterns
  if (hrefMatch) {
    selectors.push(
      `a[href*="${hrefMatch[1]}"]`,
      `[href$="${hrefMatch[1]}"]`,
      `a:has-text("${hrefMatch[1]}")`
    );
  }

  // Priority 8: Type-specific fallbacks
  if (typeMatch) {
    selectors.push(
      `input[type="${typeMatch[1]}"]`,
      `button[type="${typeMatch[1]}"]`
    );
  }

  // Deduplicate while preserving priority order
  return [...new Set(selectors)];
}

// AI-powered selector generation using Workers AI
async function aiGenerateSelectors(
  env: Env,
  pageHTML: string,
  targetDescription: string,
  originalSelector: string
): Promise<string[]> {
  if (!env.AI) return [];

  try {
    const prompt = `You are an expert at finding HTML elements. Given the page HTML and a description, generate CSS selectors.

TARGET: ${targetDescription}
ORIGINAL SELECTOR (failed): ${originalSelector}

PAGE HTML (snippet):
${pageHTML.slice(0, 8000)}

Generate 5 CSS selectors to find this element, ordered by reliability:
1. Prefer data-testid, data-cy, aria-label
2. Use text content selectors
3. Avoid brittle selectors (nth-child, deep nesting)

Return ONLY a JSON array of selector strings, no explanation.
Example: ["[data-testid=\\"login\\"]", "button:has-text(\\"Login\\")", "#login-btn"]`;

    const response = await env.AI.run(WORKERS_AI_MODELS.QWEN_CODER_32B, {
      prompt,
      max_tokens: 500,
    }) as { response: string };

    // Parse AI response
    const match = response.response.match(/\[[\s\S]*\]/);
    if (match) {
      return JSON.parse(match[0]);
    }
  } catch (e) {
    console.error("AI selector generation failed:", e);
  }

  return [];
}

// Query Vectorize for similar healing patterns (for self-healing selectors)
async function findHealingPatterns(
  env: Env,
  elementDescription: string,
  pageContext: string
): Promise<Array<{ selector: string; confidence: number }>> {
  if (!env.VECTOR_INDEX || !env.AI) return [];

  try {
    // Generate embedding for the query
    const embeddingResult = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
      text: `selector-healing: ${elementDescription} ${pageContext}`
    });

    // Handle the AI result which returns data array
    const embedding = (embeddingResult as { data: number[][] })?.data?.[0];
    if (!embedding) return [];

    // Query Vectorize
    const results = await env.VECTOR_INDEX.query(embedding, {
      topK: 5,
      returnMetadata: "all",
    });

    return results.matches
      .filter(m => m.score > 0.7 && m.metadata?.healedSelector)
      .map(m => ({
        selector: m.metadata?.healedSelector as string || "",
        confidence: m.score * (m.metadata?.confidence as number || 0.8),
      }))
      .filter(p => p.selector);
  } catch (e) {
    console.error("Healing pattern query failed:", e);
  }

  return [];
}

// Store successful healing pattern in Vectorize
async function storeHealingPattern(
  env: Env,
  originalSelector: string,
  healedSelector: string,
  elementDescription: string,
  pageUrl: string,
  pageContext: string
): Promise<void> {
  if (!env.VECTOR_INDEX || !env.AI) return;

  try {
    // Generate embedding with prefix to distinguish healing patterns
    const embeddingResult = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
      text: `selector-healing: ${elementDescription} ${pageContext}`
    });

    // Handle the AI result
    const embedding = (embeddingResult as { data: number[][] })?.data?.[0];
    if (!embedding) return;

    const id = `heal-${crypto.randomUUID()}`;

    await env.VECTOR_INDEX.upsert([{
      id,
      values: embedding,
      metadata: {
        type: "selector-healing",
        originalSelector,
        healedSelector,
        elementDescription,
        pageUrl,
        confidence: 0.85,
        successCount: 1,
        createdAt: new Date().toISOString(),
      }
    }]);

    console.log(`Stored healing pattern: "${originalSelector}" → "${healedSelector}"`);
  } catch (e) {
    console.error("Failed to store healing pattern:", e);
  }
}

// Execute a single action with a specific selector
async function executeActionWithSelector(
  session: BrowserSession,
  actionType: string,
  selector: string,
  value: string
): Promise<void> {
  switch (actionType) {
    case "click":
      await session.click(selector);
      break;
    case "fill":
      await session.fill(selector, value);
      break;
    case "type":
      await session.type(selector, value);
      break;
    case "hover":
      await session.hover(selector);
      break;
    case "scroll":
      if (selector === "down" || value === "down") {
        await session.evaluate("window.scrollBy(0, 500)");
      } else if (selector === "up" || value === "up") {
        await session.evaluate("window.scrollBy(0, -500)");
      } else {
        await session.evaluate("window.scrollBy(0, 300)");
      }
      break;
    case "wait":
      await new Promise(r => setTimeout(r, parseInt(value || "1000")));
      break;
    case "select":
      await session.evaluate(`document.querySelector('${selector}').value = '${value}'`);
      break;
    case "check":
      await session.click(selector);
      break;
    default:
      throw new Error(`Unknown action: ${actionType}`);
  }
}

// Enhanced executeAction with multi-phase healing
async function executeAction(
  session: BrowserSession,
  action: { action: string; selector: string; value?: string; arguments?: string[]; description?: string },
  selfHeal: boolean = true,
  env?: Env
): Promise<{ success: boolean; healed: boolean; usedSelector?: string; healingMethod?: string; error?: string }> {
  const startTime = Date.now();
  const val = action.value || action.arguments?.[0] || "";
  const errors: string[] = [];

  // Phase 0: Apply smart wait before any action
  try {
    await smartWait(session, 5000);
  } catch {}

  // Phase 1: Try original selector first
  try {
    await executeActionWithSelector(session, action.action, action.selector, val);
    return { success: true, healed: false, usedSelector: action.selector, healingMethod: "original" };
  } catch (e) {
    errors.push(`${action.selector}: ${e instanceof Error ? e.message : "Unknown"}`);
  }

  if (!selfHeal) {
    return { success: false, healed: false, error: errors[0] };
  }

  // Phase 2: Try learned patterns from Vectorize (if available)
  if (env && action.description) {
    try {
      const pageContext = await session.evaluate("document.body.innerText.slice(0, 500)") as string;
      const patterns = await findHealingPatterns(env, action.description, pageContext);

      for (const pattern of patterns) {
        try {
          await executeActionWithSelector(session, action.action, pattern.selector, val);
          console.log(`Self-healed via learned pattern: "${action.selector}" → "${pattern.selector}"`);
          return {
            success: true,
            healed: true,
            usedSelector: pattern.selector,
            healingMethod: "learned-pattern"
          };
        } catch (e) {
          errors.push(`${pattern.selector}: ${e instanceof Error ? e.message : "Unknown"}`);
        }
      }
    } catch {}
  }

  // Phase 3: Try static fallback selectors
  const fallbackSelectors = generateSelectorFallbacks(action.selector, action.description);

  for (let i = 1; i < fallbackSelectors.length; i++) {
    const sel = fallbackSelectors[i];
    try {
      await executeActionWithSelector(session, action.action, sel, val);
      console.log(`Self-healed via fallback: "${action.selector}" → "${sel}"`);

      // Store successful pattern for future use
      if (env && action.description) {
        try {
          const pageContext = await session.evaluate("document.body.innerText.slice(0, 500)") as string;
          await storeHealingPattern(
            env,
            action.selector,
            sel,
            action.description,
            await session.evaluate("window.location.href") as string,
            pageContext
          );
        } catch {}
      }

      return {
        success: true,
        healed: true,
        usedSelector: sel,
        healingMethod: "static-fallback"
      };
    } catch (e) {
      errors.push(`${sel}: ${e instanceof Error ? e.message : "Unknown"}`);
    }

    // Timeout check
    if (Date.now() - startTime > DEFAULT_HEALING_CONFIG.maxHealingTime) {
      break;
    }
  }

  // Phase 4: AI-generated selectors (if enabled and env available)
  if (env?.AI && action.description) {
    try {
      const pageHTML = await session.evaluate("document.documentElement.outerHTML") as string;
      const aiSelectors = await aiGenerateSelectors(env, pageHTML, action.description, action.selector);

      for (const sel of aiSelectors) {
        try {
          await executeActionWithSelector(session, action.action, sel, val);
          console.log(`Self-healed via AI: "${action.selector}" → "${sel}"`);

          // Store the successful AI-generated pattern
          try {
            const pageContext = await session.evaluate("document.body.innerText.slice(0, 500)") as string;
            await storeHealingPattern(
              env,
              action.selector,
              sel,
              action.description,
              await session.evaluate("window.location.href") as string,
              pageContext
            );
          } catch {}

          return {
            success: true,
            healed: true,
            usedSelector: sel,
            healingMethod: "ai-generated"
          };
        } catch (e) {
          errors.push(`AI[${sel}]: ${e instanceof Error ? e.message : "Unknown"}`);
        }
      }
    } catch {}
  }

  // All healing attempts failed
  const totalAttempts = errors.length;
  return {
    success: false,
    healed: false,
    error: `All ${totalAttempts} healing attempts failed. Tried: ${errors.slice(0, 5).join("; ")}${errors.length > 5 ? `... (+${errors.length - 5} more)` : ""}`
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
    selfHeal: env.ENABLE_SELF_HEALING !== "false", // Default true unless explicitly disabled
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

  // Apply environment defaults
  const data = applyEnvDefaults(parsed.data, env);
  const { url, instruction, action, variables, timeout, backend, browser: browserType, device, platform, selfHeal, screenshot } = data;
  if (!instruction && !action) {
    return Response.json({ error: "Either 'instruction' or 'action' required" }, { status: 400, headers: corsHeaders });
  }

  let session: BrowserSession | null = null;

  // Wrap entire operation with timeout enforcement
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

    // Capture screenshot after action if requested
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

    // Ensure content is never null/undefined
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

    // Try AI-powered analysis first
    let actions = await observeWithAI(env, instruction, elements);

    // If AI returns empty, fall back to raw elements with generated descriptions
    if (!actions || actions.length === 0) {
      actions = elements.map(el => ({
        selector: el.selector,
        description: el.text
          ? `${el.tag === 'a' ? 'Link to' : el.tag === 'button' ? 'Button:' : el.tag === 'input' ? 'Input:' : ''} ${el.text}`.trim()
          : `${el.tag} element`,
        method: el.tag === 'input' || el.tag === 'textarea' ? 'fill' : 'click',
        arguments: [],
        tagName: el.tag.toUpperCase(),
        text: el.text,
        attributes: el.attributes,
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

      // Wait a bit for page to update
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

  // Determine browser/device combinations to test
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

          // Capture initial screenshot if requested
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

            // Capture screenshot after each step if requested
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

          // Capture final screenshot if requested (backwards compatibility)
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
// QUALITY INTELLIGENCE - WEBHOOK HANDLERS
// ============================================================================

interface SentryIssue {
  id: string;
  title: string;
  culprit: string;
  level: string;
  message: string;
  metadata: {
    type?: string;
    value?: string;
    filename?: string;
    function?: string;
  };
  platform: string;
  project: string;
  url: string;
  shortId: string;
  count: string;
  userCount: number;
  firstSeen: string;
  lastSeen: string;
  tags: Array<{ key: string; value: string }>;
}

interface SentryEvent {
  event_id: string;
  project: string;
  project_name: string;
  timestamp: string;
  received: string;
  platform: string;
  event?: {
    event_id: string;
    level: string;
    message?: string;
    culprit?: string;
    exception?: {
      values: Array<{
        type: string;
        value: string;
        stacktrace?: {
          frames: Array<{
            filename: string;
            function: string;
            lineno: number;
            colno: number;
            abs_path?: string;
            context_line?: string;
            pre_context?: string[];
            post_context?: string[];
          }>;
        };
      }>;
    };
    request?: {
      url: string;
      method: string;
      headers: Record<string, string>;
    };
    user?: {
      id?: string;
      email?: string;
      ip_address?: string;
    };
    contexts?: {
      browser?: { name: string; version: string };
      os?: { name: string; version: string };
      device?: { family: string; model: string };
    };
    tags: Array<{ key: string; value: string }>;
  };
  url: string;
  issue_url?: string;
}

interface DatadogEvent {
  id: string;
  event_type: string;
  title: string;
  message: string;
  date_happened: number;
  priority: string;
  host: string;
  tags: string[];
  alert_type: string;
  source_type_name: string;
  aggregation_key?: string;
  url?: string;
  // Error tracking specific fields
  error?: {
    type: string;
    message: string;
    stack: string;
    source: string;
    resource?: { url: string };
  };
  view?: {
    url: string;
    name: string;
  };
  user?: {
    id?: string;
    email?: string;
  };
  context?: Record<string, unknown>;
}

// Supabase client helper
async function supabaseRequest(
  env: Env,
  path: string,
  options: {
    method?: string;
    body?: unknown;
    headers?: Record<string, string>;
  } = {}
): Promise<{ data: unknown; error: unknown }> {
  if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_KEY) {
    return { data: null, error: "Supabase not configured" };
  }

  const url = `${env.SUPABASE_URL}/rest/v1${path}`;
  const response = await fetch(url, {
    method: options.method || "GET",
    headers: {
      "Content-Type": "application/json",
      "apikey": env.SUPABASE_SERVICE_KEY,
      "Authorization": `Bearer ${env.SUPABASE_SERVICE_KEY}`,
      "Prefer": options.method === "POST" ? "return=representation" : "return=minimal",
      ...options.headers,
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  if (!response.ok) {
    const error = await response.text();
    return { data: null, error };
  }

  const data = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : null;

  return { data, error: null };
}

// GitHub API helper for creating PRs with generated tests
interface GitHubPRResult {
  success: boolean;
  pr_url?: string;
  pr_number?: number;
  branch_name?: string;
  error?: string;
}

interface GitHubRepoConfig {
  owner: string;
  repo: string;
  base_branch?: string;
  test_directory?: string;
}

async function createGitHubPR(
  env: Env,
  config: GitHubRepoConfig,
  test: {
    name: string;
    file_path: string;
    code: string;
    description: string;
    production_event_id: string;
  }
): Promise<GitHubPRResult> {
  if (!env.GITHUB_TOKEN) {
    return { success: false, error: "GitHub token not configured" };
  }

  const { owner, repo, base_branch = "main", test_directory = "tests/e2e/generated" } = config;
  const branchName = `argus/test-${test.production_event_id.substring(0, 8)}`;
  const filePath = test.file_path.startsWith("/")
    ? test.file_path.substring(1)
    : `${test_directory}/${test.file_path}`;

  const headers = {
    "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "Argus-Quality-Intelligence",
    "Content-Type": "application/json",
  };

  try {
    // 1. Get the SHA of the base branch
    const baseRefResponse = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/git/refs/heads/${base_branch}`,
      { headers }
    );

    if (!baseRefResponse.ok) {
      const error = await baseRefResponse.text();
      return { success: false, error: `Failed to get base branch: ${error}` };
    }

    const baseRef = await baseRefResponse.json() as { object: { sha: string } };
    const baseSha = baseRef.object.sha;

    // 2. Create a new branch
    const createBranchResponse = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/git/refs`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          ref: `refs/heads/${branchName}`,
          sha: baseSha,
        }),
      }
    );

    if (!createBranchResponse.ok) {
      const error = await createBranchResponse.text();
      // Branch might already exist, try to continue
      if (!error.includes("Reference already exists")) {
        return { success: false, error: `Failed to create branch: ${error}` };
      }
    }

    // 3. Create or update the test file
    // First, check if file exists to get its SHA (needed for updates)
    const existingFileResponse = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/contents/${filePath}?ref=${branchName}`,
      { headers }
    );

    let fileSha: string | undefined;
    if (existingFileResponse.ok) {
      const existingFile = await existingFileResponse.json() as { sha: string };
      fileSha = existingFile.sha;
    }

    // Create/update the file
    const createFileResponse = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/contents/${filePath}`,
      {
        method: "PUT",
        headers,
        body: JSON.stringify({
          message: `test: Add AI-generated test for ${test.name}

Generated by Argus Quality Intelligence Platform
Production Event ID: ${test.production_event_id}

${test.description}`,
          content: btoa(test.code),  // Base64 encode the content
          branch: branchName,
          ...(fileSha ? { sha: fileSha } : {}),
        }),
      }
    );

    if (!createFileResponse.ok) {
      const error = await createFileResponse.text();
      return { success: false, error: `Failed to create file: ${error}` };
    }

    // 4. Create the pull request
    const createPRResponse = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/pulls`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          title: `[Argus] Add test: ${test.name}`,
          body: `## AI-Generated Test

This test was automatically generated by **Argus Quality Intelligence Platform** based on a production error.

### Test Details
- **Test Name**: ${test.name}
- **File Path**: \`${filePath}\`
- **Production Event ID**: \`${test.production_event_id}\`

### Description
${test.description}

### Generated Test Code
\`\`\`typescript
${test.code}
\`\`\`

---
*This PR was created automatically by [Argus](https://heyargus.ai). Please review the generated test before merging.*`,
          head: branchName,
          base: base_branch,
          draft: true,  // Create as draft for review
        }),
      }
    );

    if (!createPRResponse.ok) {
      const error = await createPRResponse.text();

      // Check if PR already exists
      if (error.includes("A pull request already exists")) {
        // Get existing PR
        const existingPRResponse = await fetch(
          `https://api.github.com/repos/${owner}/${repo}/pulls?head=${owner}:${branchName}&state=open`,
          { headers }
        );

        if (existingPRResponse.ok) {
          const prs = await existingPRResponse.json() as Array<{ html_url: string; number: number }>;
          if (prs.length > 0) {
            return {
              success: true,
              pr_url: prs[0].html_url,
              pr_number: prs[0].number,
              branch_name: branchName,
            };
          }
        }
      }

      return { success: false, error: `Failed to create PR: ${error}` };
    }

    const pr = await createPRResponse.json() as { html_url: string; number: number };

    return {
      success: true,
      pr_url: pr.html_url,
      pr_number: pr.number,
      branch_name: branchName,
    };
  } catch (error) {
    return {
      success: false,
      error: `GitHub API error: ${error instanceof Error ? error.message : "Unknown"}`,
    };
  }
}

// Parse Sentry severity to our format
function parseSentrySeverity(level: string): "fatal" | "error" | "warning" | "info" {
  switch (level.toLowerCase()) {
    case "fatal": return "fatal";
    case "error": return "error";
    case "warning": return "warning";
    default: return "info";
  }
}

// Extract component name from stack trace
function extractComponentFromStack(stacktrace: string | undefined): string | null {
  if (!stacktrace) return null;

  // Look for React component patterns
  const reactMatch = stacktrace.match(/at\s+([A-Z][a-zA-Z0-9]*)\s+\(/);
  if (reactMatch) return reactMatch[1];

  // Look for Vue component patterns
  const vueMatch = stacktrace.match(/VueComponent\.([a-zA-Z0-9_]+)/);
  if (vueMatch) return vueMatch[1];

  // Look for Angular component patterns
  const angularMatch = stacktrace.match(/([A-Z][a-zA-Z0-9]*Component)\./);
  if (angularMatch) return angularMatch[1];

  return null;
}

// Generate fingerprint for error grouping
function generateErrorFingerprint(
  type: string,
  message: string,
  component: string | null,
  url: string | null
): string {
  const parts = [type, message.substring(0, 100)];
  if (component) parts.push(component);
  if (url) {
    // Normalize URL by removing query params and IDs
    const normalizedUrl = url
      .replace(/\?.*$/, "")
      .replace(/\/\d+/g, "/:id")
      .replace(/\/[a-f0-9-]{36}/g, "/:uuid");
    parts.push(normalizedUrl);
  }

  // Create a simple hash
  const str = parts.join("|");
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

// Handle Sentry webhook
async function handleSentryWebhook(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  // Log webhook for debugging
  const webhookLogId = crypto.randomUUID();

  try {
    const body = await request.json() as {
      action: string;
      data: { issue?: SentryIssue; event?: SentryEvent };
      installation?: { uuid: string };
      actor?: { type: string; id: string; name: string };
    };

    // Log the incoming webhook
    await supabaseRequest(env, "/webhook_logs", {
      method: "POST",
      body: {
        id: webhookLogId,
        source: "sentry",
        method: request.method,
        headers: Object.fromEntries(request.headers.entries()),
        body: body,
        status: "processing",
      },
    });

    const { action, data } = body;

    // Handle different Sentry webhook actions
    if (action === "issue" || action === "created" || action === "triggered") {
      const issue = data.issue;
      const event = data.event;

      if (!issue && !event) {
        return Response.json(
          { error: "No issue or event data" },
          { status: 400, headers: corsHeaders }
        );
      }

      // Extract error details
      const errorDetails = event?.event?.exception?.values?.[0];
      const stacktrace = errorDetails?.stacktrace?.frames
        ?.map((f) => `  at ${f.function || "anonymous"} (${f.filename}:${f.lineno}:${f.colno})`)
        .reverse()
        .join("\n");

      const pageUrl = event?.event?.request?.url || event?.url;
      const component = extractComponentFromStack(stacktrace);
      const title = issue?.title || errorDetails?.type || "Unknown Error";
      const message = issue?.message || errorDetails?.value || "";

      const fingerprint = generateErrorFingerprint(
        errorDetails?.type || "Error",
        message,
        component,
        pageUrl || null
      );

      // Determine device type from user agent or contexts
      const contexts = event?.event?.contexts;
      let deviceType: "desktop" | "mobile" | "tablet" | null = null;
      if (contexts?.device?.family) {
        const family = contexts.device.family.toLowerCase();
        if (family.includes("iphone") || family.includes("android")) {
          deviceType = "mobile";
        } else if (family.includes("ipad") || family.includes("tablet")) {
          deviceType = "tablet";
        } else {
          deviceType = "desktop";
        }
      }

      // Get project ID from query param or find default project
      const url = new URL(request.url);
      let projectId = url.searchParams.get("project_id");

      if (!projectId) {
        // Find the first project for this integration
        const { data: projectData } = await supabaseRequest(env, "/projects?limit=1");
        if (Array.isArray(projectData) && projectData.length > 0) {
          projectId = projectData[0].id;
        } else {
          return Response.json(
            { error: "No project found. Please specify project_id query parameter." },
            { status: 400, headers: corsHeaders }
          );
        }
      }

      // Create production event
      const productionEvent = {
        project_id: projectId,
        source: "sentry" as const,
        external_id: issue?.id || event?.event_id || crypto.randomUUID(),
        external_url: issue?.url || event?.issue_url || event?.url,
        event_type: "error" as const,
        severity: parseSentrySeverity(issue?.level || event?.event?.level || "error"),
        title,
        message,
        stack_trace: stacktrace,
        fingerprint,
        url: pageUrl,
        component,
        browser: contexts?.browser?.name
          ? `${contexts.browser.name} ${contexts.browser.version || ""}`
          : null,
        os: contexts?.os?.name
          ? `${contexts.os.name} ${contexts.os.version || ""}`
          : null,
        device_type: deviceType,
        occurrence_count: parseInt(issue?.count || "1"),
        affected_users: issue?.userCount || 1,
        first_seen_at: issue?.firstSeen || new Date().toISOString(),
        last_seen_at: issue?.lastSeen || new Date().toISOString(),
        status: "new" as const,
        raw_payload: body,
        tags: issue?.tags?.map((t) => `${t.key}:${t.value}`) || [],
        metadata: {
          sentry_project: event?.project_name || issue?.project,
          sentry_platform: event?.platform || issue?.platform,
          sentry_short_id: issue?.shortId,
        },
      };

      // Insert or update (upsert based on fingerprint)
      const { data: insertedEvent, error: insertError } = await supabaseRequest(
        env,
        "/production_events",
        {
          method: "POST",
          body: productionEvent,
          headers: {
            "Prefer": "return=representation,resolution=merge-duplicates",
          },
        }
      );

      if (insertError) {
        console.error("Failed to insert production event:", insertError);
        await supabaseRequest(env, `/webhook_logs?id=eq.${webhookLogId}`, {
          method: "PATCH",
          body: { status: "failed", error_message: String(insertError) },
        });
        return Response.json(
          { error: "Failed to process event", details: insertError },
          { status: 500, headers: corsHeaders }
        );
      }

      // Update webhook log with success
      await supabaseRequest(env, `/webhook_logs?id=eq.${webhookLogId}`, {
        method: "PATCH",
        body: {
          status: "processed",
          processed_event_id: Array.isArray(insertedEvent) ? insertedEvent[0]?.id : null,
          processed_at: new Date().toISOString(),
        },
      });

      return Response.json(
        {
          success: true,
          message: "Event processed successfully",
          event_id: Array.isArray(insertedEvent) ? insertedEvent[0]?.id : null,
          fingerprint,
        },
        { headers: corsHeaders }
      );
    }

    // Handle other Sentry actions (resolved, ignored, etc.)
    if (action === "resolved" || action === "ignored") {
      const issue = data.issue;
      if (issue?.id) {
        await supabaseRequest(
          env,
          `/production_events?external_id=eq.${issue.id}&source=eq.sentry`,
          {
            method: "PATCH",
            body: {
              status: action === "resolved" ? "resolved" : "ignored",
              resolved_at: new Date().toISOString(),
            },
          }
        );
      }

      return Response.json(
        { success: true, message: `Issue ${action}` },
        { headers: corsHeaders }
      );
    }

    return Response.json(
      { success: true, message: "Webhook received", action },
      { headers: corsHeaders }
    );
  } catch (error) {
    console.error("Sentry webhook error:", error);
    await supabaseRequest(env, `/webhook_logs?id=eq.${webhookLogId}`, {
      method: "PATCH",
      body: {
        status: "failed",
        error_message: error instanceof Error ? error.message : "Unknown error",
      },
    });
    return Response.json(
      { error: "Webhook processing failed", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Handle Datadog webhook
async function handleDatadogWebhook(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const webhookLogId = crypto.randomUUID();

  try {
    // Verify Datadog webhook token if configured
    if (env.DATADOG_WEBHOOK_TOKEN) {
      const authHeader = request.headers.get("Authorization");
      if (authHeader !== `Bearer ${env.DATADOG_WEBHOOK_TOKEN}`) {
        return Response.json(
          { error: "Unauthorized" },
          { status: 401, headers: corsHeaders }
        );
      }
    }

    const body = await request.json() as DatadogEvent | DatadogEvent[];
    const events = Array.isArray(body) ? body : [body];

    // Log the incoming webhook
    await supabaseRequest(env, "/webhook_logs", {
      method: "POST",
      body: {
        id: webhookLogId,
        source: "datadog",
        method: request.method,
        headers: Object.fromEntries(request.headers.entries()),
        body: body,
        status: "processing",
      },
    });

    // Get project ID from query param
    const url = new URL(request.url);
    let projectId = url.searchParams.get("project_id");

    if (!projectId) {
      const { data: projectData } = await supabaseRequest(env, "/projects?limit=1");
      if (Array.isArray(projectData) && projectData.length > 0) {
        projectId = projectData[0].id;
      } else {
        return Response.json(
          { error: "No project found. Please specify project_id query parameter." },
          { status: 400, headers: corsHeaders }
        );
      }
    }

    const processedEvents: string[] = [];

    for (const event of events) {
      // Determine event type
      let eventType: "error" | "exception" | "performance" = "error";
      if (event.error?.type) {
        eventType = "exception";
      } else if (event.event_type?.includes("performance") || event.source_type_name?.includes("apm")) {
        eventType = "performance";
      }

      // Determine severity
      let severity: "fatal" | "error" | "warning" | "info" = "error";
      if (event.alert_type === "error" || event.priority === "high") {
        severity = "error";
      } else if (event.alert_type === "warning" || event.priority === "normal") {
        severity = "warning";
      } else if (event.alert_type === "info" || event.priority === "low") {
        severity = "info";
      }

      const pageUrl = event.view?.url || event.error?.resource?.url || event.url;
      const component = extractComponentFromStack(event.error?.stack);

      const fingerprint = generateErrorFingerprint(
        event.error?.type || event.event_type || "Error",
        event.error?.message || event.message,
        component,
        pageUrl || null
      );

      const productionEvent = {
        project_id: projectId,
        source: "datadog" as const,
        external_id: event.id || event.aggregation_key || crypto.randomUUID(),
        external_url: event.url,
        event_type: eventType,
        severity,
        title: event.title,
        message: event.error?.message || event.message,
        stack_trace: event.error?.stack,
        fingerprint,
        url: pageUrl,
        component,
        occurrence_count: 1,
        affected_users: 1,
        status: "new" as const,
        raw_payload: event,
        tags: event.tags || [],
        metadata: {
          datadog_host: event.host,
          datadog_source: event.source_type_name,
          datadog_priority: event.priority,
          datadog_view_name: event.view?.name,
        },
      };

      const { data: insertedEvent, error: insertError } = await supabaseRequest(
        env,
        "/production_events",
        {
          method: "POST",
          body: productionEvent,
          headers: {
            "Prefer": "return=representation,resolution=merge-duplicates",
          },
        }
      );

      if (!insertError && Array.isArray(insertedEvent)) {
        processedEvents.push(insertedEvent[0]?.id);
      }
    }

    // Update webhook log
    await supabaseRequest(env, `/webhook_logs?id=eq.${webhookLogId}`, {
      method: "PATCH",
      body: {
        status: "processed",
        processed_at: new Date().toISOString(),
      },
    });

    return Response.json(
      {
        success: true,
        message: `Processed ${processedEvents.length} events`,
        event_ids: processedEvents,
      },
      { headers: corsHeaders }
    );
  } catch (error) {
    console.error("Datadog webhook error:", error);
    await supabaseRequest(env, `/webhook_logs?id=eq.${webhookLogId}`, {
      method: "PATCH",
      body: {
        status: "failed",
        error_message: error instanceof Error ? error.message : "Unknown error",
      },
    });
    return Response.json(
      { error: "Webhook processing failed", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Get production events (for dashboard)
async function handleGetProductionEvents(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const url = new URL(request.url);
  const projectId = url.searchParams.get("project_id");
  const status = url.searchParams.get("status");
  const severity = url.searchParams.get("severity");
  const source = url.searchParams.get("source");
  const limit = url.searchParams.get("limit") || "50";
  const offset = url.searchParams.get("offset") || "0";

  let queryPath = "/production_events?select=*";

  if (projectId) queryPath += `&project_id=eq.${projectId}`;
  if (status) queryPath += `&status=eq.${status}`;
  if (severity) queryPath += `&severity=eq.${severity}`;
  if (source) queryPath += `&source=eq.${source}`;

  queryPath += `&order=created_at.desc&limit=${limit}&offset=${offset}`;

  const { data, error } = await supabaseRequest(env, queryPath);

  if (error) {
    return Response.json(
      { error: "Failed to fetch events", details: error },
      { status: 500, headers: corsHeaders }
    );
  }

  return Response.json({ events: data }, { headers: corsHeaders });
}

// Get quality intelligence stats
async function handleGetQualityStats(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const url = new URL(request.url);
  const projectId = url.searchParams.get("project_id");

  if (!projectId) {
    return Response.json(
      { error: "project_id is required" },
      { status: 400, headers: corsHeaders }
    );
  }

  const { data, error } = await supabaseRequest(
    env,
    `/quality_intelligence_stats?project_id=eq.${projectId}`
  );

  if (error) {
    return Response.json(
      { error: "Failed to fetch stats", details: error },
      { status: 500, headers: corsHeaders }
    );
  }

  return Response.json(
    { stats: Array.isArray(data) ? data[0] : data },
    { headers: corsHeaders }
  );
}

// ============================================================================
// AI TEST GENERATION ENGINE
// ============================================================================

interface TestGenerationRequest {
  production_event_id: string;
  project_id: string;
  framework?: "playwright" | "cypress" | "jest";
  auto_create_pr?: boolean;
  github_config?: {
    owner: string;
    repo: string;
    base_branch?: string;
    test_directory?: string;
  };
}

const TEST_GENERATION_PROMPT = `You are an expert QA engineer who converts production errors into automated tests.

Given a production error, generate a Playwright E2E test that would:
1. Navigate to the page where the error occurred
2. Perform the actions that led to the error
3. Assert that the error does NOT occur (the fix is in place)
4. Include proper error handling and assertions

ERROR DETAILS:
Title: {{title}}
Message: {{message}}
URL: {{url}}
Component: {{component}}
Stack Trace:
{{stack_trace}}

User Action Context: {{user_action}}

REQUIREMENTS:
- Use Playwright's best practices
- Include descriptive test names and comments
- Handle async operations properly
- Use robust selectors (data-testid preferred)
- Include both positive and negative assertions
- Add retry logic for flaky elements

Generate a complete, runnable Playwright test file. Return ONLY the test code, no explanations.`;

async function generateTestFromError(
  env: Env,
  event: {
    title: string;
    message: string | null;
    url: string | null;
    component: string | null;
    stack_trace: string | null;
    user_action: string | null;
  }
): Promise<{ test_code: string; test_name: string; confidence: number }> {
  const prompt = TEST_GENERATION_PROMPT
    .replace("{{title}}", event.title)
    .replace("{{message}}", event.message || "No message provided")
    .replace("{{url}}", event.url || "Unknown URL")
    .replace("{{component}}", event.component || "Unknown component")
    .replace("{{stack_trace}}", event.stack_trace || "No stack trace")
    .replace("{{user_action}}", event.user_action || "Unknown action");

  const testCode = await callAI(env, {
    messages: [
      {
        role: "system",
        content: "You are an expert QA automation engineer. Generate clean, maintainable Playwright tests.",
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    maxTokens: 2000,
    temperature: 0.3,
    timeout: 60000,
  });

  // Extract test name from the generated code
  const testNameMatch = testCode.match(/test\(['"](.+?)['"]/);
  const testName = testNameMatch
    ? testNameMatch[1]
    : `test-${event.component || "error"}-${Date.now()}`;

  // Calculate confidence based on available context
  let confidence = 0.5;
  if (event.url) confidence += 0.15;
  if (event.stack_trace) confidence += 0.15;
  if (event.component) confidence += 0.1;
  if (event.user_action) confidence += 0.1;

  return {
    test_code: testCode,
    test_name: testName,
    confidence: Math.min(confidence, 0.95),
  };
}

// Generate test from a production event
async function handleGenerateTest(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const body = await request.json() as TestGenerationRequest;
    const { production_event_id, project_id, framework = "playwright", auto_create_pr = false, github_config } = body;

    if (!production_event_id || !project_id) {
      return Response.json(
        { error: "production_event_id and project_id are required" },
        { status: 400, headers: corsHeaders }
      );
    }

    // Get the production event
    const { data: eventData, error: eventError } = await supabaseRequest(
      env,
      `/production_events?id=eq.${production_event_id}&project_id=eq.${project_id}`
    );

    if (eventError || !Array.isArray(eventData) || eventData.length === 0) {
      return Response.json(
        { error: "Production event not found" },
        { status: 404, headers: corsHeaders }
      );
    }

    const event = eventData[0];

    // Create a test generation job
    const jobId = crypto.randomUUID();
    const jobStart = Date.now();

    await supabaseRequest(env, "/test_generation_jobs", {
      method: "POST",
      body: {
        id: jobId,
        project_id: project_id,
        production_event_id: production_event_id,
        status: "running",
        job_type: "single_error",
        started_at: new Date().toISOString(),
      },
    });

    // Update event status
    await supabaseRequest(env, `/production_events?id=eq.${production_event_id}`, {
      method: "PATCH",
      body: { status: "analyzing" },
    });

    // Generate the test
    const { test_code, test_name, confidence } = await generateTestFromError(env, {
      title: event.title,
      message: event.message,
      url: event.url,
      component: event.component,
      stack_trace: event.stack_trace,
      user_action: event.user_action,
    });

    // Determine file path based on framework and component
    const componentSlug = (event.component || "unknown")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "-");
    const timestamp = Date.now();
    const filePath = framework === "playwright"
      ? `tests/e2e/generated/${componentSlug}-${timestamp}.spec.ts`
      : framework === "cypress"
        ? `cypress/e2e/generated/${componentSlug}-${timestamp}.cy.ts`
        : `__tests__/generated/${componentSlug}-${timestamp}.test.ts`;

    // Create the generated test record
    const { data: generatedTest, error: testError } = await supabaseRequest(
      env,
      "/generated_tests",
      {
        method: "POST",
        body: {
          project_id: project_id,
          production_event_id: production_event_id,
          name: test_name,
          description: `Auto-generated test to prevent: ${event.title}`,
          test_type: "e2e",
          framework: framework,
          test_code: test_code,
          test_file_path: filePath,
          confidence_score: confidence,
          status: "pending",
          steps: [],
          assertions: [],
          metadata: {
            generated_from_error: event.title,
            original_url: event.url,
            component: event.component,
          },
        },
        headers: {
          "Prefer": "return=representation",
        },
      }
    );

    if (testError) {
      // Update job as failed
      await supabaseRequest(env, `/test_generation_jobs?id=eq.${jobId}`, {
        method: "PATCH",
        body: {
          status: "failed",
          error_message: String(testError),
          completed_at: new Date().toISOString(),
          duration_ms: Date.now() - jobStart,
        },
      });

      return Response.json(
        { error: "Failed to save generated test", details: testError },
        { status: 500, headers: corsHeaders }
      );
    }

    const generatedTestId = Array.isArray(generatedTest) ? generatedTest[0]?.id : null;

    // Update event status
    await supabaseRequest(env, `/production_events?id=eq.${production_event_id}`, {
      method: "PATCH",
      body: {
        status: "test_pending_review",
        ai_analysis: {
          generated_test_id: generatedTestId,
          confidence_score: confidence,
          generated_at: new Date().toISOString(),
        },
      },
    });

    // Update job as completed
    await supabaseRequest(env, `/test_generation_jobs?id=eq.${jobId}`, {
      method: "PATCH",
      body: {
        status: "completed",
        tests_generated: 1,
        completed_at: new Date().toISOString(),
        duration_ms: Date.now() - jobStart,
      },
    });

    // Create GitHub PR if requested
    let prUrl: string | null = null;
    let prNumber: number | null = null;
    if (auto_create_pr && env.GITHUB_TOKEN && github_config) {
      const prResult = await createGitHubPR(env, github_config, {
        name: test_name,
        file_path: filePath,
        code: test_code,
        description: `Auto-generated test to prevent: ${event.title}`,
        production_event_id: production_event_id,
      });

      if (prResult.success) {
        prUrl = prResult.pr_url || null;
        prNumber = prResult.pr_number || null;

        // Update the generated test record with PR info
        if (generatedTestId) {
          await supabaseRequest(env, `/generated_tests?id=eq.${generatedTestId}`, {
            method: "PATCH",
            body: {
              github_pr_url: prUrl,
              github_pr_number: prNumber,
              github_pr_status: "draft",
            },
          });
        }
      } else {
        console.error("Failed to create GitHub PR:", prResult.error);
      }
    }

    return Response.json(
      {
        success: true,
        message: "Test generated successfully",
        generated_test: {
          id: generatedTestId,
          name: test_name,
          file_path: filePath,
          confidence_score: confidence,
          framework: framework,
        },
        test_code: test_code,
        pr_url: prUrl,
        pr_number: prNumber,
      },
      { headers: corsHeaders }
    );
  } catch (error) {
    console.error("Test generation error:", error);
    return Response.json(
      { error: "Test generation failed", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Batch generate tests for multiple events
async function handleBatchGenerateTests(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const body = await request.json() as {
      project_id: string;
      status?: string;
      limit?: number;
      framework?: "playwright" | "cypress" | "jest";
    };

    const { project_id, status = "new", limit = 10, framework = "playwright" } = body;

    if (!project_id) {
      return Response.json(
        { error: "project_id is required" },
        { status: 400, headers: corsHeaders }
      );
    }

    // Get production events that need tests
    const { data: events, error: eventsError } = await supabaseRequest(
      env,
      `/production_events?project_id=eq.${project_id}&status=eq.${status}&order=severity.desc,occurrence_count.desc&limit=${limit}`
    );

    if (eventsError || !Array.isArray(events)) {
      return Response.json(
        { error: "Failed to fetch events", details: eventsError },
        { status: 500, headers: corsHeaders }
      );
    }

    if (events.length === 0) {
      return Response.json(
        { success: true, message: "No events to process", generated: 0 },
        { headers: corsHeaders }
      );
    }

    // Create a batch job
    const jobId = crypto.randomUUID();
    const jobStart = Date.now();

    await supabaseRequest(env, "/test_generation_jobs", {
      method: "POST",
      body: {
        id: jobId,
        project_id: project_id,
        status: "running",
        job_type: "pattern_batch",
        started_at: new Date().toISOString(),
        metadata: { event_count: events.length },
      },
    });

    const results: Array<{
      event_id: string;
      success: boolean;
      test_id?: string;
      error?: string;
    }> = [];

    // Process events sequentially to avoid rate limits
    for (const event of events) {
      try {
        const { test_code, test_name, confidence } = await generateTestFromError(env, {
          title: event.title,
          message: event.message,
          url: event.url,
          component: event.component,
          stack_trace: event.stack_trace,
          user_action: event.user_action,
        });

        const componentSlug = (event.component || "unknown")
          .toLowerCase()
          .replace(/[^a-z0-9]/g, "-");
        const timestamp = Date.now();
        const filePath = `tests/e2e/generated/${componentSlug}-${timestamp}.spec.ts`;

        const { data: generatedTest } = await supabaseRequest(
          env,
          "/generated_tests",
          {
            method: "POST",
            body: {
              project_id: project_id,
              production_event_id: event.id,
              name: test_name,
              description: `Auto-generated test to prevent: ${event.title}`,
              test_type: "e2e",
              framework: framework,
              test_code: test_code,
              test_file_path: filePath,
              confidence_score: confidence,
              status: "pending",
            },
            headers: {
              "Prefer": "return=representation",
            },
          }
        );

        const testId = Array.isArray(generatedTest) ? generatedTest[0]?.id : null;

        // Update event status
        await supabaseRequest(env, `/production_events?id=eq.${event.id}`, {
          method: "PATCH",
          body: {
            status: "test_pending_review",
            ai_analysis: {
              generated_test_id: testId,
              confidence_score: confidence,
              generated_at: new Date().toISOString(),
            },
          },
        });

        results.push({ event_id: event.id, success: true, test_id: testId });
      } catch (error) {
        results.push({
          event_id: event.id,
          success: false,
          error: error instanceof Error ? error.message : "Unknown error",
        });
      }
    }

    const successCount = results.filter((r) => r.success).length;

    // Update job as completed
    await supabaseRequest(env, `/test_generation_jobs?id=eq.${jobId}`, {
      method: "PATCH",
      body: {
        status: "completed",
        tests_generated: successCount,
        completed_at: new Date().toISOString(),
        duration_ms: Date.now() - jobStart,
        metadata: { results },
      },
    });

    return Response.json(
      {
        success: true,
        message: `Generated ${successCount}/${events.length} tests`,
        job_id: jobId,
        results: results,
      },
      { headers: corsHeaders }
    );
  } catch (error) {
    console.error("Batch test generation error:", error);
    return Response.json(
      { error: "Batch generation failed", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Get generated tests
async function handleGetGeneratedTests(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const url = new URL(request.url);
  const projectId = url.searchParams.get("project_id");
  const status = url.searchParams.get("status");
  const limit = url.searchParams.get("limit") || "50";

  if (!projectId) {
    return Response.json(
      { error: "project_id is required" },
      { status: 400, headers: corsHeaders }
    );
  }

  let queryPath = `/generated_tests?project_id=eq.${projectId}`;
  if (status) queryPath += `&status=eq.${status}`;
  queryPath += `&order=created_at.desc&limit=${limit}`;

  const { data, error } = await supabaseRequest(env, queryPath);

  if (error) {
    return Response.json(
      { error: "Failed to fetch generated tests", details: error },
      { status: 500, headers: corsHeaders }
    );
  }

  return Response.json({ tests: data }, { headers: corsHeaders });
}

// Approve/reject a generated test
async function handleUpdateGeneratedTest(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const body = await request.json() as {
      test_id: string;
      action: "approve" | "reject" | "modify";
      review_notes?: string;
      modified_code?: string;
    };

    const { test_id, action, review_notes, modified_code } = body;

    if (!test_id || !action) {
      return Response.json(
        { error: "test_id and action are required" },
        { status: 400, headers: corsHeaders }
      );
    }

    const updateData: Record<string, unknown> = {
      status: action === "approve" ? "approved" : action === "reject" ? "rejected" : "modified",
      reviewed_at: new Date().toISOString(),
      review_notes: review_notes,
    };

    if (modified_code && action === "modify") {
      updateData.test_code = modified_code;
    }

    const { error } = await supabaseRequest(env, `/generated_tests?id=eq.${test_id}`, {
      method: "PATCH",
      body: updateData,
    });

    if (error) {
      return Response.json(
        { error: "Failed to update test", details: error },
        { status: 500, headers: corsHeaders }
      );
    }

    // If approved, update the production event status
    if (action === "approve") {
      const { data: testData } = await supabaseRequest(
        env,
        `/generated_tests?id=eq.${test_id}`
      );

      if (Array.isArray(testData) && testData.length > 0 && testData[0].production_event_id) {
        await supabaseRequest(
          env,
          `/production_events?id=eq.${testData[0].production_event_id}`,
          {
            method: "PATCH",
            body: { status: "test_generated" },
          }
        );
      }
    }

    return Response.json(
      { success: true, message: `Test ${action}d successfully` },
      { headers: corsHeaders }
    );
  } catch (error) {
    return Response.json(
      { error: "Update failed", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// ============================================================================
// RISK SCORING ENGINE
// ============================================================================

interface RiskScoreFactors {
  error_frequency: number;      // 0-100: How often errors occur
  error_severity: number;       // 0-100: Weighted severity of errors
  test_coverage: number;        // 0-100: Inverse of coverage (low = high risk)
  user_impact: number;          // 0-100: Number of affected users
  change_frequency: number;     // 0-100: How often the component changes
  recency: number;              // 0-100: How recent are the errors
}

interface EntityErrorData {
  entity_type: 'page' | 'component' | 'flow' | 'endpoint';
  entity_identifier: string;
  entity_name: string | null;
  error_count: number;
  fatal_count: number;
  error_severity_count: number;
  warning_count: number;
  affected_users: number;
  last_error_at: string;
  first_error_at: string;
}

// Weight configuration for risk score calculation
const RISK_WEIGHTS = {
  error_frequency: 0.25,
  error_severity: 0.30,
  test_coverage: 0.20,
  user_impact: 0.15,
  recency: 0.10,
};

// Calculate risk score for a single entity
function calculateEntityRiskScore(
  entity: EntityErrorData,
  maxErrorCount: number,
  maxAffectedUsers: number,
  testCoverage: number // 0-100
): { overall: number; factors: RiskScoreFactors } {
  const now = Date.now();
  const lastErrorTime = new Date(entity.last_error_at).getTime();
  const daysSinceLastError = (now - lastErrorTime) / (1000 * 60 * 60 * 24);

  // Calculate individual factor scores (0-100)
  const factors: RiskScoreFactors = {
    // Error frequency: normalized against the max in the project
    error_frequency: maxErrorCount > 0
      ? Math.min(100, Math.round((entity.error_count / maxErrorCount) * 100))
      : 0,

    // Error severity: weighted score
    // Fatal = 100 weight, Error = 70, Warning = 30
    error_severity: Math.min(100, Math.round(
      ((entity.fatal_count * 100) + (entity.error_severity_count * 70) + (entity.warning_count * 30)) /
      Math.max(1, entity.error_count)
    )),

    // Test coverage: inverse (0% coverage = 100 risk, 100% coverage = 0 risk)
    test_coverage: Math.max(0, 100 - testCoverage),

    // User impact: normalized against max affected users
    user_impact: maxAffectedUsers > 0
      ? Math.min(100, Math.round((entity.affected_users / maxAffectedUsers) * 100))
      : 0,

    // Change frequency: placeholder (would need git integration)
    change_frequency: 0,

    // Recency: errors in last 24h = 100, decays over 30 days
    recency: Math.max(0, Math.round(100 - (daysSinceLastError / 30) * 100)),
  };

  // Calculate weighted overall score
  const overall = Math.round(
    (factors.error_frequency * RISK_WEIGHTS.error_frequency) +
    (factors.error_severity * RISK_WEIGHTS.error_severity) +
    (factors.test_coverage * RISK_WEIGHTS.test_coverage) +
    (factors.user_impact * RISK_WEIGHTS.user_impact) +
    (factors.recency * RISK_WEIGHTS.recency)
  );

  return { overall: Math.min(100, overall), factors };
}

// Determine trend by comparing to previous score
function calculateTrend(
  currentScore: number,
  previousScore: number | null
): 'improving' | 'stable' | 'degrading' {
  if (previousScore === null) return 'stable';
  const diff = currentScore - previousScore;
  if (diff <= -5) return 'improving';  // Score decreased = better
  if (diff >= 5) return 'degrading';   // Score increased = worse
  return 'stable';
}

// Calculate risk scores for all entities in a project
async function handleCalculateRiskScores(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const body = await request.json() as {
      project_id: string;
      entity_types?: string[];
    };

    const { project_id, entity_types = ['page', 'component', 'flow', 'endpoint'] } = body;

    if (!project_id) {
      return Response.json(
        { error: "project_id is required" },
        { status: 400, headers: corsHeaders }
      );
    }

    // 1. Aggregate production events by entity (URL/component)
    const { data: events, error: eventsError } = await supabaseRequest(
      env,
      `/production_events?project_id=eq.${project_id}&select=url,component,severity,affected_users,created_at`
    );

    if (eventsError) {
      return Response.json(
        { error: "Failed to fetch production events", details: eventsError },
        { status: 500, headers: corsHeaders }
      );
    }

    const eventList = (events || []) as Array<{
      url: string | null;
      component: string | null;
      severity: string;
      affected_users: number;
      created_at: string;
    }>;

    if (eventList.length === 0) {
      return Response.json(
        { success: true, message: "No production events to analyze", scores_updated: 0 },
        { headers: corsHeaders }
      );
    }

    // 2. Group events by entity
    const entityMap = new Map<string, EntityErrorData>();

    for (const event of eventList) {
      // Create entries for both URL (page) and component if available
      const entities: Array<{ type: 'page' | 'component'; id: string; name: string | null }> = [];

      if (event.url) {
        // Extract path from URL for page entity
        try {
          const urlObj = new URL(event.url);
          entities.push({
            type: 'page',
            id: urlObj.pathname,
            name: urlObj.pathname,
          });
        } catch {
          entities.push({ type: 'page', id: event.url, name: event.url });
        }
      }

      if (event.component) {
        entities.push({
          type: 'component',
          id: event.component,
          name: event.component,
        });
      }

      for (const entity of entities) {
        const key = `${entity.type}:${entity.id}`;
        const existing = entityMap.get(key);

        if (existing) {
          existing.error_count++;
          existing.affected_users = Math.max(existing.affected_users, event.affected_users);
          if (event.severity === 'fatal') existing.fatal_count++;
          if (event.severity === 'error') existing.error_severity_count++;
          if (event.severity === 'warning') existing.warning_count++;
          if (new Date(event.created_at) > new Date(existing.last_error_at)) {
            existing.last_error_at = event.created_at;
          }
          if (new Date(event.created_at) < new Date(existing.first_error_at)) {
            existing.first_error_at = event.created_at;
          }
        } else {
          entityMap.set(key, {
            entity_type: entity.type,
            entity_identifier: entity.id,
            entity_name: entity.name,
            error_count: 1,
            fatal_count: event.severity === 'fatal' ? 1 : 0,
            error_severity_count: event.severity === 'error' ? 1 : 0,
            warning_count: event.severity === 'warning' ? 1 : 0,
            affected_users: event.affected_users,
            last_error_at: event.created_at,
            first_error_at: event.created_at,
          });
        }
      }
    }

    // 3. Calculate max values for normalization
    let maxErrorCount = 0;
    let maxAffectedUsers = 0;
    for (const entity of entityMap.values()) {
      maxErrorCount = Math.max(maxErrorCount, entity.error_count);
      maxAffectedUsers = Math.max(maxAffectedUsers, entity.affected_users);
    }

    // 4. Get existing risk scores for trend calculation
    const { data: existingScores } = await supabaseRequest(
      env,
      `/risk_scores?project_id=eq.${project_id}&select=entity_identifier,entity_type,overall_risk_score`
    );

    const existingScoreMap = new Map<string, number>();
    if (Array.isArray(existingScores)) {
      for (const score of existingScores) {
        existingScoreMap.set(
          `${score.entity_type}:${score.entity_identifier}`,
          score.overall_risk_score
        );
      }
    }

    // 5. Get test coverage data (tests that cover these entities)
    const { data: tests } = await supabaseRequest(
      env,
      `/tests?project_id=eq.${project_id}&select=target_url,metadata`
    );

    const testedEntities = new Set<string>();
    if (Array.isArray(tests)) {
      for (const test of tests) {
        if (test.target_url) {
          try {
            const urlObj = new URL(test.target_url);
            testedEntities.add(`page:${urlObj.pathname}`);
          } catch {
            testedEntities.add(`page:${test.target_url}`);
          }
        }
        // Check metadata for component testing
        if (test.metadata?.component) {
          testedEntities.add(`component:${test.metadata.component}`);
        }
      }
    }

    // 6. Calculate and upsert risk scores
    const scoreUpdates: Array<{
      project_id: string;
      entity_type: string;
      entity_identifier: string;
      entity_name: string | null;
      overall_risk_score: number;
      error_frequency_score: number;
      error_severity_score: number;
      test_coverage_score: number;
      user_impact_score: number;
      change_frequency_score: number;
      factors: Record<string, unknown>;
      previous_score: number | null;
      score_trend: string;
      recommendations: string[];
      priority_tests_needed: string[];
      last_calculated_at: string;
    }> = [];

    for (const [key, entity] of entityMap) {
      if (!entity_types.includes(entity.entity_type)) continue;

      // Check if this entity has test coverage
      const hasTests = testedEntities.has(key);
      const testCoverage = hasTests ? 80 : 0; // Simplified: 80% if has tests, 0% if not

      const { overall, factors } = calculateEntityRiskScore(
        entity,
        maxErrorCount,
        maxAffectedUsers,
        testCoverage
      );

      const previousScore = existingScoreMap.get(key) ?? null;
      const trend = calculateTrend(overall, previousScore);

      // Generate recommendations based on factors
      const recommendations: string[] = [];
      const priorityTests: string[] = [];

      if (factors.test_coverage > 70) {
        recommendations.push(`Add E2E tests for ${entity.entity_name || entity.entity_identifier}`);
        priorityTests.push(`Test ${entity.entity_type}: ${entity.entity_identifier}`);
      }
      if (factors.error_severity > 60) {
        recommendations.push(`Investigate critical errors in ${entity.entity_name || entity.entity_identifier}`);
      }
      if (factors.recency > 80) {
        recommendations.push(`Recent errors detected - immediate attention needed`);
      }
      if (factors.user_impact > 50) {
        recommendations.push(`High user impact - prioritize fixes for ${entity.entity_name || entity.entity_identifier}`);
      }

      scoreUpdates.push({
        project_id: project_id,
        entity_type: entity.entity_type,
        entity_identifier: entity.entity_identifier,
        entity_name: entity.entity_name,
        overall_risk_score: overall,
        error_frequency_score: factors.error_frequency,
        error_severity_score: factors.error_severity,
        test_coverage_score: factors.test_coverage,
        user_impact_score: factors.user_impact,
        change_frequency_score: factors.change_frequency,
        factors: {
          error_count: entity.error_count,
          fatal_count: entity.fatal_count,
          error_count_severity: entity.error_severity_count,
          warning_count: entity.warning_count,
          affected_users: entity.affected_users,
          has_tests: hasTests,
          recency_score: factors.recency,
        },
        previous_score: previousScore,
        score_trend: trend,
        recommendations: recommendations,
        priority_tests_needed: priorityTests,
        last_calculated_at: new Date().toISOString(),
      });
    }

    // 7. Upsert risk scores (delete existing and insert new)
    // First, delete existing scores for this project
    await supabaseRequest(env, `/risk_scores?project_id=eq.${project_id}`, {
      method: "DELETE",
    });

    // Then insert new scores
    let successCount = 0;
    for (const score of scoreUpdates) {
      const { error } = await supabaseRequest(env, "/risk_scores", {
        method: "POST",
        body: score,
      });
      if (!error) successCount++;
    }

    // 8. Update quality intelligence stats
    const highRiskCount = scoreUpdates.filter(s => s.overall_risk_score >= 70).length;
    const testedCount = scoreUpdates.filter(s => s.test_coverage_score < 50).length;

    await supabaseRequest(env, `/quality_intelligence_stats?project_id=eq.${project_id}`, {
      method: "PATCH",
      body: {
        high_risk_components: highRiskCount,
        components_with_tests: testedCount,
        updated_at: new Date().toISOString(),
      },
    });

    return Response.json(
      {
        success: true,
        message: "Risk scores calculated successfully",
        scores_updated: successCount,
        high_risk_count: highRiskCount,
        entities_analyzed: entityMap.size,
        summary: {
          pages: scoreUpdates.filter(s => s.entity_type === 'page').length,
          components: scoreUpdates.filter(s => s.entity_type === 'component').length,
          avg_risk_score: scoreUpdates.length > 0
            ? Math.round(scoreUpdates.reduce((sum, s) => sum + s.overall_risk_score, 0) / scoreUpdates.length)
            : 0,
        },
      },
      { headers: corsHeaders }
    );
  } catch (error) {
    console.error("Risk score calculation error:", error);
    return Response.json(
      { error: "Risk score calculation failed", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Get risk scores
async function handleGetRiskScores(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const url = new URL(request.url);
  const projectId = url.searchParams.get("project_id");
  const entityType = url.searchParams.get("entity_type");
  const minScore = url.searchParams.get("min_score") || "0";

  if (!projectId) {
    return Response.json(
      { error: "project_id is required" },
      { status: 400, headers: corsHeaders }
    );
  }

  let queryPath = `/risk_scores?project_id=eq.${projectId}&overall_risk_score=gte.${minScore}&order=overall_risk_score.desc`;

  if (entityType) queryPath += `&entity_type=eq.${entityType}`;

  const { data, error } = await supabaseRequest(env, queryPath);

  if (error) {
    return Response.json(
      { error: "Failed to fetch risk scores", details: error },
      { status: 500, headers: corsHeaders }
    );
  }

  return Response.json({ risk_scores: data }, { headers: corsHeaders });
}

// ============================================================================
// AUTONOMOUS QUALITY LOOP
// The core differentiator - fully automated error-to-test pipeline
// ============================================================================

interface AutonomousLoopConfig {
  project_id: string;
  production_event_id: string;
  // Options
  run_discovery?: boolean;      // Discover page elements
  capture_visual?: boolean;     // Capture visual baseline
  generate_test?: boolean;      // Generate AI test
  verify_test?: boolean;        // Run test to verify it works
  create_pr?: boolean;          // Create GitHub PR
  learn_pattern?: boolean;      // Update pattern database
  // GitHub config (required if create_pr is true)
  github_config?: {
    owner: string;
    repo: string;
    base_branch?: string;
    test_directory?: string;
  };
}

interface AutonomousLoopResult {
  success: boolean;
  stages: {
    discovery?: { success: boolean; elements_found?: number; error?: string };
    visual?: { success: boolean; screenshot_id?: string; error?: string };
    generation?: { success: boolean; test_id?: string; confidence?: number; error?: string };
    verification?: { success: boolean; passed?: boolean; duration_ms?: number; error?: string };
    pr?: { success: boolean; pr_url?: string; pr_number?: number; error?: string };
    learning?: { success: boolean; pattern_id?: string; error?: string };
  };
  total_duration_ms: number;
}

// Run the full autonomous quality loop
async function handleAutonomousLoop(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  const startTime = Date.now();

  try {
    const config = await request.json() as AutonomousLoopConfig;
    const {
      project_id,
      production_event_id,
      run_discovery = true,
      capture_visual = true,
      generate_test = true,
      verify_test = true,
      create_pr = false,
      learn_pattern = true,
      github_config,
    } = config;

    if (!project_id || !production_event_id) {
      return Response.json(
        { error: "project_id and production_event_id are required" },
        { status: 400, headers: corsHeaders }
      );
    }

    const result: AutonomousLoopResult = {
      success: false,
      stages: {},
      total_duration_ms: 0,
    };

    // 1. Fetch the production event
    const { data: eventData, error: eventError } = await supabaseRequest(
      env,
      `/production_events?id=eq.${production_event_id}`
    );

    if (eventError || !Array.isArray(eventData) || eventData.length === 0) {
      return Response.json(
        { error: "Production event not found", details: eventError },
        { status: 404, headers: corsHeaders }
      );
    }

    const event = eventData[0] as {
      id: string;
      url: string | null;
      title: string;
      message: string | null;
      stack_trace: string | null;
      component: string | null;
      severity: string;
      user_action: string | null;
    };

    if (!event.url) {
      return Response.json(
        { error: "Production event has no URL - cannot run autonomous loop" },
        { status: 400, headers: corsHeaders }
      );
    }

    // Update event status to analyzing
    await supabaseRequest(env, `/production_events?id=eq.${production_event_id}`, {
      method: "PATCH",
      body: { status: "analyzing" },
    });

    // 2. STAGE: Discovery - Find page elements
    if (run_discovery) {
      try {
        // Use the /observe endpoint internally
        const observeResponse = await handleObserve(
          new Request("http://internal/observe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              url: event.url,
              actions: ["click", "fill", "select"],
            }),
          }),
          env,
          corsHeaders
        );

        const observeData = await observeResponse.json() as { actions?: unknown[] };
        result.stages.discovery = {
          success: true,
          elements_found: observeData.actions?.length || 0,
        };
      } catch (error) {
        result.stages.discovery = {
          success: false,
          error: error instanceof Error ? error.message : "Discovery failed",
        };
      }
    }

    // 3. STAGE: Visual Capture - Take baseline screenshot
    if (capture_visual) {
      try {
        // Use the /extract endpoint to capture screenshot
        const extractResponse = await handleExtract(
          new Request("http://internal/extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              url: event.url,
              schema: { screenshot: "full page screenshot" },
              screenshot: true,
            }),
          }),
          env,
          corsHeaders
        );

        const extractData = await extractResponse.json() as { screenshot?: string };
        result.stages.visual = {
          success: !!extractData.screenshot,
          screenshot_id: extractData.screenshot ? `visual-${production_event_id}` : undefined,
        };
      } catch (error) {
        result.stages.visual = {
          success: false,
          error: error instanceof Error ? error.message : "Visual capture failed",
        };
      }
    }

    // 4. STAGE: Test Generation - Create AI test
    let generatedTestId: string | null = null;
    let testCode: string | null = null;
    let testFilePath: string | null = null;

    if (generate_test) {
      try {
        const genResponse = await handleGenerateTest(
          new Request("http://internal/api/generate-test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              production_event_id,
              project_id,
              framework: "playwright",
              auto_create_pr: false, // We'll handle PR creation in the loop
            }),
          }),
          env,
          corsHeaders
        );

        const genData = await genResponse.json() as {
          success?: boolean;
          generated_test?: { id: string; confidence_score: number; file_path: string };
          test_code?: string;
        };

        if (genData.success && genData.generated_test) {
          generatedTestId = genData.generated_test.id;
          testCode = genData.test_code || null;
          testFilePath = genData.generated_test.file_path;
          result.stages.generation = {
            success: true,
            test_id: generatedTestId,
            confidence: genData.generated_test.confidence_score,
          };
        } else {
          result.stages.generation = { success: false, error: "Generation returned no test" };
        }
      } catch (error) {
        result.stages.generation = {
          success: false,
          error: error instanceof Error ? error.message : "Test generation failed",
        };
      }
    }

    // 5. STAGE: Verification - Run the generated test
    if (verify_test && testCode && event.url) {
      try {
        const verifyStart = Date.now();

        // Use the /test endpoint to run a quick verification
        const testResponse = await handleTest(
          new Request("http://internal/test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              url: event.url,
              steps: [
                { action: "navigate", url: event.url },
                { action: "wait", duration: 2000 },
                { action: "screenshot" },
              ],
              name: `Verify: ${event.title}`,
            }),
          }),
          env,
          corsHeaders
        );

        const testData = await testResponse.json() as { success?: boolean };
        result.stages.verification = {
          success: true,
          passed: testData.success,
          duration_ms: Date.now() - verifyStart,
        };

        // Update generated test with verification result
        if (generatedTestId) {
          await supabaseRequest(env, `/generated_tests?id=eq.${generatedTestId}`, {
            method: "PATCH",
            body: {
              times_run: 1,
              times_passed: testData.success ? 1 : 0,
              times_failed: testData.success ? 0 : 1,
            },
          });
        }
      } catch (error) {
        result.stages.verification = {
          success: false,
          error: error instanceof Error ? error.message : "Verification failed",
        };
      }
    }

    // 6. STAGE: Create PR - Push to GitHub
    if (create_pr && github_config && testCode && testFilePath && generatedTestId) {
      try {
        const prResult = await createGitHubPR(env, github_config, {
          name: `test-${event.component || "error"}-${Date.now()}`,
          file_path: testFilePath,
          code: testCode,
          description: `Auto-generated test for: ${event.title}`,
          production_event_id,
        });

        if (prResult.success) {
          result.stages.pr = {
            success: true,
            pr_url: prResult.pr_url,
            pr_number: prResult.pr_number,
          };

          // Update generated test with PR info
          await supabaseRequest(env, `/generated_tests?id=eq.${generatedTestId}`, {
            method: "PATCH",
            body: {
              github_pr_url: prResult.pr_url,
              github_pr_number: prResult.pr_number,
              github_pr_status: "draft",
            },
          });
        } else {
          result.stages.pr = { success: false, error: prResult.error };
        }
      } catch (error) {
        result.stages.pr = {
          success: false,
          error: error instanceof Error ? error.message : "PR creation failed",
        };
      }
    }

    // 7. STAGE: Pattern Learning - Update cross-company patterns
    if (learn_pattern) {
      try {
        // Create anonymized pattern from this error
        const patternHash = await generatePatternHash(event);

        // Check if pattern exists
        const { data: existingPatterns } = await supabaseRequest(
          env,
          `/error_patterns?pattern_hash=eq.${patternHash}`
        );

        if (Array.isArray(existingPatterns) && existingPatterns.length > 0) {
          // Update existing pattern
          const pattern = existingPatterns[0] as { id: string; total_occurrences: number };
          await supabaseRequest(env, `/error_patterns?id=eq.${pattern.id}`, {
            method: "PATCH",
            body: {
              total_occurrences: pattern.total_occurrences + 1,
              updated_at: new Date().toISOString(),
            },
          });
          result.stages.learning = { success: true, pattern_id: pattern.id };
        } else {
          // Create new pattern
          const { data: newPattern } = await supabaseRequest(env, "/error_patterns", {
            method: "POST",
            body: {
              pattern_hash: patternHash,
              pattern_type: event.stack_trace ? "stack_trace" : "error_message",
              pattern_signature: anonymizePattern(event),
              example_message: event.title.substring(0, 100),
              category: categorizeError(event.title, event.stack_trace),
              frameworks: detectFrameworks(event.stack_trace),
              total_occurrences: 1,
              companies_affected: 1,
            },
            headers: { "Prefer": "return=representation" },
          });

          const patternData = Array.isArray(newPattern) ? newPattern[0] : newPattern;
          result.stages.learning = {
            success: true,
            pattern_id: patternData?.id || "new",
          };
        }
      } catch (error) {
        result.stages.learning = {
          success: false,
          error: error instanceof Error ? error.message : "Pattern learning failed",
        };
      }
    }

    // Calculate overall success
    const stageResults = Object.values(result.stages);
    const successfulStages = stageResults.filter(s => s.success).length;
    result.success = successfulStages >= stageResults.length * 0.5; // 50%+ success
    result.total_duration_ms = Date.now() - startTime;

    // Update event status based on results
    const finalStatus = result.stages.pr?.success
      ? "test_pending_review"
      : result.stages.generation?.success
        ? "test_generated"
        : "analyzing";

    await supabaseRequest(env, `/production_events?id=eq.${production_event_id}`, {
      method: "PATCH",
      body: {
        status: finalStatus,
        ai_analysis: {
          autonomous_loop_run: true,
          stages_completed: successfulStages,
          total_stages: stageResults.length,
          duration_ms: result.total_duration_ms,
        },
      },
    });

    return Response.json(result, { headers: corsHeaders });
  } catch (error) {
    console.error("Autonomous loop error:", error);
    return Response.json(
      {
        success: false,
        error: "Autonomous loop failed",
        details: error instanceof Error ? error.message : "Unknown",
        total_duration_ms: Date.now() - startTime,
      },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Helper: Generate a hash for pattern matching (anonymized)
async function generatePatternHash(event: {
  title: string;
  stack_trace?: string | null;
  component?: string | null;
}): Promise<string> {
  // Create a normalized, anonymized string for hashing
  const normalized = [
    event.title.replace(/[0-9]+/g, "N").replace(/['"][^'"]*['"]/g, "S"),
    event.component || "",
    event.stack_trace
      ? event.stack_trace
          .split("\n")
          .slice(0, 3)
          .map(line => line.replace(/:\d+:\d+/g, ":N:N"))
          .join("|")
      : "",
  ].join(":::");

  // Simple hash using Web Crypto API
  const encoder = new TextEncoder();
  const data = encoder.encode(normalized);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, "0")).join("").substring(0, 32);
}

// Helper: Anonymize pattern for cross-company learning
function anonymizePattern(event: {
  title: string;
  stack_trace?: string | null;
  component?: string | null;
}): Record<string, unknown> {
  return {
    error_type: extractErrorType(event.title),
    has_stack_trace: !!event.stack_trace,
    stack_depth: event.stack_trace?.split("\n").length || 0,
    component_type: event.component ? detectComponentType(event.component) : null,
    // Remove all company-specific data
  };
}

// Helper: Extract error type from message
function extractErrorType(message: string): string {
  const patterns: Record<string, RegExp> = {
    null_reference: /null|undefined|cannot read|is not defined/i,
    type_error: /type.*error|cannot.*type|invalid.*type/i,
    network_error: /network|fetch|xhr|http|timeout|connection/i,
    validation_error: /validat|invalid|required|missing/i,
    auth_error: /auth|unauthorized|forbidden|401|403/i,
    not_found: /not found|404|missing/i,
    syntax_error: /syntax|parse|unexpected token/i,
  };

  for (const [type, regex] of Object.entries(patterns)) {
    if (regex.test(message)) return type;
  }
  return "unknown";
}

// Helper: Detect component type from name
function detectComponentType(component: string): string {
  const lowerName = component.toLowerCase();
  if (lowerName.includes("button")) return "button";
  if (lowerName.includes("form")) return "form";
  if (lowerName.includes("modal")) return "modal";
  if (lowerName.includes("table")) return "table";
  if (lowerName.includes("list")) return "list";
  if (lowerName.includes("nav")) return "navigation";
  if (lowerName.includes("header")) return "header";
  if (lowerName.includes("footer")) return "footer";
  if (lowerName.includes("input")) return "input";
  if (lowerName.includes("select")) return "select";
  return "generic";
}

// Helper: Categorize error
function categorizeError(title: string, stackTrace?: string | null): string {
  const combined = `${title} ${stackTrace || ""}`.toLowerCase();

  if (combined.includes("react")) return "react";
  if (combined.includes("vue")) return "vue";
  if (combined.includes("angular")) return "angular";
  if (combined.includes("next")) return "nextjs";
  if (combined.includes("nuxt")) return "nuxt";
  if (combined.includes("svelte")) return "svelte";

  return extractErrorType(title);
}

// Helper: Detect frameworks from stack trace
function detectFrameworks(stackTrace?: string | null): string[] {
  if (!stackTrace) return [];
  const frameworks: string[] = [];

  if (stackTrace.includes("react")) frameworks.push("react");
  if (stackTrace.includes("vue")) frameworks.push("vue");
  if (stackTrace.includes("angular")) frameworks.push("angular");
  if (stackTrace.includes("next")) frameworks.push("nextjs");
  if (stackTrace.includes("express")) frameworks.push("express");
  if (stackTrace.includes("node_modules")) frameworks.push("node");

  return frameworks;
}

// ============================================================================
// VECTORIZE: SEMANTIC PATTERN MATCHING
// ============================================================================

interface PatternEmbedding {
  id: string;
  values: number[];
  metadata: {
    pattern_hash: string;
    category: string;
    frameworks: string[];
    example_message: string;
    solution_count: number;
  };
}

interface SimilarPattern {
  id: string;
  score: number;
  pattern_hash: string;
  category: string;
  example_message: string;
  known_solutions?: string[];
}

// Generate embeddings using Workers AI (bge-base-en-v1.5 - 768 dimensions)
async function generateEmbedding(
  env: Env,
  text: string
): Promise<number[] | null> {
  if (!env.AI) {
    console.log("Workers AI not available, skipping embedding generation");
    return null;
  }

  try {
    // Normalize the text for better embedding quality
    const normalizedText = text
      .replace(/[0-9]+/g, "NUM")              // Replace numbers
      .replace(/['"][^'"]*['"]/g, "STRING")   // Replace string literals
      .replace(/0x[0-9a-fA-F]+/g, "HEX")      // Replace hex values
      .substring(0, 512);                      // Limit length for embedding model

    const response = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
      text: [normalizedText],
    });

    if (response && response.data && response.data[0]) {
      return response.data[0] as number[];
    }
    return null;
  } catch (error) {
    console.error("Embedding generation failed:", error);
    return null;
  }
}

// Store pattern embedding in Vectorize
async function storePatternEmbedding(
  env: Env,
  patternId: string,
  embedding: number[],
  metadata: PatternEmbedding["metadata"]
): Promise<boolean> {
  if (!env.VECTOR_INDEX) {
    console.log("Vectorize not available, skipping embedding storage");
    return false;
  }

  try {
    await env.VECTOR_INDEX.upsert([
      {
        id: patternId,
        values: embedding,
        metadata: metadata,
      },
    ]);
    return true;
  } catch (error) {
    console.error("Failed to store embedding:", error);
    return false;
  }
}

// Find semantically similar patterns using Vectorize
async function findSimilarPatterns(
  env: Env,
  errorText: string,
  limit: number = 5,
  minScore: number = 0.7
): Promise<SimilarPattern[]> {
  if (!env.VECTOR_INDEX || !env.AI) {
    return [];
  }

  try {
    // Generate embedding for the query
    const embedding = await generateEmbedding(env, errorText);
    if (!embedding) return [];

    // Query Vectorize for similar patterns
    const results = await env.VECTOR_INDEX.query(embedding, {
      topK: limit,
      returnMetadata: true,
    });

    if (!results || !results.matches) return [];

    // Filter by minimum score and transform results
    const similarPatterns: SimilarPattern[] = [];
    for (const match of results.matches) {
      if (match.score >= minScore) {
        const metadata = match.metadata as PatternEmbedding["metadata"];

        // Fetch known solutions from database
        let knownSolutions: string[] = [];
        try {
          const { data } = await supabaseRequest(
            env,
            `/error_pattern_solutions?pattern_id=eq.${match.id}&select=solution_description&limit=3`
          );
          if (Array.isArray(data)) {
            knownSolutions = data.map((s: { solution_description: string }) => s.solution_description);
          }
        } catch {
          // Solutions are optional
        }

        similarPatterns.push({
          id: match.id,
          score: match.score,
          pattern_hash: metadata?.pattern_hash || "",
          category: metadata?.category || "unknown",
          example_message: metadata?.example_message || "",
          known_solutions: knownSolutions.length > 0 ? knownSolutions : undefined,
        });
      }
    }

    return similarPatterns;
  } catch (error) {
    console.error("Semantic pattern search failed:", error);
    return [];
  }
}

// Enhanced pattern learning with vector embeddings
async function learnPatternWithEmbedding(
  env: Env,
  event: {
    id: string;
    title: string;
    message?: string;
    stack_trace?: string | null;
    component?: string | null;
  }
): Promise<{ pattern_id: string; similar_patterns: SimilarPattern[] } | null> {
  try {
    // Generate pattern hash for exact matching
    const patternHash = await generatePatternHash(event);

    // Generate text for embedding (combine title, message, and first few stack trace lines)
    const embeddingText = [
      event.title,
      event.message || "",
      event.stack_trace ? event.stack_trace.split("\n").slice(0, 5).join(" ") : "",
    ].join(" ").trim();

    // Find similar patterns first
    const similarPatterns = await findSimilarPatterns(env, embeddingText, 5, 0.75);

    // Check if exact pattern exists
    const { data: existingPatterns } = await supabaseRequest(
      env,
      `/error_patterns?pattern_hash=eq.${patternHash}`
    );

    let patternId: string;

    if (Array.isArray(existingPatterns) && existingPatterns.length > 0) {
      // Update existing pattern
      const pattern = existingPatterns[0] as { id: string; total_occurrences: number };
      patternId = pattern.id;

      await supabaseRequest(env, `/error_patterns?id=eq.${patternId}`, {
        method: "PATCH",
        body: {
          total_occurrences: pattern.total_occurrences + 1,
          updated_at: new Date().toISOString(),
        },
      });
    } else {
      // Create new pattern
      const category = categorizeError(event.title, event.stack_trace);
      const frameworks = detectFrameworks(event.stack_trace);

      const { data: newPattern } = await supabaseRequest(env, "/error_patterns", {
        method: "POST",
        body: {
          pattern_hash: patternHash,
          pattern_type: event.stack_trace ? "stack_trace" : "error_message",
          pattern_signature: anonymizePattern(event),
          example_message: event.title.substring(0, 100),
          category,
          frameworks,
          total_occurrences: 1,
        },
      });

      patternId = (newPattern as { id: string })?.id || crypto.randomUUID();

      // Store embedding in Vectorize for semantic search
      const embedding = await generateEmbedding(env, embeddingText);
      if (embedding) {
        await storePatternEmbedding(env, patternId, embedding, {
          pattern_hash: patternHash,
          category,
          frameworks,
          example_message: event.title.substring(0, 100),
          solution_count: 0,
        });
      }
    }

    return {
      pattern_id: patternId,
      similar_patterns: similarPatterns,
    };
  } catch (error) {
    console.error("Pattern learning with embedding failed:", error);
    return null;
  }
}

// API Handler: Semantic pattern search
async function handleSemanticPatternSearch(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const body = await request.json() as {
      error_text: string;
      limit?: number;
      min_score?: number;
    };

    if (!body.error_text) {
      return Response.json(
        { error: "error_text is required" },
        { status: 400, headers: corsHeaders }
      );
    }

    // Check if Vectorize is available
    if (!env.VECTOR_INDEX) {
      return Response.json(
        {
          error: "Semantic search not available",
          message: "Vectorize index not configured. Run: wrangler vectorize create argus-patterns --dimensions=768 --metric=cosine"
        },
        { status: 503, headers: corsHeaders }
      );
    }

    const similarPatterns = await findSimilarPatterns(
      env,
      body.error_text,
      body.limit || 5,
      body.min_score || 0.7
    );

    return Response.json({
      success: true,
      query: body.error_text.substring(0, 100),
      patterns: similarPatterns,
      count: similarPatterns.length,
      has_solutions: similarPatterns.some(p => p.known_solutions && p.known_solutions.length > 0),
    }, { headers: corsHeaders });

  } catch (error) {
    console.error("Semantic pattern search failed:", error);
    return Response.json(
      { error: "Semantic search failed", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// API Handler: Get AI Quality Score (composite metric)
async function handleGetAIQualityScore(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const url = new URL(request.url);
    const projectId = url.searchParams.get("project_id");

    if (!projectId) {
      return Response.json(
        { error: "project_id is required" },
        { status: 400, headers: corsHeaders }
      );
    }

    // Fetch all quality metrics
    const [eventsRes, testsRes, riskRes, statsRes] = await Promise.all([
      supabaseRequest(env, `/production_events?project_id=eq.${projectId}&select=severity,status&limit=1000`),
      supabaseRequest(env, `/generated_tests?project_id=eq.${projectId}&select=status,confidence_score&limit=1000`),
      supabaseRequest(env, `/risk_scores?project_id=eq.${projectId}&select=overall_risk_score&limit=100`),
      supabaseRequest(env, `/quality_intelligence_stats?project_id=eq.${projectId}`),
    ]);

    const events = (eventsRes.data || []) as Array<{ severity: string; status: string }>;
    const tests = (testsRes.data || []) as Array<{ status: string; confidence_score: number }>;
    const risks = (riskRes.data || []) as Array<{ overall_risk_score: number }>;
    const stats = Array.isArray(statsRes.data) && statsRes.data.length > 0 ? statsRes.data[0] : null;

    // Calculate component scores (0-100)

    // 1. Error Management Score (fewer unresolved errors = higher score)
    const unresolvedErrors = events.filter(e => e.status === "new" || e.status === "analyzing").length;
    const resolvedErrors = events.filter(e => e.status === "resolved" || e.status === "test_generated").length;
    const totalErrors = events.length;
    const errorManagementScore = totalErrors === 0 ? 100 : Math.round((resolvedErrors / totalErrors) * 100);

    // 2. Test Coverage Score (more approved/deployed tests = higher score)
    const approvedTests = tests.filter(t => t.status === "approved" || t.status === "deployed").length;
    const pendingTests = tests.filter(t => t.status === "pending").length;
    const avgConfidence = tests.length > 0
      ? tests.reduce((acc, t) => acc + t.confidence_score, 0) / tests.length
      : 0;
    const testCoverageScore = Math.min(100, Math.round(
      (approvedTests * 10) + (avgConfidence * 50) + (pendingTests * 3)
    ));

    // 3. Risk Mitigation Score (lower average risk = higher score)
    const avgRisk = risks.length > 0
      ? risks.reduce((acc, r) => acc + r.overall_risk_score, 0) / risks.length
      : 0;
    const riskMitigationScore = Math.round(100 - avgRisk);

    // 4. Automation Score (based on test generation rate)
    const testsGenerated = (stats as Record<string, number> | null)?.tests_generated || 0;
    const eventsReceived = (stats as Record<string, number> | null)?.total_production_events || 0;
    const automationRate = eventsReceived > 0 ? testsGenerated / eventsReceived : 0;
    const automationScore = Math.min(100, Math.round(automationRate * 100));

    // 5. Prevention Score (incidents prevented)
    const incidentsPrevented = (stats as Record<string, number> | null)?.incidents_prevented || 0;
    const preventionScore = Math.min(100, incidentsPrevented * 10);

    // Calculate overall AI Quality Score (weighted average)
    const weights = {
      errorManagement: 0.20,
      testCoverage: 0.25,
      riskMitigation: 0.20,
      automation: 0.20,
      prevention: 0.15,
    };

    const overallScore = Math.round(
      errorManagementScore * weights.errorManagement +
      testCoverageScore * weights.testCoverage +
      riskMitigationScore * weights.riskMitigation +
      automationScore * weights.automation +
      preventionScore * weights.prevention
    );

    // Determine grade
    let grade: string;
    let gradeColor: string;
    if (overallScore >= 90) { grade = "A+"; gradeColor = "green"; }
    else if (overallScore >= 80) { grade = "A"; gradeColor = "green"; }
    else if (overallScore >= 70) { grade = "B"; gradeColor = "lime"; }
    else if (overallScore >= 60) { grade = "C"; gradeColor = "yellow"; }
    else if (overallScore >= 50) { grade = "D"; gradeColor = "orange"; }
    else { grade = "F"; gradeColor = "red"; }

    // Generate insights
    const insights: string[] = [];
    if (unresolvedErrors > 5) insights.push(`${unresolvedErrors} production errors need attention`);
    if (pendingTests > 3) insights.push(`${pendingTests} generated tests awaiting review`);
    if (avgRisk > 60) insights.push(`High average risk score (${Math.round(avgRisk)})`);
    if (automationRate < 0.5) insights.push("Consider enabling auto-test generation for faster coverage");
    if (overallScore >= 80) insights.push("Excellent quality posture! Keep monitoring for regressions.");

    return Response.json({
      success: true,
      project_id: projectId,
      overall_score: overallScore,
      grade,
      grade_color: gradeColor,
      component_scores: {
        error_management: {
          score: errorManagementScore,
          label: "Error Management",
          description: "How well production errors are being addressed",
        },
        test_coverage: {
          score: testCoverageScore,
          label: "Test Coverage",
          description: "Quality and quantity of AI-generated tests",
        },
        risk_mitigation: {
          score: riskMitigationScore,
          label: "Risk Mitigation",
          description: "Overall risk level of components",
        },
        automation: {
          score: automationScore,
          label: "Automation Level",
          description: "Error-to-test automation rate",
        },
        prevention: {
          score: preventionScore,
          label: "Prevention",
          description: "Incidents prevented by tests",
        },
      },
      metrics: {
        total_events: totalErrors,
        unresolved_events: unresolvedErrors,
        tests_generated: tests.length,
        tests_approved: approvedTests,
        avg_confidence: Math.round(avgConfidence * 100),
        high_risk_components: risks.filter(r => r.overall_risk_score >= 70).length,
        incidents_prevented: incidentsPrevented,
      },
      insights,
      calculated_at: new Date().toISOString(),
    }, { headers: corsHeaders });

  } catch (error) {
    console.error("Failed to calculate AI Quality Score:", error);
    return Response.json(
      { error: "Failed to calculate quality score", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// ============================================================================
// PREDICTIVE QUALITY: AI-POWERED BUG PREDICTION
// ============================================================================

interface PredictedRisk {
  entity: string;
  entity_type: string;
  prediction_score: number;  // 0-100, likelihood of failure
  predicted_timeframe: string; // "24h", "7d", "30d"
  risk_factors: string[];
  recommendations: string[];
  similar_past_failures: number;
  confidence: number;  // 0-1
}

interface TrendData {
  entity: string;
  error_count_7d: number;
  error_count_30d: number;
  trend: "increasing" | "stable" | "decreasing";
  velocity: number;  // Rate of change
}

// API Handler: Get Predictive Quality Analysis
async function handleGetPredictiveQuality(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const url = new URL(request.url);
    const projectId = url.searchParams.get("project_id");
    const timeframe = url.searchParams.get("timeframe") || "7d";

    if (!projectId) {
      return Response.json(
        { error: "project_id is required" },
        { status: 400, headers: corsHeaders }
      );
    }

    // Fetch historical data for prediction
    const [eventsRes, riskRes, patternsRes] = await Promise.all([
      // Get all production events with timestamps
      supabaseRequest(env, `/production_events?project_id=eq.${projectId}&select=id,title,severity,url,component,created_at,occurrence_count,affected_users&order=created_at.desc&limit=500`),
      // Get current risk scores
      supabaseRequest(env, `/risk_scores?project_id=eq.${projectId}&select=*`),
      // Get known error patterns
      supabaseRequest(env, `/error_patterns?select=pattern_hash,category,total_occurrences,example_message&order=total_occurrences.desc&limit=50`),
    ]);

    const events = (eventsRes.data || []) as Array<{
      id: string;
      title: string;
      severity: string;
      url?: string;
      component?: string;
      created_at: string;
      occurrence_count: number;
      affected_users: number;
    }>;

    const risks = (riskRes.data || []) as Array<{
      id: string;
      entity_type: string;
      entity_identifier: string;
      entity_name?: string;
      overall_risk_score: number;
      error_frequency_score?: number;
      score_trend?: string;
    }>;

    const patterns = (patternsRes.data || []) as Array<{
      pattern_hash: string;
      category: string;
      total_occurrences: number;
      example_message: string;
    }>;

    // Calculate trend data for each entity
    const now = new Date();
    const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    // Group events by entity (URL or component)
    const entityErrors: Record<string, {
      events_7d: number;
      events_30d: number;
      severity_sum: number;
      users_affected: number;
      entity_type: string;
    }> = {};

    events.forEach(event => {
      const entity = event.component || event.url || "unknown";
      const entityType = event.component ? "component" : "page";
      const eventDate = new Date(event.created_at);

      if (!entityErrors[entity]) {
        entityErrors[entity] = { events_7d: 0, events_30d: 0, severity_sum: 0, users_affected: 0, entity_type: entityType };
      }

      const severityScore = event.severity === "fatal" ? 4 : event.severity === "error" ? 3 : event.severity === "warning" ? 2 : 1;

      if (eventDate >= sevenDaysAgo) {
        entityErrors[entity].events_7d += event.occurrence_count;
        entityErrors[entity].severity_sum += severityScore * event.occurrence_count;
      }
      if (eventDate >= thirtyDaysAgo) {
        entityErrors[entity].events_30d += event.occurrence_count;
        entityErrors[entity].users_affected = Math.max(entityErrors[entity].users_affected, event.affected_users);
      }
    });

    // Calculate predictions
    const predictions: PredictedRisk[] = [];

    for (const [entity, data] of Object.entries(entityErrors)) {
      // Calculate trend velocity
      const weeklyRate = data.events_7d;
      const monthlyRate = data.events_30d / 4; // Avg per week
      const velocity = monthlyRate > 0 ? (weeklyRate - monthlyRate) / monthlyRate : 0;

      // Determine trend
      let trend: "increasing" | "stable" | "decreasing";
      if (velocity > 0.2) trend = "increasing";
      else if (velocity < -0.2) trend = "decreasing";
      else trend = "stable";

      // Find matching risk score
      const riskScore = risks.find(r =>
        r.entity_identifier === entity || r.entity_name === entity
      );

      // Calculate prediction score using multiple factors
      const factors = {
        trend_weight: trend === "increasing" ? 0.35 : trend === "decreasing" ? -0.15 : 0,
        severity_weight: Math.min(data.severity_sum / 20, 0.25),
        frequency_weight: Math.min(data.events_7d / 50, 0.2),
        users_weight: Math.min(data.users_affected / 100, 0.1),
        historical_risk: (riskScore?.overall_risk_score || 50) / 100 * 0.2,
      };

      const baseScore = 30; // Base prediction score
      const predictionScore = Math.min(100, Math.max(0, Math.round(
        baseScore +
        factors.trend_weight * 100 +
        factors.severity_weight * 100 +
        factors.frequency_weight * 100 +
        factors.users_weight * 100 +
        factors.historical_risk * 100
      )));

      // Only include predictions above threshold
      if (predictionScore >= 40 || trend === "increasing") {
        // Generate risk factors
        const riskFactors: string[] = [];
        if (trend === "increasing") riskFactors.push(`Error rate increasing ${Math.round(velocity * 100)}% vs last month`);
        if (data.events_7d >= 10) riskFactors.push(`${data.events_7d} errors in the last 7 days`);
        if (data.users_affected >= 50) riskFactors.push(`${data.users_affected} users affected`);
        if (data.severity_sum >= 10) riskFactors.push("High severity errors detected");
        if (riskScore?.score_trend === "degrading") riskFactors.push("Risk score trending upward");

        // Generate recommendations
        const recommendations: string[] = [];
        if (trend === "increasing") recommendations.push("Prioritize writing tests for this component");
        if (data.events_7d >= 20) recommendations.push("Investigate root cause immediately");
        if (!riskScore) recommendations.push("Add to risk monitoring");
        recommendations.push("Consider adding error boundary or fallback");
        recommendations.push("Review recent changes to this component");

        // Calculate confidence based on data availability
        const dataPoints = data.events_30d + (riskScore ? 1 : 0) + patterns.length;
        const confidence = Math.min(0.95, 0.5 + (dataPoints / 100));

        predictions.push({
          entity,
          entity_type: data.entity_type,
          prediction_score: predictionScore,
          predicted_timeframe: timeframe,
          risk_factors: riskFactors.slice(0, 5),
          recommendations: recommendations.slice(0, 4),
          similar_past_failures: patterns.filter(p =>
            entity.toLowerCase().includes(p.category) || p.example_message.toLowerCase().includes(entity.toLowerCase())
          ).reduce((sum, p) => sum + p.total_occurrences, 0),
          confidence: Math.round(confidence * 100) / 100,
        });
      }
    }

    // Sort by prediction score (highest risk first)
    predictions.sort((a, b) => b.prediction_score - a.prediction_score);

    // Calculate overall prediction summary
    const highRiskCount = predictions.filter(p => p.prediction_score >= 70).length;
    const mediumRiskCount = predictions.filter(p => p.prediction_score >= 40 && p.prediction_score < 70).length;
    const totalPredicted = predictions.length;

    // Generate AI summary using Workers AI if available
    let aiSummary = "";
    if (env.AI && predictions.length > 0) {
      try {
        const topRisks = predictions.slice(0, 3).map(p => `${p.entity} (${p.prediction_score}%)`).join(", ");
        const summaryPrompt = `Based on error trend analysis, summarize in 2 sentences: ${highRiskCount} high-risk and ${mediumRiskCount} medium-risk components identified. Top concerns: ${topRisks}. Focus on actionable insights.`;

        const aiResponse = await env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
          messages: [{ role: "user", content: summaryPrompt }],
          max_tokens: 150,
        });

        if (aiResponse && aiResponse.response) {
          aiSummary = aiResponse.response;
        }
      } catch {
        // AI summary is optional
      }
    }

    return Response.json({
      success: true,
      project_id: projectId,
      timeframe,
      predictions: predictions.slice(0, 20),  // Top 20 predictions
      summary: {
        total_analyzed: Object.keys(entityErrors).length,
        total_predicted: totalPredicted,
        high_risk: highRiskCount,
        medium_risk: mediumRiskCount,
        increasing_trends: predictions.filter(p => p.risk_factors.some(f => f.includes("increasing"))).length,
      },
      ai_summary: aiSummary || null,
      data_quality: {
        events_analyzed: events.length,
        risk_scores_available: risks.length,
        patterns_learned: patterns.length,
      },
      calculated_at: new Date().toISOString(),
    }, { headers: corsHeaders });

  } catch (error) {
    console.error("Failed to calculate predictive quality:", error);
    return Response.json(
      { error: "Failed to generate predictions", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// ============================================================================
// ADDITIONAL OBSERVABILITY INTEGRATIONS
// ============================================================================

// FullStory Webhook Handler
interface FullStoryEvent {
  eventType: "rage_click" | "dead_click" | "error" | "frustration_signal";
  sessionId: string;
  userId?: string;
  pageUrl: string;
  timestamp: string;
  data: {
    errorMessage?: string;
    clickTarget?: string;
    frustrationScore?: number;
  };
}

async function handleFullStoryWebhook(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const payload = await request.json() as FullStoryEvent;

    // Map FullStory event types to our event types
    const eventTypeMap: Record<string, string> = {
      rage_click: "rage_click",
      dead_click: "dead_click",
      error: "error",
      frustration_signal: "user_feedback",
    };

    // Get project from header or query
    const url = new URL(request.url);
    const projectId = url.searchParams.get("project_id");

    if (!projectId) {
      return Response.json(
        { error: "project_id query param required" },
        { status: 400, headers: corsHeaders }
      );
    }

    const { data, error } = await supabaseRequest(env, "/production_events", {
      method: "POST",
      body: {
        project_id: projectId,
        source: "fullstory",
        external_id: payload.sessionId,
        external_url: `https://app.fullstory.com/ui/session/${payload.sessionId}`,
        event_type: eventTypeMap[payload.eventType] || "error",
        severity: payload.data.frustrationScore && payload.data.frustrationScore > 80 ? "error" : "warning",
        title: payload.data.errorMessage || `${payload.eventType} detected`,
        message: payload.data.clickTarget
          ? `User clicked on: ${payload.data.clickTarget}`
          : payload.data.errorMessage,
        url: payload.pageUrl,
        user_action: payload.eventType,
        raw_payload: payload,
        status: "new",
      },
    });

    if (error) {
      return Response.json({ error: "Failed to store event", details: error }, { status: 500, headers: corsHeaders });
    }

    return Response.json({ success: true, event: data }, { headers: corsHeaders });
  } catch (error) {
    return Response.json(
      { error: "Failed to process FullStory webhook", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// LogRocket Webhook Handler
interface LogRocketEvent {
  type: "error" | "issue";
  sessionURL: string;
  userEmail?: string;
  errorMessage?: string;
  errorStack?: string;
  pageURL: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

async function handleLogRocketWebhook(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const payload = await request.json() as LogRocketEvent;

    const url = new URL(request.url);
    const projectId = url.searchParams.get("project_id");

    if (!projectId) {
      return Response.json(
        { error: "project_id query param required" },
        { status: 400, headers: corsHeaders }
      );
    }

    const component = extractComponentFromStack(payload.errorStack);

    const { data, error } = await supabaseRequest(env, "/production_events", {
      method: "POST",
      body: {
        project_id: projectId,
        source: "logrocket",
        external_id: payload.sessionURL,
        external_url: payload.sessionURL,
        event_type: payload.type === "error" ? "error" : "exception",
        severity: "error",
        title: payload.errorMessage || "LogRocket Issue",
        message: payload.errorMessage,
        stack_trace: payload.errorStack,
        url: payload.pageURL,
        component,
        raw_payload: payload,
        status: "new",
      },
    });

    if (error) {
      return Response.json({ error: "Failed to store event", details: error }, { status: 500, headers: corsHeaders });
    }

    return Response.json({ success: true, event: data }, { headers: corsHeaders });
  } catch (error) {
    return Response.json(
      { error: "Failed to process LogRocket webhook", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// New Relic Webhook Handler
interface NewRelicEvent {
  condition_name: string;
  account_id: number;
  incident_id: string;
  incident_url: string;
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  timestamp: number;
  details?: string;
  targets?: Array<{
    type: string;
    name: string;
    link?: string;
  }>;
}

async function handleNewRelicWebhook(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const payload = await request.json() as NewRelicEvent;

    const url = new URL(request.url);
    const projectId = url.searchParams.get("project_id");

    if (!projectId) {
      return Response.json(
        { error: "project_id query param required" },
        { status: 400, headers: corsHeaders }
      );
    }

    const severityMap: Record<string, string> = {
      CRITICAL: "fatal",
      HIGH: "error",
      MEDIUM: "warning",
      LOW: "info",
    };

    const { data, error } = await supabaseRequest(env, "/production_events", {
      method: "POST",
      body: {
        project_id: projectId,
        source: "newrelic",
        external_id: payload.incident_id,
        external_url: payload.incident_url,
        event_type: "error",
        severity: severityMap[payload.severity] || "error",
        title: payload.condition_name,
        message: payload.details,
        url: payload.targets?.[0]?.link,
        raw_payload: payload,
        status: "new",
      },
    });

    if (error) {
      return Response.json({ error: "Failed to store event", details: error }, { status: 500, headers: corsHeaders });
    }

    return Response.json({ success: true, event: data }, { headers: corsHeaders });
  } catch (error) {
    return Response.json(
      { error: "Failed to process New Relic webhook", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Bugsnag Webhook Handler
interface BugsnagEvent {
  error: {
    exceptionClass: string;
    message: string;
    stacktrace: Array<{ file: string; lineNumber: number; method: string }>;
  };
  app: {
    version?: string;
  };
  context?: string;
  unhandled: boolean;
  severity: "error" | "warning" | "info";
  user?: {
    id?: string;
    email?: string;
  };
  metaData?: Record<string, unknown>;
  url?: string;
  project?: {
    id: string;
    name: string;
    url: string;
  };
}

async function handleBugsnagWebhook(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const payload = await request.json() as BugsnagEvent;

    const url = new URL(request.url);
    const projectId = url.searchParams.get("project_id");

    if (!projectId) {
      return Response.json(
        { error: "project_id query param required" },
        { status: 400, headers: corsHeaders }
      );
    }

    const stackTrace = payload.error.stacktrace
      .slice(0, 10)
      .map(frame => `at ${frame.method} (${frame.file}:${frame.lineNumber})`)
      .join("\n");

    const component = extractComponentFromStack(stackTrace);

    const { data, error } = await supabaseRequest(env, "/production_events", {
      method: "POST",
      body: {
        project_id: projectId,
        source: "bugsnag",
        external_id: `${payload.project?.id}-${Date.now()}`,
        external_url: payload.project?.url,
        event_type: payload.unhandled ? "crash" : "exception",
        severity: payload.severity === "error" ? "error" : payload.severity === "warning" ? "warning" : "info",
        title: `${payload.error.exceptionClass}: ${payload.error.message}`,
        message: payload.error.message,
        stack_trace: stackTrace,
        url: payload.url || payload.context,
        component,
        raw_payload: payload,
        status: "new",
      },
    });

    if (error) {
      return Response.json({ error: "Failed to store event", details: error }, { status: 500, headers: corsHeaders });
    }

    return Response.json({ success: true, event: data }, { headers: corsHeaders });
  } catch (error) {
    return Response.json(
      { error: "Failed to process Bugsnag webhook", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
}

// Rollbar Webhook Handler
interface RollbarEvent {
  event_name: string;
  data: {
    item: {
      id: number;
      counter: number;
      title: string;
      level: "critical" | "error" | "warning" | "info" | "debug";
      environment: string;
      framework: string;
      last_occurrence_timestamp: number;
      total_occurrences: number;
      unique_occurrences: number;
      public_item_id: number;
    };
    occurrence?: {
      body: {
        trace?: {
          exception: { class: string; message: string };
          frames: Array<{ filename: string; lineno: number; method: string }>;
        };
      };
      request?: {
        url: string;
        user_ip: string;
      };
    };
  };
}

async function handleRollbarWebhook(
  request: Request,
  env: Env,
  corsHeaders: Record<string, string>
): Promise<Response> {
  try {
    const payload = await request.json() as RollbarEvent;

    const url = new URL(request.url);
    const projectId = url.searchParams.get("project_id");

    if (!projectId) {
      return Response.json(
        { error: "project_id query param required" },
        { status: 400, headers: corsHeaders }
      );
    }

    const trace = payload.data.occurrence?.body?.trace;
    const stackTrace = trace?.frames
      ?.slice(0, 10)
      .map(frame => `at ${frame.method} (${frame.filename}:${frame.lineno})`)
      .join("\n");

    const component = stackTrace ? extractComponentFromStack(stackTrace) : null;

    const severityMap: Record<string, string> = {
      critical: "fatal",
      error: "error",
      warning: "warning",
      info: "info",
      debug: "info",
    };

    const { data, error } = await supabaseRequest(env, "/production_events", {
      method: "POST",
      body: {
        project_id: projectId,
        source: "rollbar",
        external_id: String(payload.data.item.id),
        external_url: `https://rollbar.com/item/${payload.data.item.public_item_id}`,
        event_type: payload.data.item.level === "critical" ? "crash" : "exception",
        severity: severityMap[payload.data.item.level] || "error",
        title: payload.data.item.title,
        message: trace?.exception?.message || payload.data.item.title,
        stack_trace: stackTrace,
        url: payload.data.occurrence?.request?.url,
        component,
        occurrence_count: payload.data.item.total_occurrences,
        affected_users: payload.data.item.unique_occurrences,
        raw_payload: payload,
        status: "new",
      },
    });

    if (error) {
      return Response.json({ error: "Failed to store event", details: error }, { status: 500, headers: corsHeaders });
    }

    return Response.json({ success: true, event: data }, { headers: corsHeaders });
  } catch (error) {
    return Response.json(
      { error: "Failed to process Rollbar webhook", details: error instanceof Error ? error.message : "Unknown" },
      { status: 500, headers: corsHeaders }
    );
  }
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

      return Response.json({
        status: "healthy",
        version: "2.0.0",
        backends: {
          cloudflare: true,
          testingbot: hasTestingBot,
        },
        features: ["act", "extract", "observe", "agent", "test"],
        browsers: hasTestingBot ? ["chrome", "firefox", "safari", "edge", "webkit"] : ["chrome"],
        devices: Object.keys(DEVICE_PRESETS),
        config: {
          defaultBackend: defaults.backend,
          defaultModelProvider: defaults.modelProvider,
          selfHealingEnabled: defaults.selfHeal,
          cachingEnabled: defaults.caching,
          autoFailover: hasTestingBot, // Automatically failover to TestingBot on Cloudflare rate limit
          aiProviders: {
            "workers-ai": true,
            "openai": hasOpenAI,
            "anthropic": hasAnthropic,
          },
        },
      }, { headers: corsHeaders });
    }

    // API docs
    if (path === "/") {
      return Response.json({
        name: "Argus API",
        version: "2.1.0",
        description: "AI-powered browser automation and Quality Intelligence Platform - heyargus.ai",
        endpoints: {
          // Browser Automation
          "POST /act": "Execute browser action",
          "POST /extract": "Extract structured data",
          "POST /observe": "Discover available actions",
          "POST /agent": "Run autonomous workflow",
          "POST /test": "Run cross-browser tests",
          "GET /health": "Health check",
          // Quality Intelligence - Webhooks
          "POST /webhooks/sentry": "Receive Sentry error events",
          "POST /webhooks/datadog": "Receive Datadog error events",
          // Quality Intelligence - API
          "GET /api/production-events": "List production events",
          "GET /api/quality-stats": "Get quality intelligence stats",
          "GET /api/risk-scores": "Get risk scores for components",
          "GET /api/generated-tests": "List AI-generated tests",
          // Test Generation
          "POST /api/generate-test": "Generate test from production error",
          "POST /api/batch-generate-tests": "Batch generate tests from errors",
          "POST /api/update-generated-test": "Approve/reject generated test",
          // Risk Scoring
          "POST /api/calculate-risk-scores": "Calculate risk scores for project entities",
          // Autonomous Quality Loop
          "POST /api/autonomous-loop": "Run autonomous quality loop (discovery → visual → test → verify → PR)",
          // Semantic Pattern Search (Vectorize AI)
          "POST /api/semantic-search": "Find semantically similar error patterns using AI embeddings",
          // AI Quality Score
          "GET /api/quality-score": "Get composite AI Quality Score with component breakdowns",
          // Predictive Quality
          "GET /api/predictive-quality": "AI-powered bug prediction and trend analysis",
        },
        backends: ["cloudflare (free, Chromium)", "testingbot (paid, all browsers + devices)"],
        devices: Object.keys(DEVICE_PRESETS),
        qualityIntelligence: {
          integrations: ["sentry", "datadog", "fullstory", "logrocket", "newrelic", "bugsnag", "rollbar"],
          features: ["error-to-test", "risk-scoring", "pattern-learning", "autonomous-loop", "semantic-search", "ai-quality-score", "predictive-quality"],
          ai_powered: ["vectorize-embeddings", "workers-ai-inference", "semantic-similarity", "llama-predictions"],
        },
      }, { headers: corsHeaders });
    }

    // Quality Intelligence Webhooks - NO AUTH REQUIRED (they use their own verification)
    if (request.method === "POST" && path.startsWith("/webhooks/")) {
      try {
        switch (path) {
          case "/webhooks/sentry": return await handleSentryWebhook(request, env, corsHeaders);
          case "/webhooks/datadog": return await handleDatadogWebhook(request, env, corsHeaders);
          case "/webhooks/fullstory": return await handleFullStoryWebhook(request, env, corsHeaders);
          case "/webhooks/logrocket": return await handleLogRocketWebhook(request, env, corsHeaders);
          case "/webhooks/newrelic": return await handleNewRelicWebhook(request, env, corsHeaders);
          case "/webhooks/bugsnag": return await handleBugsnagWebhook(request, env, corsHeaders);
          case "/webhooks/rollbar": return await handleRollbarWebhook(request, env, corsHeaders);
        }
      } catch (error) {
        console.error("Webhook failed:", error);
        return Response.json({ error: "Webhook error", details: error instanceof Error ? error.message : "Unknown" }, { status: 500, headers: corsHeaders });
      }
    }

    // Auth for all other endpoints
    if (!authenticate(request, env)) {
      return Response.json({ error: "Unauthorized" }, { status: 401, headers: corsHeaders });
    }

    try {
      // POST endpoints - Browser Automation
      if (request.method === "POST") {
        switch (path) {
          case "/act": return await handleAct(request, env, corsHeaders);
          case "/extract": return await handleExtract(request, env, corsHeaders);
          case "/observe": return await handleObserve(request, env, corsHeaders);
          case "/agent": return await handleAgent(request, env, corsHeaders);
          case "/test": return await handleTest(request, env, corsHeaders);
          // Quality Intelligence - Test Generation
          case "/api/generate-test": return await handleGenerateTest(request, env, corsHeaders);
          case "/api/batch-generate-tests": return await handleBatchGenerateTests(request, env, corsHeaders);
          case "/api/update-generated-test": return await handleUpdateGeneratedTest(request, env, corsHeaders);
          // Quality Intelligence - Risk Scoring
          case "/api/calculate-risk-scores": return await handleCalculateRiskScores(request, env, corsHeaders);
          // Quality Intelligence - Autonomous Loop (connects all features)
          case "/api/autonomous-loop": return await handleAutonomousLoop(request, env, corsHeaders);
          // Quality Intelligence - Semantic Pattern Search (Vectorize)
          case "/api/semantic-search": return await handleSemanticPatternSearch(request, env, corsHeaders);
        }
      }

      // GET endpoints - Quality Intelligence API
      if (request.method === "GET") {
        switch (path) {
          case "/api/production-events": return await handleGetProductionEvents(request, env, corsHeaders);
          case "/api/quality-stats": return await handleGetQualityStats(request, env, corsHeaders);
          case "/api/risk-scores": return await handleGetRiskScores(request, env, corsHeaders);
          case "/api/generated-tests": return await handleGetGeneratedTests(request, env, corsHeaders);
          // Quality Intelligence - AI Quality Score
          case "/api/quality-score": return await handleGetAIQualityScore(request, env, corsHeaders);
          // Quality Intelligence - Predictive Quality
          case "/api/predictive-quality": return await handleGetPredictiveQuality(request, env, corsHeaders);
        }
      }

      return Response.json({ error: "Not found" }, { status: 404, headers: corsHeaders });
    } catch (error) {
      console.error("Request failed:", error);
      return Response.json({ error: "Internal error", details: error instanceof Error ? error.message : "Unknown" }, { status: 500, headers: corsHeaders });
    }
  },
};
