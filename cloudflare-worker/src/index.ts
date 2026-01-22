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
import { handleKeyVaultRequest, type KeyVaultEnv } from "./key-vault";

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

  // KV Namespace for Key Vault (BYOK encryption keys)
  KEY_VAULT?: KVNamespace;

  // Key Encryption Key for envelope encryption (BYOK)
  KEK_SECRET?: string;

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

  // Vultr Browser Pool (Legacy backend)
  BROWSER_POOL_URL?: string;
  BROWSER_POOL_JWT_SECRET?: string;

  // Selenium Grid 4 (Primary backend - W3C WebDriver protocol)
  SELENIUM_GRID_URL?: string;

  // Configuration
  DEFAULT_MODEL_PROVIDER: string;
  DEFAULT_BACKEND: string;
  ENABLE_CACHING: string;
  ENABLE_SELF_HEALING: string;

  // Environment - set to "production" to disable debug logging
  ENVIRONMENT?: string;

  // Media signing secret for authenticated screenshot/video URLs
  // Used for HMAC-SHA256 signature verification
  MEDIA_SIGNING_SECRET?: string;
}

// ============================================================================
// DEBUG LOGGING
// ============================================================================

/**
 * Conditional debug logging - only logs in non-production environments.
 * Prevents information disclosure of internal operations in production.
 */
function debugLog(env: Env, message: string, data?: unknown): void {
  if (env.ENVIRONMENT !== 'production') {
    if (data !== undefined) {
      console.log(message, JSON.stringify(data, null, 2));
    } else {
      console.log(message);
    }
  }
}

/**
 * Conditional debug warning - only logs in non-production environments.
 */
function debugWarn(env: Env, message: string, data?: unknown): void {
  if (env.ENVIRONMENT !== 'production') {
    if (data !== undefined) {
      console.warn(message, data);
    } else {
      console.warn(message);
    }
  }
}

type Backend = "cloudflare" | "testingbot" | "vultr" | "selenium" | "auto";
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

/**
 * AI-controlled session configuration.
 * The Argus brain determines optimal settings based on test complexity.
 */
interface SessionConfig {
  // AI-determined timeouts (in seconds)
  maxDuration: number;      // Max total test duration (default: 300)
  idleTimeout: number;      // Max time between commands (default: 60)

  // Resource hints for browser allocation
  memoryClass: 'low' | 'standard' | 'high';  // Maps to node selection
  priority: 'low' | 'normal' | 'urgent';      // Affects queue position

  // Optional viewport override
  viewport?: { width: number; height: number };
}

// Default session config when AI doesn't provide one
const DEFAULT_SESSION_CONFIG: SessionConfig = {
  maxDuration: 300,      // 5 minutes
  idleTimeout: 60,       // 1 minute
  memoryClass: 'standard',
  priority: 'normal',
};

// ============================================================================
// JWT TOKEN SIGNING FOR VULTR BROWSER POOL
// ============================================================================

interface PoolTokenPayload {
  iss: string;      // Issuer: 'argus-api'
  sub: string;      // User/session ID
  aud: string;      // Audience: 'browser-pool'
  exp: number;      // Expiration
  iat: number;      // Issued at
  jti: string;      // Unique token ID
  action?: string;  // Action being performed
}

function base64UrlEncode(data: string | ArrayBuffer): string {
  const bytes = typeof data === 'string'
    ? new TextEncoder().encode(data)
    : new Uint8Array(data);
  const base64 = btoa(String.fromCharCode(...bytes));
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

async function signPoolToken(
  payload: Omit<PoolTokenPayload, 'iat' | 'exp' | 'jti'>,
  secret: string,
  expiresInSeconds: number = 300
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);

  const fullPayload: PoolTokenPayload = {
    ...payload,
    iat: now,
    exp: now + expiresInSeconds,
    jti: crypto.randomUUID(),
  };

  const header = { alg: 'HS256', typ: 'JWT' };
  const headerB64 = base64UrlEncode(JSON.stringify(header));
  const payloadB64 = base64UrlEncode(JSON.stringify(fullPayload));

  // Sign using Web Crypto API (Cloudflare Workers compatible)
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );

  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    encoder.encode(`${headerB64}.${payloadB64}`)
  );

  const signatureB64 = base64UrlEncode(signature);
  return `${headerB64}.${payloadB64}.${signatureB64}`;
}

// ============================================================================
// MEDIA URL SIGNING (Screenshots, Videos)
// ============================================================================

/**
 * Generate HMAC-SHA256 signature for media URLs.
 * Format: HMAC(artifactId:expiration, secret)
 */
async function generateMediaSignature(
  artifactId: string,
  expiration: number,
  secret: string
): Promise<string> {
  const encoder = new TextEncoder();
  const message = `${artifactId}:${expiration}`;

  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );

  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    encoder.encode(message)
  );

  // Convert to hex string
  return Array.from(new Uint8Array(signature))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Verify HMAC-SHA256 signature for media URLs.
 * Uses constant-time comparison to prevent timing attacks.
 */
async function verifyMediaSignature(
  artifactId: string,
  signature: string,
  expiration: string,
  secret: string
): Promise<{ valid: boolean; reason?: string }> {
  // Check expiration first
  const exp = parseInt(expiration, 10);
  if (isNaN(exp)) {
    return { valid: false, reason: 'Invalid expiration format' };
  }

  const now = Math.floor(Date.now() / 1000);
  if (now > exp) {
    return { valid: false, reason: 'URL has expired' };
  }

  // Generate expected signature
  const expected = await generateMediaSignature(artifactId, exp, secret);

  // Constant-time comparison to prevent timing attacks
  if (signature.length !== expected.length) {
    return { valid: false, reason: 'Invalid signature' };
  }

  let result = 0;
  for (let i = 0; i < signature.length; i++) {
    result |= signature.charCodeAt(i) ^ expected.charCodeAt(i);
  }

  if (result !== 0) {
    return { valid: false, reason: 'Invalid signature' };
  }

  return { valid: true };
}

// ============================================================================
// VULTR BROWSER POOL PROXY
// ============================================================================

/**
 * Proxy a request to the Vultr browser pool.
 * Used when backend="vultr" to leverage the VKE browser infrastructure.
 */
async function callVultrPool<T>(
  endpoint: string,
  body: Record<string, unknown>,
  env: Env
): Promise<{ success: true; data: T } | { success: false; error: string }> {
  const poolUrl = env.BROWSER_POOL_URL;

  if (!poolUrl) {
    return { success: false, error: "BROWSER_POOL_URL not configured" };
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  // Sign JWT if secret is configured
  if (env.BROWSER_POOL_JWT_SECRET) {
    const token = await signPoolToken(
      {
        iss: "argus-api",
        sub: "worker",
        aud: "browser-pool",
        action: endpoint.replace("/", ""),
      },
      env.BROWSER_POOL_JWT_SECRET
    );
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const response = await fetch(`${poolUrl}${endpoint}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return { success: false, error: `Pool error (${response.status}): ${errorText}` };
    }

    const data = await response.json() as T;
    return { success: true, data };
  } catch (error) {
    return { success: false, error: `Pool connection failed: ${error}` };
  }
}

/**
 * Determine if we should use Vultr browser pool based on configuration.
 */
function shouldUseVultrPool(backend: Backend, env: Env): boolean {
  // Explicit vultr backend
  if (backend === "vultr") return true;

  // Auto mode with vultr as default
  if (backend === "auto" && env.DEFAULT_BACKEND === "vultr" && env.BROWSER_POOL_URL) {
    return true;
  }

  return false;
}

// ============================================================================
// SSRF PROTECTION
// ============================================================================

/**
 * Blocked hostnames and IP patterns for SSRF protection.
 * Prevents access to internal networks, cloud metadata endpoints, etc.
 */
const BLOCKED_HOST_PATTERNS = [
  // Localhost variants
  /^localhost$/i,
  /^127\.\d+\.\d+\.\d+$/,
  /^0\.0\.0\.0$/,
  /^\[::1\]$/,
  /^::1$/,
  
  // Private IPv4 ranges (RFC 1918)
  /^10\.\d+\.\d+\.\d+$/,
  /^172\.(1[6-9]|2\d|3[01])\.\d+\.\d+$/,
  /^192\.168\.\d+\.\d+$/,
  
  // Link-local
  /^169\.254\.\d+\.\d+$/,
  
  // IPv6 private/local
  /^fc[0-9a-f]{2}:/i,
  /^fd[0-9a-f]{2}:/i,
  /^fe80:/i,
  
  // Cloud metadata endpoints
  /^metadata\.google/i,
  /^metadata\.azure/i,
  /^169\.254\.169\.254$/,  // AWS/GCP/Azure metadata
  /^100\.100\.100\.200$/,  // Alibaba Cloud metadata
  
  // Internal domain patterns
  /\.internal$/i,
  /\.local$/i,
  /\.localhost$/i,
];

const BLOCKED_SCHEMES = ['file:', 'ftp:', 'gopher:', 'data:', 'javascript:'];

/**
 * Validate that a URL is safe for browser automation (not internal/private).
 * Returns true if URL is allowed, false if it should be blocked.
 */
function isAllowedUrl(urlStr: string): boolean {
  try {
    const url = new URL(urlStr);
    
    // Block non-HTTP(S) schemes
    if (!['http:', 'https:'].includes(url.protocol)) {
      return false;
    }
    
    // Block schemes that could be dangerous
    if (BLOCKED_SCHEMES.includes(url.protocol)) {
      return false;
    }
    
    const hostname = url.hostname.toLowerCase();
    
    // Check against blocked patterns
    for (const pattern of BLOCKED_HOST_PATTERNS) {
      if (pattern.test(hostname)) {
        return false;
      }
    }
    
    // Block URLs with username:password (potential for abuse)
    if (url.username || url.password) {
      return false;
    }
    
    return true;
  } catch {
    return false;
  }
}

/**
 * Zod refinement for safe external URLs.
 */
const SafeUrlSchema = z.string().url().refine(isAllowedUrl, {
  message: "URL must be a publicly accessible HTTP/HTTPS URL. Internal networks, localhost, and cloud metadata endpoints are blocked."
});

// ============================================================================
// REQUEST SCHEMAS
// ============================================================================

const BackendSchema = z.enum(["cloudflare", "testingbot", "vultr", "selenium", "auto"]).optional().default("auto");
const BrowserSchema = z.enum(["chrome", "firefox", "safari", "edge", "webkit"]).optional().default("chrome");
const DeviceSchema = z.enum([
  "desktop", "desktop-hd", "desktop-mac",
  "tablet", "tablet-landscape", "tablet-android",
  "mobile", "mobile-android", "mobile-landscape",
  "iphone-15", "iphone-14", "pixel-8", "pixel-7", "samsung-s24"
]).optional().default("desktop");

const ActRequestSchema = z.object({
  url: SafeUrlSchema,
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
  url: SafeUrlSchema,
  instruction: z.string().optional(),
  schema: z.record(z.any()).optional(),
  selector: z.string().optional(),
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  browser: BrowserSchema,
  device: DeviceSchema,
});

const ObserveRequestSchema = z.object({
  url: SafeUrlSchema,
  instruction: z.string().optional().default("Find all interactive elements"),
  selector: z.string().optional(),
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  browser: BrowserSchema,
  device: DeviceSchema,
  projectId: z.string().uuid().optional(),  // For activity logging
  activityType: z.enum(['discovery', 'visual_test', 'test_run', 'quality_audit', 'global_test']).optional().default('discovery'),
});

const AgentRequestSchema = z.object({
  url: SafeUrlSchema,
  instruction: z.string(),
  systemPrompt: z.string().optional(),
  maxSteps: z.number().optional().default(20),
  timeout: z.number().optional().default(60000),
  backend: BackendSchema,
  browser: BrowserSchema,
  device: DeviceSchema,
  captureScreenshots: z.boolean().optional().default(false),
});

// AI-controlled session configuration schema
const SessionConfigSchema = z.object({
  maxDuration: z.number().min(30).max(1800).optional().default(300),  // 30s to 30min
  idleTimeout: z.number().min(10).max(300).optional().default(60),    // 10s to 5min
  memoryClass: z.enum(['low', 'standard', 'high']).optional().default('standard'),
  priority: z.enum(['low', 'normal', 'urgent']).optional().default('normal'),
  viewport: z.object({
    width: z.number().min(320).max(3840),
    height: z.number().min(240).max(2160),
  }).optional(),
}).optional();

const TestRequestSchema = z.object({
  url: SafeUrlSchema,
  steps: z.array(z.string()),
  screenshot: z.boolean().optional().default(false),
  captureScreenshots: z.boolean().optional().default(false),
  recordVideo: z.boolean().optional().default(false),  // Enable video recording (requires backend support)
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  browsers: z.array(BrowserSchema).optional(),
  device: DeviceSchema,
  devices: z.array(DeviceSchema).optional(),
  projectId: z.string().uuid().optional(),  // For activity logging
  activityType: z.enum(['discovery', 'visual_test', 'test_run', 'quality_audit', 'global_test']).optional().default('test_run'),
  // AI-controlled session configuration
  sessionConfig: SessionConfigSchema,
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
// SELENIUM GRID WEBDRIVER CLIENT (W3C WebDriver Protocol)
// ============================================================================

class SeleniumGridSession {
  private sessionId: string;
  private baseUrl: string;

  constructor(sessionId: string, gridUrl: string) {
    this.sessionId = sessionId;
    // Remove trailing slash and ensure /wd/hub path
    this.baseUrl = gridUrl.replace(/\/$/, '');
    if (!this.baseUrl.includes('/wd/hub')) {
      this.baseUrl = `${this.baseUrl}/wd/hub`;
    }
  }

  private async request(method: string, path: string, body?: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}/session/${this.sessionId}${path}`, {
      method,
      headers: {
        "Content-Type": "application/json",
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
    const using = selector.startsWith("#") || selector.startsWith(".") || selector.includes("[")
      ? "css selector"
      : (selector.startsWith("//") ? "xpath" : "css selector");
    const result = await this.request("POST", "/element", { using, value: selector });
    return Object.values(result)[0] as string;
  }

  async findElements(selector: string): Promise<string[]> {
    const using = selector.startsWith("#") || selector.startsWith(".") || selector.includes("[")
      ? "css selector"
      : (selector.startsWith("//") ? "xpath" : "css selector");
    const result = await this.request("POST", "/elements", { using, value: selector });
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
    // Delete session from Selenium Grid - retry up to 3 times
    const deleteUrl = `${this.baseUrl}/session/${this.sessionId}`;

    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const response = await fetch(deleteUrl, {
          method: "DELETE",
          signal: AbortSignal.timeout(5000), // 5 second timeout for cleanup
        });
        if (response.ok || response.status === 404) {
          // 404 means session already closed, which is fine
          console.log(`Selenium session ${this.sessionId} closed successfully`);
          return;
        }
        console.warn(`Failed to close Selenium session (attempt ${attempt}): ${response.status}`);
      } catch (error) {
        console.warn(`Error closing Selenium session (attempt ${attempt}): ${error}`);
      }

      // Wait before retry (100ms, 200ms, 400ms)
      if (attempt < 3) {
        await new Promise(r => setTimeout(r, 100 * Math.pow(2, attempt - 1)));
      }
    }

    console.error(`Failed to close Selenium session ${this.sessionId} after 3 attempts`);
  }
}

/**
 * Create a Selenium Grid session
 * @param env Environment with SELENIUM_GRID_URL
 * @param browserType Browser to use (chrome, firefox, etc.)
 * @param deviceConfig Device viewport configuration
 * @param sessionConfig AI-determined session configuration (timeouts, memory class)
 */
async function createSeleniumGridSession(
  env: Env,
  browserType: BrowserType,
  deviceConfig: any,
  sessionConfig?: SessionConfig
): Promise<SeleniumGridSession> {
  const gridUrl = env.SELENIUM_GRID_URL!;
  const baseUrl = gridUrl.replace(/\/$/, '');
  const hubUrl = baseUrl.includes('/wd/hub') ? baseUrl : `${baseUrl}/wd/hub`;

  // Merge with defaults
  const config = { ...DEFAULT_SESSION_CONFIG, ...sessionConfig };

  // Use viewport from sessionConfig if provided, otherwise use deviceConfig
  const viewport = config.viewport || deviceConfig.viewport || { width: 1920, height: 1080 };

  // Build capabilities for Selenium Grid (W3C WebDriver format)
  // Map browser types to Selenium Grid expected names
  const browserNameMap: Record<string, string> = {
    chrome: "chrome",
    firefox: "firefox",
    edge: "MicrosoftEdge",  // Selenium Grid uses MicrosoftEdge
    safari: "chrome",       // Safari not available, fallback to Chrome
    webkit: "chrome",       // WebKit not available, fallback to Chrome
  };

  const capabilities: Record<string, any> = {
    browserName: browserNameMap[browserType] || "chrome",

    // AI-controlled timeouts (Selenium Grid 4 capabilities)
    // se:sessionTimeout - Max total session duration in seconds
    // se:idleTimeout - Max time between commands in seconds
    "se:sessionTimeout": config.maxDuration,
    "se:idleTimeout": config.idleTimeout,

    // Resource hints for Grid routing (custom capabilities)
    "argus:memoryClass": config.memoryClass,
    "argus:priority": config.priority,

    "goog:chromeOptions": browserType === "chrome" ? {
      args: [
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        `--window-size=${viewport.width},${viewport.height}`,
      ],
    } : undefined,
    "moz:firefoxOptions": browserType === "firefox" ? {
      args: ["-headless"],
      prefs: {
        "layout.css.devPixelsPerPx": "1.0",
      },
    } : undefined,
    "ms:edgeOptions": browserType === "edge" ? {
      args: [
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        `--window-size=${viewport.width},${viewport.height}`,
      ],
    } : undefined,
  };

  // Log AI-determined config for debugging (non-production only)
  debugLog(env, `Creating Selenium session with AI config`, {
    maxDuration: config.maxDuration,
    idleTimeout: config.idleTimeout,
    memoryClass: config.memoryClass,
    priority: config.priority,
  });

  // Remove undefined options
  Object.keys(capabilities).forEach(key => {
    if (capabilities[key] === undefined) delete capabilities[key];
  });

  const response = await fetch(`${hubUrl}/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      capabilities: {
        alwaysMatch: capabilities,
        firstMatch: [{}],
      }
    }),
  });

  const data = await response.json() as any;
  if (!data.value?.sessionId) {
    throw new Error(`Failed to create Selenium Grid session: ${JSON.stringify(data)}`);
  }

  return new SeleniumGridSession(data.value.sessionId, gridUrl);
}

/**
 * Determine if we should use Selenium Grid based on configuration.
 */
function shouldUseSeleniumGrid(backend: Backend, env: Env): boolean {
  // Explicit selenium backend
  if (backend === "selenium") return true;

  // Auto mode with selenium as default (if configured)
  if (backend === "auto" && env.DEFAULT_BACKEND === "selenium" && env.SELENIUM_GRID_URL) {
    return true;
  }

  return false;
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

// Selenium Grid WebDriver wrapper
class SeleniumGridBrowserSession implements BrowserSession {
  private session: SeleniumGridSession;
  backendUsed: string;

  constructor(session: SeleniumGridSession, browserType: BrowserType) {
    this.session = session;
    this.backendUsed = `selenium:${browserType}`;
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
// ERROR CATEGORIZATION
// ============================================================================

type ErrorCategory = 'network' | 'element' | 'assertion' | 'browser' | 'timeout' | 'unknown';

interface CategorizedError {
  category: ErrorCategory;
  message: string;
  originalError: string;
  isRetryable: boolean;
  suggestedAction?: string;
}

function categorizeError(error: Error | string): CategorizedError {
  const message = typeof error === 'string' ? error : error.message;
  const lowerMessage = message.toLowerCase();

  // Network errors - usually retryable
  if (
    lowerMessage.includes('fetch failed') ||
    lowerMessage.includes('network') ||
    lowerMessage.includes('dns') ||
    lowerMessage.includes('econnrefused') ||
    lowerMessage.includes('enotfound') ||
    lowerMessage.includes('socket') ||
    lowerMessage.includes('connection reset') ||
    lowerMessage.includes('unable to connect')
  ) {
    return {
      category: 'network',
      message: 'Network connection failed. Please check your internet connection and try again.',
      originalError: message,
      isRetryable: true,
      suggestedAction: 'Check network connectivity and retry',
    };
  }

  // Browser/session errors - usually retryable
  if (
    lowerMessage.includes('browser') ||
    lowerMessage.includes('session') ||
    lowerMessage.includes('cdp') ||
    lowerMessage.includes('target closed') ||
    lowerMessage.includes('context') ||
    lowerMessage.includes('crashed') ||
    lowerMessage.includes('disconnected')
  ) {
    return {
      category: 'browser',
      message: 'Browser session error. Attempting to reconnect...',
      originalError: message,
      isRetryable: true,
      suggestedAction: 'Browser will automatically retry',
    };
  }

  // Rate limiting - retryable with backoff
  if (
    lowerMessage.includes('429') ||
    lowerMessage.includes('rate limit') ||
    lowerMessage.includes('too many requests') ||
    lowerMessage.includes('throttle')
  ) {
    return {
      category: 'network',
      message: 'Rate limited by browser service. Retrying with fallback...',
      originalError: message,
      isRetryable: true,
      suggestedAction: 'Wait and retry, or use alternate backend',
    };
  }

  // Timeout errors - sometimes retryable
  if (
    lowerMessage.includes('timeout') ||
    lowerMessage.includes('timed out') ||
    lowerMessage.includes('exceeded')
  ) {
    return {
      category: 'timeout',
      message: 'Operation timed out. The page may be slow to load.',
      originalError: message,
      isRetryable: true,
      suggestedAction: 'Increase timeout or retry',
    };
  }

  // Element errors - not retryable without fixing selector
  if (
    lowerMessage.includes('selector') ||
    lowerMessage.includes('element') ||
    lowerMessage.includes('not found') ||
    lowerMessage.includes('no node found') ||
    lowerMessage.includes('waiting for') ||
    lowerMessage.includes('locator') ||
    lowerMessage.includes('not visible') ||
    lowerMessage.includes('not interactable')
  ) {
    return {
      category: 'element',
      message: 'Element not found on page. The page structure may have changed.',
      originalError: message,
      isRetryable: false,
      suggestedAction: 'Check selector or enable self-healing',
    };
  }

  // Assertion errors - not retryable
  if (
    lowerMessage.includes('verification failed') ||
    lowerMessage.includes('assertion') ||
    lowerMessage.includes('expected') ||
    lowerMessage.includes('to contain') ||
    lowerMessage.includes('to match')
  ) {
    return {
      category: 'assertion',
      message: 'Verification failed. The page content did not match expectations.',
      originalError: message,
      isRetryable: false,
      suggestedAction: 'Review test expectations',
    };
  }

  // Unknown errors
  return {
    category: 'unknown',
    message: 'An unexpected error occurred.',
    originalError: message,
    isRetryable: false,
    suggestedAction: 'Check logs for details',
  };
}

// ============================================================================
// RETRY CONFIGURATION
// ============================================================================

const RETRY_CONFIG = {
  maxAttempts: 3,
  baseDelayMs: 1000,
  maxDelayMs: 10000,
  backoffMultiplier: 2,
};

function calculateBackoff(attempt: number): number {
  const delay = Math.min(
    RETRY_CONFIG.baseDelayMs * Math.pow(RETRY_CONFIG.backoffMultiplier, attempt),
    RETRY_CONFIG.maxDelayMs
  );
  // Add jitter (±20%)
  const jitter = delay * 0.2 * (Math.random() * 2 - 1);
  return Math.floor(delay + jitter);
}

async function withRetry<T>(
  operation: () => Promise<T>,
  operationName: string,
  shouldRetry: (error: CategorizedError) => boolean = (e) => e.isRetryable
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < RETRY_CONFIG.maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      const categorized = categorizeError(lastError);

      console.log(`[Retry] ${operationName} failed (attempt ${attempt + 1}/${RETRY_CONFIG.maxAttempts}): ${categorized.category} - ${categorized.originalError}`);

      if (!shouldRetry(categorized) || attempt === RETRY_CONFIG.maxAttempts - 1) {
        throw lastError;
      }

      const delay = calculateBackoff(attempt);
      console.log(`[Retry] Waiting ${delay}ms before retry...`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError || new Error(`${operationName} failed after ${RETRY_CONFIG.maxAttempts} attempts`);
}

// ============================================================================
// BROWSER CONNECTION
// ============================================================================

async function connectToCloudflare(env: Env, deviceConfig: any, timeout: number): Promise<CloudflareBrowserSession> {
  return await withRetry(
    async () => {
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
    },
    'Cloudflare browser connection'
  );
}

async function connectToBrowser(
  env: Env,
  options: {
    backend: Backend;
    browser: BrowserType;
    device: DeviceType;
    platform: Platform;
    timeout: number;
    sessionConfig?: SessionConfig;  // AI-controlled session configuration
  }
): Promise<BrowserSession> {
  const { backend, browser: browserType, device, timeout, sessionConfig } = options;
  let { platform } = options;
  const deviceConfig = DEVICE_PRESETS[device] || DEVICE_PRESETS.desktop;

  // Auto-select platform based on browser (Safari requires macOS)
  if (browserType === "safari" || browserType === "webkit") {
    platform = "macos";
  }

  // Priority 1: Selenium Grid (if explicitly requested or configured as default)
  if (shouldUseSeleniumGrid(backend, env)) {
    try {
      // Selenium Grid supports Chrome and Firefox (not Safari/WebKit)
      const gridBrowser = browserType === "safari" || browserType === "webkit" ? "chrome" : browserType;
      // Pass AI-determined sessionConfig to Selenium Grid
      const session = await createSeleniumGridSession(env, gridBrowser, deviceConfig, sessionConfig);
      return new SeleniumGridBrowserSession(session, gridBrowser);
    } catch (error) {
      console.warn(`Selenium Grid connection failed: ${error}, falling back...`);
      // Fall through to other backends
    }
  }

  // Priority 2: TestingBot (for real devices or Safari/Firefox)
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
  }

  // Priority 3: Cloudflare Browser Rendering (free, Chrome only)
  try {
    return await connectToCloudflare(env, deviceConfig, timeout);
  } catch (error) {
    const errorMessage = String(error);
    const isRateLimited = errorMessage.includes("429") || errorMessage.includes("Rate limit") || errorMessage.includes("rate limit");

    // Failover chain: Cloudflare -> Selenium Grid -> TestingBot
    if (isRateLimited) {
      // Try Selenium Grid as first failover
      if (env.SELENIUM_GRID_URL) {
        try {
          console.log("Cloudflare rate limited, failing over to Selenium Grid...");
          const gridBrowser = browserType === "safari" || browserType === "webkit" ? "chrome" : browserType;
          // Pass AI-determined sessionConfig to Selenium Grid failover
          const session = await createSeleniumGridSession(env, gridBrowser, deviceConfig, sessionConfig);
          return new SeleniumGridBrowserSession(session, gridBrowser);
        } catch (gridError) {
          console.warn(`Selenium Grid failover failed: ${gridError}`);
        }
      }

      // Try TestingBot as final failover
      if (env.TESTINGBOT_KEY && env.TESTINGBOT_SECRET) {
        console.log("Failing over to TestingBot...");
        const session = await createTestingBotSession(env, browserType, platform, deviceConfig, device);
        return new TestingBotBrowserSession(session, browserType, platform);
      }
    }

    throw error;
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

ACTION TYPES:
- click: Click an element
- fill: Fill a text input
- type: Type text character by character
- hover: Hover over an element
- scroll: Scroll the page
- verify: Verify something on the page (for assertions like "Verify the title", "Check if button exists")
  - For verify: selector is optional (use "page" if checking page-level content)
  - value should contain what to verify (e.g., "title contains Example", "button is visible")

RESPOND WITH ONLY VALID JSON (no explanation):
{"action": "click|fill|type|hover|scroll|verify", "selector": "valid CSS selector or 'page'", "value": "text to type OR verification condition", "description": "brief action description"}`;

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
// SELF-HEALING (AI-Powered + Static Fallbacks)
// ============================================================================

interface HealingSuggestion {
  selector: string;
  confidence: number;
  reason: string;
}

interface HealingContext {
  originalSelector: string;
  description: string;
  errorMessage: string;
  pageElements: Array<{ tag: string; text: string; selector: string; attributes: Record<string, string> }>;
  pageUrl: string;
}

// AI-powered selector healing - analyzes current page to find the right element
async function healSelectorWithAI(
  env: Env,
  context: HealingContext
): Promise<HealingSuggestion[]> {
  try {
    const elementsStr = context.pageElements
      .slice(0, 50)
      .map((e, i) => `${i + 1}. ${e.tag}: "${e.text}" [${e.selector}] ${JSON.stringify(e.attributes)}`)
      .join('\n');

    const prompt = `You are a browser automation expert. The original selector failed. Find the correct element.

FAILED SELECTOR: "${context.originalSelector}"
INTENDED ACTION: "${context.description}"
ERROR: "${context.errorMessage}"
PAGE URL: ${context.pageUrl}

CURRENT PAGE ELEMENTS:
${elementsStr}

TASK: Find the element that matches the intended action. Consider:
1. The element text/label might have changed
2. The class/id might have been renamed
3. The element might have moved to a different position
4. Look for elements with similar text, aria-labels, or roles

RESPOND WITH JSON ARRAY (up to 3 suggestions, highest confidence first):
[{"selector": "exact CSS selector from the list above", "confidence": 0.0-1.0, "reason": "why this matches"}]

IMPORTANT: Only use selectors that appear in CURRENT PAGE ELEMENTS above. Do not invent selectors.`;

    const text = await callAI(env, {
      messages: [{ role: 'user', content: prompt }],
      maxTokens: 500,
      timeout: 15000,
    });

    const match = text.match(/\[[\s\S]*\]/);
    if (match) {
      const suggestions = JSON.parse(match[0]) as HealingSuggestion[];
      return suggestions.filter(s => s.selector && s.confidence > 0.3);
    }
  } catch (e) {
    console.error('AI healing failed:', e);
  }
  return [];
}

// Generate healing suggestions for failed actions (to return to frontend)
async function generateHealingSuggestions(
  env: Env,
  session: BrowserSession,
  failedAction: { selector: string; description: string },
  errorMessage: string
): Promise<{ suggestions: HealingSuggestion[]; pageContext: string }> {
  try {
    const elements = await session.getInteractiveElements();
    const pageUrl = await session.getUrl();

    const context: HealingContext = {
      originalSelector: failedAction.selector,
      description: failedAction.description,
      errorMessage,
      pageElements: elements,
      pageUrl,
    };

    const suggestions = await healSelectorWithAI(env, context);

    // Also add static fallbacks as lower-confidence suggestions
    const staticFallbacks = generateSelectorFallbacks(failedAction.selector, failedAction.description);
    for (const selector of staticFallbacks.slice(1, 4)) {
      if (!suggestions.find(s => s.selector === selector)) {
        suggestions.push({
          selector,
          confidence: 0.4,
          reason: 'Static fallback based on selector pattern',
        });
      }
    }

    return {
      suggestions: suggestions.slice(0, 5),
      pageContext: `URL: ${pageUrl}, Elements: ${elements.length}`,
    };
  } catch (e) {
    console.error('Failed to generate healing suggestions:', e);
    return { suggestions: [], pageContext: '' };
  }
}

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
    case "verify":
      await executeVerification(session, selector, value || "");
      break;
    default:
      await session.click(selector);
  }
}

async function executeVerification(
  session: BrowserSession,
  selector: string,
  condition: string
): Promise<void> {
  const conditionLower = condition.toLowerCase();

  // Handle page-level verifications
  if (selector === "page" || selector === "") {
    const title = await session.page.title();
    const content = await session.getContent();

    // Check title conditions
    if (conditionLower.includes("title")) {
      const titleMatch = condition.match(/contains?\s+['""]?([^'""]+)['""]?/i) ||
                         condition.match(/title\s+(?:is\s+)?['""]?([^'""]+)['""]?/i);
      if (titleMatch) {
        const expected = titleMatch[1].trim();
        if (!title.toLowerCase().includes(expected.toLowerCase())) {
          throw new Error(`Title verification failed: expected to contain "${expected}", got "${title}"`);
        }
        return;
      }
      // Just check title exists
      if (title.length === 0) {
        throw new Error("Title verification failed: page has no title");
      }
      return;
    }

    // Check text content
    if (conditionLower.includes("contains") || conditionLower.includes("has text")) {
      const textMatch = condition.match(/contains?\s+['""]?([^'""]+)['""]?/i);
      if (textMatch) {
        const expected = textMatch[1].trim();
        if (!content.toLowerCase().includes(expected.toLowerCase())) {
          throw new Error(`Page content verification failed: expected to contain "${expected}"`);
        }
        return;
      }
    }

    // Page exists check
    if (content.length > 0) {
      return; // Page has content, verification passes
    }
    throw new Error("Page verification failed: page has no content");
  }

  // Element-level verifications
  const locator = session.page.locator(selector);

  if (conditionLower.includes("visible") || conditionLower.includes("exists") || conditionLower.includes("present")) {
    const isVisible = await locator.isVisible().catch(() => false);
    if (!isVisible) {
      throw new Error(`Element "${selector}" is not visible`);
    }
    return;
  }

  if (conditionLower.includes("hidden") || conditionLower.includes("not visible")) {
    const isVisible = await locator.isVisible().catch(() => false);
    if (isVisible) {
      throw new Error(`Element "${selector}" is visible but expected to be hidden`);
    }
    return;
  }

  if (conditionLower.includes("contains") || conditionLower.includes("has text")) {
    const textMatch = condition.match(/contains?\s+['""]?([^'""]+)['""]?/i);
    if (textMatch) {
      const expected = textMatch[1].trim();
      const actualText = await locator.textContent().catch(() => "");
      if (!actualText?.toLowerCase().includes(expected.toLowerCase())) {
        throw new Error(`Element text verification failed: expected to contain "${expected}", got "${actualText}"`);
      }
      return;
    }
  }

  // Default: just check element exists
  const count = await locator.count();
  if (count === 0) {
    throw new Error(`Element "${selector}" not found on page`);
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

  // Use Vultr browser pool if configured
  if (shouldUseVultrPool(backend as Backend, env)) {
    const poolResult = await callVultrPool<ActResult>("/act", {
      url, instruction, action, variables, timeout, backend: "vultr", browser: browserType, device, platform, selfHeal, screenshot
    }, env);

    if (poolResult.success) {
      return Response.json({ ...poolResult.data, backend: "vultr" }, { headers: corsHeaders });
    } else {
      // Fall through to local execution if pool fails
      console.warn(`Vultr pool failed: ${poolResult.error}, falling back to local execution`);
    }
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
    const categorized = categorizeError(message);
    return Response.json(
      {
        success: false,
        message: categorized.message,
        actions: [],
        errorDetails: {
          category: categorized.category,
          originalError: categorized.originalError,
          isRetryable: categorized.isRetryable,
          suggestedAction: categorized.suggestedAction,
        },
        timedOut: categorized.category === 'timeout',
      },
      { status: categorized.category === 'timeout' ? 408 : 500, headers: corsHeaders }
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

  // Use Vultr browser pool if configured
  if (shouldUseVultrPool(backend as Backend, env)) {
    const poolResult = await callVultrPool<Record<string, unknown>>("/extract", {
      url, instruction, schema, selector, timeout, backend: "vultr", browser: browserType, device
    }, env);

    if (poolResult.success) {
      // Ensure response matches MCP expected format: { success: true, data: {...} }
      const responseData = poolResult.data;
      if (responseData && 'success' in responseData) {
        // Pool already returned proper format
        return Response.json({ ...responseData, _backend: "vultr" }, { headers: corsHeaders });
      } else {
        // Pool returned raw data, wrap it
        return Response.json({ success: true, data: responseData, _backend: "vultr" }, { headers: corsHeaders });
      }
    } else {
      console.warn(`Vultr pool failed: ${poolResult.error}, falling back to local execution`);
    }
  }

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

    // Wrap response to match MCP expected format: { success: true, data: {...} }
    return Response.json({ success: true, data, _backend: session.backendUsed }, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Extract operation");
  } catch (error) {
    if (session) await session.close().catch(() => {});
    const message = error instanceof Error ? error.message : "Unknown";
    const categorized = categorizeError(message);
    return Response.json(
      {
        error: categorized.message,
        errorDetails: {
          category: categorized.category,
          originalError: categorized.originalError,
          isRetryable: categorized.isRetryable,
          suggestedAction: categorized.suggestedAction,
        },
        timedOut: categorized.category === 'timeout',
      },
      { status: categorized.category === 'timeout' ? 408 : 500, headers: corsHeaders }
    );
  }
}

async function handleObserve(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = ObserveRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const { url, instruction, timeout, backend, browser: browserType, device, projectId, activityType } = parsed.data;

  // Use Vultr browser pool if configured
  if (shouldUseVultrPool(backend as Backend, env)) {
    const poolResult = await callVultrPool<{ actions: Action[]; elements: unknown[] }>("/observe", {
      url, instruction, timeout, backend: "vultr", browser: browserType, device, projectId, activityType
    }, env);

    if (poolResult.success) {
      return Response.json({ ...poolResult.data, _backend: "vultr" }, { headers: corsHeaders });
    } else {
      console.warn(`Vultr pool failed: ${poolResult.error}, falling back to local execution`);
    }
  }

  let session: BrowserSession | null = null;

  // Create activity logger if projectId is provided
  const activityLogger = await createActivityLogger(env, projectId, activityType);

  const executeWithTimeout = async (): Promise<Response> => {
    await activityLogger?.logActivity('started', 'Starting observation', `Analyzing ${url}`);

    session = await connectToBrowser(env, { backend: backend as Backend, browser: browserType as BrowserType, device: device as DeviceType, platform: "windows", timeout });
    await activityLogger?.logActivity('step', 'Browser connected', `Connected via ${session.backendUsed}`);

    await session.navigate(url);
    await activityLogger?.logActivity('step', 'Page loaded', `Navigated to ${url}`);

    const elements = await session.getInteractiveElements();
    await activityLogger?.logActivity('step', 'Elements found', `Found ${elements.length} interactive elements`);

    await activityLogger?.logActivity('thinking', 'AI analyzing elements', instruction);
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

    await activityLogger?.logActivity('step', 'Analysis complete', `Identified ${actions.length} actions`);
    await session.close();
    await activityLogger?.complete(true);

    return Response.json({ actions, elements, _backend: session.backendUsed }, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Observe operation");
  } catch (error) {
    if (session) await session.close().catch(() => {});
    const message = error instanceof Error ? error.message : "Unknown";
    const categorized = categorizeError(message);
    await activityLogger?.logActivity('error', categorized.message, categorized.originalError);
    await activityLogger?.complete(false);
    return Response.json(
      {
        error: categorized.message,
        errorDetails: {
          category: categorized.category,
          originalError: categorized.originalError,
          isRetryable: categorized.isRetryable,
          suggestedAction: categorized.suggestedAction,
        },
        timedOut: categorized.category === 'timeout',
      },
      { status: categorized.category === 'timeout' ? 408 : 500, headers: corsHeaders }
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

  // Use Vultr browser pool if configured
  if (shouldUseVultrPool(backend as Backend, env)) {
    const poolResult = await callVultrPool<AgentResult>("/agent", {
      url, instruction, systemPrompt, maxSteps, timeout, backend: "vultr", browser: browserType, device, captureScreenshots
    }, env);

    if (poolResult.success) {
      return Response.json({ ...poolResult.data, backend: "vultr" }, { headers: corsHeaders });
    } else {
      console.warn(`Vultr pool failed: ${poolResult.error}, falling back to local execution`);
    }
  }

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
    const categorized = categorizeError(message);
    return Response.json(
      {
        success: false,
        message: categorized.message,
        actions: [],
        completed: false,
        usage: { inputTokens: 0, outputTokens: 0, totalSteps: 0 },
        errorDetails: {
          category: categorized.category,
          originalError: categorized.originalError,
          isRetryable: categorized.isRetryable,
          suggestedAction: categorized.suggestedAction,
        },
        timedOut: categorized.category === 'timeout',
      },
      { status: categorized.category === 'timeout' ? 408 : 500, headers: corsHeaders }
    );
  }
}

async function handleTest(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = TestRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const { url, steps, screenshot, captureScreenshots, recordVideo, timeout, backend, browsers, device, devices, projectId, activityType, sessionConfig } = parsed.data;

  // Convert Zod-parsed sessionConfig to our SessionConfig type
  const aiSessionConfig: SessionConfig | undefined = sessionConfig ? {
    maxDuration: sessionConfig.maxDuration ?? DEFAULT_SESSION_CONFIG.maxDuration,
    idleTimeout: sessionConfig.idleTimeout ?? DEFAULT_SESSION_CONFIG.idleTimeout,
    memoryClass: sessionConfig.memoryClass ?? DEFAULT_SESSION_CONFIG.memoryClass,
    priority: sessionConfig.priority ?? DEFAULT_SESSION_CONFIG.priority,
    viewport: sessionConfig.viewport,
  } : undefined;

  // Log AI-determined session config if provided (non-production only)
  if (aiSessionConfig) {
    debugLog(env, `AI-determined session config`, {
      maxDuration: aiSessionConfig.maxDuration,
      idleTimeout: aiSessionConfig.idleTimeout,
      memoryClass: aiSessionConfig.memoryClass,
      priority: aiSessionConfig.priority,
    });
  }

  // Use Vultr browser pool if configured
  if (shouldUseVultrPool(backend as Backend, env)) {
    const poolResult = await callVultrPool<TestResult>("/test", {
      url, steps, screenshot, captureScreenshots, recordVideo, timeout, backend: "vultr", browsers, device, devices, projectId, activityType, sessionConfig
    }, env);

    if (poolResult.success) {
      return Response.json({ ...poolResult.data, backend: "vultr" }, { headers: corsHeaders });
    } else {
      console.warn(`Vultr pool failed: ${poolResult.error}, falling back to local execution`);
    }
  }

  const browserList = browsers || ["chrome"];
  const deviceList = devices || [device];

  // Create activity logger for real-time progress
  const logger = await createActivityLogger(env, projectId, activityType as ActivityType);

  const executeWithTimeout = async (): Promise<Response> => {
    const results: BrowserResult[] = [];
    const allScreenshots: string[] = [];
    const stepResults: Array<{ instruction: string; success: boolean; error?: string; screenshot?: string }> = [];

    // Log start
    if (logger) {
      await logger.logActivity('started', `Starting ${activityType}`, `URL: ${url}, Steps: ${steps.length}`);
      await logger.updateSession({ currentStep: 'Initializing browser', completedSteps: 0 });
    }

    for (const browserType of browserList) {
      for (const deviceType of deviceList) {
        let session: BrowserSession | null = null;
        try {
          if (logger) {
            await logger.logActivity('step', 'Connecting to browser', `Browser: ${browserType}, Device: ${deviceType}`);
          }

          session = await connectToBrowser(env, {
            backend: backend as Backend,
            browser: browserType as BrowserType,
            device: deviceType as DeviceType,
            platform: "windows",
            timeout,
            sessionConfig: aiSessionConfig,  // Pass AI-controlled session config
          });

          if (logger) {
            await logger.logActivity('step', 'Navigating to URL', url);
            await logger.updateSession({ currentStep: 'Loading page' });
          }

          await session.navigate(url);

          if (captureScreenshots) {
            try {
              const initialScreenshot = await session.screenshot();
              allScreenshots.push(initialScreenshot);
              if (logger) {
                await logger.logActivity('screenshot', 'Initial screenshot captured', undefined, `data:image/png;base64,${initialScreenshot}`);
                await logger.updateSession({ lastScreenshotUrl: `data:image/png;base64,${initialScreenshot}` });
              }
            } catch {}
          }

          let allSuccess = true;
          for (let i = 0; i < steps.length; i++) {
            const step = steps[i];

            if (logger) {
              await logger.logActivity('thinking', 'Analyzing step', `Step ${i + 1}/${steps.length}: ${step}`);
              await logger.updateSession({ currentStep: step, completedSteps: i });
            }

            const pageHtml = await session.getContent();
            const elements = await session.getInteractiveElements();
            const elementsStr = elements.map(e => `${e.tag}: "${e.text}" [${e.selector}]`).join("\n");

            const interpreted = await interpretAction(env, step, pageHtml, elementsStr);
            if (!interpreted) {
              if (logger) {
                await logger.logActivity('error', 'Failed to interpret step', step);
              }
              stepResults.push({ instruction: step, success: false, error: "Could not interpret instruction" });
              allSuccess = false;
              break;
            }

            if (logger) {
              await logger.logActivity('action', `Executing: ${interpreted.description}`, `Selector: ${interpreted.selector}`);
            }

            const result = await executeAction(session, interpreted, true, env);

            let stepScreenshot: string | undefined;
            if (captureScreenshots) {
              try {
                stepScreenshot = await session.screenshot();
                allScreenshots.push(stepScreenshot);
                if (logger) {
                  await logger.logActivity('screenshot', `Screenshot after step ${i + 1}`, undefined, `data:image/png;base64,${stepScreenshot}`);
                  await logger.updateSession({ lastScreenshotUrl: `data:image/png;base64,${stepScreenshot}` });
                }
              } catch {}
            }

            // If action failed, generate AI healing suggestions
            let healingSuggestions: HealingSuggestion[] | undefined;
            if (!result.success && interpreted) {
              try {
                const healing = await generateHealingSuggestions(
                  env,
                  session,
                  { selector: interpreted.selector, description: interpreted.description },
                  result.error || 'Unknown error'
                );
                healingSuggestions = healing.suggestions;

                if (logger && healingSuggestions.length > 0) {
                  await logger.logActivity(
                    'thinking',
                    'AI analyzing failure',
                    `Found ${healingSuggestions.length} potential fixes: ${healingSuggestions.map(s => s.selector).join(', ')}`
                  );
                }
              } catch (e) {
                console.error('Failed to generate healing suggestions:', e);
              }
            }

            stepResults.push({
              instruction: step,
              success: result.success,
              error: result.success ? undefined : result.error,
              screenshot: stepScreenshot,
              // Include retry context for failed steps
              failedAction: !result.success && interpreted ? {
                selector: interpreted.selector,
                description: interpreted.description,
                action: interpreted.action,
                value: interpreted.value,
              } : undefined,
              healingSuggestions: !result.success ? healingSuggestions : undefined,
            });

            if (!result.success) {
              if (logger) {
                await logger.logActivity('error', `Step failed: ${step}`, result.error);
              }
              allSuccess = false;
              break;
            }

            if (logger) {
              await logger.updateSession({ completedSteps: i + 1 });
            }
          }

          let screenshotBase64: string | undefined;
          if (screenshot && !captureScreenshots) {
            try {
              screenshotBase64 = await session.screenshot();
              if (logger) {
                await logger.updateSession({ lastScreenshotUrl: `data:image/png;base64,${screenshotBase64}` });
              }
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
          const errorMessage = error instanceof Error ? error.message : "Unknown error";
          if (logger) {
            await logger.logActivity('error', 'Browser session failed', errorMessage);
          }
          results.push({
            browser: browserType,
            platform: "error",
            device: deviceType,
            success: false,
            error: errorMessage,
          });
        }
      }
    }

    const overallSuccess = results.every(r => r.success);

    // Log completion
    if (logger) {
      await logger.complete(overallSuccess);
    }

    return Response.json({
      success: overallSuccess,
      steps: stepResults.length > 0 ? stepResults : steps.map(s => ({ instruction: s, success: results.some(r => r.success) })),
      browsers: results,
      backend: backend,
      screenshots: allScreenshots.length > 0 ? allScreenshots : undefined,
      finalScreenshot: results[0]?.screenshot,
      sessionId: logger?.sessionId,
    } as TestResult & { sessionId?: string }, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, "Test operation");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown";
    const categorized = categorizeError(message);

    // Log timeout/error with categorization
    if (logger) {
      await logger.logActivity('error', categorized.message, categorized.originalError);
      await logger.complete(false);
    }

    return Response.json(
      {
        success: false,
        steps: [],
        browsers: [],
        error: categorized.message,
        errorDetails: {
          category: categorized.category,
          originalError: categorized.originalError,
          isRetryable: categorized.isRetryable,
          suggestedAction: categorized.suggestedAction,
        },
        timedOut: categorized.category === 'timeout',
        sessionId: logger?.sessionId,
      },
      { status: categorized.category === 'timeout' ? 408 : 500, headers: corsHeaders }
    );
  }
}

// ============================================================================
// ACTIVITY LOGGING (Real-time progress to Supabase)
// ============================================================================

type ActivityType = 'discovery' | 'visual_test' | 'test_run' | 'quality_audit' | 'global_test';
type EventType = 'started' | 'step' | 'screenshot' | 'thinking' | 'action' | 'error' | 'completed' | 'cancelled';

interface ActivityLogger {
  sessionId: string;
  logActivity(eventType: EventType, title: string, description?: string, screenshotUrl?: string, metadata?: Record<string, unknown>): Promise<void>;
  updateSession(updates: { currentStep?: string; completedSteps?: number; lastScreenshotUrl?: string; status?: string }): Promise<void>;
  complete(success: boolean): Promise<void>;
}

async function createActivityLogger(
  env: Env,
  projectId: string | undefined,
  activityType: ActivityType
): Promise<ActivityLogger | null> {
  if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_KEY || !projectId) {
    return null;
  }

  const sessionId = crypto.randomUUID();
  const supabaseUrl = env.SUPABASE_URL;
  const supabaseKey = env.SUPABASE_SERVICE_KEY;

  const supabaseFetch = async (endpoint: string, body: unknown) => {
    try {
      await fetch(`${supabaseUrl}/rest/v1/${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'apikey': supabaseKey,
          'Authorization': `Bearer ${supabaseKey}`,
          'Prefer': 'return=minimal',
        },
        body: JSON.stringify(body),
      });
    } catch (e) {
      console.error('Supabase activity log failed:', e);
    }
  };

  const supabasePatch = async (table: string, id: string, body: unknown) => {
    try {
      await fetch(`${supabaseUrl}/rest/v1/${table}?id=eq.${id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'apikey': supabaseKey,
          'Authorization': `Bearer ${supabaseKey}`,
          'Prefer': 'return=minimal',
        },
        body: JSON.stringify(body),
      });
    } catch (e) {
      console.error('Supabase session update failed:', e);
    }
  };

  // Create live session
  await supabaseFetch('live_sessions', {
    id: sessionId,
    project_id: projectId,
    session_type: activityType,
    status: 'active',
    total_steps: 0,
    completed_steps: 0,
    started_at: new Date().toISOString(),
  });

  return {
    sessionId,

    async logActivity(eventType: EventType, title: string, description?: string, screenshotUrl?: string, metadata?: Record<string, unknown>) {
      await supabaseFetch('activity_logs', {
        project_id: projectId,
        session_id: sessionId,
        activity_type: activityType,
        event_type: eventType,
        title,
        description,
        screenshot_url: screenshotUrl,
        metadata: metadata || {},
      });
    },

    async updateSession(updates: { currentStep?: string; completedSteps?: number; lastScreenshotUrl?: string; status?: string }) {
      const payload: Record<string, unknown> = {};
      if (updates.currentStep !== undefined) payload.current_step = updates.currentStep;
      if (updates.completedSteps !== undefined) payload.completed_steps = updates.completedSteps;
      if (updates.lastScreenshotUrl !== undefined) payload.last_screenshot_url = updates.lastScreenshotUrl;
      if (updates.status !== undefined) payload.status = updates.status;

      await supabasePatch('live_sessions', sessionId, payload);
    },

    async complete(success: boolean) {
      await supabasePatch('live_sessions', sessionId, {
        status: success ? 'completed' : 'failed',
        completed_at: new Date().toISOString(),
      });

      await supabaseFetch('activity_logs', {
        project_id: projectId,
        session_id: sessionId,
        activity_type: activityType,
        event_type: 'completed',
        title: success ? 'Completed successfully' : 'Failed',
      });
    },
  };
}

// ============================================================================
// RETRY ENDPOINT - Retry failed tests with AI-suggested fixes
// ============================================================================

const RetryRequestSchema = z.object({
  url: SafeUrlSchema,
  failedStep: z.object({
    instruction: z.string(),
    failedAction: z.object({
      selector: z.string(),
      description: z.string(),
      action: z.string(),
      value: z.string().optional(),
    }).optional(),
    healingSuggestions: z.array(z.object({
      selector: z.string(),
      confidence: z.number(),
      reason: z.string(),
    })).optional(),
  }),
  useHealingSuggestion: z.number().optional().default(0), // Index of suggestion to use (0 = highest confidence)
  timeout: z.number().optional().default(30000),
  backend: BackendSchema,
  device: DeviceSchema,
  projectId: z.string().uuid().optional(),
});

async function handleRetry(request: Request, env: Env, corsHeaders: Record<string, string>): Promise<Response> {
  const body = await request.json();
  const parsed = RetryRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "Invalid request", details: parsed.error.message }, { status: 400, headers: corsHeaders });
  }

  const { url, failedStep, useHealingSuggestion, timeout, backend, device, projectId } = parsed.data;

  // Create activity logger for retry
  const logger = await createActivityLogger(env, projectId, 'test_run');

  let session: BrowserSession | null = null;

  const executeWithTimeout = async (): Promise<Response> => {
    if (logger) {
      await logger.logActivity('started', 'Retrying failed step', `Using AI healing suggestion #${useHealingSuggestion + 1}`);
    }

    session = await connectToBrowser(env, {
      backend: backend as Backend,
      browser: 'chrome' as BrowserType,
      device: device as DeviceType,
      platform: 'windows',
      timeout,
    });

    await session.navigate(url);

    if (logger) {
      await logger.logActivity('step', 'Page loaded', url);
    }

    // Determine which selector to use
    let selectorToUse: string;
    let actionToUse = failedStep.failedAction?.action || 'click';
    let valueToUse = failedStep.failedAction?.value;
    let descriptionToUse = failedStep.failedAction?.description || failedStep.instruction;

    if (failedStep.healingSuggestions && failedStep.healingSuggestions.length > 0) {
      const suggestionIndex = Math.min(useHealingSuggestion, failedStep.healingSuggestions.length - 1);
      const suggestion = failedStep.healingSuggestions[suggestionIndex];
      selectorToUse = suggestion.selector;

      if (logger) {
        await logger.logActivity(
          'thinking',
          'Using AI suggestion',
          `Selector: ${selectorToUse} (confidence: ${Math.round(suggestion.confidence * 100)}%) - ${suggestion.reason}`
        );
      }
    } else if (failedStep.failedAction?.selector) {
      // No healing suggestions, try re-interpreting the action from fresh page state
      if (logger) {
        await logger.logActivity('thinking', 'Re-analyzing page', 'No healing suggestions, analyzing current page state');
      }

      const pageHtml = await session.getContent();
      const elements = await session.getInteractiveElements();
      const elementsStr = elements.map(e => `${e.tag}: "${e.text}" [${e.selector}]`).join('\n');

      const interpreted = await interpretAction(env, failedStep.instruction, pageHtml, elementsStr);
      if (interpreted) {
        selectorToUse = interpreted.selector;
        actionToUse = interpreted.action;
        valueToUse = interpreted.value;
        descriptionToUse = interpreted.description;

        if (logger) {
          await logger.logActivity('thinking', 'Found new action', `${actionToUse}: ${selectorToUse}`);
        }
      } else {
        selectorToUse = failedStep.failedAction.selector;
      }
    } else {
      await session.close();
      return Response.json({
        success: false,
        error: 'No selector or healing suggestions provided',
        errorDetails: {
          category: 'element',
          isRetryable: false,
          suggestedAction: 'Please provide healing suggestions or original action details',
        },
      }, { status: 400, headers: corsHeaders });
    }

    if (logger) {
      await logger.logActivity('action', `Executing: ${descriptionToUse}`, `Selector: ${selectorToUse}`);
    }

    // Execute the action with self-healing enabled
    const result = await executeAction(
      session,
      { action: actionToUse, selector: selectorToUse, value: valueToUse, description: descriptionToUse },
      true, // Enable self-healing
      env
    );

    let screenshot: string | undefined;
    try {
      screenshot = await session.screenshot();
      if (logger) {
        await logger.updateSession({ lastScreenshotUrl: `data:image/png;base64,${screenshot}` });
      }
    } catch {}

    // If still failing, generate new healing suggestions
    let newHealingSuggestions: HealingSuggestion[] | undefined;
    if (!result.success) {
      try {
        const healing = await generateHealingSuggestions(
          env,
          session,
          { selector: selectorToUse, description: descriptionToUse },
          result.error || 'Unknown error'
        );
        newHealingSuggestions = healing.suggestions;
      } catch {}
    }

    const backendUsed = session.backendUsed;
    await session.close();

    if (logger) {
      await logger.complete(result.success);
    }

    const categorized = result.success ? null : categorizeError(result.error || 'Unknown');

    return Response.json({
      success: result.success,
      message: result.success
        ? `Retry successful with healed selector: ${selectorToUse}`
        : `Retry failed: ${result.error}`,
      usedSelector: selectorToUse,
      healed: result.healed,
      healingMethod: result.healingMethod,
      screenshot,
      backend: backendUsed,
      sessionId: logger?.sessionId,
      // Include new suggestions if still failing
      ...(newHealingSuggestions && newHealingSuggestions.length > 0 && {
        newHealingSuggestions,
        errorDetails: categorized ? {
          category: categorized.category,
          originalError: categorized.originalError,
          isRetryable: categorized.isRetryable || newHealingSuggestions.length > 0,
          suggestedAction: newHealingSuggestions.length > 0
            ? `Try one of ${newHealingSuggestions.length} new AI suggestions`
            : categorized.suggestedAction,
        } : undefined,
      }),
    }, { headers: corsHeaders });
  };

  try {
    return await withTimeout(executeWithTimeout(), timeout, 'Retry operation');
  } catch (error) {
    if (session) await session.close().catch(() => {});
    const message = error instanceof Error ? error.message : 'Unknown';
    const categorized = categorizeError(message);

    if (logger) {
      await logger.logActivity('error', categorized.message, categorized.originalError);
      await logger.complete(false);
    }

    return Response.json({
      success: false,
      error: categorized.message,
      errorDetails: {
        category: categorized.category,
        originalError: categorized.originalError,
        isRetryable: categorized.isRetryable,
        suggestedAction: categorized.suggestedAction,
      },
      timedOut: categorized.category === 'timeout',
      sessionId: logger?.sessionId,
    }, { status: categorized.category === 'timeout' ? 408 : 500, headers: corsHeaders });
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

// ============================================================================
// CORS CONFIGURATION
// ============================================================================

/**
 * Allowed origins for CORS. Restricts to known Argus domains.
 */
const ALLOWED_ORIGINS = [
  'https://heyargus.ai',
  'https://www.heyargus.ai',
  'https://app.heyargus.ai',
  'https://dashboard.heyargus.ai',
  'https://argus-dashboard.vercel.app',
];

// Development origins (only allowed when not in production)
const DEV_ORIGINS = [
  'http://localhost:3000',
  'http://localhost:3001',
  'http://127.0.0.1:3000',
];

/**
 * Get CORS headers based on request origin.
 * Returns specific origin if allowed, otherwise the first allowed origin.
 */
function getCorsHeaders(request: Request, env: Env): Record<string, string> {
  const origin = request.headers.get('Origin') || '';
  
  // Check if origin is in allowed list
  let allowedOrigins = [...ALLOWED_ORIGINS];
  
  // Add dev origins if not in production
  const isProduction = env.DEFAULT_BACKEND !== 'cloudflare' || !env.BROWSER_POOL_URL?.includes('localhost');
  if (!isProduction) {
    allowedOrigins = [...allowedOrigins, ...DEV_ORIGINS];
  }
  
  const allowedOrigin = allowedOrigins.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  
  return {
    "Access-Control-Allow-Origin": allowedOrigin,
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Max-Age": "86400",
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    const corsHeaders = getCorsHeaders(request, env);

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
          "GET /screenshots/:artifact_id?sig=SIG&exp=EXP": "Serve screenshots (signed URL required when MEDIA_SIGNING_SECRET set)",
          "GET /videos/:artifact_id?sig=SIG&exp=EXP": "Serve videos (signed URL required when MEDIA_SIGNING_SECRET set)",
          "GET /storage/*": "Serve other artifacts from R2 (requires auth)",
        },
        backends: ["cloudflare (free, Chromium)", "testingbot (paid, all browsers + devices)"],
        devices: Object.keys(DEVICE_PRESETS),
        note: "For intelligence features (webhooks, test generation, quality scoring), use the Brain service.",
      }, { headers: corsHeaders });
    }

    // =========================================================================
    // KEY VAULT - BYOK Encryption Service
    // Routes: /key-vault/encrypt, /key-vault/decrypt, /key-vault/dek/*, /key-vault/health
    // Provides envelope encryption for secure API key storage
    // =========================================================================
    if (path.startsWith("/key-vault/")) {
      // Verify internal auth (only backend should call this)
      const authHeader = request.headers.get("Authorization");
      const expectedToken = env.API_TOKEN;

      if (expectedToken && authHeader !== `Bearer ${expectedToken}`) {
        return Response.json(
          { error: "Unauthorized - Key Vault requires API token" },
          { status: 401, headers: corsHeaders }
        );
      }

      const response = await handleKeyVaultRequest(request, env as KeyVaultEnv);
      // Add CORS headers to response
      const responseHeaders = new Headers(response.headers);
      Object.entries(corsHeaders).forEach(([key, value]) => {
        responseHeaders.set(key, value);
      });
      return new Response(response.body, {
        status: response.status,
        headers: responseHeaders,
      });
    }

    // =========================================================================
    // SIGNED SCREENSHOT ACCESS
    // Route: /screenshots/:artifact_id?sig=HMAC&exp=TIMESTAMP
    // Requires valid HMAC signature when MEDIA_SIGNING_SECRET is configured
    // Falls back to public access during rollout (when secret not set)
    // =========================================================================
    if (path.startsWith("/screenshots/") && request.method === "GET") {
      const artifactId = path.replace("/screenshots/", "");
      if (!artifactId) {
        return Response.json({ error: "Missing artifact ID" }, { status: 400, headers: corsHeaders });
      }

      // Validate artifact ID format (alphanumeric, underscore, hyphen only)
      if (!artifactId.match(/^[a-zA-Z0-9_\-]+$/)) {
        return Response.json({ error: "Invalid artifact ID format" }, { status: 400, headers: corsHeaders });
      }

      // Signature verification when MEDIA_SIGNING_SECRET is configured
      if (env.MEDIA_SIGNING_SECRET) {
        const url = new URL(request.url);
        const sig = url.searchParams.get('sig');
        const exp = url.searchParams.get('exp');

        if (!sig || !exp) {
          return Response.json(
            { error: "Missing signature parameters", hint: "URL requires ?sig=SIGNATURE&exp=EXPIRATION" },
            { status: 403, headers: corsHeaders }
          );
        }

        const verification = await verifyMediaSignature(artifactId, sig, exp, env.MEDIA_SIGNING_SECRET);
        if (!verification.valid) {
          debugLog(env, "Screenshot signature verification failed", { artifactId, reason: verification.reason });
          return Response.json(
            { error: verification.reason || "Invalid or expired signature" },
            { status: 403, headers: corsHeaders }
          );
        }
      }

      if (!env.ARTIFACTS) {
        return Response.json({ error: "Storage not configured" }, { status: 503, headers: corsHeaders });
      }

      try {
        // Screenshots are stored with .png extension
        const key = `screenshots/${artifactId}.png`;
        const object = await env.ARTIFACTS.get(key);

        if (!object) {
          return Response.json({ error: "Screenshot not found", artifact_id: artifactId }, { status: 404, headers: corsHeaders });
        }

        const headers = new Headers(corsHeaders);
        headers.set("Content-Type", "image/png");
        // Cache based on signature expiry - max 15 min when signed, 24h when public
        const cacheTime = env.MEDIA_SIGNING_SECRET ? 900 : 86400;
        headers.set("Cache-Control", `public, max-age=${cacheTime}`);
        headers.set("ETag", object.etag);

        return new Response(object.body, { headers });
      } catch (error) {
        console.error("Screenshot fetch error:", error);
        return Response.json({ error: "Failed to fetch screenshot" }, { status: 500, headers: corsHeaders });
      }
    }

    // =========================================================================
    // SIGNED VIDEO ACCESS
    // Route: /videos/:artifact_id?sig=HMAC&exp=TIMESTAMP
    // Same signature verification as screenshots
    // =========================================================================
    if (path.startsWith("/videos/") && request.method === "GET") {
      const artifactId = path.replace("/videos/", "");
      if (!artifactId) {
        return Response.json({ error: "Missing artifact ID" }, { status: 400, headers: corsHeaders });
      }

      // Validate artifact ID format (alphanumeric, underscore, hyphen only)
      if (!artifactId.match(/^[a-zA-Z0-9_\-]+$/)) {
        return Response.json({ error: "Invalid artifact ID format" }, { status: 400, headers: corsHeaders });
      }

      // Signature verification when MEDIA_SIGNING_SECRET is configured
      if (env.MEDIA_SIGNING_SECRET) {
        const url = new URL(request.url);
        const sig = url.searchParams.get('sig');
        const exp = url.searchParams.get('exp');

        if (!sig || !exp) {
          return Response.json(
            { error: "Missing signature parameters", hint: "URL requires ?sig=SIGNATURE&exp=EXPIRATION" },
            { status: 403, headers: corsHeaders }
          );
        }

        const verification = await verifyMediaSignature(artifactId, sig, exp, env.MEDIA_SIGNING_SECRET);
        if (!verification.valid) {
          debugLog(env, "Video signature verification failed", { artifactId, reason: verification.reason });
          return Response.json(
            { error: verification.reason || "Invalid or expired signature" },
            { status: 403, headers: corsHeaders }
          );
        }
      }

      if (!env.ARTIFACTS) {
        return Response.json({ error: "Storage not configured" }, { status: 503, headers: corsHeaders });
      }

      try {
        // Try both .mp4 (Selenium Grid) and .webm (browser recording) extensions
        let object = await env.ARTIFACTS.get(`videos/${artifactId}.mp4`);
        let contentType = "video/mp4";

        if (!object) {
          object = await env.ARTIFACTS.get(`videos/${artifactId}.webm`);
          contentType = "video/webm";
        }

        if (!object) {
          return Response.json({ error: "Video not found", artifact_id: artifactId }, { status: 404, headers: corsHeaders });
        }

        const headers = new Headers(corsHeaders);
        headers.set("Content-Type", contentType);
        // Cache based on signature expiry - max 15 min when signed, 24h when public
        const cacheTime = env.MEDIA_SIGNING_SECRET ? 900 : 86400;
        headers.set("Cache-Control", `public, max-age=${cacheTime}`);
        headers.set("ETag", object.etag);
        // Allow video seeking
        headers.set("Accept-Ranges", "bytes");

        return new Response(object.body, { headers });
      } catch (error) {
        console.error("Video fetch error:", error);
        return Response.json({ error: "Failed to fetch video" }, { status: 500, headers: corsHeaders });
      }
    }

    // =========================================================================
    // AUTHENTICATED STORAGE ACCESS (for other artifacts)
    // Route: /storage/*
    // =========================================================================
    if (path.startsWith("/storage/") && request.method === "GET") {
      // Require authentication for storage access
      if (!authenticate(request, env)) {
        return Response.json({ error: "Unauthorized - authentication required for storage access" }, { status: 401, headers: corsHeaders });
      }

      const key = path.replace("/storage/", "");
      if (!key) {
        return Response.json({ error: "Missing file key" }, { status: 400, headers: corsHeaders });
      }

      // Validate key format to prevent path traversal
      if (key.includes('..') || key.startsWith('/') || !key.match(/^[a-zA-Z0-9\-_\/\.]+$/)) {
        return Response.json({ error: "Invalid file key format" }, { status: 400, headers: corsHeaders });
      }

      if (!env.ARTIFACTS) {
        return Response.json({ error: "Storage not configured" }, { status: 503, headers: corsHeaders });
      }

      try {
        const object = await env.ARTIFACTS.get(key);
        if (!object) {
          return Response.json({ error: "File not found" }, { status: 404, headers: corsHeaders });
        }

        const headers = new Headers(corsHeaders);
        headers.set("Content-Type", object.httpMetadata?.contentType || "application/octet-stream");
        headers.set("Cache-Control", object.httpMetadata?.cacheControl || "public, max-age=31536000");
        headers.set("ETag", object.etag);

        return new Response(object.body, { headers });
      } catch (error) {
        console.error("Storage fetch error:", error);
        return Response.json({ error: "Failed to fetch file" }, { status: 500, headers: corsHeaders });
      }
    }

    // =========================================================================
    // BRAIN SERVICE PROXY - User API endpoints
    // Routes: /api/v1/users/*, /api/v1/chat/*
    // Proxies AI settings, preferences, and chat to Python backend
    // =========================================================================
    if (path.startsWith("/api/v1/users/") || path.startsWith("/api/v1/chat/")) {
      if (!env.BRAIN_URL) {
        return Response.json(
          { error: "Brain service not configured" },
          { status: 503, headers: corsHeaders }
        );
      }

      try {
        // Forward request to Brain service
        const brainUrl = new URL(path, env.BRAIN_URL);
        brainUrl.search = url.search; // Preserve query params

        const headers = new Headers(request.headers);
        headers.set("X-Forwarded-For", request.headers.get("CF-Connecting-IP") || "");
        headers.set("X-Forwarded-Proto", "https");

        const response = await fetch(brainUrl.toString(), {
          method: request.method,
          headers: headers,
          body: request.method !== "GET" && request.method !== "HEAD"
            ? await request.text()
            : undefined,
        });

        // Add CORS headers to response
        const responseHeaders = new Headers(response.headers);
        Object.entries(corsHeaders).forEach(([key, value]) => {
          responseHeaders.set(key, value);
        });

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: responseHeaders,
        });
      } catch (error) {
        console.error("Brain proxy error:", error);
        return Response.json(
          { error: "Failed to connect to Brain service" },
          { status: 502, headers: corsHeaders }
        );
      }
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
          case "/retry": return await handleRetry(request, env, corsHeaders);
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
        debugLog(env, `Processing ${type} event for project ${projectId}`);

        switch (type) {
          case 'webhook_event':
            // Process webhook events - deduplicate and forward to Brain
            if (payload.fingerprint) {
              const isDupe = await cache.isDuplicate(String(payload.fingerprint));
              if (isDupe) {
                debugLog(env, `Duplicate event skipped: ${payload.fingerprint}`);
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
            debugLog(env, `Test request for project ${projectId}`, payload);
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
            debugLog(env, `Error event for project ${projectId}`, payload);
            break;

          default:
            debugWarn(env, `Unknown event type: ${type}`);
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
