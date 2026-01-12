/**
 * Page Analyzer with Claude Vision Integration for Crawlee Discovery
 *
 * Uses Claude's vision capabilities to analyze page screenshots and extract
 * semantic information about UI elements, user flows, and testable surfaces.
 */

import Anthropic from "@anthropic-ai/sdk";
import * as crypto from "crypto";

// ============================================================================
// Types and Interfaces
// ============================================================================

/** Page categories for classification */
export type PageCategory =
  | "landing"
  | "auth_login"
  | "auth_signup"
  | "auth_reset"
  | "dashboard"
  | "settings"
  | "profile"
  | "list"
  | "detail"
  | "form"
  | "checkout"
  | "cart"
  | "search"
  | "error"
  | "help"
  | "about"
  | "contact"
  | "other";

/** Priority levels for flows and tests */
export type FlowPriority = "critical" | "high" | "medium" | "low";

/** Analysis result for a page */
export interface PageAnalysis {
  category: PageCategory;
  description: string;
  purpose: string;
  mainActions: string[];
  userFlows: UserFlow[];
  accessibilityNotes: string[];
  risks: string[];
}

/** A user flow that can be tested */
export interface UserFlow {
  name: string;
  description: string;
  steps: string[];
  priority: FlowPriority;
}

/** Analysis result for a single element */
export interface ElementAnalysis {
  purpose: string;
  importance: number; // 0-1 scale
  category: string;
  suggestedTestActions: string[];
}

/** Extracted element from the page */
export interface ExtractedElement {
  type: string; // button, link, input, form, etc.
  text: string;
  selector: string;
  tag?: string;
  href?: string;
  placeholder?: string;
  name?: string;
  inputType?: string;
  required?: boolean;
  attributes?: Record<string, string>;
}

/** A discovered page */
export interface DiscoveredPage {
  url: string;
  title: string;
  description: string;
  category?: PageCategory;
  elements: ExtractedElement[];
  screenshot?: string; // Base64
  forms: FormInfo[];
  links: string[];
  userFlows: string[];
}

/** Form information */
export interface FormInfo {
  action?: string;
  method?: string;
  inputs: FormInputInfo[];
}

/** Form input information */
export interface FormInputInfo {
  type: string;
  name?: string;
  placeholder?: string;
  required?: boolean;
}

/** A discovered flow across pages */
export interface DiscoveredFlow {
  id: string;
  name: string;
  description: string;
  startUrl: string;
  steps: FlowStep[];
  priority: FlowPriority;
  category: string;
}

/** A step in a discovered flow */
export interface FlowStep {
  page: string;
  action: string;
  target?: string;
  value?: string;
  assertion?: string;
}

// ============================================================================
// Cost Tracking
// ============================================================================

interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
}

interface CostTracker {
  totalInputTokens: number;
  totalOutputTokens: number;
  analysisCount: number;
}

const costTracker: CostTracker = {
  totalInputTokens: 0,
  totalOutputTokens: 0,
  analysisCount: 0,
};

// Pricing per 1M tokens (as of 2024)
const PRICING = {
  "claude-3-5-sonnet-latest": { input: 3.0, output: 15.0 },
  "claude-3-5-sonnet-20241022": { input: 3.0, output: 15.0 },
  "claude-3-haiku-20240307": { input: 0.25, output: 1.25 },
  "claude-sonnet-4-5-20250514": { input: 3.0, output: 15.0 },
};

type ModelId = keyof typeof PRICING;

function trackUsage(usage: TokenUsage): void {
  costTracker.totalInputTokens += usage.inputTokens;
  costTracker.totalOutputTokens += usage.outputTokens;
  costTracker.analysisCount += 1;
}

/**
 * Get the total cost of all analyses performed
 * @param model The model used (defaults to claude-3-5-sonnet)
 * @returns Cost breakdown
 */
export function getAnalysisCost(model: ModelId = "claude-3-5-sonnet-latest"): {
  inputTokens: number;
  outputTokens: number;
  inputCost: number;
  outputCost: number;
  totalCost: number;
  analysisCount: number;
} {
  const pricing = PRICING[model] || PRICING["claude-3-5-sonnet-latest"];
  const inputCost = (costTracker.totalInputTokens / 1_000_000) * pricing.input;
  const outputCost =
    (costTracker.totalOutputTokens / 1_000_000) * pricing.output;

  return {
    inputTokens: costTracker.totalInputTokens,
    outputTokens: costTracker.totalOutputTokens,
    inputCost,
    outputCost,
    totalCost: inputCost + outputCost,
    analysisCount: costTracker.analysisCount,
  };
}

/**
 * Reset the cost tracker
 */
export function resetCostTracker(): void {
  costTracker.totalInputTokens = 0;
  costTracker.totalOutputTokens = 0;
  costTracker.analysisCount = 0;
}

// ============================================================================
// Caching
// ============================================================================

interface CacheEntry<T> {
  value: T;
  timestamp: number;
}

const analysisCache = new Map<string, CacheEntry<PageAnalysis>>();
const elementCache = new Map<string, CacheEntry<ElementAnalysis>>();
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

function generateCacheKey(data: Buffer | string, context?: string): string {
  const hash = crypto.createHash("sha256");
  if (Buffer.isBuffer(data)) {
    hash.update(data);
  } else {
    hash.update(data);
  }
  if (context) {
    hash.update(context);
  }
  return hash.digest("hex").slice(0, 32);
}

function getCached<T>(
  cache: Map<string, CacheEntry<T>>,
  key: string
): T | null {
  const entry = cache.get(key);
  if (!entry) return null;

  const age = Date.now() - entry.timestamp;
  if (age > CACHE_TTL_MS) {
    cache.delete(key);
    return null;
  }

  return entry.value;
}

function setCache<T>(
  cache: Map<string, CacheEntry<T>>,
  key: string,
  value: T
): void {
  cache.set(key, { value, timestamp: Date.now() });
}

/**
 * Clear all caches
 */
export function clearAnalysisCache(): void {
  analysisCache.clear();
  elementCache.clear();
}

// ============================================================================
// API Client
// ============================================================================

let anthropicClient: Anthropic | null = null;

function getClient(): Anthropic {
  if (!anthropicClient) {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error(
        "ANTHROPIC_API_KEY environment variable is not set. " +
          "Please set it to use Claude Vision analysis."
      );
    }
    anthropicClient = new Anthropic({ apiKey });
  }
  return anthropicClient;
}

// ============================================================================
// Retry Logic
// ============================================================================

interface RetryOptions {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxRetries: 3,
  initialDelayMs: 1000,
  maxDelayMs: 30000,
  backoffMultiplier: 2,
};

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withRetry<T>(
  fn: () => Promise<T>,
  options: Partial<RetryOptions> = {}
): Promise<T> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: Error | null = null;
  let delay = opts.initialDelayMs;

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      // Check for rate limit errors
      const isRateLimit =
        error instanceof Anthropic.RateLimitError ||
        (error instanceof Error && error.message.includes("rate limit"));

      // Check for overloaded errors
      const isOverloaded =
        error instanceof Anthropic.APIError &&
        (error as InstanceType<typeof Anthropic.APIError>).status === 529;

      if (!isRateLimit && !isOverloaded) {
        throw error;
      }

      if (attempt < opts.maxRetries) {
        console.warn(
          `API rate limited or overloaded, retrying in ${delay}ms (attempt ${attempt + 1}/${opts.maxRetries})`
        );
        await sleep(delay);
        delay = Math.min(delay * opts.backoffMultiplier, opts.maxDelayMs);
      }
    }
  }

  throw lastError || new Error("Max retries exceeded");
}

// ============================================================================
// JSON Parsing
// ============================================================================

function robustJsonParse<T>(content: string): T | null {
  // Extract JSON from markdown code blocks
  let jsonStr = content;

  if (jsonStr.includes("```json")) {
    jsonStr = jsonStr.split("```json")[1].split("```")[0];
  } else if (jsonStr.includes("```")) {
    jsonStr = jsonStr.split("```")[1].split("```")[0];
  }

  jsonStr = jsonStr.trim();

  // Try direct parse
  try {
    return JSON.parse(jsonStr) as T;
  } catch {
    // Continue to cleanup
  }

  // Remove trailing commas
  jsonStr = jsonStr.replace(/,\s*([}\]])/g, "$1");

  // Remove comments
  jsonStr = jsonStr.replace(/\/\/.*$/gm, "");
  jsonStr = jsonStr.replace(/\/\*[\s\S]*?\*\//g, "");

  try {
    return JSON.parse(jsonStr) as T;
  } catch {
    // Continue to extract
  }

  // Try to find JSON object
  const start = jsonStr.indexOf("{");
  const end = jsonStr.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    try {
      return JSON.parse(jsonStr.slice(start, end + 1)) as T;
    } catch {
      // Failed
    }
  }

  console.warn(
    "Failed to parse JSON from response:",
    content.slice(0, 200) + "..."
  );
  return null;
}

// ============================================================================
// Main Analysis Functions
// ============================================================================

/**
 * Analyze a page screenshot with Claude Vision to understand its purpose and structure
 *
 * @param screenshot - Screenshot buffer (PNG/JPEG)
 * @param elements - Extracted elements from the page
 * @param url - The page URL
 * @returns Page analysis with category, flows, and risks
 */
export async function analyzeWithClaudeVision(
  screenshot: Buffer,
  elements: ExtractedElement[],
  url: string
): Promise<PageAnalysis> {
  const cacheKey = generateCacheKey(screenshot, url);
  const cached = getCached(analysisCache, cacheKey);
  if (cached) {
    return cached;
  }

  const client = getClient();

  // Summarize elements for the prompt
  const elementsSummary = summarizeElements(elements);

  const prompt = `Analyze this web page screenshot and provide a comprehensive analysis.

URL: ${url}

EXTRACTED ELEMENTS:
${elementsSummary}

Provide your analysis as JSON with the following structure:
{
  "category": "<one of: landing, auth_login, auth_signup, auth_reset, dashboard, settings, profile, list, detail, form, checkout, cart, search, error, help, about, contact, other>",
  "description": "<brief description of what this page is>",
  "purpose": "<the main purpose of this page from a user perspective>",
  "mainActions": ["<list of main actions a user can take on this page>"],
  "userFlows": [
    {
      "name": "<flow name>",
      "description": "<what the user accomplishes>",
      "steps": ["<step 1>", "<step 2>", "..."],
      "priority": "<critical|high|medium|low>"
    }
  ],
  "accessibilityNotes": ["<any accessibility concerns or missing attributes>"],
  "risks": ["<potential risks for testing - flaky selectors, dynamic content, etc.>"]
}

Focus on:
1. Categorizing the page type accurately
2. Identifying user flows that start, pass through, or end on this page
3. Noting any accessibility issues (missing labels, poor contrast, etc.)
4. Identifying risks that could cause test flakiness`;

  const screenshotBase64 = screenshot.toString("base64");
  const mediaType = detectMediaType(screenshot);

  const response = await withRetry(async () => {
    return client.messages.create({
      model: "claude-sonnet-4-5-20250514",
      max_tokens: 2000,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType,
                data: screenshotBase64,
              },
            },
            {
              type: "text",
              text: prompt,
            },
          ],
        },
      ],
    });
  });

  // Track token usage
  if (response.usage) {
    trackUsage({
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    });
  }

  const content =
    response.content[0].type === "text" ? response.content[0].text : "";

  const parsed = robustJsonParse<PageAnalysis>(content);

  const result: PageAnalysis = parsed || {
    category: "other",
    description: "Unable to analyze page",
    purpose: "Unknown",
    mainActions: [],
    userFlows: [],
    accessibilityNotes: [],
    risks: ["Analysis failed - manual review recommended"],
  };

  // Cache the result
  setCache(analysisCache, cacheKey, result);

  return result;
}

/**
 * Analyze a single element with Claude to understand its purpose
 *
 * @param screenshot - Screenshot buffer showing the element
 * @param element - The extracted element information
 * @param pageContext - Context about the page (description, URL, etc.)
 * @returns Element analysis with purpose and suggested test actions
 */
export async function analyzeElement(
  screenshot: Buffer,
  element: ExtractedElement,
  pageContext: string
): Promise<ElementAnalysis> {
  const cacheKey = generateCacheKey(
    screenshot,
    JSON.stringify(element) + pageContext
  );
  const cached = getCached(elementCache, cacheKey);
  if (cached) {
    return cached;
  }

  const client = getClient();

  const prompt = `Analyze this UI element shown in the screenshot.

ELEMENT DETAILS:
- Type: ${element.type}
- Text: ${element.text || "N/A"}
- Selector: ${element.selector || "N/A"}
- Tag: ${element.tag || "N/A"}
${element.placeholder ? `- Placeholder: ${element.placeholder}` : ""}
${element.href ? `- Href: ${element.href}` : ""}

PAGE CONTEXT:
${pageContext}

Provide your analysis as JSON:
{
  "purpose": "<what this element is for>",
  "importance": <0.0-1.0 importance score>,
  "category": "<navigation|action|input|display|feedback>",
  "suggestedTestActions": ["<specific test actions to perform on this element>"]
}

Consider:
1. How critical is this element to user journeys?
2. What test scenarios should include this element?
3. What interactions should be tested?`;

  const screenshotBase64 = screenshot.toString("base64");
  const mediaType = detectMediaType(screenshot);

  const response = await withRetry(async () => {
    return client.messages.create({
      model: "claude-3-haiku-20240307", // Use Haiku for cost efficiency on element analysis
      max_tokens: 500,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType,
                data: screenshotBase64,
              },
            },
            {
              type: "text",
              text: prompt,
            },
          ],
        },
      ],
    });
  });

  // Track token usage
  if (response.usage) {
    trackUsage({
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    });
  }

  const content =
    response.content[0].type === "text" ? response.content[0].text : "";

  const parsed = robustJsonParse<ElementAnalysis>(content);

  const result: ElementAnalysis = parsed || {
    purpose: "Unknown",
    importance: 0.5,
    category: "action",
    suggestedTestActions: ["click", "verify visible"],
  };

  // Cache the result
  setCache(elementCache, cacheKey, result);

  return result;
}

/**
 * Infer user flows by analyzing multiple pages together
 *
 * @param pages - Array of discovered pages
 * @param elements - Array of all extracted elements across pages
 * @returns Array of discovered flows spanning multiple pages
 */
export async function inferFlows(
  pages: DiscoveredPage[],
  elements: ExtractedElement[]
): Promise<DiscoveredFlow[]> {
  if (pages.length === 0) {
    return [];
  }

  const client = getClient();

  // Create a summary of pages for analysis
  const pagesSummary = pages.map((page) => ({
    url: page.url,
    title: page.title,
    description: page.description,
    category: page.category || "unknown",
    formCount: page.forms.length,
    linkCount: page.links.length,
    elementTypes: [...new Set(page.elements.map((e) => e.type))],
    userFlows: page.userFlows,
  }));

  // Group elements by type for summary
  const elementsByType: Record<string, number> = {};
  elements.forEach((el) => {
    elementsByType[el.type] = (elementsByType[el.type] || 0) + 1;
  });

  const prompt = `Analyze these discovered pages and identify complete user flows/journeys.

DISCOVERED PAGES:
${JSON.stringify(pagesSummary, null, 2)}

ELEMENT SUMMARY:
${JSON.stringify(elementsByType, null, 2)}

Identify user flows that span multiple pages. For each flow:
1. Give it a clear, descriptive name
2. Explain what the user accomplishes
3. List the specific steps across pages
4. Assign a priority based on business importance

Provide your analysis as JSON:
{
  "flows": [
    {
      "id": "<unique-id>",
      "name": "<flow name>",
      "description": "<what user accomplishes>",
      "startUrl": "<starting page URL>",
      "category": "<authentication|checkout|profile|navigation|form_submission|search|other>",
      "priority": "<critical|high|medium|low>",
      "steps": [
        {
          "page": "<page URL>",
          "action": "<what user does>",
          "target": "<element selector or description if applicable>",
          "assertion": "<what to verify>"
        }
      ]
    }
  ]
}

Prioritization guide:
- Critical: Authentication, checkout, core business features
- High: User profile, settings that affect experience
- Medium: Secondary features, nice-to-have flows
- Low: Edge cases, rarely used features`;

  const response = await withRetry(async () => {
    return client.messages.create({
      model: "claude-sonnet-4-5-20250514",
      max_tokens: 4000,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    });
  });

  // Track token usage
  if (response.usage) {
    trackUsage({
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    });
  }

  const content =
    response.content[0].type === "text" ? response.content[0].text : "";

  interface FlowsResponse {
    flows: Array<{
      id?: string;
      name?: string;
      description?: string;
      startUrl?: string;
      category?: string;
      priority?: string;
      steps?: FlowStep[];
    }>;
  }

  const parsed = robustJsonParse<FlowsResponse>(content);

  if (!parsed || !parsed.flows) {
    return [];
  }

  return parsed.flows.map((flow, index) => ({
    id: flow.id || `flow-${index + 1}`,
    name: flow.name || "Unnamed Flow",
    description: flow.description || "",
    startUrl: flow.startUrl || pages[0]?.url || "/",
    category: flow.category || "other",
    priority: (flow.priority as FlowPriority) || "medium",
    steps: flow.steps || [],
  }));
}

// ============================================================================
// Helper Functions
// ============================================================================

function summarizeElements(elements: ExtractedElement[]): string {
  const grouped: Record<string, ExtractedElement[]> = {};

  elements.forEach((el) => {
    const type = el.type;
    if (!grouped[type]) {
      grouped[type] = [];
    }
    grouped[type].push(el);
  });

  const lines: string[] = [];

  for (const [type, els] of Object.entries(grouped)) {
    lines.push(`\n${type.toUpperCase()}S (${els.length}):`);
    els.slice(0, 10).forEach((el) => {
      const parts = [`  - ${el.text || el.placeholder || el.name || "unnamed"}`];
      if (el.selector) parts.push(`[${el.selector}]`);
      if (el.href) parts.push(`-> ${el.href}`);
      lines.push(parts.join(" "));
    });
    if (els.length > 10) {
      lines.push(`  ... and ${els.length - 10} more`);
    }
  }

  return lines.join("\n");
}

function detectMediaType(buffer: Buffer): "image/png" | "image/jpeg" | "image/webp" | "image/gif" {
  // Check PNG signature
  if (
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47
  ) {
    return "image/png";
  }

  // Check JPEG signature
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return "image/jpeg";
  }

  // Check WebP signature
  if (
    buffer[0] === 0x52 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46 &&
    buffer[3] === 0x46
  ) {
    return "image/webp";
  }

  // Check GIF signature
  if (
    buffer[0] === 0x47 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46
  ) {
    return "image/gif";
  }

  // Default to PNG
  return "image/png";
}

// ============================================================================
// Batch Analysis Functions
// ============================================================================

/**
 * Analyze multiple pages in batch for efficiency
 *
 * @param pages - Array of pages with screenshots to analyze
 * @returns Map of URL to PageAnalysis
 */
export async function analyzePagesBatch(
  pages: Array<{ url: string; screenshot: Buffer; elements: ExtractedElement[] }>
): Promise<Map<string, PageAnalysis>> {
  const results = new Map<string, PageAnalysis>();

  // Process in parallel with concurrency limit
  const CONCURRENCY = 3;
  const chunks: Array<typeof pages> = [];

  for (let i = 0; i < pages.length; i += CONCURRENCY) {
    chunks.push(pages.slice(i, i + CONCURRENCY));
  }

  for (const chunk of chunks) {
    const promises = chunk.map(async (page) => {
      const analysis = await analyzeWithClaudeVision(
        page.screenshot,
        page.elements,
        page.url
      );
      return { url: page.url, analysis };
    });

    const chunkResults = await Promise.all(promises);
    chunkResults.forEach(({ url, analysis }) => {
      results.set(url, analysis);
    });
  }

  return results;
}

/**
 * Quick categorization of a page without full analysis
 * Uses a simpler prompt for faster response
 *
 * @param screenshot - Screenshot buffer
 * @param url - Page URL
 * @returns Just the page category
 */
export async function quickCategorize(
  screenshot: Buffer,
  url: string
): Promise<PageCategory> {
  const client = getClient();

  const prompt = `What type of web page is this? Respond with ONLY one of these categories:
landing, auth_login, auth_signup, auth_reset, dashboard, settings, profile, list, detail, form, checkout, cart, search, error, help, about, contact, other

URL: ${url}`;

  const screenshotBase64 = screenshot.toString("base64");
  const mediaType = detectMediaType(screenshot);

  const response = await withRetry(async () => {
    return client.messages.create({
      model: "claude-3-haiku-20240307",
      max_tokens: 50,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType,
                data: screenshotBase64,
              },
            },
            {
              type: "text",
              text: prompt,
            },
          ],
        },
      ],
    });
  });

  if (response.usage) {
    trackUsage({
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    });
  }

  const content =
    response.content[0].type === "text"
      ? response.content[0].text.trim().toLowerCase()
      : "other";

  const validCategories: PageCategory[] = [
    "landing",
    "auth_login",
    "auth_signup",
    "auth_reset",
    "dashboard",
    "settings",
    "profile",
    "list",
    "detail",
    "form",
    "checkout",
    "cart",
    "search",
    "error",
    "help",
    "about",
    "contact",
    "other",
  ];

  return validCategories.includes(content as PageCategory)
    ? (content as PageCategory)
    : "other";
}

// ============================================================================
// Default Export
// ============================================================================

export default {
  analyzeWithClaudeVision,
  analyzeElement,
  inferFlows,
  getAnalysisCost,
  resetCostTracker,
  clearAnalysisCache,
  analyzePagesBatch,
  quickCategorize,
};
