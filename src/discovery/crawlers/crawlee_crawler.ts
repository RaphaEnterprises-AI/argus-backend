/**
 * Crawlee Crawler Implementation for Discovery Intelligence Platform
 *
 * This module provides a production-ready web crawler using Crawlee with
 * Playwright for intelligent page discovery, element extraction, and
 * optional Claude Vision AI analysis.
 */

import {
  PlaywrightCrawler,
  createPlaywrightRouter,
  Dataset,
  type PlaywrightCrawlingContext,
  type RouterHandler,
  EnqueueStrategy,
  Configuration,
  LogLevel,
} from 'crawlee';
import type { Page } from 'playwright';
import { extractElements, type ElementExtractionResult } from './element_extractor';
import { analyzeWithClaudeVision, type VisionAnalysisResult } from './page_analyzer';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Configuration for the discovery crawl.
 */
export interface CrawlConfig {
  /** Starting URL for the crawl */
  startUrl: string;
  /** Maximum number of pages to crawl */
  maxPages: number;
  /** Maximum depth from start URL */
  maxDepth: number;
  /** URL patterns to include (glob patterns) */
  includePatterns: string[];
  /** URL patterns to exclude (glob patterns) */
  excludePatterns: string[];
  /** Crawl strategy: breadth-first, depth-first, or priority-based */
  strategy: 'bfs' | 'dfs' | 'priority';
  /** Whether to capture screenshots of each page */
  captureScreenshots: boolean;
  /** Whether to use Claude Vision AI for page analysis */
  useVisionAI: boolean;
  /** Optional authentication configuration */
  authConfig?: AuthConfig;
  /** Optional custom headers */
  customHeaders?: Record<string, string>;
  /** Optional viewport configuration */
  viewport?: { width: number; height: number };
  /** Timeout for page navigation in seconds */
  navigationTimeoutSecs?: number;
  /** Maximum concurrency */
  maxConcurrency?: number;
}

/**
 * Authentication configuration for protected pages.
 */
export interface AuthConfig {
  /** Authentication type */
  type: 'basic' | 'form' | 'cookie' | 'bearer';
  /** Credentials for basic auth or form login */
  credentials?: {
    username: string;
    password: string;
  };
  /** Login page URL for form-based auth */
  loginUrl?: string;
  /** CSS selectors for form fields */
  selectors?: {
    usernameField: string;
    passwordField: string;
    submitButton: string;
  };
  /** Cookie to inject */
  cookies?: Array<{
    name: string;
    value: string;
    domain: string;
    path?: string;
  }>;
  /** Bearer token for API auth */
  bearerToken?: string;
}

/**
 * Represents a discovered page with all extracted information.
 */
export interface DiscoveredPage {
  /** Unique identifier for this page */
  id: string;
  /** Full URL of the page */
  url: string;
  /** Page title */
  title: string;
  /** Depth from start URL */
  depth: number;
  /** Parent page URL (if any) */
  parentUrl?: string;
  /** URLs of child pages linked from this page */
  childUrls: string[];
  /** Timestamp of discovery */
  discoveredAt: string;
  /** Time taken to load the page in milliseconds */
  loadTimeMs: number;
  /** HTTP status code */
  statusCode: number;
  /** Content type of the page */
  contentType: string;
  /** Extracted elements from the page */
  elements: ElementExtractionResult;
  /** Base64 encoded screenshot (if captured) */
  screenshot?: string;
  /** Claude Vision AI analysis result (if enabled) */
  visionAnalysis?: VisionAnalysisResult;
  /** Detected page type */
  pageType: string;
  /** Meta description */
  metaDescription?: string;
  /** Open Graph metadata */
  ogMetadata?: Record<string, string>;
  /** Canonical URL */
  canonicalUrl?: string;
  /** Response headers */
  headers: Record<string, string>;
  /** Any errors encountered during crawl */
  errors: string[];
}

/**
 * Result of a discovery crawl.
 */
export interface CrawlResult {
  /** All discovered pages */
  pages: DiscoveredPage[];
  /** Total number of pages discovered */
  totalPages: number;
  /** Page relationship graph (adjacency list) */
  graph: PageGraph;
  /** Crawl statistics */
  stats: CrawlStats;
}

/**
 * Graph representation of page relationships.
 */
export interface PageGraph {
  /** Adjacency list: URL -> list of linked URLs */
  adjacencyList: Record<string, string[]>;
  /** Root URLs */
  roots: string[];
  /** Leaf URLs (no outgoing links) */
  leaves: string[];
  /** Strongly connected components */
  components?: string[][];
}

/**
 * Statistics about the crawl.
 */
export interface CrawlStats {
  startTime: string;
  endTime: string;
  durationMs: number;
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  totalBytesDownloaded: number;
  averageLoadTimeMs: number;
  maxDepthReached: number;
  uniqueDomains: string[];
}

// ============================================================================
// Logger
// ============================================================================

const logger = {
  info: (message: string, data?: Record<string, unknown>) => {
    const timestamp = new Date().toISOString();
    console.error(JSON.stringify({ level: 'INFO', timestamp, message, ...data }));
  },
  warn: (message: string, data?: Record<string, unknown>) => {
    const timestamp = new Date().toISOString();
    console.error(JSON.stringify({ level: 'WARN', timestamp, message, ...data }));
  },
  error: (message: string, data?: Record<string, unknown>) => {
    const timestamp = new Date().toISOString();
    console.error(JSON.stringify({ level: 'ERROR', timestamp, message, ...data }));
  },
  debug: (message: string, data?: Record<string, unknown>) => {
    if (process.env.DEBUG) {
      const timestamp = new Date().toISOString();
      console.error(JSON.stringify({ level: 'DEBUG', timestamp, message, ...data }));
    }
  },
};

// ============================================================================
// Authentication Setup
// ============================================================================

/**
 * Set up authentication for the crawler.
 */
async function setupAuthentication(
  page: Page,
  authConfig: AuthConfig
): Promise<void> {
  logger.info('Setting up authentication', { type: authConfig.type });

  switch (authConfig.type) {
    case 'basic':
      if (authConfig.credentials) {
        await page.context().setExtraHTTPHeaders({
          Authorization: `Basic ${Buffer.from(
            `${authConfig.credentials.username}:${authConfig.credentials.password}`
          ).toString('base64')}`,
        });
      }
      break;

    case 'form':
      if (authConfig.loginUrl && authConfig.credentials && authConfig.selectors) {
        await page.goto(authConfig.loginUrl, { waitUntil: 'networkidle' });

        await page.fill(
          authConfig.selectors.usernameField,
          authConfig.credentials.username
        );
        await page.fill(
          authConfig.selectors.passwordField,
          authConfig.credentials.password
        );
        await page.click(authConfig.selectors.submitButton);

        // Wait for navigation after login
        await page.waitForLoadState('networkidle');
        logger.info('Form login completed');
      }
      break;

    case 'cookie':
      if (authConfig.cookies) {
        await page.context().addCookies(authConfig.cookies);
        logger.info('Cookies injected', { count: authConfig.cookies.length });
      }
      break;

    case 'bearer':
      if (authConfig.bearerToken) {
        await page.context().setExtraHTTPHeaders({
          Authorization: `Bearer ${authConfig.bearerToken}`,
        });
      }
      break;

    default:
      logger.warn('Unknown authentication type', { type: authConfig.type });
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generate a unique page ID from URL.
 */
function generatePageId(url: string): string {
  const hash = Buffer.from(url).toString('base64').replace(/[^a-zA-Z0-9]/g, '').slice(0, 16);
  return `page_${hash}_${Date.now()}`;
}

/**
 * Map crawl strategy to Crawlee's EnqueueStrategy.
 */
function getEnqueueStrategy(strategy: CrawlConfig['strategy']): EnqueueStrategy {
  switch (strategy) {
    case 'bfs':
      return EnqueueStrategy.SameDomain;
    case 'dfs':
      return EnqueueStrategy.SameDomain;
    case 'priority':
      return EnqueueStrategy.SameDomain;
    default:
      return EnqueueStrategy.SameDomain;
  }
}

/**
 * Extract metadata from the page.
 */
async function extractPageMetadata(page: Page): Promise<{
  metaDescription?: string;
  ogMetadata: Record<string, string>;
  canonicalUrl?: string;
}> {
  return page.evaluate(() => {
    const getMetaContent = (name: string): string | undefined => {
      const meta = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
      return meta?.getAttribute('content') || undefined;
    };

    const ogMetadata: Record<string, string> = {};
    document.querySelectorAll('meta[property^="og:"]').forEach((meta) => {
      const property = meta.getAttribute('property');
      const content = meta.getAttribute('content');
      if (property && content) {
        ogMetadata[property] = content;
      }
    });

    const canonical = document.querySelector('link[rel="canonical"]');

    return {
      metaDescription: getMetaContent('description'),
      ogMetadata,
      canonicalUrl: canonical?.getAttribute('href') || undefined,
    };
  });
}

/**
 * Determine page type from URL and title.
 */
function inferPageType(url: string, title: string): string {
  const urlLower = url.toLowerCase();
  const titleLower = title.toLowerCase();

  const patterns: [RegExp[], string][] = [
    [[/login/, /signin/, /sign-in/], 'login'],
    [[/signup/, /sign-up/, /register/], 'signup'],
    [[/dashboard/, /admin/], 'dashboard'],
    [[/settings/, /preferences/], 'settings'],
    [[/profile/, /account/], 'profile'],
    [[/search/, /results/], 'search'],
    [[/checkout/, /cart/, /basket/], 'checkout'],
    [[/blog/, /article/, /post/], 'blog'],
    [[/docs/, /documentation/, /api/, /reference/], 'documentation'],
    [[/error/, /404/, /500/, /not-found/], 'error'],
    [[/product/, /item/], 'detail'],
    [[/products/, /catalog/, /list/], 'listing'],
  ];

  for (const [regexes, type] of patterns) {
    if (regexes.some((re) => re.test(urlLower) || re.test(titleLower))) {
      return type;
    }
  }

  // Check if it's a landing page (root or home)
  const urlObj = new URL(url);
  if (urlObj.pathname === '/' || urlObj.pathname === '/home') {
    return 'landing';
  }

  return 'other';
}

// ============================================================================
// Graph Building
// ============================================================================

/**
 * Build graph relationships from the page map.
 */
export function buildGraphRelationships(
  pageGraph: Map<string, DiscoveredPage>
): PageGraph {
  const adjacencyList: Record<string, string[]> = {};
  const inDegree: Record<string, number> = {};

  // Initialize adjacency list and in-degree count
  for (const [url, page] of pageGraph) {
    adjacencyList[url] = page.childUrls.filter((childUrl) => pageGraph.has(childUrl));
    inDegree[url] = inDegree[url] || 0;

    for (const childUrl of adjacencyList[url]) {
      inDegree[childUrl] = (inDegree[childUrl] || 0) + 1;
    }
  }

  // Find roots (no incoming links) and leaves (no outgoing links)
  const roots = Object.keys(adjacencyList).filter(
    (url) => !inDegree[url] || inDegree[url] === 0
  );

  const leaves = Object.keys(adjacencyList).filter(
    (url) => adjacencyList[url].length === 0
  );

  return {
    adjacencyList,
    roots,
    leaves,
  };
}

// ============================================================================
// Main Crawler Function
// ============================================================================

/**
 * Run a discovery crawl with the specified configuration.
 *
 * @param config - Crawl configuration
 * @returns Crawl result with discovered pages and graph
 */
export async function runDiscoveryCrawl(config: CrawlConfig): Promise<CrawlResult> {
  const startTime = new Date();
  logger.info('Starting discovery crawl', {
    startUrl: config.startUrl,
    maxPages: config.maxPages,
    maxDepth: config.maxDepth,
    strategy: config.strategy,
  });

  // Initialize page tracking
  const pageGraph = new Map<string, DiscoveredPage>();
  let totalBytesDownloaded = 0;
  let successfulRequests = 0;
  let failedRequests = 0;
  let maxDepthReached = 0;
  const uniqueDomains = new Set<string>();
  let authSetupDone = false;

  // Configure Crawlee
  Configuration.getGlobalConfig().set('logLevel', LogLevel.WARNING);

  // Create router
  const router = createPlaywrightRouter();

  // Default handler for all pages
  router.addDefaultHandler(async (context: PlaywrightCrawlingContext) => {
    const { page, request, enqueueLinks, log } = context;
    const url = request.loadedUrl || request.url;
    const depth = (request.userData?.depth as number) || 0;
    const parentUrl = request.userData?.parentUrl as string | undefined;
    const crawlStartTime = Date.now();
    const errors: string[] = [];

    logger.debug('Processing page', { url, depth });

    try {
      // Track domain
      const urlObj = new URL(url);
      uniqueDomains.add(urlObj.hostname);

      // Update max depth
      if (depth > maxDepthReached) {
        maxDepthReached = depth;
      }

      // Set up authentication on first request if needed
      if (config.authConfig && !authSetupDone) {
        await setupAuthentication(page, config.authConfig);
        authSetupDone = true;
      }

      // Wait for page to be fully loaded
      await page.waitForLoadState('networkidle').catch(() => {
        errors.push('Network idle timeout');
      });

      // Capture screenshot if enabled
      let screenshot: string | undefined;
      if (config.captureScreenshots) {
        try {
          const screenshotBuffer = await page.screenshot({
            fullPage: true,
            type: 'png',
          });
          screenshot = screenshotBuffer.toString('base64');
        } catch (err) {
          errors.push(`Screenshot failed: ${err}`);
          logger.warn('Failed to capture screenshot', { url, error: String(err) });
        }
      }

      // Extract page title
      const title = await page.title().catch(() => 'Untitled');

      // Extract page metadata
      const metadata = await extractPageMetadata(page).catch(() => ({
        metaDescription: undefined,
        ogMetadata: {},
        canonicalUrl: undefined,
      }));

      // Extract elements
      let elements: ElementExtractionResult = {
        interactive: [],
        forms: [],
        navigation: [],
        inputs: [],
        buttons: [],
        links: [],
        images: [],
        headings: [],
        landmarks: [],
      };

      try {
        elements = await extractElements(page);
      } catch (err) {
        errors.push(`Element extraction failed: ${err}`);
        logger.warn('Failed to extract elements', { url, error: String(err) });
      }

      // Run Claude Vision AI analysis if enabled
      let visionAnalysis: VisionAnalysisResult | undefined;
      if (config.useVisionAI && screenshot) {
        try {
          visionAnalysis = await analyzeWithClaudeVision(
            screenshot,
            url,
            elements
          );
        } catch (err) {
          errors.push(`Vision analysis failed: ${err}`);
          logger.warn('Failed to analyze with Vision AI', { url, error: String(err) });
        }
      }

      // Determine page type
      const pageType = visionAnalysis?.pageType || inferPageType(url, title);

      // Get response headers (from the request context if available)
      const headers: Record<string, string> = {};
      // Note: Crawlee doesn't directly expose response headers in context
      // We'd need to intercept them via page.on('response') for full access

      // Build discovered page object
      const loadTimeMs = Date.now() - crawlStartTime;
      const discoveredPage: DiscoveredPage = {
        id: generatePageId(url),
        url,
        title,
        depth,
        parentUrl,
        childUrls: [], // Will be populated by enqueueLinks
        discoveredAt: new Date().toISOString(),
        loadTimeMs,
        statusCode: 200, // Assume success if we got here
        contentType: 'text/html',
        elements,
        screenshot,
        visionAnalysis,
        pageType,
        metaDescription: metadata.metaDescription,
        ogMetadata: metadata.ogMetadata,
        canonicalUrl: metadata.canonicalUrl,
        headers,
        errors,
      };

      // Save to Dataset
      await Dataset.pushData(discoveredPage);

      // Add to page graph
      pageGraph.set(url, discoveredPage);
      successfulRequests++;

      logger.info('Page processed', {
        url,
        depth,
        title,
        pageType,
        elementsFound: elements.interactive.length,
        loadTimeMs,
      });

      // Enqueue links if not at max depth
      if (depth < config.maxDepth) {
        const enqueueOptions: Parameters<typeof enqueueLinks>[0] = {
          strategy: getEnqueueStrategy(config.strategy),
          userData: {
            depth: depth + 1,
            parentUrl: url,
          },
          transformRequestFunction: (req) => {
            // Track child URLs
            discoveredPage.childUrls.push(req.url);
            return req;
          },
        };

        // Add include patterns (globs)
        if (config.includePatterns.length > 0) {
          enqueueOptions.globs = config.includePatterns;
        }

        // Add exclude patterns
        if (config.excludePatterns.length > 0) {
          enqueueOptions.exclude = config.excludePatterns;
        }

        await enqueueLinks(enqueueOptions);
      }

    } catch (err) {
      failedRequests++;
      logger.error('Failed to process page', { url, error: String(err) });
      throw err;
    }
  });

  // Create the crawler
  const crawler = new PlaywrightCrawler({
    requestHandler: router,
    maxRequestsPerCrawl: config.maxPages,
    maxConcurrency: config.maxConcurrency ?? 5,
    requestHandlerTimeoutSecs: config.navigationTimeoutSecs ?? 60,
    navigationTimeoutSecs: config.navigationTimeoutSecs ?? 30,
    useSessionPool: true,
    maxRequestRetries: 2,
    launchContext: {
      launchOptions: {
        headless: true,
      },
    },
    browserPoolOptions: {
      useFingerprints: false,
    },
    preNavigationHooks: [
      async ({ page }, gotoOptions) => {
        // Set viewport if specified
        if (config.viewport) {
          await page.setViewportSize(config.viewport);
        }

        // Set custom headers if specified
        if (config.customHeaders) {
          await page.setExtraHTTPHeaders(config.customHeaders);
        }

        // Disable service workers to avoid caching issues
        await page.context().route('**/*', (route) => {
          const url = route.request().url();
          if (url.includes('service-worker') || url.includes('sw.js')) {
            return route.abort();
          }
          return route.continue();
        });
      },
    ],
  });

  // Run the crawler
  try {
    await crawler.run([
      {
        url: config.startUrl,
        userData: {
          depth: 0,
          parentUrl: undefined,
        },
      },
    ]);
  } catch (err) {
    logger.error('Crawler run failed', { error: String(err) });
  }

  const endTime = new Date();
  const durationMs = endTime.getTime() - startTime.getTime();

  // Build graph relationships
  const graph = buildGraphRelationships(pageGraph);

  // Calculate statistics
  const pages = Array.from(pageGraph.values());
  const averageLoadTimeMs =
    pages.length > 0
      ? pages.reduce((sum, p) => sum + p.loadTimeMs, 0) / pages.length
      : 0;

  const stats: CrawlStats = {
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    durationMs,
    totalRequests: successfulRequests + failedRequests,
    successfulRequests,
    failedRequests,
    totalBytesDownloaded,
    averageLoadTimeMs,
    maxDepthReached,
    uniqueDomains: Array.from(uniqueDomains),
  };

  logger.info('Crawl completed', {
    totalPages: pages.length,
    durationMs,
    successfulRequests,
    failedRequests,
  });

  return {
    pages,
    totalPages: pages.length,
    graph,
    stats,
  };
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Main entry point when running as a standalone script.
 * Reads configuration from command line arguments and outputs JSON to stdout.
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error('Usage: ts-node crawlee_crawler.ts <config-json>');
    console.error('');
    console.error('Config JSON format:');
    console.error(JSON.stringify({
      startUrl: 'https://example.com',
      maxPages: 100,
      maxDepth: 3,
      includePatterns: ['**/*'],
      excludePatterns: [],
      strategy: 'bfs',
      captureScreenshots: false,
      useVisionAI: false,
    }, null, 2));
    process.exit(1);
  }

  try {
    // Parse config from command line argument
    const configArg = args[0];
    let config: CrawlConfig;

    try {
      config = JSON.parse(configArg);
    } catch {
      // Try reading as a file path
      const fs = await import('fs');
      const configContent = fs.readFileSync(configArg, 'utf-8');
      config = JSON.parse(configContent);
    }

    // Validate required fields
    if (!config.startUrl) {
      throw new Error('startUrl is required');
    }

    // Set defaults
    config.maxPages = config.maxPages ?? 100;
    config.maxDepth = config.maxDepth ?? 3;
    config.includePatterns = config.includePatterns ?? [];
    config.excludePatterns = config.excludePatterns ?? [];
    config.strategy = config.strategy ?? 'bfs';
    config.captureScreenshots = config.captureScreenshots ?? false;
    config.useVisionAI = config.useVisionAI ?? false;

    logger.info('Starting crawl with config', { config });

    // Run the crawl
    const result = await runDiscoveryCrawl(config);

    // Output result as JSON to stdout
    console.log(JSON.stringify(result, null, 2));

  } catch (error) {
    logger.error('Fatal error', { error: String(error) });
    console.error(JSON.stringify({
      error: true,
      message: String(error),
    }));
    process.exit(1);
  }
}

// Run main if this is the entry point
if (require.main === module) {
  main().catch((err) => {
    console.error('Unhandled error:', err);
    process.exit(1);
  });
}

// Export for use as a module
export default {
  runDiscoveryCrawl,
  buildGraphRelationships,
};
